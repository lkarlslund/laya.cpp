#!/usr/bin/env python3
"""Resumable, CPU-only issue #7 parity and paired inference measurements."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from cpu_process import CPUProcess, peak_rss_bytes, physical_cpu_ids, thread_environment
from format_benchmark_v1 import format_report
from identity import file_hash, native_hash
from report_v1 import compare_public
from run import percentile

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'benchmarks/cases/performance-issue7-v1/manifest.json'
ACCEPTANCE = ROOT / 'benchmarks/cases/acceptance-250.json'
VARIANTS = ('english', 'multilingual', 'typed-decisions')
SIZES = (1, 2, 4, 8)
HARNESS_FILES = ('cpu_process.py', 'cpu_worker.py', 'run_cpu_issue7.py', 'validate.py',
                 'oracle.py', 'native.py', 'report_v1.py', 'format_benchmark_v1.py',
                 'identity.py', 'run.py', 'schema/validation-v1.schema.json',
                 'schema/benchmark-v1.schema.json', 'schema/issue7-run-v1.schema.json')


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def harness_hash():
    digest = hashlib.sha256()
    for name in HARNESS_FILES:
        digest.update(name.encode())
        digest.update(file_hash(ROOT / 'benchmarks' / name).encode())
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load(path):
    value = json.loads(path.read_text())
    value['_path'] = str(path)
    return value


def accepted(path, *, corpus, weights, binary, variant, threads, baseline_commit, candidate_commit, requests):
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
        identity = report['identity']
        schema = json.loads((ROOT / 'benchmarks/schema/validation-v1.schema.json').read_text())
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
        return (report['status'] == 'passed' and identity['corpus_sha256'] == corpus and
                identity['weights_sha256'] == weights and identity['candidate_sha256'] == binary and
                identity['variant'] == variant and identity['backend'] == 'cpu' and
                identity['precision'] == 'fp32' and identity['baseline_commit'] == baseline_commit and
                identity['candidate_commit'] == candidate_commit and
                identity['baseline_threads'] == threads and identity['candidate_threads'] == threads and
                set(report['acceptance']['batch_sizes']) == set(SIZES) and
                len(report['cases']) == requests * len(SIZES) and
                all(case['passed'] and not case['differences'] for case in report['cases']) and
                all(len({case['request_id'] for case in report['cases'] if case['batch_size'] == size}) == requests
                    for size in SIZES))
    except (KeyError, ValueError, TypeError, OSError, ValidationError):
        return False


def validate(cases, output, *, variant, model, source, executable, threads, identity, env):
    legacy = output.with_name(output.stem.replace('-v1', '') + '-legacy.json')
    if accepted(output, corpus=file_hash(cases), weights=file_hash(model / 'model.safetensors'),
                binary=identity['candidate_sha256'], variant=variant, threads=threads,
                baseline_commit=identity['baseline_commit'], candidate_commit=identity['candidate_commit'],
                requests=len(json.loads(cases.read_text()))):
        print(f'validated: {output}', flush=True)
        return
    command = [sys.executable, str(ROOT / 'benchmarks/validate.py'), '--backend', 'cpu',
               '--executable', str(executable), '--source', str(source), '--model', str(model),
               '--cases', str(cases), '--batch-sizes', *map(str, SIZES), '--threads', str(threads),
               '--output', str(legacy), '--v1-output', str(output),
               '--baseline-commit', identity['baseline_commit'],
               '--candidate-commit', identity['candidate_commit']]
    log = output.with_suffix('.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'validating: {variant} t{threads} {cases.name}', flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'Validation failed; see {log}')
    if not accepted(output, corpus=file_hash(cases), weights=file_hash(model / 'model.safetensors'),
                    binary=identity['candidate_sha256'], variant=variant, threads=threads,
                    baseline_commit=identity['baseline_commit'], candidate_commit=identity['candidate_commit'],
                    requests=len(json.loads(cases.read_text()))):
        raise RuntimeError(f'Validation report is incomplete: {output}')


def prewarm_weights(model):
    with (model / 'model.safetensors').open('rb') as stream:
        while stream.read(8 * 1024 * 1024):
            pass


def cpu_frequency(cpus):
    values, governors = [], set()
    for cpu in cpus:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq')
        try:
            values.append(int((base / 'scaling_cur_freq').read_text().strip()) / 1000)
            governors.add((base / 'scaling_governor').read_text().strip())
        except (OSError, ValueError):
            continue
    return (statistics.median(values) if values else None,
            ','.join(sorted(governors)) if governors else None)


def timed_stratum(cases, *, model, source, executable, threads, cpus, env, native_first=False):
    prewarm_weights(model)
    baseline_command = [sys.executable, str(ROOT / 'benchmarks/cpu_worker.py'),
                        '--source', str(source), '--model', str(model), '--threads', str(threads)]
    native_command = [str(executable), '--model', str(model), '--cpu', '--fp32', '--no-flash']
    cpu_before, policy = cpu_frequency(cpus)
    with ExitStack() as stack:
        def launch(command, ready_stream):
            return stack.enter_context(CPUProcess(command, threads=threads, cpus=cpus,
                                                  ready_stream=ready_stream, env=env))
        if native_first:
            candidate = launch(native_command, 'stderr')
            baseline = launch(baseline_command, 'stdout')
        else:
            baseline = launch(baseline_command, 'stdout')
            candidate = launch(native_command, 'stderr')
        rows = []
        for size in SIZES:
            baseline_times, candidate_times = [], []
            for start in range(0, len(cases), size):
                group = cases[start:start + size]
                expected = baseline.call(group)['results']
                candidate_response = candidate.call(group)
                if candidate_response.get('backend') != 'CPU':
                    raise RuntimeError('Candidate selected a non-CPU backend')
                actual = candidate_response['results']
                if len(expected) != len(actual):
                    raise RuntimeError(f'Candidate result count differs at request {start}')
                for request, left, right in zip(group, expected, actual):
                    _, differences = compare_public(left, right)
                    if differences:
                        raise RuntimeError(f'{request["id"]}: {differences[0]}')
                for _ in range(3):
                    baseline.call(group)
                    candidate.call(group)
                for iteration in range(5):
                    order = (('baseline', baseline), ('native', candidate)) if iteration % 2 == 0 else (
                        ('native', candidate), ('baseline', baseline))
                    for label, process in order:
                        elapsed = process.call(group)['elapsed_ms']
                        if elapsed <= 0:
                            raise RuntimeError('Nonpositive timed inference sample')
                        (baseline_times if label == 'baseline' else candidate_times).append(elapsed)
            def stats(samples):
                questions = sum(len(case['questions']) for case in cases) * 5
                return dict(samples_ms=samples, p50_ms=statistics.median(samples),
                            p95_ms=percentile(samples, .95),
                            questions_per_second=questions * 1000 / sum(samples))
            rows.append(dict(batch_size=size, baseline=stats(baseline_times),
                             native=stats(candidate_times), failures=[],
                             speedup=sum(baseline_times) / sum(candidate_times)))
            print(f'batch {size}: {rows[-1]["native"]["questions_per_second"]:.1f} native q/s', flush=True)
        startup = dict(baseline_load_ms=baseline.load_ms, candidate_load_ms=candidate.load_ms,
                       baseline_peak_rss_bytes=peak_rss_bytes(baseline.process.pid),
                       candidate_peak_rss_bytes=peak_rss_bytes(candidate.process.pid),
                       missing_reason=None)
    cpu_after, _ = cpu_frequency(cpus)
    observed = [value for value in (cpu_before, cpu_after) if value is not None]
    return rows, startup, statistics.median(observed) if observed else None, policy


def benchmark(cases_path, manifest_path, stratum, acceptance_path, workload_path, output,
              *, variant, model, source, executable, threads, cpus, identity, env):
    if output.is_file():
        try:
            old = json.loads(output.read_text())
            if (old['status'] == 'passed' and old['identity']['candidate_sha256'] == identity['candidate_sha256']
                    and old['identity']['weights_sha256'] == file_hash(model / 'model.safetensors')
                    and old['identity']['baseline_commit'] == identity['baseline_commit']
                    and old['identity']['candidate_commit'] == identity['candidate_commit']
                    and old['identity']['acceptance_report_sha256'] == file_hash(acceptance_path)
                    and old['identity']['workload_parity_report_sha256'] == file_hash(workload_path)
                    and old['identity']['corpus_manifest_sha256'] == file_hash(manifest_path)
                    and old['settings']['baseline_threads'] == threads
                    and old['settings']['candidate_threads'] == threads
                    and all(old['startup'][key] is not None for key in
                            ('baseline_load_ms', 'candidate_load_ms', 'baseline_peak_rss_bytes',
                             'candidate_peak_rss_bytes'))
                    and {row['batch_size'] for row in old['rows']} == set(SIZES)
                    and all(row['parity_passed'] for row in old['rows'])):
                print(f'measured: {output}', flush=True)
                return
        except (KeyError, ValueError, OSError):
            pass
    cases = json.loads(cases_path.read_text())
    print(f'measuring: {variant} t{threads} {stratum}', flush=True)
    rows, startup, mhz, governor = timed_stratum(cases, model=model, source=source,
                                                  executable=executable, threads=threads,
                                                  cpus=cpus, env=env,
                                                  native_first=(cases[0]['outcome_variant'] + threads) % 2 == 1)
    sweep = dict(backend='cpu', precision='fp32', weights_sha256=file_hash(model / 'model.safetensors'),
                 native_build_sha256=identity['candidate_sha256'], native_device=load(acceptance_path)['identity']['candidate_device'],
                 cases_sha256=file_hash(cases_path), warmup=3, iterations=5,
                 batch_sizes=list(SIZES), threads=threads, rows=rows, passed=True,
                 ggml_revision=identity['ggml_revision'])
    report = format_report(sweep, load(acceptance_path), load(workload_path),
                           load(manifest_path), stratum, executable)
    report['startup'] = startup
    report['host']['cpu_frequency_mhz'] = mhz
    report['host']['frequency_policy'] = governor
    report['host']['affinity'] = ','.join(map(str, cpus))
    report['settings']['page_cache_state'] = 'warm'
    report['settings']['timing_scope'] = 'in-process preparation, inference, calibration and answer construction'
    if any(report['startup'][key] is None for key in ('baseline_load_ms', 'candidate_load_ms',
             'baseline_peak_rss_bytes', 'candidate_peak_rss_bytes')):
        raise RuntimeError('Required startup or RSS measurement is missing')
    schema = json.loads((ROOT / 'benchmarks/schema/benchmark-v1.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
    save(output, report)


def update_index(output, manifest, identity, variants, thread_budgets, physical_cores):
    entries = []
    for variant in variants:
        for threads in thread_budgets:
            root = output / variant / f't{threads}'
            acceptance_path = root / 'acceptance-v1.json'
            acceptance_ok = accepted(acceptance_path, corpus=identity['acceptance_sha256'],
                                   weights=identity['weights_sha256'][variant],
                                   binary=identity['candidate_sha256'], variant=variant,
                                   threads=threads, baseline_commit=identity['baseline_commit'],
                                   candidate_commit=identity['candidate_commit'], requests=250)
            reports = []
            for stratum in manifest['strata']:
                path = root / (Path(stratum['file']).stem + '-benchmark-v1.json')
                if path.is_file():
                    try:
                        value = json.loads(path.read_text())
                    except ValueError:
                        continue
                    if (value.get('status') == 'passed' and
                            value.get('identity', {}).get('candidate_sha256') == identity['candidate_sha256'] and
                            value.get('identity', {}).get('corpus_manifest_sha256') == identity['corpus_manifest_sha256'] and
                            value.get('settings', {}).get('candidate_threads') == threads and
                            all(value.get('startup', {}).get(key) is not None for key in
                                ('baseline_load_ms', 'candidate_load_ms', 'baseline_peak_rss_bytes',
                                 'candidate_peak_rss_bytes')) and
                            {row['batch_size'] for row in value.get('rows', [])} == set(SIZES)):
                        reports.append(dict(stratum=stratum['file'], path=str(path.relative_to(output)),
                                            sha256=file_hash(path)))
            entries.append(dict(variant=variant, threads=threads,
                                acceptance_path=str(acceptance_path.relative_to(output)) if acceptance_ok else None,
                                acceptance_sha256=file_hash(acceptance_path) if acceptance_ok else None,
                                reports=reports,
                                complete=len(reports) == len(manifest['strata']) and acceptance_ok))
    full_matrix = (set(variants) == set(VARIANTS) and set(thread_budgets) == {1, physical_cores})
    status = 'passed' if full_matrix and all(entry['complete'] for entry in entries) else 'incomplete'
    index = dict(schema_version=1, kind='issue7-run-index', status=status,
                 corpus_id=manifest['corpus_id'],
                 corpus_manifest_sha256=identity['corpus_manifest_sha256'],
                 candidate_sha256=identity['candidate_sha256'],
                 candidate_commit=identity['candidate_commit'],
                 baseline_commit=identity['baseline_commit'], entries=entries)
    schema = json.loads((ROOT / 'benchmarks/schema/issue7-run-v1.schema.json').read_text())
    Draft202012Validator(schema).validate(index)
    save(output / 'run-index.json', index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'models/laya')
    parser.add_argument('--source', type=Path, default=ROOT / 'research/laya')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build-cpu/bin/laya-cli')
    parser.add_argument('--manifest', type=Path, default=CORPUS)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/issue7')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--threads', nargs='+', type=int, default=None,
                        help='Default: one thread and all available physical cores')
    parser.add_argument('--strata', nargs='+', help='Restrict strata for development; resulting index remains incomplete')
    args = parser.parse_args()
    cpus = physical_cpu_ids()
    thread_budgets = args.threads or [1, len(cpus)]
    if any(value < 1 or value > len(cpus) for value in thread_budgets):
        parser.error('Thread budgets must fit the available physical cores')
    manifest = json.loads(args.manifest.read_text())
    chosen = [item for item in manifest['strata'] if not args.strata or item['file'] in args.strata]
    if len(chosen) != len(args.strata or chosen):
        parser.error('Unknown or duplicate stratum')
    args.executable = args.executable.resolve()
    args.model_root = args.model_root.resolve()
    args.source = args.source.resolve()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain',
                                     '--untracked-files=normal'], text=True)
    if dirty:
        raise RuntimeError('Commit the source tree before starting a reproducible CPU campaign')
    import torch
    weights = {}
    for variant in VARIANTS:
        model = args.model_root if variant == 'english' else args.model_root / variant
        if not (model / 'model.safetensors').is_file():
            raise FileNotFoundError(model / 'model.safetensors')
        weights[variant] = file_hash(model / 'model.safetensors')
    snapshot = dict(schema_version=1, candidate_commit=revision(ROOT), baseline_commit=revision(args.source),
                    candidate_sha256=native_hash(args.executable), ggml_revision=revision(ROOT / 'third_party/ggml'),
                    harness_sha256=harness_hash(), corpus_manifest_sha256=file_hash(args.manifest),
                    acceptance_sha256=file_hash(ACCEPTANCE), weights_sha256=weights,
                    python=platform.python_version(), torch=torch.__version__)
    identity_path = args.output / 'identity.json'
    if identity_path.is_file():
        prior = json.loads(identity_path.read_text())
        for key in snapshot:
            if prior[key] != snapshot[key]:
                raise RuntimeError(f'Existing run uses a different {key}; choose another output directory')
        snapshot = prior
    else:
        save(identity_path, snapshot)
    update_index(args.output, manifest, snapshot, args.variants, thread_budgets, len(cpus))
    for variant in args.variants:
        model = args.model_root if variant == 'english' else args.model_root / variant
        if not (model / 'model.safetensors').is_file():
            raise FileNotFoundError(model / 'model.safetensors')
        for threads in thread_budgets:
            root = args.output / variant / f't{threads}'
            env = thread_environment(threads)
            acceptance_path = root / 'acceptance-v1.json'
            validate(ACCEPTANCE, acceptance_path, variant=variant, model=model,
                     source=args.source, executable=args.executable, threads=threads,
                     identity=snapshot, env=env)
            for entry in chosen:
                stratum = entry['file']
                cases_path = args.manifest.parent / stratum
                name = Path(stratum).stem
                workload_path = root / f'{name}-validation-v1.json'
                validate(cases_path, workload_path, variant=variant, model=model,
                         source=args.source, executable=args.executable, threads=threads,
                         identity=snapshot, env=env)
                report_path = root / f'{name}-benchmark-v1.json'
                benchmark(cases_path, args.manifest, stratum, acceptance_path, workload_path,
                          report_path, variant=variant, model=model, source=args.source,
                          executable=args.executable, threads=threads, cpus=cpus[:threads],
                          identity=snapshot, env=env)
                print(f'complete: {report_path}', flush=True)
                update_index(args.output, manifest, snapshot, args.variants, thread_budgets, len(cpus))
    print('Selected runs completed', flush=True)


if __name__ == '__main__':
    main()
