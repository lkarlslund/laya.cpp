#!/usr/bin/env python3
"""Run one bounded, paired CPU comparison after a separate acceptance gate."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

from jsonschema import Draft202012Validator, FormatChecker

from cpu_process import CPUProcess, peak_rss_bytes, physical_cpu_ids, thread_environment
from format_benchmark_v1 import build_flags, host
from identity import file_hash, native_hash
from make_quick_corpus import OUTPUT as QUICK_CORPUS, SELECTION_INDICES
from report_v1 import compare_public, revision
from run import percentile
from run_cpu_issue7 import ACCEPTANCE, ROOT, SIZES, VARIANTS, accepted, prewarm_weights, save

SAMPLE_INDICES = SELECTION_INDICES
CORPUS = QUICK_CORPUS / 'manifest.json'
SCHEMA = ROOT / 'benchmarks/schema/cpu-quick-v1.schema.json'


def sample_stratum(manifest, stratum):
    entry = next((row for row in manifest['strata'] if row['file'] == stratum), None)
    if entry is None:
        raise ValueError(f'Unknown stratum: {stratum}')
    path = QUICK_CORPUS / stratum
    if file_hash(path) != entry['sha256']:
        raise ValueError(f'Stratum hash changed: {stratum}')
    cases = json.loads(path.read_text())
    if len(cases) != 8 or entry['requests'] != 8:
        raise ValueError('Quick selection requires the fixed eight-request stratum')
    kinds = {q['type'] for case in cases for q in case['questions'].values()}
    if kinds != {'choice', 'score', 'noul'}:
        raise ValueError(f'Selection lost an output type: {kinds}')
    return entry, cases, file_hash(path)


def acceptance_identity(path, model, executable, variant, threads, source):
    report = json.loads(path.read_text())
    identity = report['identity']
    if not accepted(path, corpus=file_hash(ACCEPTANCE), weights=file_hash(model / 'model.safetensors'),
                    binary=native_hash(executable), variant=variant, threads=threads,
                    baseline_commit=revision(source), candidate_commit=identity['candidate_commit'],
                    requests=250):
        raise ValueError('A complete, passing acceptance-250 report for this build, model and thread budget is required')
    return identity


def measurement(samples, questions):
    elapsed = [sample['elapsed_ms'] for sample in samples]
    return dict(samples_ms=elapsed, total_wall_ms=sum(elapsed),
                questions_per_second=questions * 1000 / sum(elapsed),
                p50_ms=statistics.median(elapsed), p95_ms=percentile(elapsed, .95))


def run(args):
    started = time.monotonic()
    entry, cases, sample_hash = sample_stratum(json.loads(CORPUS.read_text()), args.stratum)
    band = entry['length_band']
    budget = 300 if band in ('short', 'medium') else 900
    deadline = started + budget
    model = args.model_root if args.variant == 'english' else args.model_root / args.variant
    cpus = physical_cpu_ids()[:args.threads]
    if len(cpus) != args.threads:
        raise ValueError('Thread budget exceeds available physical cores')
    acceptance = args.acceptance or args.acceptance_root / args.variant / f't{args.threads}' / 'acceptance-v1.json'
    aid = acceptance_identity(acceptance, model, args.executable, args.variant, args.threads, args.source)
    env = thread_environment(args.threads)
    prewarm_weights(model)
    candidate_command = [str(args.executable), '--model', str(model), '--cpu', '--fp32', '--no-flash']
    baseline_command = [sys.executable, str(ROOT / 'benchmarks/cpu_worker.py'),
                        '--source', str(args.source), '--model', str(model), '--threads', str(args.threads)]
    groups = [cases[i:i + args.batch_size] for i in range(0, len(cases), args.batch_size)]
    samples, answers = [], []
    error = None
    failure = False
    startup = None

    def call(process, group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Wall-clock budget exhausted')
        return process.call(group, timeout=remaining)

    try:
        with ExitStack() as stack:
            baseline = stack.enter_context(CPUProcess(baseline_command, threads=args.threads,
                                  cpus=cpus, ready_stream='stdout', env=env))
            candidate = stack.enter_context(CPUProcess(candidate_command, threads=args.threads,
                                  cpus=cpus, ready_stream='stderr', env=env))
            startup = dict(baseline_load_ms=baseline.load_ms, candidate_load_ms=candidate.load_ms)
            # Warm one small call per process. The timed calls also check parity;
            # there is no second pass over the workload.
            call(baseline, cases[:1])
            call(candidate, cases[:1])
            for index, group in enumerate(groups):
                order = (('baseline', baseline), ('candidate', candidate)) if index % 2 == 0 else (
                         ('candidate', candidate), ('baseline', baseline))
                responses = {label: call(process, group) for label, process in order}
                if responses['candidate'].get('backend') != 'CPU':
                    raise RuntimeError('Candidate selected a non-CPU backend')
                expected, actual = responses['baseline']['results'], responses['candidate']['results']
                if len(expected) != len(group) or len(actual) != len(group):
                    raise RuntimeError('Result count differs from request count')
                for request, left, right in zip(group, expected, actual):
                    maximum, differences = compare_public(left, right)
                    answers.append(dict(request_id=request['id'], expected=left, actual=right,
                                        max_numeric_absolute_error=maximum, differences=differences))
                    if differences:
                        raise ValueError(f'{request["id"]}: {differences[0]}')
                samples.append(dict(request_ids=[case['id'] for case in group],
                    baseline=dict(elapsed_ms=responses['baseline']['elapsed_ms']),
                    candidate=dict(elapsed_ms=responses['candidate']['elapsed_ms'])))
            startup['baseline_peak_rss_bytes'] = peak_rss_bytes(baseline.process.pid)
            startup['candidate_peak_rss_bytes'] = peak_rss_bytes(candidate.process.pid)
    except (TimeoutError, ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {exc}'
        failure = not isinstance(exc, TimeoutError)

    flags, build_type = build_flags(args.executable)
    host_info = host('cpu', aid['candidate_device'], build_type)
    elapsed_wall = time.monotonic() - started
    if elapsed_wall > budget and error is None:
        error = f'Wall-clock budget of {budget}s exceeded'
    complete = len(samples) == len(groups) and len(answers) == len(cases) and error is None
    if not complete and error is None:
        error = 'The timed pass did not cover every selected request'
    baseline_samples = [item['baseline'] for item in samples]
    candidate_samples = [item['candidate'] for item in samples]
    if complete:
        questions = sum(len(case['questions']) for case in cases)
        baseline_stats = measurement(baseline_samples, questions)
        candidate_stats = measurement(candidate_samples, questions)
        speedup = baseline_stats['total_wall_ms'] / candidate_stats['total_wall_ms']
    else:
        baseline_stats = candidate_stats = speedup = None
    report = dict(schema_version=1, kind='cpu-quick-benchmark',
                  status='passed' if complete else 'failed' if failure else 'incomplete',
                  reason=error,
                  identity=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      variant=args.variant, backend='cpu', precision='fp32',
                      source_commit=aid['baseline_commit'], candidate_commit=aid['candidate_commit'],
                      candidate_sha256=aid['candidate_sha256'], weights_sha256=aid['weights_sha256'],
                      acceptance_report_sha256=file_hash(acceptance),
                      corpus_manifest_sha256=file_hash(CORPUS), stratum=args.stratum,
                      stratum_sha256=entry['sha256'], selected_sha256=sample_hash),
                  host=host_info,
                  settings=dict(threads=args.threads, affinity=cpus, batch_size=args.batch_size,
                      wall_budget_seconds=budget, wall_elapsed_seconds=elapsed_wall,
                      selection_indices=list(SAMPLE_INDICES), warmup_calls_per_runtime=1,
                      timed_passes=1, candidate_build_flags=flags,
                      includes_transport=False, includes_queue=False, page_cache_state='warm'),
                  startup=startup, selected_request_ids=[case['id'] for case in cases],
                  questions=sum(len(case['questions']) for case in cases),
                  answers=answers, samples=samples,
                  baseline=baseline_stats, candidate=candidate_stats, speedup=speedup)
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
    save(args.output, report)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--stratum', choices=[entry['file'] for entry in json.loads(CORPUS.read_text())['strata']], required=True)
    parser.add_argument('--batch-size', type=int, choices=SIZES, default=4)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'models/laya')
    parser.add_argument('--source', type=Path, default=ROOT / 'research/laya')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build-cpu/bin/laya-cli')
    parser.add_argument('--acceptance-root', type=Path, default=ROOT / 'results/issue7')
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    args.source = args.source.resolve()
    args.model_root = args.model_root.resolve()
    if not args.output:
        args.output = ROOT / 'results/cpu-quick' / args.variant / f't{args.threads}' / (
            Path(args.stratum).stem + f'-b{args.batch_size}.json')
    if not run(args):
        raise SystemExit(f'Incomplete quick benchmark: {args.output}')
    print(f'Passed: {args.output}')


if __name__ == '__main__':
    main()
