#!/usr/bin/env python3
"""Measure paired PyTorch CPU and Core ML throughput after fail-closed validation."""
import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from compare import compare_values
from identity import file_hash, native_hash
from native import Native
from validate_coreml import CPUOracle, coreml_environment, grouped_requests, model_path, select_bucket


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='build-coreml/bin/laya-cli')
    parser.add_argument('--source', default='research/laya')
    parser.add_argument('--model', default='models/laya')
    parser.add_argument('--variant', choices=('english', 'multilingual', 'typed-decisions'),
                        default='english')
    parser.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--validation', type=Path,
                        default=Path('results/coreml-validation.json'))
    parser.add_argument('--cache-root', type=Path, default=Path('results/coreml-cache'))
    parser.add_argument('--output', type=Path, default=Path('results/coreml-sweep.json'))
    return parser.parse_args(argv)


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def timing_stats(samples, questions, iterations):
    if not samples or questions < 1 or iterations < 1:
        raise ValueError('Timing statistics require samples, questions and iterations')
    if not all(math.isfinite(value) and value >= 0 for value in samples):
        raise ValueError('Timing samples must be finite and nonnegative')
    total = sum(samples)
    if total <= 0:
        raise ValueError('Total measured time must be positive')
    return dict(p50_ms=statistics.median(samples), p95_ms=percentile(samples, .95),
                questions_per_second=questions * iterations * 1000 / total,
                samples_ms=samples)


def revision(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_name():
    if sys.platform == 'darwin':
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def normalize_backend(value):
    return re.sub(r'[^a-z0-9]', '', str(value).casefold())


def validated_variant(validation, *, variant, cases_hash, build_hash, model,
                      model_revision, precision, weights_hash, manifest_hash, batch_sizes):
    if validation.get('schema_version') != 2:
        raise ValueError('Core ML validation schema 2 is required; rerun validate_coreml.py')
    if validation.get('backend') != 'coreml' or not validation.get('passed') or not validation.get('complete'):
        raise ValueError('A complete passing Core ML validation report is required')
    expected = {
        'cases_sha256': cases_hash,
        'native_build_sha256': build_hash,
    }
    for field, value in expected.items():
        if validation.get(field) != value:
            raise ValueError(f'Validation {field} does not match the benchmark input')
    matches = [item for item in validation.get('variants', []) if item.get('variant') == variant]
    if len(matches) != 1:
        raise ValueError(f'Validation must contain exactly one {variant!r} result')
    result = matches[0]
    identity = {
        'model': str(Path(model).resolve()),
        'model_revision': model_revision,
        'precision': precision,
        'weights_sha256': weights_hash,
        'coreml_manifest_sha256': manifest_hash,
    }
    for field, value in identity.items():
        if result.get(field) != value:
            raise ValueError(f'Validation variant {field} does not match the benchmark model')
    validated = {item.get('batch_size') for item in result.get('batches', []) if item.get('passed')}
    if not set(batch_sizes).issubset(validated):
        raise ValueError('Every requested batch size must have passed Core ML validation')
    return result


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


def main(argv=None):
    args = arguments(argv)
    if args.iterations < 1 or args.warmup < 0 or any(size < 1 for size in args.batch_sizes):
        raise ValueError('Iterations and batch sizes must be positive; warmup must be nonnegative')
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError('Batch sizes must be unique')

    cases = json.loads(args.cases.read_text())
    if not cases:
        raise ValueError('Benchmark cases must be nonempty')
    reference_model = model_path(args.model, args.variant).resolve()
    manifest_path = reference_model / 'coreml' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('variant') != args.variant:
        raise ValueError('Core ML manifest variant does not match --variant')

    cases_hash = file_hash(args.cases)
    build_hash = native_hash(args.executable)
    weights_hash = file_hash(reference_model / 'model.safetensors')
    manifest_hash = file_hash(manifest_path)
    model_revision = (reference_model / 'REVISION').read_text().strip()
    validation = json.loads(args.validation.read_text())
    validation_variant = validated_variant(
        validation, variant=args.variant, cases_hash=cases_hash, build_hash=build_hash,
        model=reference_model, model_revision=model_revision, precision=manifest.get('precision'),
        weights_hash=weights_hash, manifest_hash=manifest_hash,
        batch_sizes=args.batch_sizes)

    oracle = CPUOracle(args.source, reference_model)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for the CPU baseline') from exc

    report = dict(
        schema_version=2, backend='coreml', baseline_backend='pytorch-cpu',
        variant=args.variant, cases_sha256=cases_hash,
        model=str(reference_model), model_revision=model_revision,
        precision=validation_variant['precision'], baseline_precision='fp32',
        native_build_sha256=build_hash,
        weights_sha256=weights_hash, coreml_manifest_sha256=manifest_hash,
        device=None, cpu=cpu_name(), torch=torch.__version__,
        python=platform.python_version(), platform=platform.platform(),
        iterations=args.iterations, warmup=args.warmup,
        batch_sizes=args.batch_sizes, complete=False, passed=True, rows=[],
        project_revision=revision('.'), ggml_revision=revision('third_party/ggml'),
        baseline_revision=revision(args.source), export_tools=manifest.get('export_tools'))

    answer_tolerance = validation.get('answer_atol', 0.0001)
    if (not isinstance(answer_tolerance, (int, float)) or not math.isfinite(answer_tolerance)
            or answer_tolerance < 0 or answer_tolerance > 0.0001):
        raise ValueError('Validation answer tolerance exceeds the public acceptance contract')
    states = []
    tasks_by_bucket = {}
    for batch_size in args.batch_sizes:
        groups = grouped_requests(cases, batch_size, manifest)
        state = dict(batch_size=batch_size, groups=groups, baseline_times=[], native_times=[],
                     failures=[], group_ordinal=0,
                     question_count=sum(item[2] for bucket_groups in groups.values() for item in bucket_groups))
        states.append(state)
        for bucket, bucket_groups in groups.items():
            for start, requests, questions in bucket_groups:
                tasks_by_bucket.setdefault(bucket, []).append((state, start, requests, questions))

    write_report(args.output, report)
    # A process sees only one compiled bucket, and every use of that bucket is
    # scheduled together. This avoids both model switching and repeated reopen.
    for bucket, bucket_tasks in tasks_by_bucket.items():
        environment = coreml_environment(args.cache_root, bucket)
        with Native(args.executable, reference_model, backend='coreml', env=environment) as native:
            for state, start, requests, _questions in bucket_tasks:
                inputs = oracle.prepare(requests)
                expected = oracle.format(requests, oracle.forward(inputs))
                response = native.call(requests)
                if normalize_backend(response.get('backend')) != 'coreml':
                    raise RuntimeError(f"Core ML backend was not selected: {response.get('backend')!r}")
                device = response.get('device')
                if report['device'] is None:
                    report['device'] = device
                elif device != report['device']:
                    raise RuntimeError(f'Core ML device changed from {report["device"]!r} to {device!r}')
                actual = response.get('results', [])
                if len(actual) != len(expected):
                    state['failures'].append(dict(start=start, bucket=bucket, error='result count differs'))
                else:
                    for request, left, right in zip(requests, expected, actual):
                        try:
                            compare_values(left, right, answer_tolerance)
                        except ValueError as exc:
                            state['failures'].append(dict(id=request.get('id'), start=start,
                                                          bucket=bucket, error=str(exc)))

                for _ in range(args.warmup):
                    oracle.predict_batch(requests)
                    native.call(requests)
                for iteration in range(args.iterations):
                    order = ('baseline', 'native') if (iteration + state['group_ordinal']) % 2 == 0 else ('native', 'baseline')
                    for backend in order:
                        if backend == 'baseline':
                            before = time.perf_counter()
                            oracle.predict_batch(requests)
                            state['baseline_times'].append((time.perf_counter() - before) * 1000)
                        else:
                            elapsed = native.call(requests).get('elapsed_ms')
                            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                                raise RuntimeError(f'Invalid native elapsed_ms: {elapsed!r}')
                            state['native_times'].append(float(elapsed))
                state['group_ordinal'] += 1

    for state in states:
        baseline = timing_stats(state['baseline_times'], state['question_count'], args.iterations)
        native = timing_stats(state['native_times'], state['question_count'], args.iterations)
        failures = state['failures']
        row = dict(batch_size=state['batch_size'],
                   buckets={name: len(items) for name, items in state['groups'].items()},
                   baseline=baseline, native=native, failures=failures,
                   speedup=(sum(state['baseline_times']) / sum(state['native_times'])) if not failures else None)
        report['rows'].append(row)
        report['passed'] &= not failures
        write_report(args.output, report)
        print(f'batch={state["batch_size"]} baseline={baseline["questions_per_second"]:.1f} q/s '
              f'coreml={native["questions_per_second"]:.1f} q/s '
              f'correctness_failures={len(failures)}', flush=True)

    report['complete'] = True
    write_report(args.output, report)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
