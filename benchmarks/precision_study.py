#!/usr/bin/env python3
"""Compare native precision modes to matching baselines without weakening acceptance gates.

Timing a failing mode is diagnostic only; it never makes that mode eligible for
deployment. Cross-precision drift is reported separately from port correctness.
"""
import argparse
import contextlib
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from compare import compare_values
from identity import file_hash, native_hash
from native import Native


def agreement():
    return dict(questions=0, question_failures=0, choice_mismatches=0,
                metadata_failures=0, nonfinite_failures=0, max_numeric_error=0.0, failures=[], passed=True)


def numeric_error(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return max((numeric_error(left[k], right[k]) for k in left.keys() & right.keys()), default=0.0)
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(left-right) if math.isfinite(left) and math.isfinite(right) else math.inf
    return 0.0


def compare_outputs(summary, expected, actual, requests):
    if len(expected) != len(actual):
        raise RuntimeError('Result count mismatch')
    for want, got, request in zip(expected, actual, requests):
        try:
            compare_values({k: want[k] for k in ('model', 'usage')},
                           {k: got[k] for k in ('model', 'usage')}, 0)
            if want['answers'].keys() != got['answers'].keys():
                raise ValueError('Answer IDs differ')
        except (ValueError, KeyError) as error:
            summary['metadata_failures'] += 1
            summary['failures'].append(dict(id=request.get('id'), error=str(error)))
        for key, left in want['answers'].items():
            summary['questions'] += 1
            right = got.get('answers', {}).get(key)
            if left.get('type') == 'choice' and (not isinstance(right, dict) or left['choice'] != right.get('choice')):
                summary['choice_mismatches'] += 1
            error = numeric_error(left, right)
            if math.isfinite(error):
                summary['max_numeric_error'] = max(summary['max_numeric_error'], error)
            else:
                summary['nonfinite_failures'] += 1
            try:
                compare_values(left, right, 0.0001)
            except ValueError as mismatch:
                summary['question_failures'] += 1
                summary['failures'].append(dict(id=key, error=str(mismatch)))
    summary['passed'] = summary['question_failures'] == 0 and summary['metadata_failures'] == 0


@contextlib.contextmanager
def precision(oracle, fp32, defaults):
    import torch
    old = oracle.fp32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        oracle.fp32 = fp32
        torch.backends.cuda.matmul.allow_tf32 = False if fp32 else defaults[0]
        torch.backends.cudnn.allow_tf32 = False if fp32 else defaults[1]
        yield
    finally:
        oracle.fp32 = old
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def stats(samples, questions, iterations):
    from run import percentile
    return dict(p50_ms=statistics.median(samples), p95_ms=percentile(samples, .95),
                questions_per_second=questions*iterations*1000/sum(samples), samples_ms=samples)


def study(args, variant):
    import torch
    from oracle import Oracle
    model = Path(args.model_root) / (variant if variant != 'english' else '')
    cases = json.loads(Path(args.cases).read_text())
    question_count = sum(len(case['questions']) for case in cases)
    defaults = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    oracle = Oracle(args.source, model, fp32=False)
    if oracle.agent.dtype != torch.bfloat16:
        raise RuntimeError('This study requires a BF16 default CUDA baseline')
    report = dict(schema_version=1, variant=variant, complete=False,
                  acceptance='exact_categories_absolute_numeric_0.0001_at_matching_precision',
                  native_build_sha256=native_hash(args.executable),
                  cases_sha256=file_hash(args.cases), weights_sha256=file_hash(model/'model.safetensors'),
                  model_revision=(model/'REVISION').read_text().strip(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  default_tf32=dict(matmul=defaults[0], cudnn=defaults[1]),
                  exclusive_gpu=False, warmup=args.warmup, iterations=args.iterations,
                  timing_scope='preprocessing + inference + formatting; excludes loading and native JSON transport',
                  rows=[])
    for key, directory in [('project_revision', '.'), ('ggml_revision', 'third_party/ggml'), ('baseline_revision', args.source)]:
        report[key] = subprocess.check_output(['git', '-C', directory, 'rev-parse', 'HEAD'], text=True).strip()
    output = Path(args.output)/f'{variant}.json'
    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    # One additional native checkpoint at a time; the same baseline stays resident.
    for mode in args.modes:
        fp32 = mode == 'fp32'
        with Native(args.executable, model, fp32=fp32, flash=True, tensor_core=fp32) as native:
            for size in args.batch_sizes:
                paired, cross = agreement(), agreement()
                replay_failures = 0
                times = {key: [] for key in ('baseline_fp32', 'baseline_bf16', 'native')}
                for offset in range(0, len(cases), size):
                    requests = cases[offset:offset+size]
                    inputs = oracle.prepare(requests)
                    references = {}
                    for baseline_fp32 in (False, True):
                        with precision(oracle, baseline_fp32, defaults):
                            raw = oracle.forward(inputs)
                            if not all(torch.isfinite(value).all() for value in raw):
                                raise RuntimeError('Nonfinite baseline tensors')
                            expected = oracle.format(requests, raw)
                            if oracle.finish(requests, inputs, raw) != expected:
                                raise RuntimeError('Batch formatter differs from the public API')
                            references[baseline_fp32] = expected
                        del raw
                    del inputs
                    # Cached unused tensor allocations must not starve the other process.
                    torch.cuda.empty_cache()
                    actual = native.call(requests)['results']
                    compare_outputs(paired, references[fp32], actual, requests)
                    compare_outputs(cross, references[False], references[True], requests)
                    for _ in range(3):
                        if native.call(requests)['results'] != actual:
                            replay_failures += 1
                    def run(backend):
                        if backend == 'native':
                            return native.call(requests)['elapsed_ms']
                        with precision(oracle, backend == 'baseline_fp32', defaults):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            oracle.predict_batch(requests)
                            torch.cuda.synchronize()
                            return (time.perf_counter()-start)*1000
                    backends = list(times)
                    for _ in range(args.warmup):
                        for backend in backends:
                            run(backend)
                    # Rotate and reverse order to share clock/thermal conditions.
                    for iteration in range(args.iterations):
                        index = (iteration + offset//size) % len(backends)
                        order = backends[index:] + backends[:index]
                        if iteration % 2:
                            order.reverse()
                        for backend in order:
                            times[backend].append(run(backend))
                passed = paired['passed'] and replay_failures == 0
                timing = {key: stats(value, question_count, args.iterations) for key, value in times.items()}
                paired_baseline = 'baseline_fp32' if fp32 else 'baseline_bf16'
                row = dict(native_precision=mode, baseline_precision=mode, batch_size=size,
                           matching_precision=paired, baseline_cross_precision_drift=cross,
                           replay_failures=replay_failures, passed=passed,
                           eligible_for_deployment=passed, timing=timing,
                           accepted_speedup=sum(times[paired_baseline])/sum(times['native']) if passed else None,
                           diagnostic_speedup_vs_default_bf16=sum(times['baseline_bf16'])/sum(times['native']))
                report['rows'].append(row)
                save()
                print(f'{variant} {mode} B{size}: matching failures={paired["question_failures"]}/{paired["questions"]}, '
                      f'choice mismatches={paired["choice_mismatches"]}, max error={paired["max_numeric_error"]:.6g}, '
                      f'native={timing["native"]["questions_per_second"]:.1f} q/s, '
                      f'baseline BF16={timing["baseline_bf16"]["questions_per_second"]:.1f} q/s, '
                      f'baseline FP32={timing["baseline_fp32"]["questions_per_second"]:.1f} q/s', flush=True)
        torch.cuda.empty_cache()
    report['complete'] = True
    report['eligible_modes'] = [mode for mode in args.modes
                                if all(r['passed'] for r in report['rows'] if r['native_precision'] == mode)]
    save()
    del oracle
    gc.collect()
    torch.cuda.empty_cache()
    return dict(variant=variant, complete=True, eligible_modes=report['eligible_modes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='build-cuda/bin/laya-cli')
    parser.add_argument('--source', default='research/laya')
    parser.add_argument('--model-root', default='models/laya')
    parser.add_argument('--cases', default='benchmarks/cases/acceptance-250.json')
    parser.add_argument('--variants', nargs='+', choices=['english', 'multilingual', 'typed-decisions'],
                        default=['english', 'multilingual', 'typed-decisions'])
    parser.add_argument('--modes', nargs='+', choices=['fp32', 'bf16'], default=['fp32', 'bf16'])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--output', default='results/precision-study')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error('Invalid warmup, iteration or batch count')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    manifest = {'complete': False, 'variants': []}
    summary = Path(args.output)/'summary.json'
    summary.write_text(json.dumps(manifest, indent=2)+'\n')
    for variant in args.variants:
        manifest['variants'].append(study(args, variant))
        summary.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest['complete'] = True
    summary.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
