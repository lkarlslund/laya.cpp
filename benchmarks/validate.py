#!/usr/bin/env python3
"""Fail-closed token, raw tensor, and typed answer parity across fixed batch sizes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from native import Native
from oracle import Oracle
from compare import compare_values
from identity import file_hash, native_hash, matching_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', default='build-cuda/bin/laya-cli')
    p.add_argument('--backend', choices=['cuda', 'vulkan', 'cpu'], default='cuda')
    p.add_argument('--source', default='research/laya')
    p.add_argument('--model', default='models/laya')
    p.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--fp32', action='store_true', default=True)
    p.add_argument('--bf16', '--experimental-bf16', dest='fp32', action='store_false')
    p.add_argument('--no-flash', action='store_true')
    p.add_argument('--tensor-core-fp32', action='store_true')
    # Unnormalized action logits can exceed 4000. A scale-aware FP32 bound
    # avoids treating a few ULPs there like the same error near zero.
    p.add_argument('--raw-atol', type=float, default=0.001)
    p.add_argument('--raw-rtol', type=float, default=0.00001)
    p.add_argument('--answer-atol', type=float, default=0.0001)
    p.add_argument('--output', type=Path, default=Path('results/validation.json'))
    a = p.parse_args()
    if a.backend != 'cuda':
        if not a.fp32: p.error('This backend currently requires FP32')
        if a.backend == 'cpu' and a.tensor_core_fp32: p.error('CPU requires plain FP32')
        a.no_flash = True
    if any(x < 1 for x in a.batch_sizes): p.error('batch sizes must be positive')
    if not all(math.isfinite(x) and x >= 0 for x in (a.raw_atol,a.raw_rtol,a.answer_atol)):
        p.error('Tolerances must be finite and nonnegative')
    if a.answer_atol > 0.0001: p.error('Answer tolerance cannot exceed the acceptance contract')
    cases = json.loads(a.cases.read_text())
    oracle = Oracle(a.source, a.model, a.fp32)
    report = dict(backend=a.backend, cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(), precision='fp32' if a.fp32 else 'bf16',
                  native_build_sha256=native_hash(a.executable), weights_sha256=file_hash(Path(a.model)/'model.safetensors'),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  python_runtime='rocm' if torch.version.hip else 'cuda',
                  python_runtime_version=torch.version.hip or torch.version.cuda,
                  fused_attention=not a.no_flash, tensor_core_fp32=a.tensor_core_fp32,
                  acceptance='exact_categories_absolute_numeric_0.0001', raw_atol=a.raw_atol, raw_rtol=a.raw_rtol, answer_atol=a.answer_atol, passed=True, batches=[])
    # Run one native process at a time to avoid unnecessary duplicate device weights.
    with Native(a.executable, a.model, raw=True, fp32=a.fp32, flash=not a.no_flash, tensor_core=a.tensor_core_fp32, backend=a.backend) as native:
        report['native_device'] = matching_device(native.call(cases[:1]), a.backend, report['gpu'])
        for batch_size in a.batch_sizes:
            summary = dict(batch_size=batch_size, questions=0, max_logit_error=0., max_action_error=0., failures=[], raw_diagnostics=[])
            for start in range(0, len(cases), batch_size):
                requests = cases[start:start+batch_size]
                inputs = oracle.prepare(requests)
                expected_inputs = oracle.expected_inputs(inputs)
                torch.cuda.empty_cache()
                output = native.call(requests)['results']
                if output['inputs'] != expected_inputs:
                    summary['failures'].append(dict(start=start, error='input tensor mismatch'))
                    continue
                expected = oracle.forward(inputs)
                errors = []
                for key, reference in zip(('logits', 'actions'), expected):
                    actual = np.array(output[key], dtype=np.float32).reshape(reference.shape)
                    target = reference.float().cpu().numpy()
                    error = float(np.max(np.abs(actual-target)))
                    summary['max_logit_error' if key == 'logits' else 'max_action_error'] = max(
                        summary['max_logit_error' if key == 'logits' else 'max_action_error'], error)
                    if not np.isfinite(actual).all():
                        summary['failures'].append(dict(start=start, error=f'Nonfinite {key}'))
                    if not np.allclose(actual,target,atol=a.raw_atol,rtol=a.raw_rtol):
                        errors.append(f'{key} max error {error}')
                if errors: summary['raw_diagnostics'].append(dict(start=start, ids=[r['id'] for r in requests], error='; '.join(errors)))
                summary['questions'] += sum(len(r['questions']) for r in requests)
            report['batches'].append(summary)
            print({k: (len(v) if k in ('failures', 'raw_diagnostics') else v) for k, v in summary.items()}, flush=True)
            if summary['failures']: report['passed'] = False
    # Independently compare the actual native public API against baseline formatting.
    with Native(a.executable, a.model, fp32=a.fp32, flash=not a.no_flash, tensor_core=a.tensor_core_fp32, backend=a.backend) as native:
        for batch_size, summary in zip(a.batch_sizes, report['batches']):
            summary['answer_failures'] = []
            for start in range(0, len(cases), batch_size):
                requests = cases[start:start+batch_size]
                expected = oracle.format(requests, oracle.forward(oracle.prepare(requests)))
                torch.cuda.empty_cache()
                actual = native.call(requests)['results']
                # Repeated calls exercise persistent buffers and CUDA graph replay.
                for _ in range(3):
                    repeated = native.call(requests)['results']
                    if repeated != actual: raise RuntimeError('Native graph replay changed the answer')
                if len(actual) != len(expected): raise RuntimeError('Native result count differs')
                for i, (left, right) in enumerate(zip(expected, actual)):
                    try: compare_values(left, right, a.answer_atol)
                    except ValueError as exc:
                        summary['answer_failures'].append(dict(id=requests[i]['id'], error=str(exc)))
            if summary['answer_failures']: report['passed'] = False
            print(f"batch {batch_size}: {len(summary['answer_failures'])} answer failures", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)

if __name__ == '__main__': main()
