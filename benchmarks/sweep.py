#!/usr/bin/env python3
"""Measure fixed request groups at several GPU batch sizes, gating each on correctness."""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
import torch
from compare import compare_values
from native import Native
from oracle import Oracle
from run import percentile
from identity import file_hash, native_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', default='build-cuda/bin/laya-cli')
    p.add_argument('--source', default='research/laya')
    p.add_argument('--model', default='models/laya')
    p.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1,2,4,8])
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--fp32', action='store_true', default=True)
    p.add_argument('--experimental-bf16', dest='fp32', action='store_false')
    p.add_argument('--no-flash', action='store_true')
    p.add_argument('--tensor-core-fp32', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/sweep.json'))
    p.add_argument('--validation', type=Path, default=Path('results/validation-250-fp32.json'))
    a = p.parse_args()
    if a.iterations < 1 or a.warmup < 0 or any(x < 1 for x in a.batch_sizes): p.error('Invalid iteration or batch count')
    cases = json.loads(a.cases.read_text())
    validation = json.loads(a.validation.read_text())
    binary_hash = native_hash(a.executable)
    weights_hash = file_hash(Path(a.model)/'model.safetensors')
    if (not validation['passed'] or validation['cases_sha256'] != hashlib.sha256(a.cases.read_bytes()).hexdigest()
            or validation['native_build_sha256'] != binary_hash or validation['weights_sha256'] != weights_hash
            or validation['gpu'] != torch.cuda.get_device_name() or validation['torch'] != torch.__version__
            or validation['fused_attention'] != (not a.no_flash) or validation['tensor_core_fp32'] != a.tensor_core_fp32
            or validation['precision'] != ('fp32' if a.fp32 else 'bf16')
            or not set(a.batch_sizes).issubset({b['batch_size'] for b in validation['batches']})):
        p.error('A passing validation report for this corpus, binary, weights, precision and batch sizes is required')
    oracle = Oracle(a.source, a.model, a.fp32)
    report = dict(schema_version=2, cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(),
                  model_revision=(Path(a.model)/'REVISION').read_text().strip(), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, precision='fp32' if a.fp32 else 'bf16', iterations=a.iterations,
                  fused_attention=not a.no_flash, tensor_core_fp32=a.tensor_core_fp32,
                  native_build_sha256=binary_hash, weights_sha256=weights_hash,
                  warmup=a.warmup, batch_sizes=a.batch_sizes, rows=[], passed=True)
    for label,path in [('project_revision','.'),('ggml_revision','third_party/ggml'),('baseline_revision',a.source)]:
        report[label] = subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
    with Native(a.executable, a.model, fp32=a.fp32, flash=not a.no_flash, tensor_core=a.tensor_core_fp32) as native:
        for batch_size in a.batch_sizes:
            base_times, native_times, failures = [], [], []
            for start in range(0,len(cases),batch_size):
                requests = cases[start:start+batch_size]
                inputs = oracle.prepare(requests)
                raw = oracle.forward(inputs)
                expected = oracle.format(requests, raw)
                if oracle.finish(requests, inputs, raw) != expected:
                    raise RuntimeError('Batch formatter differs from public API')
                actual = native.call(requests)['results']
                if len(actual) != len(expected): raise RuntimeError('Native result count differs')
                for case, left, right in zip(requests, expected, actual):
                    try: compare_values(left,right,0.0001)
                    except ValueError as exc: failures.append(dict(id=case['id'],error=str(exc)))
                for _ in range(a.warmup):
                    oracle.predict_batch(requests); native.call(requests)
                torch.cuda.synchronize()
                for iteration in range(a.iterations):
                    # Alternate order to reduce systematic clock/thermal bias.
                    for backend in (('baseline','native') if iteration % 2 == 0 else ('native','baseline')):
                        if backend == 'baseline':
                            torch.cuda.synchronize(); before=time.perf_counter()
                            oracle.predict_batch(requests); torch.cuda.synchronize()
                            base_times.append((time.perf_counter()-before)*1000)
                        else: native_times.append(native.call(requests)['elapsed_ms'])
            def stats(times):
                return dict(p50_ms=statistics.median(times),p95_ms=percentile(times,.95),
                            questions_per_second=len(cases)*a.iterations*1000/sum(times), samples_ms=times)
            entry=dict(batch_size=batch_size,baseline=stats(base_times),native=stats(native_times),failures=failures)
            entry['speedup'] = sum(base_times)/sum(native_times) if not failures else None
            report['rows'].append(entry)
            report['passed'] &= not failures
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(f"batch={batch_size} baseline={entry['baseline']['questions_per_second']:.1f} q/s native={entry['native']['questions_per_second']:.1f} q/s correctness failures={len(failures)}",flush=True)
    raise SystemExit(0 if report['passed'] else 1)

if __name__ == '__main__': main()
