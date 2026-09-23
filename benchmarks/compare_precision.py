#!/usr/bin/env python3
"""Compare accepted native FP32 and BF16 modes with alternating timed calls."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from identity import file_hash, native_hash
from native import Native
from run import percentile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True)
    p.add_argument('--model', default='models/laya')
    p.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    p.add_argument('--fp32-validation', type=Path, required=True)
    p.add_argument('--bf16-validation', type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 0 or a.iterations < 1 or any(b < 1 for b in a.batch_sizes):
        p.error('Invalid warmup, iteration or batch count')
    identity = dict(native_build_sha256=native_hash(a.executable), cases_sha256=file_hash(a.cases),
                    weights_sha256=file_hash(Path(a.model)/'model.safetensors'),
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    for mode, path in [('fp32', a.fp32_validation), ('bf16', a.bf16_validation)]:
        v = json.loads(path.read_text())
        if (not v['passed'] or any(v.get(k) != value for k, value in identity.items()) or
                v['precision'] != mode or not v['fused_attention'] or
                v['tensor_core_fp32'] != (mode == 'fp32') or v.get('allow_truncation') is not True or v['answer_atol'] > .0001 or
                not set(a.batch_sizes).issubset({b['batch_size'] for b in v['batches']})):
            p.error(f'A matching passing {mode} validation report is required')
    cases = json.loads(a.cases.read_text())
    questions = sum(len(c['questions']) for c in cases)
    report = dict(**identity, allow_truncation=True, warmup=a.warmup, iterations=a.iterations, rows=[], complete=False,
                  comparison='native optimized FP32 versus native BF16; separately accepted at matching precision')
    with Native(a.executable, a.model, allow_truncation=True, flash=True, tensor_core=True) as fp32, \
            Native(a.executable, a.model, allow_truncation=True, fp32=False, flash=True) as bf16:
        clients = [fp32, bf16]
        for batch in a.batch_sizes:
            times = [[], []]
            for start in range(0, len(cases), batch):
                requests = cases[start:start+batch]
                for _ in range(a.warmup):
                    for client in clients:
                        client.call(requests)
                for iteration in range(a.iterations):
                    for index in ([0, 1] if (iteration + start // batch) % 2 == 0 else [1, 0]):
                        times[index].append(clients[index].call(requests)['elapsed_ms'])
            def stats(values):
                return dict(questions_per_second=questions*a.iterations*1000/sum(values),
                            p50_ms=statistics.median(values), p95_ms=percentile(values, .95), samples_ms=values)
            row = dict(batch_size=batch, fp32=stats(times[0]), bf16=stats(times[1]),
                       bf16_speedup=sum(times[0])/sum(times[1]))
            report['rows'].append(row)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2)+'\n')
            print(f"batch={batch} fp32={row['fp32']['questions_per_second']:.1f} bf16={row['bf16']['questions_per_second']:.1f} speedup={row['bf16_speedup']:.3f}", flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
