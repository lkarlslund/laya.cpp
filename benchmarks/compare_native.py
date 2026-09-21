#!/usr/bin/env python3
"""Compare two standalone native builds at matching precision on identical request groups."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from compare import compare_values
from identity import file_hash, native_hash, matching_device
from native import Native
from run import percentile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bf16', action='store_true', help='Compare BF16 builds instead of optimized FP32')
    p.add_argument('--before', required=True, help='Executable with its own preserved shared libraries')
    p.add_argument('--after', default='build-cuda/bin/laya-cli')
    p.add_argument('--before-backend', choices=['cuda','vulkan'], default='cuda')
    p.add_argument('--after-backend', choices=['cuda','vulkan'], default='cuda')
    p.add_argument('--before-tensor-core-fp32', action='store_true')
    p.add_argument('--model', default='models/laya')
    p.add_argument('--cases', type=Path, default=Path('benchmarks/cases/acceptance-250.json'))
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1,2,4,8])
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('results/native-comparison.json'))
    a=p.parse_args()
    if a.iterations<1 or a.warmup<0 or any(b<1 for b in a.batch_sizes): p.error('Invalid iteration or batch count')
    validation=json.loads(a.validation.read_text())
    corpus_hash=file_hash(a.cases)
    after_hash=native_hash(a.after)
    weights_hash=file_hash(Path(a.model)/'model.safetensors')
    if (validation.get('backend','cuda')!=a.after_backend or not validation['passed'] or validation['native_build_sha256']!=after_hash or
        validation['cases_sha256']!=corpus_hash or validation['weights_sha256']!=weights_hash or
        validation['gpu']!=torch.cuda.get_device_name() or validation['torch']!=torch.__version__ or
        validation['precision']!=('bf16' if a.bf16 else 'fp32') or
        validation['answer_atol']>0.0001 or
        not set(a.batch_sizes).issubset({b['batch_size'] for b in validation['batches']})):
        p.error('A matching passing validation report for the selected precision is required')
    cases=json.loads(a.cases.read_text())
    questions=sum(len(c['questions']) for c in cases)
    before_flash = a.before_backend=='cuda' or a.bf16
    before_tensor = not a.bf16 and (a.before_backend=='cuda' or a.before_tensor_core_fp32)
    after_flash = validation['fused_attention']
    after_tensor = validation['tensor_core_fp32']
    report=dict(before_backend=a.before_backend,after_backend=a.after_backend,before_fused_attention=before_flash,after_fused_attention=after_flash,before_tensor_core_fp32=before_tensor,after_tensor_core_fp32=after_tensor,before_sha256=native_hash(a.before),after_sha256=after_hash,weights_sha256=weights_hash,
                cases_sha256=corpus_hash,gpu=torch.cuda.get_device_name(),torch=torch.__version__,iterations=a.iterations,warmup=a.warmup,precision='bf16' if a.bf16 else 'fp32',rows=[],passed=True)
    with Native(a.before,a.model,fp32=not a.bf16,flash=before_flash,tensor_core=before_tensor,backend=a.before_backend) as before, Native(a.after,a.model,fp32=not a.bf16,flash=after_flash,tensor_core=after_tensor,backend=a.after_backend) as after:
        report['before_device']=matching_device(before.call(cases[:1]),a.before_backend,report['gpu'])
        report['after_device']=matching_device(after.call(cases[:1]),a.after_backend,report['gpu'])
        if a.after_backend=='vulkan' and report['after_device']!=validation.get('native_device'):
            p.error('Native GPU differs from the validated device')
        for batch in a.batch_sizes:
            times=[[],[]]
            failures=[]
            for start in range(0,len(cases),batch):
                requests=cases[start:start+batch]
                left=before.call(requests)['results']; right=after.call(requests)['results']
                if len(left)!=len(right): raise RuntimeError('Result count differs')
                for case,x,y in zip(requests,left,right):
                    try: compare_values(x,y,0.0001)
                    except ValueError as exc: failures.append(dict(id=case['id'],error=str(exc)))
                for _ in range(a.warmup): before.call(requests); after.call(requests)
                for iteration in range(a.iterations):
                    for index in ([0,1] if iteration%2==0 else [1,0]):
                        response=(before if index==0 else after).call(requests)
                        times[index].append(response['elapsed_ms'])
            def stats(values):
                return dict(p50_ms=statistics.median(values),p95_ms=percentile(values,.95),
                            questions_per_second=questions*a.iterations*1000/sum(values),samples_ms=values)
            row=dict(batch_size=batch,before=stats(times[0]),after=stats(times[1]),failures=failures,
                     speedup=sum(times[0])/sum(times[1]) if not failures else None)
            report['rows'].append(row); report['passed'] &= not failures
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(f"batch={batch} before={row['before']['questions_per_second']:.1f} after={row['after']['questions_per_second']:.1f} speedup={row['speedup']} failures={len(failures)}",flush=True)
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__': main()
