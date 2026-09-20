#!/usr/bin/env python3
"""Run the fixed acceptance corpus and optional throughput sweep on each checkpoint."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,default=Path('models/laya'))
    p.add_argument('--variants',nargs='+',choices=['english','multilingual','typed-decisions'],default=['english','multilingual','typed-decisions'])
    p.add_argument('--executable',default='build-cuda/bin/laya-cli')
    p.add_argument('--source',default='research/laya')
    p.add_argument('--cases',default='benchmarks/cases/acceptance-250.json')
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--strict-fp32',action='store_true')
    p.add_argument('--sweep',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/models'))
    a=p.parse_args()
    if any(b<1 for b in a.batch_sizes): p.error('Batch sizes must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,complete=False,variants=a.variants,models=[])
    for variant in a.variants:
        directory=a.model_root if variant=='english' else a.model_root/variant
        common=['--model',str(directory),'--executable',a.executable,'--source',a.source,'--cases',a.cases,
                '--batch-sizes',*map(str,a.batch_sizes)]
        if not a.strict_fp32: common+=['--tensor-core-fp32']
        validation=a.output/(variant+'-validation.json')
        entry=dict(variant=variant,model=str(directory),validation=str(validation))
        commands=[('validate.py',common+['--output',str(validation)])]
        if a.sweep: commands.append(('sweep.py',common+['--validation',str(validation),'--output',str(a.output/(variant+'-sweep.json'))]))
        for script,arguments in commands:
            print(f'{variant}: {script}',flush=True)
            with (a.output/(variant+'-'+script+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*arguments],stdout=log,stderr=subprocess.STDOUT)
            entry[script]=result.returncode
            if result.returncode:
                report['passed']=False
                print(f'{variant}: FAILED; see {log.name}',flush=True)
                break
        report['models'].append(entry)
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__': main()
