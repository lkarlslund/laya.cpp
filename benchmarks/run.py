#!/usr/bin/env python3
"""Synchronized end-to-end decision benchmark with a pluggable runtime."""
import argparse
import hashlib
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend-path', type=Path, required=True)
    p.add_argument('--factory', default='laya:load', help='module:factory; factory(model, device=...) returns predict(state, questions)')
    p.add_argument('--model', type=Path, default=ROOT / 'models/laya')
    p.add_argument('--cases', type=Path, default=ROOT / 'benchmarks/cases/smoke.json')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 0 or a.iterations < 1:
        p.error('warmup must be nonnegative and iterations positive')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    sys.path.insert(0, str(a.backend_path.resolve()))
    module, name = a.factory.split(':')
    factory = getattr(importlib.import_module(module), name)
    cases = json.loads(a.cases.read_text())
    if not cases or len({c['id'] for c in cases}) != len(cases):
        p.error('cases must be nonempty with unique ids')
    sync = torch.cuda.synchronize if a.device == 'cuda' else lambda: None
    sync()
    start = time.perf_counter()
    agent = factory(str(a.model.resolve()), device=a.device)
    sync()
    load_ms = (time.perf_counter() - start) * 1000
    def check_device():
        if str(agent.device).split(':')[0] != a.device:
            raise RuntimeError(f'Runtime fell back to {agent.device}; requested {a.device}')
    check_device()
    rows = []
    with torch.inference_mode():
        for case in cases:
            for _ in range(a.warmup):
                agent.predict(case['state'], case['questions'])
            check_device()
            sync()
            if a.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(a.iterations):
                sync()
                start = time.perf_counter()
                answer = agent.predict(case['state'], case['questions'])
                sync()
                times.append((time.perf_counter() - start) * 1000)
                check_device()
            rows.append(dict(id=case['id'], latency_ms=dict(p50=statistics.median(times), p95=percentile(times, .95), mean=statistics.mean(times)),
                             questions_per_second=len(case['questions']) * 1000 / statistics.mean(times), samples_ms=times,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None, result=answer))
    def revision(path):
        r = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    report = dict(schema_version=1, factory=a.factory, backend_revision=revision(a.backend_path), project_revision=revision(ROOT),
                  model=str(a.model.resolve()), model_revision=(a.model / 'REVISION').read_text().strip() if (a.model / 'REVISION').exists() else None,
                  cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(), device=a.device,
                  gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None, torch=torch.__version__, cuda=torch.version.cuda,
                  python=platform.python_version(), platform=platform.platform(), warmup=a.warmup, iterations=a.iterations,
                  load_ms=load_ms, rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print(f"{row['id']}: p50={row['latency_ms']['p50']:.3f} ms p95={row['latency_ms']['p95']:.3f} ms")

if __name__ == '__main__':
    main()
