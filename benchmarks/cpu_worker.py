#!/usr/bin/env python3
"""Isolated, internally timed CPU baseline for paired benchmark runs."""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        parser.error('--threads must be from 1 to 256')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != str(args.threads):
            parser.error(f'{name} must equal --threads before importing PyTorch')

    import torch
    from oracle import Oracle
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    oracle = Oracle(args.source, args.model, fp32=True, device='cpu')
    print(json.dumps(dict(ready=True, device=oracle.agent.device.type,
                          threads=torch.get_num_threads(), torch=torch.__version__)), flush=True)
    for line in sys.stdin:
        try:
            requests = json.loads(line)
            started = time.perf_counter()
            results = oracle.predict_batch(requests)
            elapsed_ms = (time.perf_counter() - started) * 1000
            print(json.dumps(dict(results=results, elapsed_ms=elapsed_ms), allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')), flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
