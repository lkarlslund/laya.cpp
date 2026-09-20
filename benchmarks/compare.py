#!/usr/bin/env python3
"""Gate paired benchmark reports on output agreement and report latency ratios."""
import argparse
import json
import math
from pathlib import Path


def compare_values(a, b, tolerance, path='result'):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f'{path}: keys differ')
        for key in a:
            compare_values(a[key], b[key], tolerance, f'{path}.{key}')
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        error = abs(a-b)
        if not math.isfinite(a) or not math.isfinite(b) or (error > tolerance and not math.isclose(error,tolerance,rel_tol=1e-9,abs_tol=1e-12)):
            raise ValueError(f'{path}: {a} != {b} (tolerance {tolerance})')
    elif a != b:
        raise ValueError(f'{path}: {a!r} != {b!r}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--atol', type=float, default=0.001)
    a = p.parse_args()
    if not math.isfinite(a.atol) or a.atol < 0:
        p.error('atol must be finite and nonnegative')
    baseline, candidate = [json.loads(x.read_text()) for x in (a.baseline, a.candidate)]
    for key in ('schema_version', 'cases_sha256', 'model_revision', 'device', 'gpu'):
        if baseline[key] != candidate[key]:
            p.error(f'incompatible {key}')
    old = {r['id']: r for r in baseline['rows']}
    new = {r['id']: r for r in candidate['rows']}
    if not old or old.keys() != new.keys():
        p.error('case IDs differ or are empty')
    try:
        for key in old:
            for field in ('answers', 'usage'):
                compare_values(old[key]['result'][field], new[key]['result'][field], a.atol, f'{key}.{field}')
    except ValueError as exc:
        p.error(str(exc))
    for key in old:
        speedup = old[key]['latency_ms']['p50'] / new[key]['latency_ms']['p50']
        print(f'{key}: outputs agree; p50 speedup {speedup:.3f}x')

if __name__ == '__main__':
    main()
