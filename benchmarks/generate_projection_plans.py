#!/usr/bin/env python3
"""Compress measured FP16/BF16 projection modes into a C++ geometry table."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--output', type=Path, default=Path('src/vulkan/projection_plans.hpp'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.profile.read_text().splitlines()]
    plans = {'fp16': {}, 'bf16': {}}
    for row in rows:
        key = (row['k'], row['m'], row['bias'], row['n'])
        scheme = row['scheme']
        if scheme not in ('NONE', 'REDUCTION_SCHEME_OUTPUT_TYPE',
                          'REDUCTION_SCHEME_COMPUTE_TYPE', 'REDUCTION_SCHEME_INPLACE'):
            raise ValueError(f'Unsupported reduction scheme: {scheme}')
        mode = (row['chunk'], scheme == 'REDUCTION_SCHEME_OUTPUT_TYPE',
                scheme == 'REDUCTION_SCHEME_INPLACE', row['round_before_bias'])
        target = plans[row['precision']]
        if key in target:
            raise ValueError(f'Duplicate geometry: {key}')
        target[key] = mode
    if not plans['fp16'] or plans['fp16'] != plans['bf16']:
        raise ValueError('Both precision profiles must be complete and identical')
    table = plans['fp16']
    shapes = sorted({key[:3] for key in table})
    limit = max(key[3] for key in table)
    lines = ['#pragma once', '#include <cstdint>', 'namespace laya::vulkan_precision {',
             'struct projection_plan { int chunk=0; bool round_partial=false; bool serial=false; bool bias_after_storage=false; };',
             '// Geometry and rounding modes measured with binary probes on RTX PRO 6000,',
             '// PyTorch 2.11 / CUDA 13.0. Both 16-bit formats select the same plans.',
             'inline projection_plan select_projection_plan(int64_t k,int64_t m,int64_t n,bool bias) {']
    for k, m, bias in shapes:
        records = [table[k, m, bias, n] for n in range(1, limit + 1)]
        if not any(any(mode) for mode in records):
            continue
        lines.append(f'    if (k=={k} && m=={m} && bias=={str(bias).lower()}) {{')
        start = 1
        while start <= limit:
            end = start
            while end < limit and records[end] == records[start - 1]:
                end += 1
            chunk, rounded, serial, stored = records[start - 1]
            if any(records[start - 1]):
                mode = ','.join([str(chunk), *map(lambda x: str(x).lower(), (rounded, serial, stored))])
                lines.append(f'        if (n>={start} && n<={end}) return {{{mode}}};')
            start = end + 1
        lines.append('    }')
    lines.extend(['    return {};', '}', '}', ''])
    args.output.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
