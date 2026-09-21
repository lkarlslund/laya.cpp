#!/usr/bin/env python3
"""Generate the 257-key CUDA SDPA fixture for Vulkan partition tests."""
import argparse
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('tests/vulkan_partition_golden.hpp'))
    args=parser.parse_args()
    if torch.version.hip or not torch.cuda.is_available():
        parser.error('This fixture requires CUDA PyTorch')
    length,heads,width=257,2,64
    q=torch.zeros((1,heads,length,width)); q[:,:,:,0]=1
    k=torch.zeros_like(q)
    tokens=torch.arange(length)
    k[:,:,:,0]=(tokens%13-6)/8
    values=(tokens[:,None]*3+torch.arange(width)[None,:]*5)
    v=torch.stack([((values+h*7)%31-15)/64 for h in range(heads)])[None]
    q,k,v=(value.cuda() for value in (q,k,v))
    lines=[f'// PyTorch {torch.__version__}, {torch.cuda.get_device_name()}; 257-key SDPA.',
           '#pragma once','#include <cstdint>']
    for name,dtype in [('bf16',torch.bfloat16),('fp16',torch.float16)]:
        y=torch.nn.functional.scaled_dot_product_attention(q.to(dtype),k.to(dtype),v.to(dtype))
        if not torch.equal(y,y[:,:,:1,:].expand_as(y)):
            raise RuntimeError('Identical queries produced different fixture outputs')
        bits=y[0,:,0,:].view(torch.uint16).cpu().flatten().tolist()
        lines.append(f'inline constexpr uint16_t partition_{name}[128] = {{')
        for i in range(0,128,8): lines.append('    '+','.join(hex(n) for n in bits[i:i+8])+',')
        lines.append('};')
    args.output.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
