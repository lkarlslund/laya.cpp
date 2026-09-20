# RTX performance

Measured on 2026-09-20 with an RTX PRO 6000 Blackwell Workstation Edition.
The optimized native mode uses `--tensor-core-fp32 --flash-fp32`. The comparison
baseline uses FP32 with autocast and TF32 disabled. These results do not establish
a speedup over BF16 execution. Another application occupied the GPU, so these
are shared-device measurements rather than isolated hardware limits.

| Batch | FP32 baseline questions/s | Native questions/s | Throughput speedup |
|---:|---:|---:|---:|
| 1 | 150.4 | 312.1 | 2.07× |
| 2 | 208.0 | 389.0 | 1.87× |
| 4 | 244.8 | 405.7 | 1.66× |
| 8 | 242.4 | 359.0 | 1.48× |

All 250 fixed questions passed at every listed batch size: exact categorical
outputs and numeric absolute error no greater than 0.0001. Tokenization and batch
input tensors matched exactly. Repeated graph execution returned identical
outputs. Twelve additional edge requests passed at batches 1 and 4.
Raw tensor differences remain diagnostic and are not bitwise identity claims.

The sweep used three warmups and five timed iterations per request group,
alternating backend order. Times include preprocessing and output construction;
model loading and JSON transport are excluded. Aggregate throughput includes
the entire corpus, including long inputs and partial batches.

The primary changes are fused activation splitting/recombination, packed
Tensor Core products with FP32 accumulation, fused rotary Q/K/V packing, and
length-dependent attention dispatch. Learned projection weights retain their
checkpoint FP16 values exactly. Activation splitting is approximate and remains
subject to the public output acceptance gate.

See [measurement metadata](measurements/rtx-pro-6000-fp32.json) for hashes and
summary statistics, and [benchmarking](benchmarking.md) to reproduce the run.
