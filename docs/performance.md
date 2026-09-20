# RTX performance

The latest optimization fuses encoder MLP product recombination, exact-erf GELU
gating, and activation splitting. Attention masks are now generated on the device
from sequence lengths, avoiding quadratic CPU work and host-to-device mask copies.
The run command is unchanged: `--tensor-core-fp32 --flash-fp32`.

## Incremental improvement

Measured on 2026-09-20 on an RTX PRO 6000 Blackwell Workstation Edition, comparing
against the preceding native build (`e71c846`). Both executables retained their own
shared libraries. Each received the same requests, with alternating timing order.

| Batch | Previous native questions/s | Updated native questions/s | Throughput gain |
|---:|---:|---:|---:|
| 1 | 293.9 | 343.7 | 16.9% |
| 2 | 236.7 | 278.9 | 17.8% |
| 4 | 214.6 | 230.7 | 7.5% |
| 8 | 230.5 | 259.7 | 12.6% |

The GPU was shared with another active application. Contention varied substantially
during these runs; use the paired comparison above for the incremental improvement.
Do not multiply these ratios by historical results to estimate an isolated speedup.

## Fresh FP32 comparison

A separate sweep compared the updated native executable against an FP32 baseline
with autocast and TF32 disabled. It does not establish a speedup over BF16 execution.

| Batch | FP32 baseline questions/s | Native questions/s | Throughput speedup |
|---:|---:|---:|---:|
| 1 | 117.7 | 326.2 | 2.77× |
| 2 | 152.4 | 373.2 | 2.45× |
| 4 | 114.2 | 187.1 | 1.64× |
| 8 | 221.2 | 385.3 | 1.74× |

All 250 fixed questions passed at every listed batch size: exact categorical
outputs and numeric absolute error no greater than 0.0001. Tokenization and batch
input tensors matched exactly. Repeated graph execution returned identical
outputs. Raw tensor differences remain diagnostic, rather than bitwise identity
claims. Dedicated CPU and CUDA tests cover the fused MLP and changing mask lengths
across graph replays, including short, padded, and maximum-length sequences.

Both sweeps used three warmups and five timed iterations per request group.
Times include preprocessing and output construction; loading and JSON transport
are excluded. Throughput includes the full corpus, long inputs, and partial batches.

See [latest measurement metadata](measurements/rtx-pro-6000-fused-mlp.json),
[earlier measurements](measurements/rtx-pro-6000-fp32.json), and
[benchmarking instructions](benchmarking.md) for reproduction details.
