# RTX performance

The 2026-09-20 rerun supersedes the earlier measurements affected by GPU contention.
The RTX PRO 6000 Blackwell reported 0% compute utilization before starting. Another
model remained resident in VRAM; per-process utilization was unavailable, so this
was not a verified exclusive-device run. Aggregate telemetry was recorded every
500 ms. The configured power limit was 450 W; software power limiting was active
and hardware thermal slowdown inactive in two spot checks.

The implementation is `08fa472`, using `--tensor-core-fp32 --flash-fp32`. No runtime
code changed for this rerun. Both comparisons used five warmups and ten timed
iterations per request group, with alternating backend order.

## FP32 comparison

The baseline uses FP32 with autocast and TF32 disabled. These figures do not
establish a speedup over default BF16 execution.

| Batch | FP32 baseline questions/s | Native questions/s | Throughput speedup |
|---:|---:|---:|---:|
| 1 | 148.5 | 350.6 | 2.36× |
| 2 | 201.3 | 441.4 | 2.19× |
| 4 | 228.7 | 446.5 | 1.95× |
| 8 | 226.9 | 394.4 | 1.74× |

## Last optimization versus the previous native build

Both native executables retained their own shared libraries. The previous build
is `e71c846`; the current build adds fused encoder MLP processing and device-side
attention mask generation. Each received identical request groups.

| Batch | Previous native questions/s | Current native questions/s | Throughput gain |
|---:|---:|---:|---:|
| 1 | 301.1 | 345.1 | 14.6% |
| 2 | 370.7 | 424.1 | 14.4% |
| 4 | 373.6 | 436.0 | 16.7% |
| 8 | 333.5 | 387.1 | 16.1% |

All 250 questions passed again at every batch size: exact categorical outputs and
numeric absolute error no greater than 0.0001. The sweeps also required the existing
passing validation report to match the executable, weights, corpus, and settings.

Times include preprocessing and output construction; model loading and JSON
transport are excluded. Throughput includes the full corpus, long inputs, and
partial batches. Aggregate throughput and p50 latency summarize different aspects
of this mixed workload.

See [rerun metadata](measurements/rtx-pro-6000-rerun.json),
[superseded measurements](measurements/rtx-pro-6000-fused-mlp.json), and
[benchmarking instructions](benchmarking.md) for reproducibility details.
