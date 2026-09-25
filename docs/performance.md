# Performance

Latest recorded measurements for each comparison, across all three models and
the 250 fixed questions at batches 1, 2, 4 and 8. Rates are questions/second.

NVIDIA: RTX PRO 6000 Blackwell (96 GB), capped at **450 W**.
AMD: Radeon 8060S, with same-GPU Python ROCm baselines.

The full-corpus comparisons below alternate execution order, with three warmups
and five timed iterations per request group. Timing includes preprocessing,
inference and output formatting; model loading and native JSON transport are
excluded. Background services retained GPU allocations for those measurements.
All recorded public-answer checks pass exact categories and numeric absolute
tolerance **0.0001** at matching precision.

These are separate measured runs, not one simultaneous benchmark. Compare speeds
within a table row. Build, weights, corpus and environment identities are linked
below each table. Historical optimization experiments are omitted.

## CPU-calibrated CUDA quick corpus (2026-09-25)

This is a **small-sample diagnostic**, separate from the 250-question matrix
below. It uses eight fixed requests in each of 16 length/output-shape strata,
batch size 4, one warmup and one timed pass. Each row has only two timed batch
calls, so the rates are useful for spotting broad regressions, not for stable
tail-latency claims. Full acceptance-250 passed at batch sizes 1, 2, 4 and 8
for all three checkpoints; every selected quick answer also passed exact
categories and numeric absolute error at most 0.0001.

NVIDIA RTX PRO 6000 Blackwell, 450 W limit, driver 615.71.09. The NInfer server
was stopped during measurement; the Laya server retained its GPU allocation.
Optimized FP32 CUDA used `--tensor-core-fp32 --flash-fp32` on the same binary.

| Model | Short q1 speedup | Limit q8 speedup | Range over all 16 strata |
| --- | ---: | ---: | ---: |
| english | 2.36× | 1.46× | 1.42–2.36× |
| multilingual | 2.44× | 1.01× | 1.01–2.55× |
| typed-decisions | 2.36× | 1.29× | 1.22–2.36× |

The [quick-run manifest](measurements/cuda-quick-2026-09-25/manifest.json)
links all 48 timed reports, their per-request parity JSON, three full acceptance
reports, and the fixed corpus hash. Within each stratum, compare the paired
baseline and native rates rather than combining different lengths into one
throughput number.

## CUDA versus Python

The FP32 mode uses `--tensor-core-fp32 --flash-fp32`; Python FP32 disables
autocast and TF32. BF16 uses `--bf16`. Native CUDA FP16 is unsupported.

| Model | Precision | Batch | Python | CUDA | CUDA/Python |
|---|---|---:|---:|---:|---:|
| english | FP32 | 1 | 147.9 | 341.7 | 2.31× |
| english | FP32 | 2 | 201.6 | 420.6 | 2.09× |
| english | FP32 | 4 | 232.9 | 437.2 | 1.88× |
| english | FP32 | 8 | 231.7 | 385.7 | 1.66× |
| english | BF16 | 1 | 149.2 | 365.8 | 2.45× |
| english | BF16 | 2 | 267.5 | 585.6 | 2.19× |
| english | BF16 | 4 | 459.8 | 761.1 | 1.66× |
| english | BF16 | 8 | 662.5 | 809.8 | 1.22× |
| multilingual | FP32 | 1 | 191.7 | 475.3 | 2.48× |
| multilingual | FP32 | 2 | 299.8 | 610.0 | 2.03× |
| multilingual | FP32 | 4 | 388.5 | 673.4 | 1.73× |
| multilingual | FP32 | 8 | 408.0 | 552.1 | 1.35× |
| multilingual | BF16 | 1 | 178.9 | 485.4 | 2.71× |
| multilingual | BF16 | 2 | 320.7 | 816.0 | 2.54× |
| multilingual | BF16 | 4 | 550.2 | 1164.0 | 2.12× |
| multilingual | BF16 | 8 | 828.1 | 1246.4 | 1.51× |
| typed-decisions | FP32 | 1 | 130.6 | 263.4 | 2.02× |
| typed-decisions | FP32 | 2 | 169.9 | 302.4 | 1.78× |
| typed-decisions | FP32 | 4 | 178.6 | 284.1 | 1.59× |
| typed-decisions | FP32 | 8 | 164.3 | 238.7 | 1.45× |
| typed-decisions | BF16 | 1 | 142.7 | 303.9 | 2.13× |
| typed-decisions | BF16 | 2 | 248.0 | 498.8 | 2.01× |
| typed-decisions | BF16 | 4 | 399.2 | 610.3 | 1.53× |
| typed-decisions | BF16 | 8 | 529.0 | 587.7 | 1.11× |

[Measurement metadata](measurements/readme-performance.json). This is the latest
paired CUDA/Python matrix; a newer native-only comparison appears below.

## Vulkan versus Python: NVIDIA

FP32 compensated uses `--tensor-core-fp32`. FP16 and BF16 use `--fp16`
and `--bf16`, respectively. Each uses matching-precision Python on the same GPU.

| Model | Mode | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | FP32 plain | 1 | 141.2 | 62.3 | 0.44× |
| english | FP32 plain | 2 | 205.2 | 104.7 | 0.51× |
| english | FP32 plain | 4 | 244.9 | 137.3 | 0.56× |
| english | FP32 plain | 8 | 236.0 | 166.5 | 0.71× |
| english | FP32 compensated | 1 | 147.0 | 117.6 | 0.80× |
| english | FP32 compensated | 2 | 204.9 | 178.1 | 0.87× |
| english | FP32 compensated | 4 | 247.3 | 240.2 | 0.97× |
| english | FP32 compensated | 8 | 246.6 | 280.1 | 1.14× |
| multilingual | FP32 plain | 1 | 183.4 | 109.6 | 0.60× |
| multilingual | FP32 plain | 2 | 294.0 | 164.5 | 0.56× |
| multilingual | FP32 plain | 4 | 395.4 | 254.7 | 0.64× |
| multilingual | FP32 plain | 8 | 419.5 | 284.1 | 0.68× |
| multilingual | FP32 compensated | 1 | 188.9 | 114.3 | 0.60× |
| multilingual | FP32 compensated | 2 | 304.4 | 204.3 | 0.67× |
| multilingual | FP32 compensated | 4 | 405.3 | 305.9 | 0.75× |
| multilingual | FP32 compensated | 8 | 432.1 | 358.2 | 0.83× |
| typed-decisions | FP32 plain | 1 | 126.8 | 57.2 | 0.45× |
| typed-decisions | FP32 plain | 2 | 169.6 | 95.5 | 0.56× |
| typed-decisions | FP32 plain | 4 | 183.5 | 113.3 | 0.62× |
| typed-decisions | FP32 plain | 8 | 166.9 | 122.1 | 0.73× |
| typed-decisions | FP32 compensated | 1 | 133.8 | 98.1 | 0.73× |
| typed-decisions | FP32 compensated | 2 | 177.5 | 144.0 | 0.81× |
| typed-decisions | FP32 compensated | 4 | 191.7 | 172.8 | 0.90× |
| typed-decisions | FP32 compensated | 8 | 169.9 | 187.8 | 1.11× |
| english | FP16 | 1 | 147.6 | 139.8 | 0.95× |
| english | FP16 | 2 | 265.7 | 248.1 | 0.93× |
| english | FP16 | 4 | 451.5 | 391.0 | 0.87× |
| english | FP16 | 8 | 668.5 | 513.1 | 0.77× |
| multilingual | FP16 | 1 | 177.1 | 178.8 | 1.01× |
| multilingual | FP16 | 2 | 318.6 | 316.0 | 0.99× |
| multilingual | FP16 | 4 | 543.0 | 487.3 | 0.90× |
| multilingual | FP16 | 8 | 841.0 | 623.8 | 0.74× |
| typed-decisions | FP16 | 1 | 145.3 | 133.3 | 0.92× |
| typed-decisions | FP16 | 2 | 250.3 | 215.6 | 0.86× |
| typed-decisions | FP16 | 4 | 404.4 | 314.4 | 0.78× |
| typed-decisions | FP16 | 8 | 549.2 | 351.1 | 0.64× |
| english | BF16 | 1 | 147.6 | 142.4 | 0.96× |
| english | BF16 | 2 | 264.9 | 248.8 | 0.94× |
| english | BF16 | 4 | 453.9 | 398.6 | 0.88× |
| english | BF16 | 8 | 660.1 | 515.9 | 0.78× |
| multilingual | BF16 | 1 | 176.9 | 180.2 | 1.02× |
| multilingual | BF16 | 2 | 316.8 | 314.5 | 0.99× |
| multilingual | BF16 | 4 | 543.7 | 482.4 | 0.89× |
| multilingual | BF16 | 8 | 826.4 | 615.3 | 0.74× |
| typed-decisions | BF16 | 1 | 145.0 | 134.8 | 0.93× |
| typed-decisions | BF16 | 2 | 246.6 | 213.3 | 0.86× |
| typed-decisions | BF16 | 4 | 403.1 | 310.3 | 0.77× |
| typed-decisions | BF16 | 8 | 549.2 | 352.0 | 0.64× |

Measurement metadata: [FP32](measurements/vulkan-packed-fp32-performance.json),
[FP16/BF16](measurements/vulkan-final-nvidia-python-performance.json).

## Vulkan versus Python: AMD

FP32 compensated uses `--tensor-core-fp32`. FP16 and BF16 use `--fp16`
and `--bf16`, respectively. Each uses matching-precision Python on the same GPU.

| Model | Mode | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | FP32 plain | 1 | 16.8 | 10.2 | 0.61× |
| english | FP32 plain | 2 | 18.1 | 16.5 | 0.91× |
| english | FP32 plain | 4 | 17.7 | 20.1 | 1.14× |
| english | FP32 plain | 8 | 15.7 | 20.2 | 1.29× |
| english | FP32 compensated | 1 | 16.1 | 20.2 | 1.26× |
| english | FP32 compensated | 2 | 17.2 | 26.3 | 1.53× |
| english | FP32 compensated | 4 | 16.9 | 27.6 | 1.63× |
| english | FP32 compensated | 8 | 15.1 | 27.9 | 1.85× |
| multilingual | FP32 plain | 1 | 35.5 | 33.6 | 0.95× |
| multilingual | FP32 plain | 2 | 37.6 | 51.2 | 1.36× |
| multilingual | FP32 plain | 4 | 36.2 | 59.0 | 1.63× |
| multilingual | FP32 plain | 8 | 31.3 | 57.2 | 1.82× |
| multilingual | FP32 compensated | 1 | 35.7 | 64.3 | 1.80× |
| multilingual | FP32 compensated | 2 | 38.2 | 80.6 | 2.11× |
| multilingual | FP32 compensated | 4 | 36.8 | 84.6 | 2.30× |
| multilingual | FP32 compensated | 8 | 31.4 | 76.9 | 2.45× |
| typed-decisions | FP32 plain | 1 | 13.5 | 9.3 | 0.69× |
| typed-decisions | FP32 plain | 2 | 14.9 | 14.6 | 0.98× |
| typed-decisions | FP32 plain | 4 | 13.8 | 17.0 | 1.23× |
| typed-decisions | FP32 plain | 8 | 12.0 | 16.1 | 1.35× |
| typed-decisions | FP32 compensated | 1 | 13.5 | 17.4 | 1.28× |
| typed-decisions | FP32 compensated | 2 | 14.7 | 22.7 | 1.55× |
| typed-decisions | FP32 compensated | 4 | 13.7 | 23.0 | 1.68× |
| typed-decisions | FP32 compensated | 8 | 11.8 | 22.0 | 1.86× |
| english | FP16 | 1 | 33.1 | 28.1 | 0.85× |
| english | FP16 | 2 | 37.8 | 29.6 | 0.78× |
| english | FP16 | 4 | 48.8 | 35.3 | 0.72× |
| english | FP16 | 8 | 48.5 | 38.6 | 0.79× |
| multilingual | FP16 | 1 | 56.2 | 70.1 | 1.25× |
| multilingual | FP16 | 2 | 63.7 | 77.1 | 1.21× |
| multilingual | FP16 | 4 | 73.7 | 85.5 | 1.16× |
| multilingual | FP16 | 8 | 68.7 | 79.5 | 1.16× |
| typed-decisions | FP16 | 1 | 25.5 | 23.9 | 0.94× |
| typed-decisions | FP16 | 2 | 31.1 | 24.9 | 0.80× |
| typed-decisions | FP16 | 4 | 35.1 | 28.1 | 0.80× |
| typed-decisions | FP16 | 8 | 31.1 | 27.2 | 0.88× |
| english | BF16 | 1 | 32.2 | 15.9 | 0.49× |
| english | BF16 | 2 | 37.4 | 19.7 | 0.53× |
| english | BF16 | 4 | 48.3 | 24.2 | 0.50× |
| english | BF16 | 8 | 48.5 | 27.6 | 0.57× |
| multilingual | BF16 | 1 | 55.7 | 38.9 | 0.70× |
| multilingual | BF16 | 2 | 63.4 | 48.9 | 0.77× |
| multilingual | BF16 | 4 | 73.0 | 60.0 | 0.82× |
| multilingual | BF16 | 8 | 68.0 | 62.7 | 0.92× |
| typed-decisions | BF16 | 1 | 24.9 | 14.3 | 0.57× |
| typed-decisions | BF16 | 2 | 30.9 | 17.3 | 0.56× |
| typed-decisions | BF16 | 4 | 35.1 | 20.5 | 0.59× |
| typed-decisions | BF16 | 8 | 31.9 | 20.6 | 0.65× |

Measurement metadata: [FP32](measurements/vulkan-packed-fp32-performance.json),
[FP16](measurements/vulkan-amd-16bit-python-performance.json) (FP16 entries only),
[BF16](measurements/vulkan-amd-bf16-parallel-scan-performance.json).

## CUDA versus Vulkan: NVIDIA BF16

The latest paired native-backend comparison uses BF16 on both sides.

| Model | Batch | CUDA | Vulkan | Vulkan/CUDA |
|---|---:|---:|---:|---:|
| english | 1 | 364.6 | 139.2 | 0.38× |
| english | 2 | 580.7 | 248.6 | 0.43× |
| english | 4 | 762.3 | 400.3 | 0.53× |
| english | 8 | 815.1 | 517.7 | 0.64× |
| multilingual | 1 | 483.0 | 184.8 | 0.38× |
| multilingual | 2 | 803.5 | 319.3 | 0.40× |
| multilingual | 4 | 1165.9 | 489.4 | 0.42× |
| multilingual | 8 | 1262.7 | 621.0 | 0.49× |
| typed-decisions | 1 | 306.9 | 135.8 | 0.44× |
| typed-decisions | 2 | 505.8 | 215.1 | 0.43× |
| typed-decisions | 4 | 622.8 | 319.4 | 0.51× |
| typed-decisions | 8 | 601.7 | 351.2 | 0.58× |

[Measurement metadata](measurements/vulkan-final-cuda-performance.json).

See [benchmarking](benchmarking.md) to reproduce the runs and
[Vulkan support](vulkan.md) for precision profiles and validation limits.
