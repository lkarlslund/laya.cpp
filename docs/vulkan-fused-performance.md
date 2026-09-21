# Fused Vulkan FP32 performance

The fused projection split/merge build passes all 3,000 fixed-corpus comparisons
on both tested GPUs, with exact categories and numeric error at most 0.0001.
These measurements compare native Vulkan and Python FP32 on the same GPU.

Each batch setting uses all 250 fixed questions, three warmups and five timed
runs per request group, alternating execution order. Timings include tokenization,
inference and formatting; model loading and native JSON transport are excluded.
No other builds or GPU tests ran during these measurements. Background services
remained running, so this is not an exclusive-machine claim.

## AMD Radeon 8060S Graphics

Mesa 26.2.2-arch3.2; PyTorch 2.11.0+rocm7.13.0 (HIP 7.13.99004).
Rates are questions/second.

| Model | Batch | Python FP32 | Vulkan compensated FP32 | Speedup |
|---|---:|---:|---:|---:|
| english | 1 | 16.8 | 19.4 | 1.16× |
| english | 2 | 18.0 | 25.6 | 1.42× |
| english | 4 | 17.3 | 27.0 | 1.56× |
| english | 8 | 15.5 | 27.4 | 1.76× |
| multilingual | 1 | 35.4 | 58.9 | 1.66× |
| multilingual | 2 | 38.0 | 73.7 | 1.94× |
| multilingual | 4 | 36.3 | 76.6 | 2.11× |
| multilingual | 8 | 31.4 | 68.8 | 2.19× |
| typed-decisions | 1 | 13.4 | 16.8 | 1.25× |
| typed-decisions | 2 | 14.7 | 22.2 | 1.51× |
| typed-decisions | 4 | 13.6 | 22.8 | 1.67× |
| typed-decisions | 8 | 11.8 | 20.7 | 1.76× |

## NVIDIA RTX PRO 6000 Blackwell Workstation Edition

Power capped at **450 W**, driver 610.43.03; PyTorch 2.11.0+cu130.
Rates are questions/second.

| Model | Batch | Python FP32 | Vulkan compensated FP32 | Speedup |
|---|---:|---:|---:|---:|
| english | 1 | 144.3 | 76.4 | 0.53× |
| english | 2 | 207.0 | 140.0 | 0.68× |
| english | 4 | 251.3 | 214.9 | 0.85× |
| english | 8 | 258.1 | 259.8 | 1.01× |
| multilingual | 1 | 181.4 | 93.1 | 0.51× |
| multilingual | 2 | 299.3 | 172.6 | 0.58× |
| multilingual | 4 | 405.3 | 270.2 | 0.67× |
| multilingual | 8 | 443.0 | 333.1 | 0.75× |
| typed-decisions | 1 | 132.2 | 66.9 | 0.51× |
| typed-decisions | 2 | 181.3 | 117.5 | 0.65× |
| typed-decisions | 4 | 197.6 | 158.9 | 0.80× |
| typed-decisions | 8 | 179.3 | 177.3 | 0.99× |

The native CUDA/Vulkan pair was timed separately from the Python/Vulkan pair.

| Model | Batch | CUDA optimized FP32 | Vulkan compensated FP32 |
|---|---:|---:|---:|
| english | 1 | 338.7 | 76.3 |
| english | 2 | 441.0 | 139.8 |
| english | 4 | 477.3 | 212.2 |
| english | 8 | 413.3 | 253.5 |
| multilingual | 1 | 477.4 | 96.4 |
| multilingual | 2 | 632.1 | 175.0 |
| multilingual | 4 | 719.0 | 271.7 |
| multilingual | 8 | 590.0 | 331.7 |

Measurements are added after each model finishes all four batch settings.
Further model and GPU comparisons are in progress.

See [validation](measurements/vulkan-fused-fp32.json) and
[measurement metadata](measurements/vulkan-fused-performance.json) for the
source revision, binary fingerprints, model identities and full statistics.
These are FP32 results; 16-bit Vulkan inference is still under development.
