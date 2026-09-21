# Fused Vulkan FP32 performance

The fused projection split/merge build passes all 3,000 fixed-corpus comparisons
on both tested GPUs, with exact categories and numeric error at most 0.0001.
These measurements compare native Vulkan and Python FP32 on the same GPU.

Each batch setting uses all 250 fixed questions, three warmups and five timed
runs per request group, alternating execution order. Timings include tokenization,
inference and formatting; model loading and native JSON transport are excluded.
No other builds or GPU tests ran during these measurements. Background services
remained running, so this is not an exclusive-machine claim.

## AMD Radeon 8060S

Mesa 26.2.2-arch3.2; PyTorch 2.11.0 with ROCm 7.13.0. Rates are questions/second.

| Model | Batch | Python FP32 | Vulkan compensated FP32 | Speedup |
|---|---:|---:|---:|---:|
| english | 1 | 16.8 | 19.4 | 1.16× |
| english | 2 | 18.0 | 25.6 | 1.42× |
| english | 4 | 17.3 | 27.0 | 1.56× |
| english | 8 | 15.5 | 27.4 | 1.76× |

Measurements are added after each model finishes all four batch settings.
The remaining AMD models and the NVIDIA sweep are in progress.

See [validation](measurements/vulkan-fused-fp32.json) and
[measurement metadata](measurements/vulkan-fused-performance.json) for the
source revision, binary fingerprints, model identities and full statistics.
These are FP32 results; 16-bit Vulkan inference is still under development.
