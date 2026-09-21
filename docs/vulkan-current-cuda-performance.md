# Native CUDA and Vulkan BF16 performance

Warmed comparisons on RTX PRO 6000 Blackwell capped at **450 W**, using all 250
fixed questions at each batch size, three warmups and five timed iterations per
group. The harness alternates execution order and checks exact categories and
numeric absolute tolerance 0.0001. Model loading and JSON transport are excluded.
Background services retained GPU allocations; no other GPU experiments or builds ran concurrently.

The Vulkan build passed all 6,000 matching-precision Python answer comparisons.
These measurements compare native backends in BF16; native CUDA FP16 is unsupported.

Rates are questions/second.

| Model | Batch | CUDA BF16 | Vulkan BF16 | Vulkan/CUDA |
|---|---:|---:|---:|---:|
| english | 1 | 364.0 | 136.2 | 0.37× |
| english | 2 | 580.9 | 243.5 | 0.42× |
| english | 4 | 758.6 | 384.1 | 0.51× |
| english | 8 | 816.6 | 508.0 | 0.62× |
| multilingual | 1 | 483.2 | 182.0 | 0.38× |
| multilingual | 2 | 811.9 | 317.0 | 0.39× |
| multilingual | 4 | 1166.1 | 482.0 | 0.41× |
| multilingual | 8 | 1264.2 | 615.3 | 0.49× |

Measurements for the remaining models are pending.

See [measurement metadata](measurements/vulkan-current-cuda-performance.json) for
build, weight, corpus and report identities.
