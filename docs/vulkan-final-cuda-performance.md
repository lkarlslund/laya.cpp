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

See [measurement metadata](measurements/vulkan-final-cuda-performance.json) for
build, weight, corpus and report identities.
