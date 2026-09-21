# Fused QKV packing performance

Warmed native Vulkan comparisons on RTX PRO 6000 Blackwell capped at **450 W**.
Each row covers all 250 fixed questions, three warmups and five timed iterations
per request group, alternating the two builds. Loading and native JSON transport
are excluded. Both builds retain their own shared libraries. Background services
retained GPU allocations; no other GPU experiments or builds ran concurrently.

All measured answers meet exact categories and absolute numeric tolerance 0.0001.
The after build also passed the [6,000-case Python comparison](measurements/vulkan-packed-16bit-validation.json).
This table measures the packing change; it does not measure speed relative to Python.

Rates are questions/second.

| Model | Precision | Batch | Before | Fused packing | Speedup |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 111.7 | 127.5 | 1.14× |
| english | FP16 | 2 | 199.0 | 227.3 | 1.14× |
| english | FP16 | 4 | 320.1 | 363.2 | 1.13× |
| english | FP16 | 8 | 425.5 | 491.2 | 1.15× |
| english | BF16 | 1 | 112.7 | 129.0 | 1.14× |
| english | BF16 | 2 | 201.0 | 229.4 | 1.14× |
| english | BF16 | 4 | 320.6 | 364.8 | 1.14× |
| english | BF16 | 8 | 425.9 | 491.7 | 1.15× |
| multilingual | FP16 | 1 | 145.9 | 169.1 | 1.16× |
| multilingual | FP16 | 2 | 257.1 | 297.5 | 1.16× |
| multilingual | FP16 | 4 | 404.6 | 459.9 | 1.14× |
| multilingual | FP16 | 8 | 523.3 | 593.1 | 1.13× |
| multilingual | BF16 | 1 | 146.1 | 169.5 | 1.16× |
| multilingual | BF16 | 2 | 255.5 | 295.2 | 1.16× |
| multilingual | BF16 | 4 | 406.1 | 461.9 | 1.14× |
| multilingual | BF16 | 8 | 517.6 | 589.8 | 1.14× |

Packing improves throughput by 13–16% in the completed runs.
Measurements for the remaining model/precision pairs are pending.
See [measurement metadata](measurements/vulkan-packed-performance.json) for build,
weight, corpus and report identities. Matching Python timings for this build and
AMD packing timings remain pending.
