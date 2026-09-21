# Fused projection-storage performance

Warmed native Vulkan comparisons on RTX PRO 6000 Blackwell capped at **450 W**.
Each row covers all 250 fixed questions, three warmups and five timed iterations
per request group, alternating the two builds. Loading and native JSON transport
are excluded. Both builds retain their own shared libraries. Background services
retained GPU allocations; no other GPU experiments or builds ran concurrently.

All measured answers meet exact categories and absolute numeric tolerance 0.0001.
The after build also passed the [6,000-case Python comparison](measurements/vulkan-finish-16bit-validation.json).
This table measures the projection-storage change; it does not measure speed relative to Python.

Rates are questions/second.

| Model | Precision | Batch | Before | Fused storage | Speedup |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 128.1 | 131.4 | 1.03× |
| english | FP16 | 2 | 228.6 | 234.8 | 1.03× |
| english | FP16 | 4 | 367.4 | 377.7 | 1.03× |
| english | FP16 | 8 | 485.0 | 498.9 | 1.03× |
| english | BF16 | 1 | 128.6 | 131.8 | 1.03× |
| english | BF16 | 2 | 229.4 | 237.3 | 1.03× |
| english | BF16 | 4 | 366.4 | 378.1 | 1.03× |
| english | BF16 | 8 | 488.6 | 503.7 | 1.03× |
| multilingual | FP16 | 1 | 168.6 | 175.4 | 1.04× |
| multilingual | FP16 | 2 | 296.4 | 307.2 | 1.04× |
| multilingual | FP16 | 4 | 460.7 | 475.0 | 1.03× |
| multilingual | FP16 | 8 | 593.3 | 614.2 | 1.04× |
| multilingual | BF16 | 1 | 168.8 | 176.3 | 1.04× |
| multilingual | BF16 | 2 | 294.8 | 306.7 | 1.04× |
| multilingual | BF16 | 4 | 460.0 | 474.7 | 1.03× |
| multilingual | BF16 | 8 | 590.2 | 610.9 | 1.03× |

Projection storage improves throughput by 2.5–4.4% in the completed runs.
Measurements for the remaining model/precision pairs are pending.
See [measurement metadata](measurements/vulkan-finish-performance.json) for build,
weight, corpus and report identities. Matching Python timings for this build remain pending.
AMD 16-bit timing awaits its correctness gate.
