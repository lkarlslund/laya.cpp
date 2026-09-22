# Fused normalization-rounding performance

Warmed native Vulkan comparisons on RTX PRO 6000 Blackwell capped at **450 W**.
Each row covers all 250 fixed questions, three warmups and five timed iterations
per request group, alternating the two builds. Loading and native JSON transport
are excluded. Both builds retain their own shared libraries. Background services
retained GPU allocations; no other GPU experiments or builds ran concurrently.

All measured answers meet exact categories and absolute numeric tolerance 0.0001.
The after build also passed the [6,000-case Python comparison](measurements/vulkan-norm-16bit-validation.json).
This table measures the normalization-rounding change; it does not measure speed relative to Python.

Rates are questions/second.

| Model | Precision | Batch | Before | Fused normalization rounding | Speedup |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 135.4 | 137.1 | 1.01× |
| english | FP16 | 2 | 241.9 | 243.2 | 1.01× |
| english | FP16 | 4 | 382.7 | 386.6 | 1.01× |
| english | FP16 | 8 | 507.4 | 513.5 | 1.01× |
| english | BF16 | 1 | 133.7 | 134.7 | 1.01× |
| english | BF16 | 2 | 235.8 | 237.8 | 1.01× |
| english | BF16 | 4 | 385.5 | 390.3 | 1.01× |
| english | BF16 | 8 | 512.7 | 520.7 | 1.02× |
| multilingual | BF16 | 1 | 181.8 | 184.0 | 1.01× |
| multilingual | BF16 | 2 | 314.7 | 316.0 | 1.00× |
| multilingual | BF16 | 4 | 479.8 | 487.8 | 1.02× |
| multilingual | BF16 | 8 | 609.0 | 618.1 | 1.01× |

Measured throughput change is 0.4–1.7% in the completed runs.
Measurements for the remaining model/precision pairs are pending.
See [measurement metadata](measurements/vulkan-norm-performance.json) for build,
weight, corpus and report identities. Matching Python timings for this build remain pending.
AMD 16-bit timing awaits its correctness gate.
