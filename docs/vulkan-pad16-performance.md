# Bit-preserving weight-padding performance

Warmed native Vulkan comparisons on RTX PRO 6000 Blackwell capped at **450 W**.
Each row covers all 250 fixed questions, three warmups and five timed iterations
per request group, alternating the two builds. Loading and native JSON transport
are excluded. Both builds retain their own shared libraries. Background services
retained GPU allocations; no other GPU experiments or builds ran concurrently.

All measured answers meet exact categories and absolute numeric tolerance 0.0001.
The after build also passed the [6,000-case Python comparison](measurements/vulkan-pad16-16bit-validation.json).
This table measures the weight-padding change; it does not measure speed relative to Python.

Rates are questions/second.

| Model | Precision | Batch | Before | Bit-preserving padding | Speedup |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 134.3 | 139.3 | 1.04× |
| english | FP16 | 2 | 243.7 | 248.5 | 1.02× |
| english | FP16 | 4 | 384.8 | 390.7 | 1.02× |
| english | FP16 | 8 | 514.1 | 515.5 | 1.00× |
| english | BF16 | 1 | 138.0 | 144.3 | 1.05× |
| english | BF16 | 2 | 245.3 | 252.5 | 1.03× |
| english | BF16 | 4 | 392.4 | 400.0 | 1.02× |
| english | BF16 | 8 | 521.2 | 521.9 | 1.00× |
| multilingual | FP16 | 1 | 182.6 | 183.0 | 1.00× |
| multilingual | FP16 | 2 | 319.0 | 317.9 | 1.00× |
| multilingual | FP16 | 4 | 492.9 | 492.6 | 1.00× |
| multilingual | FP16 | 8 | 631.5 | 631.8 | 1.00× |
| multilingual | BF16 | 1 | 184.1 | 184.7 | 1.00× |
| multilingual | BF16 | 2 | 318.2 | 319.3 | 1.00× |
| multilingual | BF16 | 4 | 490.1 | 489.8 | 1.00× |
| multilingual | BF16 | 8 | 624.6 | 624.8 | 1.00× |
| typed-decisions | BF16 | 1 | 132.0 | 136.2 | 1.03× |
| typed-decisions | BF16 | 2 | 211.9 | 216.6 | 1.02× |
| typed-decisions | BF16 | 4 | 312.6 | 318.0 | 1.02× |
| typed-decisions | BF16 | 8 | 355.2 | 357.5 | 1.01× |

Measured throughput change is -0.3–4.6% in the completed runs.
Measurements for the remaining model/precision pairs are pending.
See [measurement metadata](measurements/vulkan-pad16-performance.json) for build,
weight, corpus and report identities. Matching Python timings for this build remain pending.
AMD 16-bit timing awaits its correctness gate.
