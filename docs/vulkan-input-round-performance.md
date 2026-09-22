# Fused input-rounding performance

Warmed native Vulkan comparisons on RTX PRO 6000 Blackwell capped at **450 W**.
Each row covers all 250 fixed questions, three warmups and five timed iterations
per request group, alternating the two builds. Loading and native JSON transport
are excluded. Both builds retain their own shared libraries. Background services
retained GPU allocations; no other GPU experiments or builds ran concurrently.

All measured answers meet exact categories and absolute numeric tolerance 0.0001.
The after build also passed the [6,000-case Python comparison](measurements/vulkan-input-round-16bit-validation.json).
This table measures the input-rounding change; it does not measure speed relative to Python.

Rates are questions/second.

| Model | Precision | Batch | Before | Fused input rounding | Speedup |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 132.5 | 135.5 | 1.02× |
| english | FP16 | 2 | 237.7 | 241.4 | 1.02× |
| english | FP16 | 4 | 376.9 | 382.8 | 1.02× |
| english | FP16 | 8 | 505.4 | 509.6 | 1.01× |

Input rounding improves throughput by 0.8–2.3% in the completed runs.
This archived experiment contains only the completed pairs shown above.
See [measurement metadata](measurements/vulkan-input-round-performance.json) for build,
weight, corpus and report identities. This intermediate-build comparison is retained for its optimization history;
see [the optimized NVIDIA build versus Python](vulkan-final-nvidia-performance.md) for the complete comparison.
Completed AMD FP16/BF16 baselines are recorded in [AMD measurements](vulkan-amd-16bit-performance.md).
