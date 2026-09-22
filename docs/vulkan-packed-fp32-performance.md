# Vulkan FP32 performance after QKV packing

These paired measurements use the validated QKV-packing build, preserved with
its shared libraries. Python runs on the same GPU as Vulkan: CUDA on the RTX PRO
6000 Blackwell capped at **450 W**, and ROCm on the Radeon 8060S. Plain FP32 and
compensated projections each use matching Python FP32 as their baseline.

Each row covers all 250 fixed questions, three warmups and five timed iterations
per group. Execution order alternates. Tokenization, inference and formatting
are timed; model loading and native JSON transport are excluded. Background
services retain allocations; no other GPU experiments or builds run concurrently.
Every recorded answer comparison passes exact categories and absolute numeric
tolerance 0.0001.

Rates are questions/second.

| GPU | Model | Mode | Batch | Python FP32 | Vulkan | Vulkan/Python |
|---|---|---|---:|---:|---:|---:|
| NVIDIA | english | plain | 1 | 141.2 | 62.3 | 0.44× |
| NVIDIA | english | plain | 2 | 205.2 | 104.7 | 0.51× |
| NVIDIA | english | plain | 4 | 244.9 | 137.3 | 0.56× |
| NVIDIA | english | plain | 8 | 236.0 | 166.5 | 0.71× |
| NVIDIA | english | compensated | 1 | 147.0 | 117.6 | 0.80× |
| NVIDIA | english | compensated | 2 | 204.9 | 178.1 | 0.87× |
| NVIDIA | english | compensated | 4 | 247.3 | 240.2 | 0.97× |
| NVIDIA | english | compensated | 8 | 246.6 | 280.1 | 1.14× |
| NVIDIA | multilingual | plain | 1 | 183.4 | 109.6 | 0.60× |
| NVIDIA | multilingual | plain | 2 | 294.0 | 164.5 | 0.56× |
| NVIDIA | multilingual | plain | 4 | 395.4 | 254.7 | 0.64× |
| NVIDIA | multilingual | plain | 8 | 419.5 | 284.1 | 0.68× |
| NVIDIA | multilingual | compensated | 1 | 188.9 | 114.3 | 0.60× |
| NVIDIA | multilingual | compensated | 2 | 304.4 | 204.3 | 0.67× |
| NVIDIA | multilingual | compensated | 4 | 405.3 | 305.9 | 0.75× |
| NVIDIA | multilingual | compensated | 8 | 432.1 | 358.2 | 0.83× |
| NVIDIA | typed-decisions | compensated | 1 | 133.8 | 98.1 | 0.73× |
| NVIDIA | typed-decisions | compensated | 2 | 177.5 | 144.0 | 0.81× |
| NVIDIA | typed-decisions | compensated | 4 | 191.7 | 172.8 | 0.90× |
| NVIDIA | typed-decisions | compensated | 8 | 169.9 | 187.8 | 1.11× |

Measurements for remaining GPU/model/mode combinations are pending.

See [measurement metadata](measurements/vulkan-packed-fp32-performance.json) for
binary, weight, corpus and report identities.
