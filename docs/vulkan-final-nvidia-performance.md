# NVIDIA Vulkan 16-bit performance

Matching-precision Python CUDA and native Vulkan on RTX PRO 6000 Blackwell capped at 450 W. Each batch
setting uses all 250 fixed questions, three warmups and five timed iterations
per request group, alternating execution order. Timing includes tokenization,
inference and formatting; it excludes model loading and native JSON transport.
No concurrent GPU experiments or builds ran during timing.

Rates are questions/second. Only completed, passing runs appear below.

| Model | Precision | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 147.6 | 139.8 | 0.95× |
| english | FP16 | 2 | 265.7 | 248.1 | 0.93× |
| english | FP16 | 4 | 451.5 | 391.0 | 0.87× |
| english | FP16 | 8 | 668.5 | 513.1 | 0.77× |
| multilingual | FP16 | 1 | 177.1 | 178.8 | 1.01× |
| multilingual | FP16 | 2 | 318.6 | 316.0 | 0.99× |
| multilingual | FP16 | 4 | 543.0 | 487.3 | 0.90× |
| multilingual | FP16 | 8 | 841.0 | 623.8 | 0.74× |
| typed-decisions | FP16 | 1 | 145.3 | 133.3 | 0.92× |
| typed-decisions | FP16 | 2 | 250.3 | 215.6 | 0.86× |
| typed-decisions | FP16 | 4 | 404.4 | 314.4 | 0.78× |
| typed-decisions | FP16 | 8 | 549.2 | 351.1 | 0.64× |

See [measurement identities](measurements/vulkan-final-nvidia-python-performance.json)
and [production acceptance](measurements/vulkan-pad16-16bit-validation.json).
