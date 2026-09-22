# AMD Vulkan 16-bit performance

Matching-precision Python ROCm and native Vulkan on Radeon 8060S. Each batch
setting uses all 250 fixed questions, three warmups and five timed iterations
per request group, alternating execution order. Timing includes tokenization,
inference and formatting; it excludes model loading and native JSON transport.
No concurrent GPU experiments or builds ran during timing.

Rates are questions/second. Only completed, passing runs appear below.

| Model | Precision | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | FP16 | 1 | 33.1 | 28.1 | 0.85× |
| english | FP16 | 2 | 37.8 | 29.6 | 0.78× |
| english | FP16 | 4 | 48.8 | 35.3 | 0.72× |
| english | FP16 | 8 | 48.5 | 38.6 | 0.79× |
| multilingual | FP16 | 1 | 56.2 | 70.1 | 1.25× |
| multilingual | FP16 | 2 | 63.7 | 77.1 | 1.21× |
| multilingual | FP16 | 4 | 73.7 | 85.5 | 1.16× |
| multilingual | FP16 | 8 | 68.7 | 79.5 | 1.16× |
| typed-decisions | FP16 | 1 | 25.5 | 23.9 | 0.94× |
| typed-decisions | FP16 | 2 | 31.1 | 24.9 | 0.80× |
| typed-decisions | FP16 | 4 | 35.1 | 28.1 | 0.80× |
| typed-decisions | FP16 | 8 | 31.1 | 27.2 | 0.88× |
| english | BF16 | 1 | 31.9 | 12.2 | 0.38× |
| english | BF16 | 2 | 37.2 | 14.0 | 0.38× |
| english | BF16 | 4 | 48.0 | 14.5 | 0.30× |
| english | BF16 | 8 | 48.4 | 16.2 | 0.33× |
| multilingual | BF16 | 1 | 55.3 | 27.8 | 0.50× |
| multilingual | BF16 | 2 | 62.9 | 33.9 | 0.54× |
| multilingual | BF16 | 4 | 73.0 | 38.9 | 0.53× |
| multilingual | BF16 | 8 | 67.8 | 37.6 | 0.55× |

See [measurement identities](measurements/vulkan-amd-16bit-python-performance.json)
and production acceptance for [FP16](measurements/vulkan-amd-fp16-runtime-validation.json)
and [BF16](measurements/vulkan-amd-bf16-runtime-validation.json).
