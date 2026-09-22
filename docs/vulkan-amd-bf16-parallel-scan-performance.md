# AMD Vulkan BF16 performance after parallel scaling scans

Matching-precision Python ROCm and native Vulkan on Radeon 8060S. Each batch
setting uses all 250 fixed questions, three warmups and five timed iterations
per request group, alternating execution order. Timing includes tokenization,
inference and formatting; it excludes model loading and native JSON transport.
No concurrent GPU experiments or builds ran during timing.

Rates are questions/second. Only completed, passing runs appear below.

| Model | Precision | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | BF16 | 1 | 32.2 | 15.9 | 0.49× |
| english | BF16 | 2 | 37.4 | 19.7 | 0.53× |
| english | BF16 | 4 | 48.3 | 24.2 | 0.50× |
| english | BF16 | 8 | 48.5 | 27.6 | 0.57× |
| multilingual | BF16 | 1 | 55.7 | 38.9 | 0.70× |
| multilingual | BF16 | 2 | 63.4 | 48.9 | 0.77× |
| multilingual | BF16 | 4 | 73.0 | 60.0 | 0.82× |
| multilingual | BF16 | 8 | 68.0 | 62.7 | 0.92× |

See [measurement identities](measurements/vulkan-amd-bf16-parallel-scan-performance.json)
and [BF16 acceptance](measurements/vulkan-amd-bf16-parallel-scan-validation.json).
