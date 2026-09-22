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

See [measurement identities](measurements/vulkan-amd-bf16-parallel-scan-performance.json)
and [BF16 acceptance](measurements/vulkan-amd-bf16-parallel-scan-validation.json).
