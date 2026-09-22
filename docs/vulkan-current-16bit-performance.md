# Current Vulkan 16-bit performance

These warmed measurements compare native Vulkan with matching-precision Python
on RTX PRO 6000 Blackwell capped at **450 W**. All measured groups pass exact
categories and numeric absolute tolerance 0.0001. The [combined correctness gate](measurements/vulkan-input-round-16bit-validation.json)
covers all three models and both precisions. Timing rows below include only completed runs.

Each batch setting uses all 250 fixed questions, three warmups and five timed
iterations per request group, alternating execution order. Timing includes
tokenization, inference and formatting, excluding model loading and native JSON
transport. Background services retained GPU allocations. No concurrent GPU
experiments or builds ran during these measurements.

Rates are questions/second.

| Model | Precision | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | BF16 | 1 | 145.5 | 134.8 | 0.93× |
| english | BF16 | 2 | 263.1 | 240.2 | 0.91× |
| english | BF16 | 4 | 449.0 | 384.3 | 0.86× |
| english | BF16 | 8 | 671.8 | 508.1 | 0.76× |
| english | FP16 | 1 | 145.4 | 133.6 | 0.92× |
| english | FP16 | 2 | 264.3 | 238.9 | 0.90× |
| english | FP16 | 4 | 447.1 | 379.5 | 0.85× |
| english | FP16 | 8 | 662.2 | 501.7 | 0.76× |
| multilingual | BF16 | 1 | 176.3 | 177.7 | 1.01× |
| multilingual | BF16 | 2 | 316.0 | 312.8 | 0.99× |
| multilingual | BF16 | 4 | 539.9 | 478.5 | 0.89× |
| multilingual | BF16 | 8 | 829.5 | 607.7 | 0.73× |
| multilingual | FP16 | 1 | 175.7 | 178.9 | 1.02× |
| multilingual | FP16 | 2 | 317.2 | 314.1 | 0.99× |
| multilingual | FP16 | 4 | 543.6 | 479.1 | 0.88× |
| multilingual | FP16 | 8 | 827.1 | 615.8 | 0.74× |
| typed-decisions | BF16 | 1 | 144.4 | 128.4 | 0.89× |
| typed-decisions | BF16 | 2 | 246.8 | 206.4 | 0.84× |
| typed-decisions | BF16 | 4 | 409.0 | 306.1 | 0.75× |
| typed-decisions | BF16 | 8 | 546.5 | 346.0 | 0.63× |
| typed-decisions | FP16 | 1 | 142.1 | 126.9 | 0.89× |
| typed-decisions | FP16 | 2 | 250.2 | 207.9 | 0.83× |
| typed-decisions | FP16 | 4 | 403.1 | 304.1 | 0.75× |
| typed-decisions | FP16 | 8 | 532.3 | 342.6 | 0.64× |

See [measurement metadata](measurements/vulkan-current-16bit-performance.json) for binary,
weight and corpus identities. Completed AMD FP16/BF16 baselines are recorded in [AMD measurements](vulkan-amd-16bit-performance.md).
