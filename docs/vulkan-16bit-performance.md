# Vulkan 16-bit performance

These warmed measurements compare native Vulkan with matching-precision Python
on RTX PRO 6000 Blackwell capped at **450 W**. All measured groups pass exact
categories and numeric absolute tolerance 0.0001. The [combined correctness gate](measurements/vulkan-nvidia-16bit-validation.json)
covers all three models and both precisions; timing runs for the remaining models are still in progress.

Each batch setting uses all 250 fixed questions, three warmups and five timed
iterations per request group, alternating execution order. Timing includes
tokenization, inference and formatting, excluding model loading and native JSON
transport. Background services retained GPU allocations. No concurrent GPU
experiments or builds ran during these measurements.

Rates are questions/second.

| Model | Precision | Batch | Python | Vulkan | Vulkan/Python |
|---|---|---:|---:|---:|---:|
| english | BF16 | 1 | 143.9 | 110.4 | 0.77× |
| english | BF16 | 2 | 256.4 | 196.6 | 0.77× |
| english | BF16 | 4 | 441.6 | 316.5 | 0.72× |
| english | BF16 | 8 | 645.1 | 419.4 | 0.65× |
| english | FP16 | 1 | 143.3 | 110.6 | 0.77× |
| english | FP16 | 2 | 258.7 | 197.4 | 0.76× |
| english | FP16 | 4 | 439.7 | 315.3 | 0.72× |
| english | FP16 | 8 | 634.7 | 416.1 | 0.66× |

Vulkan currently trails Python in these English-model runs. This is the baseline
for further dispatch and conversion optimizations, not a completed speed target.
See [measurement metadata](measurements/vulkan-16bit-performance.json) for binary,
weight and corpus identities. AMD 16-bit timing awaits its correctness gate.
