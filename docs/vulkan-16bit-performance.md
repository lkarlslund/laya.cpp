# Vulkan 16-bit performance

These warmed measurements compare native Vulkan with matching-precision Python
on RTX PRO 6000 Blackwell capped at **450 W**. All measured groups pass exact
categories and numeric absolute tolerance 0.0001. The [combined correctness gate](measurements/vulkan-nvidia-16bit-validation.json)
covers all three models and both precisions, as do these timing runs.

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
| multilingual | BF16 | 1 | 172.4 | 143.7 | 0.83× |
| multilingual | BF16 | 2 | 308.8 | 252.8 | 0.82× |
| multilingual | BF16 | 4 | 537.3 | 400.6 | 0.75× |
| multilingual | BF16 | 8 | 829.0 | 513.9 | 0.62× |
| multilingual | FP16 | 1 | 171.4 | 143.8 | 0.84× |
| multilingual | FP16 | 2 | 309.0 | 253.1 | 0.82× |
| multilingual | FP16 | 4 | 529.3 | 401.2 | 0.76× |
| multilingual | FP16 | 8 | 820.9 | 518.5 | 0.63× |
| typed-decisions | BF16 | 1 | 141.7 | 106.4 | 0.75× |
| typed-decisions | BF16 | 2 | 241.4 | 174.4 | 0.72× |
| typed-decisions | BF16 | 4 | 394.8 | 257.2 | 0.65× |
| typed-decisions | BF16 | 8 | 528.8 | 295.6 | 0.56× |
| typed-decisions | FP16 | 1 | 139.6 | 105.3 | 0.75× |
| typed-decisions | FP16 | 2 | 242.7 | 174.0 | 0.72× |
| typed-decisions | FP16 | 4 | 391.7 | 257.8 | 0.66× |
| typed-decisions | FP16 | 8 | 517.9 | 294.0 | 0.57× |

Vulkan currently reaches 56–84% of Python throughput in these runs. This is the baseline
for further dispatch and conversion optimizations, not a completed speed target.
See [measurement metadata](measurements/vulkan-16bit-performance.json) for binary,
weight and corpus identities. AMD 16-bit timing awaits its correctness gate.

For before/after native comparisons, use benchmarks/compare_native.py with
--bf16 or --fp16 and Vulkan selected on both sides. The --before-library-path
and --after-library-path options set separate loader paths for the two child
processes, so archived executables retain their archived shared libraries.
The harness hashes the libraries resolved under each environment. The after
build still requires a passing matching-precision validation report.
