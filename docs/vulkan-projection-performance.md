# Vulkan FP32 projection performance

The additional NVIDIA projection partitions pass all 3,000 fixed-corpus
comparisons across the three models. Categories match exactly and numeric
answers differ by at most 0.0001. These timing runs compare native Vulkan FP32
with Python FP32 on the same RTX PRO 6000 Blackwell, capped at **450 W**.

Driver 610.43.03; PyTorch 2.11.0+cu130. Each batch setting uses all 250 questions,
three warmups and five timed iterations per group, alternating execution order.
Timing includes tokenization, inference and formatting, excluding model loading
and native JSON transport. No other builds or GPU tests ran during timing;
background services retained their allocations.

Rates are questions/second.

| Model | Batch | Python FP32 | Vulkan compensated FP32 | Speedup |
|---|---:|---:|---:|---:|
| english | 1 | 149.0 | 106.5 | 0.72× |
| english | 2 | 209.3 | 166.5 | 0.80× |
| english | 4 | 253.2 | 226.6 | 0.89× |
| english | 8 | 260.1 | 267.9 | 1.03× |
| multilingual | 1 | 187.2 | 107.0 | 0.57× |
| multilingual | 2 | 307.0 | 191.6 | 0.62× |
| multilingual | 4 | 412.2 | 288.7 | 0.70× |
| multilingual | 8 | 449.6 | 340.1 | 0.76× |
| typed-decisions | 1 | 135.3 | 90.6 | 0.67× |
| typed-decisions | 2 | 182.7 | 135.5 | 0.74× |
| typed-decisions | 4 | 199.4 | 164.6 | 0.83× |
| typed-decisions | 8 | 180.2 | 181.7 | 1.01× |

This change applies to NVIDIA projections. The earlier [paired GPU measurements](vulkan-fused-performance.md)
include the AMD results and separate native CUDA comparisons. These timings
should not be treated as a simultaneous before/after comparison with that run.

See [validation](measurements/vulkan-projection-split-fp32.json) and
[measurement metadata](measurements/vulkan-projection-performance.json) for source
and binary identities. These results do not establish support for 16-bit Vulkan.
