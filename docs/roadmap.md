# Development roadmap

1. **Foundation (present):** native calibration core, pinned model downloads,
   repeatable end-to-end measurement and output comparison.
2. **Correctness:** checkpoint tensor inventory and loader; tokenizer and exact
   sequence formatting; native encoder and typed heads; layerwise numerical tests.
3. **RTX execution:** BF16 cuBLASLt projections, local/global attention kernels,
   persistent allocation, shape buckets, fused operations, CUDA Graph replay.
4. **Validation:** raw tensor parity and complete output parity; separate model-only
   and end-to-end timing; throughput and latency sweeps across supported RTX GPUs.
5. **Serving:** reusable C/C++ API, JSON CLI, batching, model variants, packaging.

Quantization follows a working BF16 implementation and accuracy measurements.
Do not claim speedups from postprocessing microbenchmarks as model speedups.
