# Vulkan projection precision

Low-precision projection results depend on where intermediate sums are rounded. The NVIDIA plan table covers the 20 biased and unbiased weight geometries used by the three models, with 1–8192 input columns. FP16 and BF16 selected identical plans in 327,680 synthetic measurements with PyTorch 2.11 / CUDA 13.0 on RTX PRO 6000 Blackwell.

Plans distinguish unsplit products, parallel partitions with either FP32 or low-precision partials, and serial partitions that round the running sum after every partition. For parallel biased projections, the stored-product flag selects rounding before adding bias. For serial projections, that same measured flag identifies the separate-bias algorithm, which adds bias to the first partition; the other serial algorithm adds it to the last partition. Cancellation probes distinguish these positions: first and last products are +1 and -1, with a quarter-ULP bias that is lost only when added to the first partition. All 72 measured serial range representatives pass this check in both precisions. These are geometry-based rules; no question-specific corrections are used.

The bias-first correction also makes every captured intermediate of a failing multilingual BF16 batch of four match exactly. Operator tests cover first-partition, last-partition, and post-storage bias on NVIDIA and AMD.

The combined native runtime passes all 6,000 fixed-corpus comparisons on RTX PRO 6000 Blackwell: English, multilingual and typed-decisions, each in FP16 and BF16, at batches 1, 2, 4 and 8. Categories match exactly and numeric public outputs differ by at most 0.0001. The [combined validation record](measurements/vulkan-nvidia-16bit-validation.json) identifies the tested binaries, weights, corpus and Python environment. Raw tensors remain diagnostic and can differ even when public answers pass.

Batched BF16 scalar heads use 64-element FP32 partial dot products to limit cancellation error before final rounding. FP16 retains its original reduction order. The combined regression checks both policies together. AMD low-precision validation remains incomplete; it does not use this NVIDIA projection policy.

The profiles describe the measured device and library version. Untested geometries fall back to an unsplit product, without a matching-precision guarantee. Reprofile when targeting another device or math-library version.

Measurement metadata and hashes are in [the projection record](measurements/vulkan-low-precision-projection-plans.json). Regenerate the header from the retained JSONL profile with:

```sh
python benchmarks/generate_projection_plans.py path/to/profile.jsonl
```

Each profile row contains `precision`, `k`, `m`, `n`, `bias`, `chunk`, `scheme`, and `round_before_bias`. The generator requires identical, complete FP16/BF16 geometry coverage and rejects unknown reduction schemes.

ROCm GELU patches preserve signed zeros as well as nonzero values. Their generator
compares all 65,536 input bit patterns in each 16-bit format, ignoring only NaN
payload differences. Numerical equality alone missed 13,942 zero-sign differences;
an isolated AMD matrix test showed that those signs can affect rounded projection
outputs. Regenerate and check the tables with `benchmarks/generate_gelu_tables.py`
and `--check` in the matching ROCm environment. This operator check does not establish
full-model AMD 16-bit acceptance.

The projection-storage operator combines optional bias addition, FP16/BF16
rounding, and an optional FP32 residual addition in one dispatch. Bias precedes
rounding; the residual follows it. Independent bitwise operator tests pass on
NVIDIA and AMD for all three storage formats, biased and unbiased inputs,
residuals, rounding midpoints, signed zeros, and subnormals. The exhaustive GPU
activation tests also check zero signs. The operator is integrated into 16-bit
projections; all 6,000 NVIDIA public-answer comparisons pass for the three models
at batches 1, 2, 4 and 8 in FP16 and BF16
([record](measurements/vulkan-finish-16bit-validation.json)).
[Projection-storage timing results](vulkan-finish-performance.md) compare this
build with the preceding packing build.

Explicitly marked NVIDIA low-precision matrix products can retain FP16/BF16
inputs while preserving FP32 accumulation and the requested unsplit reduction.
Bitwise tests compare these products with the rounded-FP32-input path at small
and large batch dimensions, including signed zeros and cancellation. Scalar
projections retain their existing policy. Runtime integration of this conversion
optimization remains pending.

The fused QKV packing operator combines layout conversion, rotary multiplication,
and storage rounding in one dispatch. Its operator tests pass on NVIDIA and AMD
for FP32, FP16 and BF16, with batched inputs, rotary positions enabled and
disabled, subnormal inputs, and large values. Rotary products round separately
before addition. All 6,000 NVIDIA FP16/BF16 public-answer comparisons also pass with the
fused operator integrated ([record](measurements/vulkan-packed-16bit-validation.json)).
The NVIDIA plain and compensated FP32 modes also pass all 6,000 comparisons
([record](measurements/vulkan-packed-fp32-nvidia-validation.json)). AMD plain and
compensated FP32 pass all 6,000 comparisons against Python on the same GPU
([record](measurements/vulkan-packed-fp32-amd-validation.json)).
[Warmed before/after measurements](vulkan-packed-performance.md) show a 12–16%
throughput gain from packing across all three models in FP16 and BF16 on NVIDIA.
