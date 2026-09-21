# Vulkan projection precision

Low-precision projection results depend on where intermediate sums are rounded. The NVIDIA plan table covers the 20 biased and unbiased weight geometries used by the three models, with 1–8192 input columns. FP16 and BF16 selected identical plans in 327,680 synthetic measurements with PyTorch 2.11 / CUDA 13.0 on RTX PRO 6000 Blackwell.

Plans distinguish unsplit products, parallel partitions with either FP32 or low-precision partials, and serial partitions that round the running sum after every partition. For parallel biased projections, the stored-product flag selects rounding before adding bias. For serial projections, that same measured flag identifies the separate-bias algorithm, which adds bias to the first partition; the other serial algorithm adds it to the last partition. Cancellation probes distinguish these positions: first and last products are +1 and -1, with a quarter-ULP bias that is lost only when added to the first partition. All 72 measured serial range representatives pass this check in both precisions. These are geometry-based rules; no question-specific corrections are used.

The bias-first correction also makes every captured intermediate of a failing multilingual BF16 batch of four match exactly. Operator tests cover first-partition, last-partition, and post-storage bias on NVIDIA and AMD.

This coverage fixes a traced typed-decisions BF16 batch of two long requests: all captured intermediates, including all 28 encoder outputs and both head outputs, match exactly. Vulkan operator tests pass. The English and multilingual FP16 prototypes also pass all 250 fixed questions at batches 1, 2, 4, and 8 with exact categories and numeric absolute tolerance 0.0001; see the [English](measurements/vulkan-english-fp16-validation.json) and [multilingual](measurements/vulkan-multilingual-fp16-validation.json) validation records. The typed-decisions prototype also passes all 1,000 comparisons in both [BF16](measurements/vulkan-typed-decisions-bf16-validation.json) and [FP16](measurements/vulkan-typed-decisions-fp16-validation.json). Thus all three FP16 model variants pass on this NVIDIA device. After the bias-first correction, multilingual BF16 also passes all 1,000 comparisons ([record](measurements/vulkan-multilingual-bf16-validation.json)). English BF16 still has one batch-4 numeric failure; these results do not establish general FP16/BF16 support. AMD does not use this NVIDIA-specific policy.

The profiles describe the measured device and library version. Untested geometries fall back to an unsplit product, without a matching-precision guarantee. Reprofile when targeting another device or math-library version.

Measurement metadata and hashes are in [the projection record](measurements/vulkan-low-precision-projection-plans.json). Regenerate the header from the retained JSONL profile with:

```sh
python benchmarks/generate_projection_plans.py path/to/profile.jsonl
```

Each profile row contains `precision`, `k`, `m`, `n`, `bias`, `chunk`, `scheme`, and `round_before_bias`. The generator requires identical, complete FP16/BF16 geometry coverage and rejects unknown reduction schemes.
