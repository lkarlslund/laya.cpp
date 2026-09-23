# Validation and performance

`benchmarks/cases/acceptance-250.json` is a fixed, committed corpus containing
100 choice, 75 ordinal score, and 75 boolean questions across 25 scenarios.
Every question has a distinct instruction. Cases include 2–12 options, long
states that reach truncation, mixed sequence lengths, structured state and
criteria, mixed-language text, and quoted control-like text. The generator is
deterministic; tests enforce count, uniqueness, type distribution, and exact
reproduction of the committed corpus.

This corpus measures implementation agreement and performance, not factual
accuracy or general model quality. Changing the corpus changes its SHA-256 and
invalidates previous acceptance reports.

## Correctness gate

```sh
python benchmarks/validate.py --allow-truncation --tensor-core-fp32 --batch-sizes 1 2 4 8 \
  --output results/validation.json
```

The fixed corpus includes long states and over-budget input designed to exercise
the historical preprocessing limits. Validation commands therefore pass
`--allow-truncation` explicitly; its report records this policy, and the sweep
requires an exact policy match. Remove that option to validate strict rejection
on corpora that fit within the model budgets.

The validation tool accepts a local baseline package with `--source`, a checkpoint
with `--model`, and the executable with `--executable`. Select `--backend vulkan`
for Vulkan FP32 (with no CUDA precision flags); the default is CUDA. It tests:

- Exact token IDs, lengths, option markers, masks, types, and batch ordering.
- Finite raw logits. Raw differences above `1e-3 + 1e-5*abs(expected)` are
  recorded as diagnostics and do not independently reject a run.
- Exact discrete answers, legends, usage, and output structure; numeric answer
  fields within 0.0001 (one unit at the public output precision).
- Identical answers over repeated calls, including CUDA graph replay.

Acceptance requires exact categorical outputs and absolute numeric output error
no greater than 0.0001. Raw action logits can have magnitudes in the thousands;
their diagnostics help investigate numerical differences. This is not bitwise
identity or a proof for every possible input. The report preserves observed
maximum raw errors and every failing case. Select `--bf16` for matching BF16
validation; the compatible build profile is described in [precision](precision.md).

Correctness is defined at matching precision: native FP32 is compared with the
FP32 baseline, and native BF16 with the BF16 baseline. Different answers between
FP32 and BF16 do not, by themselves, indicate a porting bug. The objective is to
preserve the model's behavior at the selected precision, not to require every
mode to reproduce higher-precision arithmetic.

Tokenizer tests additionally cover 253 fixtures including added tokens,
whitespace, combining characters, multiple scripts, emoji, and control characters.
They run through CTest when the local tokenizer model is present.

## Batch sweep

```sh
python benchmarks/sweep.py --allow-truncation --tensor-core-fp32 --validation results/validation.json \
  --batch-sizes 1 2 4 8 --warmup 3 --iterations 5
```

Batch size is the number of fixed one-question requests in one forward pass;
it is GPU batching, not multiple processes competing for the device. Each backend
receives identical groups in corpus order, including the final partial group.
The tool verifies public results before measuring each group. It also checks that
the batched baseline formatter agrees with its public API formatter.

Unused baseline GPU allocator cache is released before native graph allocation
for each group. Warmups repopulate it before timing; cache cleanup is outside
all measured intervals. Each group is warmed before timing. Backend order alternates on successive
iterations. CUDA synchronization brackets baseline timing; native inference
returns only after outputs are available on the host. Native timing includes
native preprocessing, transfers, inference, calibration, and JSON value creation,
and excludes stdin/stdout serialization and transport. Baseline timing includes
its equivalent in-process work. Model loading is outside the measured loop.

Reports include all samples, p50/p95 batch latency, aggregate questions/second,
precision, GPU identity, software/source revisions, and corpus identity. The sweep
requires a passing validation report for the exact executable, loaded CUDA math
libraries, selected backend, weight hashes, precision, corpus, and batch sizes. No speedup is reported for a failed group.
Use an idle GPU for publishable measurements; shared-GPU results are exploratory.
The FP32 comparison disables baseline autocast and TF32. It does not measure
the baseline package's default BF16 serving configuration. Always state precision
when comparing runs. FP32 and BF16 timings are different
operating points and must not be presented as interchangeable.

The smaller `smoke.json` corpus and `benchmarks/run.py` remain useful for quick
single-request baseline measurements. Detailed reports are stored in ignored
`results/`.

## Apple Core ML sweep

Core ML uses a dedicated fail-closed validator and sweep because its reference
backend is PyTorch CPU rather than CUDA. After producing a passing schema-2
validation report, run:

```sh
python benchmarks/sweep_coreml.py \
  --validation results/coreml-validation.json \
  --cache-root results/coreml-cache \
  --batch-sizes 1 2 4 8 --warmup 3 --iterations 5 \
  --output results/coreml-sweep.json
```

The `rows` use the same `baseline`, `native`, `speedup`, `failures`, p50/p95 and
questions-per-second fields as `sweep.py`. The report additionally records the
compiled bucket used by each group and binds timing to the Core ML manifest.
See [Apple Core ML](coreml.md) for setup, validation and interpretation details.

## Comparing native builds

Preserve the earlier executable together with its own shared libraries. Ensure
its runtime library paths resolve to that snapshot, then compare it with a new
build that has passed acceptance:

```sh
python benchmarks/compare_native.py --before /path/to/previous/laya-cli \
  --validation results/validation.json
```

Both builds use the optimized FP32 mode by default. Add `--bf16` to compare
two BF16 builds with matching BF16 acceptance. They receive identical request
groups and alternating timing order. Each group must meet the output tolerance before its speedup is
reported. The report records both build fingerprints and requires a matching
acceptance report for the new build. This separates incremental native improvements
from changes in hardware contention between measurements.

The preserved `--before` executable runs without the new truncation switch, so
its built-in legacy shortening remains active. The current `--after` executable
receives `--allow-truncation`; the comparison report records this flag asymmetry
and the legacy truncation policy used on both sides.

## Comparing native precision modes

```sh
python benchmarks/compare_precision.py --executable build-bf16/bin/laya-cli \
  --fp32-validation results/fp32/english-validation.json \
  --bf16-validation results/bf16/english-validation.json \
  --output results/native-precision.json
```

This compares optimized FP32 and BF16 from the same executable and loaded math
libraries. Both modes must have passing matching-precision validation for that
exact build, checkpoint, corpus and GPU. Each group is warmed before timing;
mode order alternates across iterations and groups. Model loading and JSON
transport are excluded. Differences in answers between FP32 and BF16 are not
used as a cross-precision correctness gate. Each mode is instead accepted against
its own same-precision baseline before timing.

Keep the GPU workload stable and compare the paired results rather than dividing
throughput numbers from separate runs. Shared-GPU results remain exploratory.

## All-model matrix

`python benchmarks/models.py --sweep` runs the same committed 250-question corpus
on English, multilingual, and typed-decisions, sequentially to avoid unnecessary
GPU memory pressure. It writes separate validation and timing reports under
`results/models/`. Use `--variants multilingual`, `--batch-sizes`, or `--strict-fp32`
to narrow a run. Use `--bf16` for BF16 validation and timing across all three models.
The optimized FP32 CUDA path is selected by default in this matrix. The matrix
opts into legacy truncation for both validation and sweep because the committed
corpus deliberately includes over-budget inputs.
Each checkpoint has its own weight fingerprint and independent correctness gate;
a passing report for one checkpoint cannot authorize timing another.

Tokenizer CTests cover all locally downloaded variants. The multilingual fixtures
exercise its distinct normalization, metaspace splitting, non-Latin text, special
tokens, and byte fallback. These tests measure implementation equivalence, not
language understanding or task accuracy.

`tests/model-requests.json` adds multilingual, multi-question, special-token, and
long-state coverage. Run it independently with:

```sh
python benchmarks/models.py --cases tests/model-requests.json \
  --batch-sizes 1 2 --output results/model-edges
```

## Precision study

```sh
python benchmarks/precision_study.py
```

The study checks both native modes against their matching baselines on all three
checkpoints and all 250 questions at batches 1, 2, 4 and 8. It records numeric
errors, category changes, metadata agreement and repeated-call stability.
Cross-precision baseline drift is recorded separately and does not affect a
native mode's acceptance result.

Each group receives three warmups and five timed iterations by default. The
execution order rotates between the native candidate, the FP32 baseline and the
BF16 baseline. Timing includes preprocessing, inference and output formatting;
loading, native JSON transport and correctness checks are excluded. Public
baseline formatting is independently checked before timing its batched formatter.

Failing candidates are timed for diagnosis but have no accepted speedup and are
not eligible for deployment. This study does not weaken the validation gates in
`validate.py` or `sweep.py`. Detailed outputs stay in ignored `results/`.
Exit status zero means the study completed, not that every mode passed; consult
the per-mode results and `eligible_modes` in its reports.
# Comparing native GPU backends

`benchmarks/compare_native.py` accepts `--before-backend cuda|vulkan` and
`--after-backend cuda|vulkan`. The candidate's precision and attention options
come from its passing validation report. A Vulkan baseline uses plain FP32 unless
`--before-tensor-core-fp32` is specified. CUDA baselines retain the optimized
FP32 defaults. Both processes keep their weights resident and alternate timed
calls on identical request groups; each group must also pass answer agreement.
Vulkan device identity must match the Python GPU and the validation report.

## Vulkan 16-bit development

The comparison tools accept `--backend vulkan --fp16` and `--bf16` as distinct
precision modes. FP16 explicitly selects FP16 autocast in the Python baseline;
it is never labeled BF16. These options support development builds and do not
imply that a Vulkan 16-bit implementation has passed acceptance. `sweep.py`
requires a passing report with matching precision, binary, model, corpus, device
and batch sizes before reporting performance.

The fused Vulkan activation kernel has exhaustive numerical checks for all
65,536 inputs in each 16-bit format, plus gated multiplication checks. Its
compiled numerical tables are generated by `benchmarks/generate_gelu_tables.py`
and can be checked with `--check` in the recorded PyTorch/CUDA environment.
Python is needed only to regenerate or verify those development artifacts.
