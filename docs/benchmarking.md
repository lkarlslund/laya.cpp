# Validation and performance

The versioned corpus and JSON report contract for issue #7 is in
[benchmark contract](benchmark-contract.md). The commands below describe the
current runners. They still write their historical summary JSON, and can also
produce the versioned reports.

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
python benchmarks/validate.py --tensor-core-fp32 --batch-sizes 1 2 4 8 \
  --output results/validation.json
```

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
python benchmarks/sweep.py --tensor-core-fp32 --validation results/validation.json \
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

## Versioned local reports

Run `validate.py` with `--v1-output` to preserve the complete expected and
actual public answer for every request and batch size. It validates the output
against `benchmarks/schema/validation-v1.schema.json`. Run it once on the fixed
acceptance corpus and again on each performance stratum. The acceptance report
must contain 250 requests at each of batch sizes 1, 2, 4, and 8. The workload
parity report must cover every batch size to be timed.

```sh
python benchmarks/validate.py --backend cpu --executable build-cpu/bin/laya-cli \
  --model models/laya --threads 4 --batch-sizes 1 2 4 8 \
  --output results/cpu-acceptance.json \
  --v1-output results/cpu-acceptance-v1.json
```

CPU validation and timing use the CPU baseline and set the same explicit thread
budget in PyTorch and the native process through `LAYA_CPU_THREADS`. `sweep.py`
rejects a validation report from a different thread budget. For a full CPU
comparison, repeat acceptance and timing with `--threads 1` and a stated
physical-core count. GPU runs omit `--threads`.

For each performance stratum, use its path from
`benchmarks/cases/performance-v1/manifest.json` with `--cases` in both
`validate.py` and `sweep.py`. Then convert the passing sweep and its two parity
reports to the benchmark-v1 JSON:

```sh
python benchmarks/format_benchmark_v1.py \
  --sweep results/cpu-short-q1-sweep.json \
  --acceptance results/cpu-acceptance-v1.json \
  --workload results/cpu-short-q1-validation-v1.json \
  --manifest benchmarks/cases/performance-v1/manifest.json \
  --stratum short-q1.json --executable build-cpu/bin/laya-cli \
  --output results/cpu-short-q1-benchmark-v1.json
```

The converter checks corpus and build hashes, all acceptance cases, workload
parity, batch coverage, warmup and timed-pass counts, and the number of raw
samples. It validates the result against `benchmark-v1.schema.json`. The
current sweep does not measure startup time or peak process RSS, so those fields
are `null` with a reason. Record those separately before presenting a complete
hardware comparison. Reports from a small local sample are format checks and
must not be presented as results for the fixed 8,192-request corpus.

## Issue #7 CPU campaign

The issue #7 runner uses the fixed 1,024-request subset documented in the
[benchmark contract](benchmark-contract.md). It runs all three FP32 checkpoints
at one thread and the available physical-core count (16 on the local Ryzen AI
MAX+ 395). Each checkpoint/thread setting gets full 250-case acceptance,
per-stratum workload parity, then three warmups and five alternating timed
passes at batch sizes 1, 2, 4, and 8. Runs are serial, checkpointed after each
stratum, and may take multiple days. Run on an otherwise idle host.

```sh
python benchmarks/run_cpu_issue7.py --executable build-cpu/bin/laya-cli \
  --model-root models/laya --source research/laya \
  --output results/issue7
```

The runner requires a clean source tree and freezes the code, harness, binary,
weights, baseline, and corpus identities in `results/issue7/identity.json`.
Rerunning the same command resumes only reports whose identities and parity
still match. `run-index.json` stays `incomplete` until all six
checkpoint/thread combinations and all 16 strata have passed. `--variants`,
`--threads`, and `--strata` can narrow development runs, but they cannot make
the full index pass.

For timing, a persistent CPU baseline worker and the native executable each
keep one checkpoint resident. Both use the same affinity and explicit PyTorch,
ggml, OpenMP, MKL, OpenBLAS, and NumExpr thread budget. Each worker measures its
own inference call internally, excluding JSON serialization, transport, and
queue time. Process start to readiness and Linux `VmHWM` peak RSS are recorded
for each worker. Checkpoint weights are read before each process pair starts,
so startup times are labeled page-cache-warm. CPU frequency and governor are
sampled where the host exposes them.

After `run-index.json` says `passed`, stage only the verified JSON evidence:

```sh
python benchmarks/publish_issue7.py --source results/issue7 \
  --destination docs/measurements/issue7
```

The staging command rejects missing rows, parity failures, changed hashes,
missing load or RSS data, and an incomplete matrix. Link the staged index and
derive human-readable tables from its reports; do not substitute a table for
the raw samples.

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
The optimized FP32 CUDA path is selected by default in this matrix.
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
