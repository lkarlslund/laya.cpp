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
python benchmarks/validate.py --tensor-core-fp32 --batch-sizes 1 2 4 8 \
  --output results/validation.json
```

The validation tool accepts a local baseline package with `--source`, a checkpoint
with `--model`, and the executable with `--executable`. It tests:

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
maximum raw errors and every failing case. BF16 has a separate experimental mode
and must pass its own gates before it can become a supported default.

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
requires a passing validation report for the exact executable and weight hashes,
precision, corpus, and batch sizes. No speedup is reported for a failed group.
Use an idle GPU for publishable measurements; shared-GPU results are exploratory.
The FP32 comparison disables baseline autocast and TF32. It does not measure
the baseline package's default BF16 serving configuration. Always state precision
when comparing runs. FP32 and BF16 timings are different
operating points and must not be presented as interchangeable.

The smaller `smoke.json` corpus and `benchmarks/run.py` remain useful for quick
single-request baseline measurements. Detailed reports are stored in ignored
`results/`.

## Comparing native builds

Preserve the earlier executable together with its own shared libraries. Ensure
its runtime library paths resolve to that snapshot, then compare it with a new
build that has passed acceptance:

```sh
python benchmarks/compare_native.py --before /path/to/previous/laya-cli \
  --validation results/validation.json
```

Both builds use the optimized CUDA mode, identical request groups, and alternating
timing order. Each group must meet the output tolerance before its speedup is
reported. The report records both build fingerprints and requires a matching
acceptance report for the new build. This separates incremental native improvements
from changes in hardware contention between measurements.

## All-model matrix

`python benchmarks/models.py --sweep` runs the same committed 250-question corpus
on English, multilingual, and typed-decisions, sequentially to avoid unnecessary
GPU memory pressure. It writes separate validation and timing reports under
`results/models/`. Use `--variants multilingual`, `--batch-sizes`, or `--strict-fp32`
to narrow a run. The optimized CUDA path is selected by default in this matrix.
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
