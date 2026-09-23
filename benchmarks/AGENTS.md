# Benchmark and validation guidance

Read the root `AGENTS.md` and `docs/benchmarking.md` before changing a harness or reporting a result. The scripts here compare native inference against the local Python baseline and produce evidence tied to particular code, models, precision, hardware, and request groups. They are not general model-quality evaluations.

## Script map

- `validate.py` is the correctness gate for one checkpoint: it checks prepared inputs, finite raw outputs, public answer parity, and repeated-call stability at selected batch sizes. `models.py` runs that workflow across `english`, `multilingual`, and `typed-decisions` and can request sweeps.
- `sweep.py` measures paired baseline/native throughput after a matching passing validation report. `compare_native.py` compares two preserved native builds. `compare_precision.py` compares precision modes from one build; each mode needs its own accepted baseline comparison.
- `http_validate.py` checks checkpoint-backed CLI/HTTP parity and dynamically grouped calls. `native.py`, `oracle.py`, `compare.py`, and `identity.py` supply shared process, baseline, comparison, and identity logic.
- `cases/acceptance-250.json` is the fixed committed corpus. `make_corpus.py` generates it deterministically; `cases/smoke.json` is a quick probe. Changing the acceptance corpus changes its identity and invalidates earlier reports.

## Evidence rules

- Compare the same precision, model, request groups, and batch sizes. The public acceptance rule is exact categories and absolute numeric error at most 0.0001. Raw-logit tolerances in `validate.py` are diagnostic and do not replace that public-output gate.
- Preserve the fail-closed sweep checks for a passing report matching the selected executable, weights, corpus, backend, precision, device, relevant runtime libraries, and batch sizes. Do not reuse a report after any of those inputs change.
- Warm both sides, use paired groups and alternating execution order, and keep model loading outside timed intervals. State whether timings include transport. Shared-GPU measurements are exploratory; use an idle GPU for publishable performance comparisons.
- Keep generated reports in ignored `results/`. `docs/measurements/` contains recorded runs; cite their identities when discussing them and do not treat them as measurements of the current checkout.

The default script paths expect local `research/laya`, `models/laya`, and a built `laya-cli`; these are not committed. Check availability before running and report missing prerequisites rather than treating a skipped comparison as a pass.
