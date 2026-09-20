# Benchmarking

The harness loads a pluggable Python adapter once. The adapter factory accepts a
local model directory and `device`; its result exposes `device` and
`predict(state, questions)`. Prediction returns `answers` and `usage`. A native
runtime can participate through a Python binding with this interface.

```sh
python benchmarks/run.py --backend-path /path/to/backend --factory package:load \
  --model models/laya --output results/run.json
python benchmarks/compare.py results/baseline.json results/candidate.json
```

The default factory is `laya:load`. Use `--device cpu` only for explicit CPU runs.
GPU runs fail on device fallback. The smoke suite covers choice, ordinal score,
and boolean questions, both singly and in a question batch.

Each case receives five warmups and thirty measured iterations by default.
Latency covers host preprocessing, tokenization, transfers, inference and output
formatting. CUDA is synchronized before and after every measured request. Model
load time is separate. Reports include all timing samples, p50/p95/mean latency,
questions per second, peak allocated PyTorch GPU memory, actual GPU name, software
versions, checkpoint revision, backend commit, and a case-file hash. PyTorch memory
statistics do not include allocations made directly by a future native runtime.

The comparison command checks case identity, hardware/device and checkpoint
revision, exact categorical agreement, and numeric absolute error (default 0.001).
It reports p50 speedup only after all outputs pass. Checkpoint revisions are
provenance labels, not checksums of local weight files; do not edit weights between
paired runs. Outputs are rounded by some adapters, so use raw tensor comparisons
as an additional gate when developing kernels.

Run paired measurements on an idle GPU with the same power configuration and
software environment. Shared-GPU measurements are exploratory. The smoke suite
is a wiring check, not evidence of quality or performance across workloads. Expand
coverage with long states, varied option counts, mixed sequence lengths, structured
criteria, multilingual text, and boundary/truncation cases before release.
