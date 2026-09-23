# Working in laya.cpp

This repository implements native C++20 inference for Laya typed decisions. It loads Laya safetensors checkpoints, tokenizes and prepares JSON requests, executes the model through ggml, and returns calibrated answers. CUDA and Vulkan are the primary GPU backends. Python is optional tooling for downloading models, validation, and measurement; serving and inference run in C++.

## Start here

- Read `README.md` for supported models, build commands, and a request example.
- Read `docs/architecture.md` before changing tokenization, graph execution, precision, or batching. `docs/models.md`, `docs/precision.md`, `docs/vulkan.md`, and `docs/http.md` describe the corresponding contracts.
- Treat `docs/benchmarking.md` as the source for acceptance and performance methodology. `docs/performance.md` points to current reported measurements; `docs/measurements/` contains recorded evidence. Do not present a recorded result as a measurement of a new build.
- Check the current tree and Git status before editing. Preserve unrelated work. Model checkpoints, research files, build trees, results, and local deployment settings are intentionally outside Git.

## Request and code map

1. `src/main.cpp` parses CLI, JSON-lines, and server options, selects one checkpoint and backend, then constructs `laya::agent`.
2. `src/protocol.cpp` validates requests, builds question rows and option markers, enforces model token budgets, calls `runtime::forward`, and formats calibrated answers. `src/tokenizer.cpp` implements the English/typed-decisions byte-level BPE and multilingual metaspace BPE paths.
3. `src/runtime.cpp` loads and validates checkpoint tensors, owns the resident ggml model and reusable graphs, and runs the encoder and decision heads. `src/decision.cpp` contains standalone calibration logic. Public interfaces are in `include/laya/`.
4. `src/precision.cpp`, the CUDA sources (`src/*.cu`), `src/vulkan_*`, and `src/vulkan/` implement backend and precision-specific operations. The pinned ggml checkout is a submodule; CUDA integration generates a modified dispatcher in the build directory from that pinned source. Do not edit a submodule to work around this integration without tracing the CMake path.
5. `src/http.cpp` implements `/v1/systemone`, `/predict`, `/v1/models`, and `/health`. It admits HTTP calls to a bounded queue and runs one serialized inference worker that can combine whole calls into a GPU batch. Direct calls to one `agent` must also be serialized by the caller.

The API uses `choice`, `score`, and `noul` questions. The JEV-compatible HTTP route follows that protocol's envelope while identifying the loaded Laya checkpoint; it does not execute JEV weights. A process loads one explicit model variant (`english`, `multilingual`, or `typed-decisions`); there is no automatic language routing or checkpoint switching.

## Build and validation

- Initialize the pinned `third_party/ggml` and `third_party/cpp-httplib` submodules before configuring CMake. Host dependencies include CMake 3.24+, a C++20 compiler, ICU, and nlohmann-json. CUDA is enabled by default; use `-DLAYA_CUDA=OFF -DLAYA_VULKAN=ON` for a Vulkan build. See `README.md` and `docs/vulkan.md` for backend dependencies.
- Run `ctest --test-dir <build-dir> --output-on-failure` after relevant changes. `http` and `decision` are model-independent tests; tokenizer tests are registered only when the corresponding local model tokenizer exists. GPU tests depend on the configured backend, and some skip when hardware is unavailable. CTest success alone does not establish model-output parity.
- For model-facing changes, use the fixed `benchmarks/cases/acceptance-250.json` corpus and `benchmarks/validate.py` at matching precision and batch grouping, following `docs/benchmarking.md`. Acceptance requires exact categorical answers and public numeric absolute error at most 0.0001. Check all affected variants and batch sizes; `tests/model-requests.json` covers additional structured and multilingual requests.
- For HTTP behavior, run `tests/http.cpp` through CTest; use `benchmarks/http_validate.py` when actual checkpoint-backed transport and dynamic batching parity matter. For performance claims, require a passing validation report for the exact build, model, backend, precision, corpus, and libraries before using the paired sweep methodology in `docs/benchmarking.md`.

## Correctness constraints

- Keep request object insertion order where it determines choice order. Preserve tokenizer normalization, mask-token neutralization, truncation, per-variant token budgets, and model-specific rotary settings.
- Strict FP32 is the default. Optimized FP32, BF16, and FP16 have distinct arithmetic and support constraints. Compare FP32 and BF16 with their respective matching-precision baselines; cross-precision answer differences alone are not a porting failure. Read `docs/precision.md` and the relevant Vulkan validation notes before changing rounding boundaries or compiler/library selection.
- Graph reuse depends on shape and, for BF16, padding state. Attention masks must reflect each call's lengths, including a reused graph. Do not infer correctness from a single warm run.
- The committed acceptance corpus is fixed evidence. Changing it invalidates prior report identity. Keep generated results under ignored `results/`, and distinguish recorded hardware measurements from results reproduced on the current machine.

## Pull requests

Keep PR descriptions brief: state the change and the verification that matters to reviewers. Link detailed evidence or limitations only when they affect the review decision.
