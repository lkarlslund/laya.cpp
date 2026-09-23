# Test guidance

Read the root `AGENTS.md` for the project map and `docs/benchmarking.md` for the model-output acceptance contract. C++ tests here are registered in the root `CMakeLists.txt`; Python `test_*.py` files run through `unittest` discovery. Add coverage at the layer whose behavior changed, then validate the actual model or HTTP path when that behavior depends on a checkpoint.

## What runs where

- `decision.cpp` tests standalone calibration logic. `http.cpp` tests routes, authentication, envelopes, limits, queue batching, overload, and shutdown with a predictor callback; it does not load model weights.
- `tokenizer.cpp` uses `tokenizer-cases.json` for English and typed-decisions and `tokenizer-multilingual.json` for multilingual. CMake registers each tokenizer test only if that variant's local tokenizer file exists at configure time. Reconfigure after adding model files; an absent tokenizer test is not a pass.
- CUDA tests cover BF16, attention, fused operations, and compensated matmul. Vulkan tests cover SPIR-V, precision, projection, attention, softmax, and BF16 range behavior. They are built only for enabled backends; several return 77 to report an unsupported device or unavailable hardware as a CTest skip. Inspect skips when judging coverage.
- `backend_replay.cpp` is registered per available checkpoint only in a combined CUDA and Vulkan build. It needs local safetensors and is marked serial. It checks a narrower replay surface than the full acceptance corpus.
- Python `test_*.py` files check corpus identity, comparison rules, precision study, and related harness behavior. Run them with `python3 -m unittest discover -s tests -p 'test_*.py'` when changing those utilities.

## Verification expectations

Run `ctest --test-dir <build-dir> --output-on-failure` for relevant C++ changes and inspect the configured test list with `ctest --test-dir <build-dir> -N` when dependencies or hardware affect registration. For request formatting, tokenizer, numerical, or backend changes, follow the matching-precision `benchmarks/validate.py` workflow across affected variants and batch sizes. For real server/model integration and dynamic batching, use `benchmarks/http_validate.py` in addition to the callback-based HTTP test. The CI workflow builds CPU-only and Vulkan configurations and runs CTest plus Python unit tests; it does not establish CUDA or checkpoint-backed acceptance.

Keep fixtures deterministic and small. Do not weaken categories, numeric tolerance, repeated-call checks, or fixture identity to make an implementation change appear correct. If a test cannot run because a submodule, model, backend, or device is unavailable, state that limitation explicitly.
