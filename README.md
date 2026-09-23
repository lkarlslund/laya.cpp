# laya.cpp

Native C++ inference for Laya typed decisions, powered by ggml with CUDA and
Vulkan and Apple Core ML backends. Tokenization, inference and JSON output run without Python.
Supports the `english`, `multilingual` and `typed-decisions` models, plus a
JEV-compatible HTTP server with automatic request batching.

[Binary releases](https://github.com/lkarlslund/laya.cpp/releases) provide Windows
and Linux x64 CUDA/Vulkan executables plus a macOS arm64 Core ML executable. See
[runtime requirements](docs/releases.md) for drivers, Windows CUDA DLLs and
compiled Core ML model buckets.

## Performance

Latest paired comparisons against matching-precision Python, using 250 fixed
questions across all three models at batch sizes 1, 2, 4 and 8. Higher is better;
**1× means equal throughput**. NVIDIA measurements use an **RTX PRO 6000
Blackwell (96 GB), capped at 450 W**; AMD measurements use a **Radeon 8060S**.

| GPU | Backend / mode | Throughput relative to Python |
|---|---|---:|
| NVIDIA | CUDA optimized FP32 | 1.35–2.48× |
| NVIDIA | CUDA BF16 | 1.11–2.71× |
| NVIDIA | Vulkan plain FP32 | 0.44–0.73× |
| NVIDIA | Vulkan compensated FP32 | 0.60–1.14× |
| NVIDIA | Vulkan FP16 / BF16 | 0.64–1.02× |
| AMD | Vulkan plain FP32 | 0.61–1.82× |
| AMD | Vulkan compensated FP32 | 1.26–2.45× |
| AMD | Vulkan FP16 | 0.72–1.25× |
| AMD | Vulkan BF16 | 0.49–0.92× |

All measured answer checks pass: exact categories and numeric absolute error
at most **0.0001**. Timings include preprocessing, inference and formatting,
excluding model loading and JSON transport.

See [latest measurements](docs/performance.md) for per-model throughput,
Vulkan FP32 results, direct CUDA/Vulkan comparisons and measurement identities.

## Build

Requires a C++20 compiler, CMake 3.24+, ICU and nlohmann-json, plus the chosen GPU
backend's dependencies. On Debian-like systems, install `libicu-dev` and
`nlohmann-json3-dev` for the host dependencies.

```sh
git submodule update --init --recursive
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --parallel 8
```

Architecture 120 targets RTX Blackwell; select the architecture for your GPU.
For Vulkan, install the Vulkan loader/headers, `glslc` and SPIR-V headers, then:

```sh
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAYA_CUDA=OFF -DLAYA_VULKAN=ON
cmake --build build-vulkan --parallel 8
```

On Apple Silicon, Core ML uses the CPU, GPU and Neural Engine through
`MLComputeUnitsAll`. The build requires macOS 12 or newer, full Xcode (the
Command Line Tools alone are not sufficient), and the host dependencies:

```sh
brew install cmake ninja icu4c nlohmann-json
scripts/build_coreml.sh
```

The script selects the standard `/Applications/Xcode.app`, locates the Homebrew
packages, initializes submodules, configures Release for arm64 and macOS 12, and
builds `build-coreml/bin/laya-cli`. Pass `--test` to run CTest, or `--fresh` to
discard a stale CMake cache. Run `--help` for all options.

The checkpoint must first be exported and compiled. See [Core ML](docs/coreml.md)
for model preparation, manual build commands, bucket choices and M1 validation.

## Run

Download the models with the optional Python tooling, then run native inference:

```sh
python scripts/download_model.py --variant all
build-cuda/bin/laya-cli --model models/laya --variant english \
  --tensor-core-fp32 --flash-fp32 --input benchmarks/cases/smoke.json
```

Choose `--variant multilingual` or `--variant typed-decisions` for another model.
Requests that exceed the selected checkpoint's token budgets are rejected by
default. Pass `--allow-truncation` to retain the older behavior for clients that
depend on silently shortened input.
For Vulkan, use `build-vulkan/bin/laya-cli --vulkan` and omit `--flash-fp32`.
Strict FP32 is the default; `--tensor-core-fp32` enables compensated FP32
projections. Both GPU backends support `--bf16`; Vulkan also supports `--fp16`.
See [precision](docs/precision.md) and [Vulkan support](docs/vulkan.md) for tested
hardware and build requirements.

For Core ML, run `build-coreml/bin/laya-cli --coreml`; model precision is fixed
at export time, so do not combine it with `--fp16` or `--bf16`.

Without `--input`, the CLI accepts one JSON request or request array per line:

```json
{"state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}
```

To serve HTTP on `127.0.0.1:8080`:

```sh
build-cuda/bin/laya-cli --server --port 8080 --variant english \
  --tensor-core-fp32 --flash-fp32
```

The JEV-compatible endpoint is `POST /v1/systemone`. Concurrent requests are
batched automatically. See [HTTP serving](docs/http.md) for examples and settings.

## Documentation

- [Models and validation](docs/models.md)
- [Latest performance measurements](docs/performance.md)
- [Benchmarking and the fixed 250-question corpus](docs/benchmarking.md)
- [Vulkan support](docs/vulkan.md)
- [Apple Core ML support](docs/coreml.md)
- [Architecture](docs/architecture.md)

## License

[MIT](LICENSE). Dependencies and model files retain their own licenses.
