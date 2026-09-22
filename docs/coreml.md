# Apple Core ML

The Core ML backend targets Apple Silicon and macOS 12 or newer. Each exported
ML Program has fixed shapes and FP32 compute by default. The native runtime uses
`MLComputeUnitsAll`, allowing Core ML to place supported operations on the CPU,
GPU and Neural Engine. Placement remains a Core ML decision; it is not a promise
that every operation runs on the Neural Engine.

## Prerequisites

Install full Xcode from Apple. The standalone Command Line Tools are not enough:
a selected Command Line Tools SDK can be newer than its linker and fail with a
TAPI or `unknown architecture` error. The build script selects the standard
Xcode installation without changing the system-wide `xcode-select` setting.

Install the remaining tools and host dependencies with Homebrew:

```sh
brew install cmake ninja icu4c nlohmann-json uv
```

Xcode installed elsewhere can be selected for the current shell:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcodebuild -version
```

## Export

Export requires the original PyTorch graph as well as the downloaded checkpoint.
Use the dedicated, pinned environment; the benchmark environment intentionally
uses a newer PyTorch release and is not compatible with this conversion path.

```sh
uv venv .venv-coreml --python 3.12
uv pip install --python .venv-coreml/bin/python -r requirements-coreml.txt
git clone --branch research https://github.com/NandhaKishorM/laya.git research/laya
python scripts/download_model.py --variant english
.venv-coreml/bin/python scripts/export_coreml.py \
  --source research/laya --model models/laya --variant english --precision fp32
```

The defaults export batches 1, 2, 4 and 8, using the checkpoint maximum length
and 12 options (the largest acceptance-corpus case).
The runtime pads each request to the smallest compatible bucket. Add repeated
`--bucket BxLxO` arguments when a workload needs more than 12 options or a
different maximum batch. Every extra bucket embeds another copy of the weights,
so start with the smallest set that covers the workload.

`--precision fp16` is an opt-in performance profile. On the M1 Pro smoke set it
preserved categories but exceeded the repository's strict public numeric error
contract (0.0002 observed versus 0.0001 allowed). Keep FP32 unless the complete
workload validator passes with the tolerances required by the application.

Compile every generated package before running the native backend:

```sh
xcrun coremlcompiler compile models/laya/coreml/b1-l512-o12/Laya.mlpackage models/laya/coreml/b1-l512-o12
xcrun coremlcompiler compile models/laya/coreml/b2-l512-o12/Laya.mlpackage models/laya/coreml/b2-l512-o12
xcrun coremlcompiler compile models/laya/coreml/b4-l512-o12/Laya.mlpackage models/laya/coreml/b4-l512-o12
xcrun coremlcompiler compile models/laya/coreml/b8-l512-o12/Laya.mlpackage models/laya/coreml/b8-l512-o12
```

FP32 packages and compiled models are large (about 1.6 GiB each for the English
checkpoint observed here). After successful compilation, the native runtime
needs only `Laya.mlmodelc`; the corresponding `.mlpackage` may be archived or
removed and regenerated later. Core ML can also create a similarly large E5RT
cache under `~/Library/Caches/laya-cli`, so keep several additional GiB free
during validation.

Repeat export and compilation under each variant directory for multilingual or
typed-decisions.

## Build and run

The build script is the recommended path:

```sh
scripts/build_coreml.sh
build-coreml/bin/laya-cli --coreml --model models/laya \
  --input benchmarks/cases/smoke.json
```

It verifies macOS and arm64, selects full Xcode, locates ICU and nlohmann-json,
initializes the Git submodules, configures CMake, and builds the Release
executable. Subsequent executions are incremental.

Useful options:

```sh
scripts/build_coreml.sh --test
scripts/build_coreml.sh --fresh
scripts/build_coreml.sh --build-dir build-coreml-debug --jobs 4
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  scripts/build_coreml.sh
```

`--test` runs CTest after the build. `--fresh` discards a stale SDK or compiler
selection from an earlier Command Line Tools build. `--build-dir` accepts an
absolute path or a path relative to the repository root. Run
`scripts/build_coreml.sh --help` for the complete interface.

For troubleshooting or integration into another build system, the equivalent
manual commands are:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
ICU_PREFIX="$(brew --prefix icu4c)"
JSON_PREFIX="$(brew --prefix nlohmann-json)"
git submodule update --init --recursive
cmake --fresh -S . -B build-coreml -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_PREFIX_PATH="${ICU_PREFIX};${JSON_PREFIX}" \
  -DLAYA_CUDA=OFF -DLAYA_VULKAN=OFF -DLAYA_COREML=ON
cmake --build build-coreml --parallel 8
build-coreml/bin/laya-cli --coreml --model models/laya \
  --input benchmarks/cases/smoke.json
```

Homebrew bottles are built for the current macOS release. A binary intended for
macOS 12 distribution must therefore link an ICU build produced with a macOS 12
deployment target, rather than relying on a newer Homebrew ICU bottle.

## Validate on M1 Pro

The validator is fail-closed: it uses a CPU PyTorch oracle, compares raw logits
and actions, verifies public results and determinism, and writes a JSON report.

```sh
.venv-coreml/bin/python benchmarks/validate_coreml.py \
  --executable build-coreml/bin/laya-cli \
  --source research/laya --model models/laya --variants english \
  --batch-sizes 1 2 4 8 --output results/coreml-validation.json
```

Do not publish performance numbers until this report has `passed: true` and
`complete: true`. Use Instruments or `powermetrics` separately to inspect the
actual accelerator placement.

The validation report is bound to the native build, checkpoint weights, Core ML
manifest, corpus and batch sizes. Regenerate it after rebuilding the executable
or exporting the model. E5RT caches are isolated by compiled bucket under
`results/coreml-cache/`; this avoids cross-shape MPSGraph cache failures and
allows the benchmark to reuse the validation caches. The directory can consume
several additional GiB and may be removed when no validation or sweep is active.

## Benchmark on Apple Silicon

Run the paired CPU/Core ML sweep only after the complete validation succeeds:

```sh
.venv-coreml/bin/python benchmarks/sweep_coreml.py \
  --executable build-coreml/bin/laya-cli \
  --source research/laya --model models/laya --variant english \
  --validation results/coreml-validation.json \
  --cache-root results/coreml-cache \
  --batch-sizes 1 2 4 8 --warmup 3 --iterations 5 \
  --output results/coreml-sweep.json
```

The sweep uses the same committed 250-question corpus and report shape as the
CUDA/Vulkan sweep. It records paired FP32 PyTorch CPU and Core ML p50/p95 latency,
questions per second, all timing samples and the Core ML speedup. Preprocessing,
inference and result formatting are measured; model loading and JSON transport
are excluded. Execution order alternates to reduce thermal and ordering bias.

Each Core ML process loads only one compiled bucket. A final partial request
group may use a smaller bucket in a separate process. Correctness is checked
again before timing every group, and no speedup is reported for a failing batch.
Use an idle Mac on external power and keep Low Power Mode disabled for comparable
results. The CPU baseline and Core ML use the same machine, but this paired
speedup is not directly interchangeable with CUDA/PyTorch measurements from a
different host.

## Migration and rollback

Core ML is opt-in. Existing CUDA, Vulkan and CPU behavior is unchanged unless
`-DLAYA_COREML=ON` and `--coreml` are selected. To roll back, rebuild without
`LAYA_COREML` and stop passing `--coreml`; generated `coreml/` directories can
remain beside the checkpoints because other backends ignore them.
