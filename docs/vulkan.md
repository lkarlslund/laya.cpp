# Vulkan inference

Vulkan runs the English, multilingual and typed-decisions checkpoints through
native C++ and ggml GPU shaders. It supports the JSON-lines CLI and the same
JEV-compatible HTTP server and bounded batching queue as CUDA.

## Build and run

Install a C++20 compiler, CMake, ICU, nlohmann-json, Vulkan loader/development
headers, SPIR-V headers, and `glslc`. A working Vulkan driver is required at runtime.
On Debian-like systems the Vulkan build packages are `libvulkan-dev glslc
spirv-headers`.

```sh
git submodule update --init --recursive
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAYA_CUDA=OFF -DLAYA_VULKAN=ON
cmake --build build-vulkan --parallel 8
ctest --test-dir build-vulkan --output-on-failure

build-vulkan/bin/laya-cli --vulkan --model models/laya \
  --variant multilingual --input benchmarks/cases/smoke.json

build-vulkan/bin/laya-cli --vulkan --server --host 127.0.0.1 --port 8080
```

This build does not require the CUDA toolkit or Python. CUDA and Vulkan can also
be compiled together using `-DLAYA_CUDA=ON -DLAYA_VULKAN=ON`. Backend selection is
explicit: `--cuda` (the CLI default), `--vulkan`, or `--cpu`. A Vulkan-only build
still requires `--vulkan` at launch. An unavailable backend produces an error;
there is no automatic CPU fallback.

The CLI startup message and each JSON-lines response identify the physical GPU
(`device`) separately from the backend (`Vulkan0`). Validation records both the
native GPU and the Python GPU; timing requires the same native device as its
validation report. This matters on systems with both integrated and discrete GPUs.

The first visible Vulkan device is used. The pinned ggml backend accepts
`GGML_VK_VISIBLE_DEVICES` to select visible device indices. In the C++ API, use
`laya::backend_type::vulkan` in the `agent` or `runtime` constructor. The existing
boolean CUDA/CPU constructors remain supported.

## Precision and implementation

Vulkan supports plain FP32 and compensated FP32 (`--tensor-core-fp32`). It
currently rejects `--bf16` and `--flash-fp32` combinations.

Compensated FP32 splits each projection input into two FP16 components, computes
both products with FP32 accumulators, and combines them in FP32. The checkpoints'
stored FP16 projection weights are represented exactly. This enables cooperative
matrix hardware without reducing each input to a single FP16 value. Attention,
normalization and residuals remain FP32. The option name is shared with CUDA;
Vulkan uses the device's cooperative-matrix implementation, including AMD's.

Each input split and output merge uses a single GPU dispatch. The split explicitly
rounds FP16 values to nearest-even, including subnormals, and scales the residual
by 1024. Operator tests check both packed components and the merged FP32 results
bit-for-bit across multiple matrix shapes and large activation values on both
tested GPUs.

A build-time extension of the pinned Vulkan backend supplies true FP32 matrix
kernels alongside cooperative half-precision kernels. It prevents implicit
FP16 conversion of explicitly FP32 matrix operands. The dependency checkout
remains unchanged. A matrix-level precision test exercises both paths on the
same device, including non-aligned dimensions and values with significant bits
that a single FP16 conversion would lose.

On AMD, compensated products accumulate cooperative-matrix tile results with
separate FP32 additions. This reduces the accumulation drift observed when a
long dot product stays entirely inside cooperative-matrix multiply-adds. The
regression check includes the multilingual numeric answer that exceeded the
0.0001 tolerance with the original accumulation path.

The encoder, attention, decision layers and action projections execute on Vulkan.
Q/K/V packing and rotary multiplication use portable ggml operations. The host
prepares and uploads attention masks for each call; tokenizer preprocessing and
action statistics use the existing host implementation. All graph operations are
checked for backend support before execution.

FP32 feed-forward layers use a fused GEGLU shader, avoiding separate gate copies,
activation and multiplication dispatches. Small decision projections run at their
actual batch size; the CUDA-specific minimum-column padding is not applied to
Vulkan.

The development kernels also retain FP32 output accumulation for 16-bit fused
attention. Operator tests cover unaligned key lengths, uniform attention,
nonuniform masked attention and empty masked rows on both tested GPUs. This
operator-level coverage does not establish full-model 16-bit acceptance; the
supported serving modes above remain the release contract.

Dense FP32 attention uses more memory than fused attention at long sequence
lengths. Begin with the default eight-question HTTP limit and reduce it on
smaller GPUs. Cold calls also include Vulkan shader/pipeline creation; benchmark
warmed calls separately. The README performance table describes CUDA only.

## Measured FP32 throughput

[Current measurements of the fused split/merge build](vulkan-fused-performance.md)
include Python on both GPUs and native CUDA on RTX. The NVIDIA tables below
retain the earlier recorded build.

On an RTX PRO 6000 Blackwell capped at **450 W**, compensated Vulkan FP32
produced the following warmed questions/second against Python FP32. Each row
covers the fixed 250-question corpus, with three warmups and five timed runs per
request group. Both sides run on the same GPU, alternating execution order.

| Model | Batch | Python FP32 | Vulkan compensated FP32 |
|---|---:|---:|---:|
| english | 1 | 128.4 | 70.0 |
| english | 2 | 188.0 | 123.9 |
| english | 4 | 242.9 | 187.6 |
| english | 8 | 241.5 | 220.9 |
| multilingual | 1 | 170.5 | 84.7 |
| multilingual | 2 | 286.3 | 154.3 |
| multilingual | 4 | 366.1 | 233.2 |
| multilingual | 8 | 428.8 | 298.2 |
| typed-decisions | 1 | 125.7 | 62.1 |
| typed-decisions | 2 | 176.1 | 106.1 |
| typed-decisions | 4 | 194.0 | 142.3 |
| typed-decisions | 8 | 174.8 | 153.6 |

CUDA remains faster on this NVIDIA card. In a separate paired native run:

| Model | Batch | CUDA optimized FP32 | Vulkan compensated FP32 |
|---|---:|---:|---:|
| english | 1 | 346.5 | 70.4 |
| english | 8 | 431.0 | 225.8 |
| multilingual | 1 | 459.9 | 85.8 |
| multilingual | 8 | 601.1 | 298.3 |
| typed-decisions | 1 | 268.5 | 61.8 |
| typed-decisions | 8 | 261.4 | 154.6 |

All measured groups pass the public-answer tolerance. Timings include
preprocessing, inference and formatting, excluding loading and native JSON
transport. Other services retained GPU allocations; the measurements do not
claim exclusive access. These are measurements of the recorded binaries, not a
guarantee for another driver, build or GPU. Full batch results, runtime versions
and binary fingerprints are in the [measurement metadata](measurements/vulkan-nvidia-performance.json).
The Vulkan-versus-Python and CUDA-versus-Vulkan pairs were timed separately.

## Validation and benchmarking

```sh
python benchmarks/models.py --backend vulkan \
  --executable build-vulkan/bin/laya-cli --output results/vulkan

python benchmarks/models.py --backend vulkan --tensor-core-fp32 \
  --executable build-vulkan/bin/laya-cli --output results/vulkan-compensated

python benchmarks/models.py --backend vulkan \
  --executable build-vulkan/bin/laya-cli --cases tests/edge-requests.json \
  --output results/vulkan-edges

python benchmarks/http_validate.py --backend vulkan --tensor-core-fp32 \
  --executable build-vulkan/bin/laya-cli --output results/vulkan-http/validation.json
```

The acceptance contract is exact categories and absolute numeric output error
at most 0.0001 against Python FP32 using identical request groups. Validation
also checks tokenization, finite raw tensors and deterministic graph replays.
Raw tensor differences are reported separately from the public-answer gate.
The Python comparison harness requires its existing PyTorch/CUDA environment;
this is not a requirement of the Vulkan inference executable.

For an AMD comparison, run the same scripts with a separate ROCm-enabled PyTorch
environment and select the AMD Vulkan device with `GGML_VK_VISIBLE_DEVICES`.
Always check the CLI's reported device name: loader layers can make ggml's indices
differ from `vulkaninfo --summary`. The harness rejects a Vulkan GPU name that
does not match the Python GPU. PyTorch on ROCm uses the `torch.cuda` API too; reports
identify its runtime as `rocm` and record the runtime version. A CUDA Python run
on NVIDIA is not an AMD Python baseline. BF16 validation rejects a baseline that
silently selects another autocast dtype.

`benchmarks/models.py --sweep` and `benchmarks/sweep.py --backend vulkan` support
paired performance measurements after validation. A CUDA validation report
cannot authorize a Vulkan benchmark, even for a binary containing both backends.

Hardware validation is on NVIDIA RTX PRO 6000 Blackwell with driver 610.43.03.
Plain FP32 also passes all 3,000 public-answer comparisons on AMD Radeon 8060S
with Mesa 26.2.2, against Python running on that same AMD GPU through ROCm:
[AMD validation](measurements/vulkan-amd-fp32.json). Intel devices have not been
tested in this environment.

The [recorded validation](measurements/vulkan-validation.json) passes 3,000
fixed-corpus and 144 edge-case comparisons across all three checkpoints at batch
sizes 1, 2, 4 and 8. All 6,000 HTTP question evaluations match CLI replay exactly;
the 3,000 dynamically grouped HTTP evaluations also pass the Python FP32 gate.
The combined CUDA+Vulkan build additionally tests both backends in one process,
comparing answers and replaying cached graphs after changing padding lengths.

The [compensated FP32 validation](measurements/vulkan-optimized-nvidia.json)
also passes all 3,000 public-answer comparisons on NVIDIA, with exact categories
and numeric error at most 0.0001. This gate concerns public answers; raw tensor
differences remain separate diagnostics. The matrix-level FP32/FP16 test passes
on both RTX PRO 6000 Blackwell and Radeon 8060S.

The [current compensated FP32 acceptance](measurements/vulkan-compensated-fp32.json)
passes all 3,000 comparisons on each tested GPU, including AMD against Python
ROCm on the same device. It uses a clean snapshot of the recorded source commit.
The residual component uses a scale of 1024 to avoid FP16 overflow at large
activations; regression tests include values near half-precision rounding
boundaries above 32,000.

The [fused split/merge build](measurements/vulkan-fused-fp32.json) also passes all
3,000 fixed-corpus comparisons on each GPU. Its NVIDIA regression checks include
144 edge-case comparisons and a small HTTP batching smoke test across all three
models, checking CLI replay and Python answers. Operator tests additionally
isolate attention reciprocal rounding and ordered reduction of split projection
products; these checks do not change the full-model requirements for 16-bit modes.
