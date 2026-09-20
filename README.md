# laya.cpp

Standalone C++ inference for Laya typed decisions on NVIDIA RTX GPUs, using ggml
and its CUDA backend. Model loading, Unicode/BPE tokenization, transformer
inference, decision heads, and JSON output all run natively.

All three checkpoints are supported:

| Variant | Context | Encoder width / layers | Tokenizer |
|---|---:|---:|---|
| `english` | 512 | 1024 / 28 | NFC byte-level BPE |
| `multilingual` | 1024 | 768 / 22 | Metaspace BPE with byte fallback |
| `typed-decisions` | 1024 | 1024 / 28 | NFC byte-level BPE |

Strict FP32 is the default. The optimized CUDA path uses exact checkpoint FP16
weights, paired activation components, FP32 accumulation, and fused packing
kernels. Enable it with `--tensor-core-fp32 --flash-fp32`. BF16 remains experimental.
Correctness is checked against the baseline at matching precision; see
[precision comparisons](docs/precision.md).

## Build

Requires a C++20 compiler, CMake 3.24+, CUDA, ICU, and nlohmann-json. On Debian-like
systems the host dependencies are `libicu-dev` and `nlohmann-json3-dev`.

```sh
git submodule update --init --recursive
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
```

Architecture 120 targets RTX Blackwell. Select the architecture appropriate to
your GPU and a host compiler supported by your CUDA toolkit. For a CPU build,
configure with `-DLAYA_CUDA=OFF` and run the CLI with `--cpu`.

## Run

Download the checkpoint files with `python scripts/download_model.py --variant all`, then:

```sh
build-cuda/bin/laya-cli --model models/laya --tensor-core-fp32 --flash-fp32 \
  --input benchmarks/cases/smoke.json
```

Select another model with `--variant multilingual` or `--variant typed-decisions`.
`--model` specifies the model-store root when combined with `--variant`; without
`--variant`, it can also name a checkpoint directory directly. Selection is
explicit. Each process keeps its selected checkpoint resident.

Without `--input`, the process accepts one JSON request (or an array of requests)
per input line and keeps weights resident between calls. A request contains
`state` and a `questions` object. Responses contain `results`, `elapsed_ms`, and
`backend`. `results` is an array, including for a single request.

```json
{"state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}
```

Use `--raw` to inspect uncalibrated logits and `--prepare` to inspect input tensors.
The executable does not require Python, PyTorch, or an inference server.

For native HTTP serving with the JEV-compatible `POST /v1/systemone` endpoint:

```sh
build-cuda/bin/laya-cli --server --port 8080 --variant english \
  --tensor-core-fp32 --flash-fp32
```

The default listener is `127.0.0.1:8080`. It also provides `/health`, `/v1/models`,
and `/predict` for batched requests. See [HTTP serving](docs/http.md) for request
examples, model aliases, concurrency, limits and optional bearer authentication.

## Models and tooling

The optional Python tooling requirements are in `requirements-bench.txt`:

```sh
python scripts/download_model.py --variant all
python benchmarks/validate.py --tensor-core-fp32 --output results/validation.json
python benchmarks/sweep.py --tensor-core-fp32 --validation results/validation.json \
  --batch-sizes 1 2 4 8
```

To validate and benchmark all three checkpoints sequentially:

```sh
python benchmarks/models.py --sweep
```

The fixed acceptance corpus contains exactly 250 different questions. The sweep
requires a passing validation report matching the corpus, weights, and binary.
Model files, local research, build products, and detailed benchmark reports stay
outside Git in `models/`, `research/`, `build*/`, and `results/`.

See [model support and validation](docs/models.md), [measured performance](docs/performance.md), [architecture](docs/architecture.md), [benchmarking](docs/benchmarking.md), and
[development roadmap](docs/roadmap.md).

## License

The project is licensed under the [MIT License](LICENSE). Dependencies and
downloaded model files retain their own licenses.
