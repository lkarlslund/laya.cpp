# laya.cpp

C++ inference for Laya typed decisions, targeting NVIDIA RTX GPUs.

Early development: the C++ library currently implements calibrated probabilities,
choice selection, expected scores, and entropy confidence. Full model loading,
tokenization, encoder inference, decision heads, and CUDA acceleration are pending.
There is no native end-to-end inference executable yet.

## Build

Requires CMake 3.24+, a C++20 compiler, and optionally Ninja.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Model storage

Install the Python tooling dependencies from `requirements-bench.txt` into your
chosen environment, then fetch the pinned English checkpoint:

```sh
python scripts/download_model.py
```

Use `--variant multilingual` or `--variant typed-decisions` for other checkpoints.
Models live under `models/`, excluded from Git. Scratch research lives under
`research/`, also excluded. Build products and benchmark reports are local only.

See [architecture](docs/architecture.md), [benchmarking](docs/benchmarking.md), and
[development roadmap](docs/roadmap.md).
