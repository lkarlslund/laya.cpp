# Model support

All three checkpoints execute through the native C++/ggml CUDA runtime.

| Variant | Context | Option/instruction budget | Heads | Layers |
|---|---:|---:|---:|---:|
| `english` | 512 | 192 | 16 | 28 |
| `multilingual` | 1024 | 256 | 12 | 22 |
| `typed-decisions` | 1024 | 256 | 16 | 28 |

Use `python scripts/download_model.py --variant all` to download the pinned
checkpoints. Select a variant explicitly when starting the process:

```sh
build-cuda/bin/laya-cli --variant multilingual --tensor-core-fp32 --flash-fp32
build-cuda/bin/laya-cli --variant typed-decisions --tensor-core-fp32 --flash-fp32
```

Each process loads one checkpoint. The C++ library accepts a checkpoint directory
directly. Checkpoints and research files remain outside Git.

## Acceptance and throughput

Each checkpoint passed the same 250 fixed questions at batches 1, 2, 4, and 8:
exact categories, numeric absolute error at most 0.0001, exact prepared input
tensors, and stable repeated graph execution. Reports are independent per model.

The following native throughput was measured on the RTX PRO 6000 Blackwell using
five timed iterations after three warmups. Another model remained resident on the
GPU. Each checkpoint uses its own tokenizer and context limit, so token counts
differ across models for the same request text.

| Variant | Batch 1 questions/s | Batch 2 | Batch 4 | Batch 8 |
|---|---:|---:|---:|---:|
| `english` | 349.3 | 454.4 | 470.0 | 413.4 |
| `multilingual` | 486.0 | 602.2 | 694.9 | 570.3 |
| `typed-decisions` | 272.5 | 319.5 | 301.5 | 249.4 |

All performance comparisons use an FP32 baseline with autocast and TF32 disabled.
These results do not establish a speedup over BF16 execution. Full summary metadata
is in [the model matrix](measurements/three-models.json).

The multilingual tokenizer has 270 exact fixtures, and English/typed-decisions
each pass the 253 shared byte-level fixtures. All models additionally pass 36 structured, multilingual, long-context, and
multi-question cases at request batch sizes 1 and 2, defined in
`tests/model-requests.json`.
