# Architecture

The serving runtime is a C++20 library and JSON-lines executable. It links ggml,
CUDA/cuBLAS, ICU, and nlohmann-json. Python is used only by optional model-download,
validation, and measurement tools.

## Checkpoint and tokenizer

The loader reads safetensors directly, validates tensor names, shapes, byte ranges,
and finite values, and uploads persistent weights. It currently accepts the
English 28-layer ModernBERT-large architecture: hidden width 1024, 16 heads,
intermediate width 2624, vocabulary 50368, and two decision layers. Unsupported
architectures and tokenizer features fail explicitly. Automatic multilingual
routing is not implemented.

The tokenizer performs NFC normalization, added-token matching, Unicode-aware
byte-level pretokenization, and ranked BPE merges. JSON object insertion order is
preserved because it defines choice ordering. Input formatting uses a 192-token
question/option budget and the checkpoint's serving sequence limit (512 for the
English checkpoint), with right truncation of state text. Literal mask tokens in
user text are neutralized before encoding.

## Computation

Each question becomes a batch row. State text may differ across rows. The encoder
uses global attention every third layer and bidirectional local attention with
an inclusive half-window of 64 elsewhere. Global and local rotary frequency
bases are 160000 and 10000. Rotary tables are persistent graph inputs, protected
from temporary-buffer reuse. Padded query rows receive a harmless valid key when
needed to avoid an all-masked softmax; padded keys remain excluded from valid
queries at every layer.

Encoder outputs receive a question-type embedding and two pre-normalized
transformer layers. A normalized MLP scores option markers. The action MLP uses
the first-token embedding and four statistics derived from the uncalibrated
option distribution. The small action feature calculation runs on the CPU;
the action MLP runs on the selected ggml backend.

Public option probabilities use question-type/option-count temperature buckets.
Choice selection uses the first maximum; scores are ordinal expectations.
Choice/score confidence is normalized entropy; boolean confidence is max(p,1-p).
Public numeric fields are rounded to four decimal places.

## Precision and memory

Strict FP32 is the default. It disables CUDA TF32 permission before backend
initialization and keeps small projections on a full-precision matrix path.
Initialize the library before other CUDA users in an embedding process; the
standalone executable controls initialization order. Learned tensors are not
quantized. The opt-in `--tensor-core-fp32` path retains stored FP16 projection
weights exactly and splits FP32 activations into a leading FP16 component and a
scaled residual. One packed Tensor Core multiplication computes both products
with FP32 accumulation and output; a fused kernel combines them. This is an
approximation to FP32 activations, evaluated against the public output tolerance.
The scorer and action projections remain FP32. Nonfinite results are rejected by
acceptance testing; BF16 remains experimental.

Q/K/V packing and rotary multiplication are fused into one native kernel, with
separate rounded multiplies and addition. `--flash-fp32` enables full-FP32 fused
attention for sequences up to 128 tokens; longer sequences use cuBLAS attention.
The build generates a narrowly modified CUDA dispatcher from the pinned ggml
source to provide FP32 accumulation/output for FP16 inputs and dispatch the
custom packing operations. The dependency checkout stays unchanged.

The runtime holds one encoder graph for the current batch/sequence/option shape
and one action graph for the current batch size. A shape change rebuilds the
corresponding graph; repeated shapes reuse allocations and ggml CUDA Graphs.
Weights stay resident. Calls to one agent must be serialized by the caller.
There is no automatic CPU fallback on a CUDA error.

For layer diagnostics, set `LAYA_TRACE_DIR` to an output directory. Intermediate
FP32 tensors are written as row-major `.f32` files after inference. This mode
retains additional activations and is unsuitable for performance measurement.
