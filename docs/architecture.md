# Architecture

The serving runtime is a C++20 library and JSON-lines executable. It links ggml,
CUDA/cuBLAS, ICU, and nlohmann-json. Python is used only by optional model-download,
validation, and measurement tools.

## Checkpoint and tokenizer

The loader reads safetensors directly, validates tensor names, shapes, byte ranges,
and finite values, and uploads persistent weights. It accepts two ModernBERT
profiles: width 1024 / 28 layers / 16 heads / intermediate width 2624 / vocabulary
50368, and width 768 / 22 layers / 12 heads / intermediate width 1152 / vocabulary
256000. Both have two decision layers and head dimension 64. Checkpoint metadata
selects dimensions, serving context (512 or 1024), token budget, and calibration.
Unsupported configurations fail explicitly. Model selection is explicit through
`--variant` or a checkpoint directory; automatic language routing is not implemented.

The English and typed-decisions tokenizers perform NFC normalization, added-token
matching, Unicode-aware byte-level pretokenization, and ranked BPE merges. The
multilingual tokenizer replaces spaces with metaspace markers, preserves the
checkpoint's normalization behavior, and applies Unicode BPE with byte fallback.
All tokenization runs in C++; tokenizer vocabulary parsing uses a map-based JSON
representation, while request objects preserve insertion order for choice ordering.

Input formatting uses the checkpoint's question/option budget (192 or 256 tokens)
and serving sequence limit, with right truncation of state text. Literal mask
tokens in user text are neutralized before encoding.

## Computation

Each question becomes a batch row. State text may differ across rows. The encoder
uses global attention every third layer and bidirectional local attention with
an inclusive half-window of 64 elsewhere. Global and local rotary frequency
bases are 160000 and 10000 for English/typed-decisions, and 160000 for both
attention types in the multilingual checkpoint. Rotary tables are persistent graph inputs, protected
from temporary-buffer reuse. Padded query rows receive a harmless valid key when
needed to avoid an all-masked softmax; padded keys remain excluded from valid
queries at every layer. Mask generation runs on the selected backend from a small
vector of sequence lengths; quadratic host-side masks are not constructed or
transferred. Masks are regenerated for every call, including graph replay with
changed lengths.

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
The encoder MLP fuses product recombination, exact-erf GELU gating, and activation
splitting into one kernel, avoiding the intermediate activation copies.
The scorer and action projections remain FP32. Nonfinite results are rejected by
acceptance testing. Native mixed BF16 is selected with `--bf16` and uses dedicated
projection, normalization, activation, rotary and attention kernels. Its strict
rounding boundaries and validated toolchain are documented in [precision](precision.md).

Q/K/V packing and rotary multiplication are fused into one native kernel, with
separate rounded multiplies and addition. `--flash-fp32` enables full-FP32 fused
attention for sequences up to 128 tokens; longer sequences use cuBLAS attention.
The build generates a narrowly modified CUDA dispatcher from the pinned ggml
source to provide FP32 accumulation/output for FP16 inputs and dispatch the
custom packing operations. The dependency checkout stays unchanged.

The runtime holds one encoder graph for the current batch/sequence/option shape
and one action graph for the current batch size. A shape change, or a change in
whether BF16 inputs contain padding, rebuilds the corresponding graph; repeated shapes reuse allocations and ggml CUDA Graphs.
Weights stay resident. Calls to one agent must be serialized by the caller.
There is no automatic CPU fallback on a CUDA error.

The HTTP mode uses the pinned cpp-httplib dependency for parsing and connections.
A bounded worker pool handles HTTP requests; a mutex serializes prediction calls
on the resident agent. `/v1/systemone` returns the JEV answer envelope and
`/predict` exposes native request batches. See [HTTP serving](http.md).

For layer diagnostics, set `LAYA_TRACE_DIR` to an output directory. Intermediate
FP32 tensors are written as row-major `.f32` files after inference. This mode
retains additional activations and is unsuitable for performance measurement.
