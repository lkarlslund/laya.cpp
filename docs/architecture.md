# Architecture

Laya evaluates typed questions in a bidirectional transformer pass. Each question
is a batch row; options occupy marked token positions in that row. It does not
decode a stream of generated tokens and does not need an autoregressive KV cache.

The initial target is the English checkpoint: 28 ModernBERT layers, hidden width
1024, 16 attention heads, intermediate width 2624, and a 50368-token vocabulary.
Global attention appears every third layer, interleaved with local attention
(window 128). Rotary position parameters differ between global and local layers.
The serving limit is 512 tokens per question with a 192-token question/option
budget. Encoder capacity alone must not override the serving configuration.

The encoder output receives a question-type embedding and two pre-normalized
transformer head layers. A normalized MLP scores option-marker positions. An
action MLP consumes the first-token embedding plus probability summary features.
The action features use uncalibrated option probabilities. Output probabilities
use temperature scaling with question-type and option-count buckets. Boolean
confidence is max(p, 1-p); choice and score confidence use normalized entropy.

The native `laya::calibrate` API accepts valid option logits and an already selected
temperature. It returns full-precision probabilities, first-maximum choice index,
expected ordinal score, and entropy confidence. Public output formatting and
boolean-specific confidence belong to the future serving layer.

RTX optimization priorities are persistent device weights and workspaces, BF16
Tensor Core matrix products, efficient bidirectional attention, fused normalization
and activation operations, and CUDA Graph replay for stable shape buckets. Every
precision or fusion change must pass decision-output agreement checks before its
speedup is accepted.
