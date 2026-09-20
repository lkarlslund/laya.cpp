# Precision and port correctness

Correctness means agreement with the baseline at the **same precision**:
FP32 against FP32, and mixed BF16 against mixed BF16. Categories must match
exactly and public numeric outputs must differ by no more than 0.0001.
Comparisons use identical request groups, batch sizes and padding. Differences
between precision modes are model behavior, not porting errors.

## Native BF16

Select `--bf16` for native mixed-precision inference. The older
`--experimental-bf16` spelling remains an alias. Inference runs entirely in C++
and CUDA through ggml; Python is only needed for optional benchmark tooling.
BF16 requires fused CUDA attention and a GPU with compute capability 8.0 or newer.
It cannot be combined with `--cpu`, `--no-flash`, or `--tensor-core-fp32`.

The validated build profile uses **NVCC 13.0.88 and cuBLAS 13.1.0.3**
(the library version API reports 13.1.0). Validation hardware is an RTX PRO 6000
Blackwell. Compiler and library selection affect rounding and GEMM algorithms;
CUDA 13.4 / cuBLAS 13.7 did not reproduce this profile's outputs. BF16 therefore
checks for a CUDA 13.0 compiler and cuBLAS 13.1.0 at startup. Other GPUs and
baseline software versions need their own acceptance run; a version check is
not a substitute for that run.

Use a CUDA 13.0 toolkit and matching cuBLAS installation. For example, with
`CUDA_ROOT` and `CUBLAS_ROOT` pointing to those installations:

```sh
cmake -S . -B build-bf16 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDA_cudart_LIBRARY="$CUDA_ROOT/lib64/libcudart.so" \
  -DCUDA_cublas_LIBRARY="$CUBLAS_ROOT/lib64/libcublas.so" \
  -DCUDA_cublasLt_LIBRARY="$CUBLAS_ROOT/lib64/libcublasLt.so"
cmake --build build-bf16 --parallel 8
ctest --test-dir build-bf16 --output-on-failure
build-bf16/bin/laya-cli --bf16 --model models/laya \
  --input benchmarks/cases/smoke.json
python benchmarks/models.py --bf16 --executable build-bf16/bin/laya-cli \
  --sweep --output results/bf16
```

Select a host compiler supported by the toolkit. The build records selected
math-library directories in its runtime search path. Benchmark fingerprints
also include the loaded cuBLAS, cuBLASLt and CUDA runtime libraries, so changing
them invalidates an earlier acceptance report.

### Arithmetic contract

BF16 is mixed precision, not a cast of every tensor to BF16:

- Projections use BF16 inputs and weights, FP32 accumulation, and BF16 outputs.
  Bias fusion follows the projection layout, including the batched decision head.
- Normalization uses FP32 Welford reductions and affine arithmetic. Residual
  additions stay FP32.
- GELU and rotary packing preserve rounding boundaries and strict device math;
  they compile separately from fast-math FP32 kernels.
- Attention uses BF16 Tensor Core products, FP32 softmax statistics and residual
  accumulators, and BF16 probabilities before the value product. Masked,
  unmasked and sequence-partitioned paths preserve their reduction orders.
- BF16 projection inputs and intermediates stay in BF16 where possible. Fused
  GELU gating preserves both the activation rounding and the product rounding.
  Projection output conversion can include the following FP32 residual addition.
- Local attention skips only entirely masked tiles, retains the padded-query
  fallback, and preserves the order of active tiles. Shared-memory padding
  reduces bank conflicts without changing arithmetic.
- Graph reuse distinguishes padded from unpadded groups as well as tensor shapes.

Kernel tests cover activation rounding boundaries, projections, normalization,
masked and unmasked attention through 1,024 tokens, and graph replay. GPU memory
and race checking are run separately with CUDA graphs disabled; normal CTests
exercise graph replay.

## FP32

Strict FP32 remains the default. The optimized mode
(`--tensor-core-fp32 --flash-fp32`) uses exact stored FP16 projection weights and
paired activation components with FP32 accumulation. It is assessed against
FP32 outputs and does not require the BF16 compiler/library profile.

## Measurement method

The fixed 250-question corpus is evaluated on all three checkpoints at batches
1, 2, 4 and 8: 3,000 comparisons per precision. Validation includes metadata,
public outputs, raw tensor diagnostics and three repeated-call checks. A sweep
requires matching passing validation, uses three warmups and five timed
iterations per request group, and alternates native and baseline execution.

One baseline and one additional native checkpoint are resident at a time. Times
include preprocessing, inference and formatting, excluding model loading and
native JSON transport. Shared-GPU timings are exploratory, not exclusive-device
measurements. Corpus agreement is not a proof for every possible input or a
measure of model task accuracy.

## Optimized BF16 versus optimized FP32

Both native modes were measured from the same executable, CUDA compiler and
math libraries, with alternating calls on identical request groups. Each mode
first passed its own same-precision baseline comparison. The GPU was shared;
small differences near 1.00× should be treated as approximate parity.

| Checkpoint | Batch | Optimized FP32 q/s | BF16 q/s | BF16 speedup |
|---|---:|---:|---:|---:|
| english | 1 | 368.1 | 376.8 | 1.02× |
| english | 2 | 445.8 | 580.6 | 1.30× |
| english | 4 | 453.9 | 746.8 | 1.65× |
| english | 8 | 405.3 | 770.8 | 1.90× |
| multilingual | 1 | 498.8 | 493.2 | 0.99× |
| multilingual | 2 | 653.0 | 823.3 | 1.26× |
| multilingual | 4 | 660.6 | 1145.4 | 1.73× |
| multilingual | 8 | 552.5 | 1201.6 | 2.17× |
| typed-decisions | 1 | 282.3 | 311.0 | 1.10× |
| typed-decisions | 2 | 315.3 | 491.4 | 1.56× |
| typed-decisions | 4 | 296.9 | 586.7 | 1.98× |
| typed-decisions | 8 | 248.6 | 561.6 | 2.26× |

BF16 is approximately tied with optimized FP32 at batch 1 for English and
multilingual, and about 10% faster for typed-decisions. At batches 2–8, BF16 is
26–126% faster across the three checkpoints. The FP32 mode already uses FP16
Tensor Core products with compensation, so this is not a comparison against
scalar FP32 multiplication. Small workloads can remain limited by kernel and
request overhead rather than matrix throughput.

The optimization removes redundant casts and contiguous copies, keeps projection
and MLP intermediates in BF16, fuses rounded GELU gating and FP32 residual addition,
skips entirely masked local-attention tiles, and pads shared-memory rows to
reduce bank conflicts. None of these changes relaxes the numerical contract.
The final build passes all 3,000 BF16 corpus comparisons and 216 multi-question
edge comparisons with zero observed raw tensor differences. Optimized FP32 also
passes its 3,000-comparison regression matrix.

A separate paired English before/after run measured a 42–74% BF16 throughput
improvement over `da9fe32` across batches 1–8.

The [measurement summary](measurements/bf16-optimized.json) includes both native
modes, their acceptance reports and build/library identities, and a paired
before/after BF16 comparison. These measurements supersede comparisons made by
dividing native throughputs from different builds or different timing runs.

## Initial BF16 acceptance on RTX PRO 6000 Blackwell

The initial corrected BF16 build (`da9fe32`) passes all 3,000 question comparisons across English,
multilingual and typed-decisions at batches 1, 2, 4 and 8. All observed raw logits
and action tensors are identical to the matching BF16 baseline on this corpus;
metadata, public answers and repeated-call checks also pass. The public contract
remains exact categories and numeric error at most 0.0001. The separate
multi-question edge corpus also passes all 216 comparisons across the three
checkpoints at batches 1 and 2, with zero observed raw tensor differences.
The optimized FP32 regression run passes all 3,000 public-output comparisons
on the same build.

The [earlier precision study](measurements/precision-study.json) records the
pre-fix BF16 prototype, which failed 2,750 of 3,000 comparisons. Those diagnostic
timings and failures describe the old implementation, not the current BF16 path.

The [BF16 measurement summary](measurements/bf16-parity.json) records validation,
loaded-library build fingerprints, checkpoint identities and paired timings.
Throughput below is questions/second, measured on a shared GPU:

| Checkpoint | Batch | BF16 baseline | Native BF16 | Speedup |
|---|---:|---:|---:|---:|
| english | 1 | 146.4 | 260.5 | 1.78× |
| english | 2 | 267.0 | 382.9 | 1.43× |
| english | 4 | 457.5 | 461.1 | 1.01× |
| english | 8 | 681.0 | 460.8 | 0.68× |
| multilingual | 1 | 177.2 | 347.9 | 1.96× |
| multilingual | 2 | 318.6 | 542.7 | 1.70× |
| multilingual | 4 | 544.2 | 692.5 | 1.27× |
| multilingual | 8 | 838.6 | 676.2 | 0.81× |
| typed-decisions | 1 | 146.9 | 207.1 | 1.41× |
| typed-decisions | 2 | 252.2 | 308.0 | 1.22× |
| typed-decisions | 4 | 408.0 | 347.8 | 0.85× |
| typed-decisions | 8 | 538.9 | 319.2 | 0.59× |

In that initial build, native BF16 was faster at batches 1 and 2 for all three
checkpoints, but trailed the BF16 baseline at batch 8 and typed-decisions batch 4.
These historical comparisons are BF16 versus BF16. The newer optimization and
paired native-mode measurements are recorded above.
