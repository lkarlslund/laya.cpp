# Precision and port correctness

Correctness means agreement with the baseline at the **same precision**:

- FP32 against FP32.
- BF16 mixed precision against BF16 mixed precision.

For either pair, categories must match exactly and public numeric outputs must
differ by no more than 0.0001. Agreement is tested on identical request groups,
including their batch size and padding. Cross-precision differences are recorded
as model behavior, not counted as porting errors. The criterion does not require
BF16 to reproduce FP32 results or establish either precision's task accuracy.

The optimized FP32 mode (`--tensor-core-fp32 --flash-fp32`) uses exact stored FP16
projection weights and paired activation components with FP32 accumulation.
It is assessed against FP32 outputs. The experimental BF16 mode uses BF16
projections and explicit rounding; its use of BF16 does not by itself establish
agreement with the baseline's mixed-precision operators.

## Measurement method

Run `python benchmarks/precision_study.py` for all three checkpoints. The fixed
250-question corpus is evaluated at batches 1, 2, 4 and 8. Each candidate receives
three repeated-call checks, three warmups and five timed iterations per request
group. Execution order rotates between both baseline precisions and the native
candidate. The public and batched baseline formatters are checked for agreement.

One baseline and one additional native checkpoint are resident at a time. Native
modes are measured in separate phases, each with paired baseline timings. The GPU
is shared; these are not exclusive-device measurements. Times include
preprocessing, inference and formatting, excluding model loading and native JSON
transport. Failed modes retain diagnostic timing, but have no accepted speedup.

## Results on RTX PRO 6000 Blackwell

The 2026-09-20 study used the optimized FP32 mode and experimental BF16 mode.
Optimized FP32 passed all 3,000 matching-precision question comparisons across
the three checkpoints, with exact categories and maximum numeric error 0.0001.

Each checkpoint has 1,000 comparisons per mode (250 questions at four batch sizes):

| Checkpoint | FP32 failed comparisons | BF16 failed comparisons | BF16 changed choices |
|---|---:|---:|---:|
| English | 0 | 921 | 3 |
| Multilingual | 0 | 890 | 4 |
| Typed-decisions | 0 | 939 | 2 |

All metadata checks and repeated-call checks passed in both modes. BF16 failed
the numeric and categorical agreement gate against the BF16 baseline; its
largest numeric deviations were 0.2555, 0.0820 and 0.0344 respectively. These are
port compatibility failures, not evidence that the BF16 format is unsuitable.
Differences between the two baseline precisions were excluded from these counts.

For English, throughput in questions per second was:

| Batch | FP32 baseline | BF16 baseline | Native optimized FP32 |
|---:|---:|---:|---:|
| 1 | 152.1 | 146.9 | 351.1 |
| 2 | 210.2 | 262.9 | 460.6 |
| 4 | 250.0 | 435.3 | 494.7 |
| 8 | 247.6 | 630.0 | 426.7 |

These three columns were timed together during the FP32 candidate phase. Native
FP32 is faster than its matching FP32 baseline at every tested batch size, but
does not beat the BF16 baseline at batch 8. Precision must therefore remain
explicit in speed claims. Native BF16 requires separate matching-precision
acceptance before its performance can justify promotion.

The existing BF16 prototype was slower than optimized FP32 in every tested
configuration. Its English throughput was 197.2, 259.2, 279.6 and 260.9 questions
per second at batches 1, 2, 4 and 8, respectively. Those measurements are diagnostic,
not accepted performance results. Optimized FP32 remains the supported choice;
matching and accelerating the BF16 path is further implementation work.

The [complete summary](measurements/precision-study.json) records both precision
comparisons, all checkpoint/batch timings, binary and weight identities, and
cross-precision baseline drift. Detailed samples and failing question IDs remain
in local `results/precision-study/` reports.
