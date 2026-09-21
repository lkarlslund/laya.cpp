# HTTP serving

Start a native listener with one checkpoint resident on the GPU:

```sh
build-cuda/bin/laya-cli --server --host 127.0.0.1 --port 8080 \
  --model models/laya --variant english --tensor-core-fp32 --flash-fp32
```

Use `--variant multilingual` or `--variant typed-decisions` for the other
checkpoints. `--model` is the model-store root. Run separate processes on different
ports to serve multiple checkpoints simultaneously, subject to available VRAM.
The server does not switch checkpoints or detect languages automatically.
Without `--variant`, `--model` can instead point directly to any of the three
checkpoint directories; discovery uses that checkpoint's metadata.

## JEV-compatible evaluation

`POST /v1/systemone` accepts the [JEV request and answer schema](https://docs.typesafe.ai/api):
`state`, `questions`, and an optional `model`. State and instructions accept text,
objects or arrays; questions can mix `choice`, `score`, and `noul`.

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-latest","state":"Please refund the duplicate charge.",
       "questions":{"refund":{"type":"noul",
       "instructions":"Does the customer ask for a refund?"}}}'
```

The response is a single object containing `model`, `answers`, and `usage`,
without the CLI's `results` envelope. Answers retain Laya's additional `action`
metadata and Noul confidence. Probabilities and scores use the same native
calibration and rounding as the CLI. This is protocol compatibility: inference
uses Laya weights, context limits and token accounting.

`jev-latest`, `laya-latest`, and `laya-rl-agent` are aliases for the selected
checkpoint. Its canonical name (`laya`, `laya-multilingual`, or
`laya-typed-decisions`) and variant name are also accepted. An omitted model uses
the selected checkpoint. Other names return HTTP 422, including a different
checkpoint that is not loaded. The response identifies the canonical Laya name;
an alias does not claim that Jev weights were executed.

Existing TypeSafe clients can point their base URL at `http://127.0.0.1:8080`.
The evaluation route and response structure follow the System One API.
`GET /v1/models` lists the loaded checkpoint, with the native service release
date. Responses include `x-typesafe-request-id`.

For example, with `typesafe-sdk` (verified with version 0.7.0):

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8080", api_key="local") as client:
    result = client.system_one(
        model="jev-latest",
        state="Please refund the duplicate charge.",
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
    print(result.answers["refund"].noul)
```

When authentication is enabled, replace `local` with the server's configured key.

## Batches and concurrency

`POST /predict` accepts one request object or an array of request objects. It
returns the CLI envelope: `results` (always an array), `elapsed_ms`, and `backend`.
This is a local extension for batching; `/v1/systemone` accepts one object.

The default limit is eight total questions per HTTP call, across all requests.
Change it with `--max-questions N`, allowing for GPU memory use at larger batches
and sequence lengths. Eight has been validated across all three checkpoints.
Bodies are limited to 1 MiB. Existing model context and option-budget limits still
apply; long state is truncated by the model's preprocessing.

Concurrent requests to both prediction routes enter a bounded FIFO queue. One
inference worker combines whole HTTP calls into a GPU batch and routes each
result back to its caller. The tokenizer and runtime remain serialized.

| Option | Default | Meaning |
|---|---|---|
| `--max-batch-questions N` | `--max-questions` (8) | Total questions per GPU batch; must be at least the per-call limit |
| `--batch-wait-ms N` | 2 | Maximum collection window from the oldest request's arrival, in milliseconds (0–1000) |
| `--max-pending-requests N` | 32 | Maximum admitted HTTP calls, including active inference (1–256) |
| `--no-batching` | off | Execute HTTP calls separately; explicit request arrays still work |

A full batch or a next request that cannot fit ends collection immediately.
Zero wait combines already queued work without intentionally delaying a batch.
Waiting behind active inference can exceed the collection window. Batches are
bounded by question count; the model's context limit bounds each question's
sequence length. Larger limits need additional GPU memory.

Admission beyond the pending limit returns JSON HTTP 503 with `Retry-After: 1`.
There are eight more HTTP workers than the pending limit and a separate
32-connection socket backlog; excess sockets close. Socket reads and writes
time out after ten seconds; idle keep-alive connections after two seconds.
Slow clients can still occupy HTTP workers before inference admission.

Successful predictions include `X-Laya-Batch-Id` and `X-Laya-Batch-Offset`
headers. The offset counts request objects within the batch, not questions.
IDs are scoped to a server process. These headers allow clients to reconstruct
the exact grouping for validation without exposing other callers' payloads.
The JEV JSON envelope is unchanged. `elapsed_ms` on `/predict` measures the
whole shared batch's preprocessing, inference and formatting, excluding queue
wait and HTTP transfer.

Batch shape and padding can change floating-point results. Compare correctness
at the same precision and with the same batch grouping. Use `--no-batching`
when fixed per-call grouping is required. Late input errors are isolated by
retrying the affected batch's HTTP calls individually; a runtime failure fails
the whole batch with 500.

## Authentication and lifecycle

The default bind address is loopback. Set `--host 0.0.0.0` to listen on all IPv4
interfaces. Set `LAYA_API_KEY` in the server environment to require
`Authorization: Bearer <key>` on prediction and model discovery routes. If unset,
authentication is disabled. Use a TLS reverse proxy for HTTPS deployments.

Keep machine-specific launch scripts, service units, proxy configuration and
credentials outside the repository or under ignored `local/`. These deployment
files are not part of the source distribution.

`GET /health` is unauthenticated and reports readiness, the loaded model, backend,
batching limits, and current pending/queued call counts. The listener starts after
model loading. SIGINT and SIGTERM
stop admission, return 503 to queued calls, and let the selected batch finish
before releasing the model.

Errors are JSON objects under `error`, with `message` and numeric `status`:

| Status | Meaning |
|---|---|
| 400 | Malformed JSON or HTTP request |
| 401 | Missing or invalid configured bearer token |
| 404 / 405 | Unknown route / unsupported method |
| 413 | Body or total question limit exceeded |
| 422 | Invalid request, question or model selection |
| 503 | Queue full or server stopping |
| 500 | Inference or internal server failure |

## Validation

`ctest --test-dir build-cuda --output-on-failure` includes model-independent HTTP
tests for routes, envelopes, authentication, question budgets, cross-route
batching, overload, error isolation, response offsets and shutdown.
The same tests run in a CPU-only build.

```sh
python benchmarks/http_validate.py
```

The transport validation compares all 250 fixed questions against CLI answers
on each checkpoint at batch sizes 1, 2, 4 and 8, then exercises the JEV endpoint
with 1, 2, 4 and 8 concurrent clients, reconstructing actual GPU batches from
response headers. It requires exact public JSON equality
(apart from the documented HTTP model identity), and checks clean SIGTERM
shutdown. It also compares those same batches against the Python baseline,
requiring exact categories and numeric error no greater than 0.0001. Pass
`--bf16` to check BF16 on both sides; the default checks FP32. Performance
measurements remain in the [benchmark harness](benchmarking.md). The HTTP report
also includes observed batch sizes and single-pass client throughput/p50/p95
latencies. Use `--no-batching` for a separate run without aggregation; these
transport timings include collection waits and are not the warmed GPU benchmark.

The earlier, pre-queue [recorded HTTP validation](measurements/http-validation.json) passed all three
checkpoints: 6,000 question evaluations with exact CLI parity across both routes,
all tested batch/concurrency levels, and clean shutdowns.

The [queue validation](measurements/http-queue-validation.json) passes all three
models in FP32 and BF16: 12,000 HTTP question evaluations across both routes,
with exact CLI parity. The 6,000 dynamically grouped evaluations also pass the
same-precision Python acceptance rule. All six server runs shut down cleanly.

For Vulkan serving, build with `-DLAYA_VULKAN=ON` and launch with `--vulkan`:

```sh
build-vulkan/bin/laya-cli --vulkan --server --host 127.0.0.1 --port 8080
```

The queue and JEV routes are shared across backends. To validate Vulkan transport
and dynamic batch routing, pass `--backend vulkan --executable build-vulkan/bin/laya-cli`
to `benchmarks/http_validate.py`.
