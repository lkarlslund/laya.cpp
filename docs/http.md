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

Eight HTTP workers accept concurrent clients. Inference is serialized around the
shared tokenizer and CUDA runtime. Concurrent HTTP requests are not automatically
combined into a GPU batch; use `/predict` or several questions in one System One
request for batched inference. Up to 32 additional connections can wait in the
socket queue; excess connections are closed. Socket reads and writes time out
after ten seconds, and idle keep-alive connections after two seconds.

`elapsed_ms` on `/predict` measures preprocessing, inference and result formatting,
excluding queue wait and HTTP transfer. Measure client latency when comparing
different HTTP concurrency levels.

## Authentication and lifecycle

The default bind address is loopback. Set `--host 0.0.0.0` to listen on all IPv4
interfaces. Set `LAYA_API_KEY` in the server environment to require
`Authorization: Bearer <key>` on prediction and model discovery routes. If unset,
authentication is disabled. Use a TLS reverse proxy for HTTPS deployments.

Keep machine-specific launch scripts, service units, proxy configuration and
credentials outside the repository or under ignored `local/`. These deployment
files are not part of the source distribution.

`GET /health` is unauthenticated and reports readiness, the loaded model, backend
and question limit. The listener starts after model loading. SIGINT and SIGTERM
stop listening and let accepted work finish before releasing the model.

Errors are JSON objects under `error`, with `message` and numeric `status`:

| Status | Meaning |
|---|---|
| 400 | Malformed JSON or HTTP request |
| 401 | Missing or invalid configured bearer token |
| 404 / 405 | Unknown route / unsupported method |
| 413 | Body or total question limit exceeded |
| 422 | Invalid request, question or model selection |
| 500 | Inference or internal server failure |

## Validation

`ctest --test-dir build-cuda --output-on-failure` includes model-independent HTTP
tests for routes, envelopes, authentication, limits, errors and serialization.
The same tests run in a CPU-only build.

```sh
python benchmarks/http_validate.py
```

The transport validation compares all 250 fixed questions against CLI answers
on each checkpoint at batch sizes 1, 2, 4 and 8, then exercises the JEV endpoint
with 1, 2, 4 and 8 concurrent clients. It requires exact public JSON equality
(apart from the documented HTTP model identity), and checks clean SIGTERM
shutdown. It tests transport correctness; model acceptance and performance
measurements remain separate in the [benchmark harness](benchmarking.md).

The [recorded HTTP validation](measurements/http-validation.json) passed all three
checkpoints: 6,000 question evaluations with exact CLI parity across both routes,
all tested batch/concurrency levels, and clean shutdowns.
