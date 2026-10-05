# Resolved implementation constants

These values close the open Phase 0 contracts. They are implementation defaults and
acceptance thresholds, not evidence that the GPU memory margin or parity envelope has
passed its later T4 validation gate.

Values for the future worker/service are required contracts, not existing configuration fields. See [current status](STATUS.md) for delivered behavior.

## Model and software baseline

| Constant               | Value                                      |
| ---------------------- | ------------------------------------------ |
| Model                  | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`       |
| Model revision         | `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Tokenizer revision     | `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Reference Python       | `3.11.15`                                  |
| Reference PyTorch      | `2.7.1`                                    |
| CUDA wheel runtime     | `12.6` (`cu126`)                           |
| Reference Transformers | `4.52.4`                                   |

The model and tokenizer must both be loaded with the immutable revision above. The
reference Python version is recorded in `.python-version`; `pyproject.toml` keeps
the supported 3.11.x package range, and `uv.lock` freezes the resolved CPU and CUDA
dependency graphs. The `cuda` extra selects PyTorch `2.7.1+cu126`; the `cpu` extra
selects PyTorch `2.7.1+cpu` for non-GPU development.

## Memory and lifecycle defaults

| Constant                   |    Value | Contract                                                         |
| -------------------------- | -------: | ---------------------------------------------------------------- |
| `kv_pool_mib`              |   `8192` | Initial FP16 KV-pool size on the T4 target                       |
| `workspace_margin_mib`     |   `1024` | Memory kept free for model work and temporary tensors            |
| `max_request_body_bytes`   | `262144` | Limit for the complete HTTP request body; reject with `413`      |
| `shutdown_timeout_seconds` |     `10` | Cooperative shutdown waiting budget; escalate through process termination |
| `max_preprocessing_requests` |      `4` | Concurrent body-read/validation/tokenization slots; reject overflow |
| `stream_write_timeout_seconds` |     `10` | Per-send deadline; stalled delivery closes as `slow_consumer` |

The future pool startup check uses actual free device memory after model loading. Validate dense workspace during parity, pool/batched workspace during paged decode, and final worst-case service memory before official measurements. The defaults do not certify any GPU memory margin. A graceful waiting budget does not promise safe interruption of a running thread or CUDA operation.

## Reproducibility controls

`pagedcore.determinism.seed_everything(seed)` seeds Python's `random` generator and
the available PyTorch CPU and CUDA generators. Inference code should run its forward
pass inside `pagedcore.determinism.inference_mode()` so autograd state and its memory
bookkeeping are disabled.

The test suite applies seed `0` before every test. Seeding does not guarantee bitwise
identical CUDA results across GPU models, drivers, CUDA versions, or PyTorch versions.
CUDA kernels can also have no deterministic implementation or can change numerical
results through FP16 reductions. GPU parity evidence must therefore record the full
environment and compare the documented tensor/logit tolerances; these controls are
not a performance or cross-machine determinism guarantee.

## Configuration loading

`load_config()` returns an immutable `PagedCoreConfig` and reads optional process
environment values with the `PAGEDCORE_` prefix. It does not load a `.env` file.

| Setting | Environment variable | Default |
|---|---|---:|
| Model ID | `PAGEDCORE_MODEL_ID` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Model revision | `PAGEDCORE_MODEL_REVISION` | `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Context limit | `PAGEDCORE_CONTEXT_LIMIT` | `2048` |
| Minimum requested output | `PAGEDCORE_MIN_NEW_TOKENS` | `1` |
| Default requested output | `PAGEDCORE_DEFAULT_MAX_NEW_TOKENS` | `128` |
| Maximum requested output | `PAGEDCORE_MAX_NEW_TOKENS` | `256` |
| Tokens per KV block | `PAGEDCORE_BLOCK_TOKENS` | `16` |
| KV pool size in MiB | `PAGEDCORE_KV_POOL_MIB` | `8192` |
| Workspace margin in MiB | `PAGEDCORE_WORKSPACE_MARGIN_MIB` | `1024` |
| Pending request limit | `PAGEDCORE_MAX_PENDING_REQUESTS` | `64` |
| Active sequence limit | `PAGEDCORE_MAX_ACTIVE_SEQUENCES` | `32` |
| Per-request token queue size | `PAGEDCORE_TOKEN_QUEUE_SIZE` | `32` |
| Bind address | `PAGEDCORE_BIND_ADDRESS` | `localhost` |
| Log level | `PAGEDCORE_LOG_LEVEL` | `INFO` |

Model identity, context and output bounds, and block size are fixed by the MVP
contract. Setting their environment variables to another value fails configuration
loading. Pool, queue, active-sequence, bind, and log settings are configurable; numeric
capacity settings must be greater than zero. Accepted log levels are `DEBUG`, `INFO`,
`WARNING`, `ERROR`, and `CRITICAL`.

The current `PagedCoreConfig` does not yet expose `max_request_body_bytes`, `shutdown_timeout_seconds`, `max_preprocessing_requests`, or `stream_write_timeout_seconds`. Add and validate them with service implementation, using the defaults above; planned environment names are respectively `PAGEDCORE_MAX_REQUEST_BODY_BYTES`, `PAGEDCORE_SHUTDOWN_TIMEOUT_SECONDS`, `PAGEDCORE_MAX_PREPROCESSING_REQUESTS`, and `PAGEDCORE_STREAM_WRITE_TIMEOUT_SECONDS`. Keep the body limit fixed at `262144` for the MVP. Preprocessing capacity is a positive integer excluding booleans; shutdown/write durations must be finite and positive. Freeze all resolved limits for benchmarks. Their absence today is not evidence that the future service bounds are implemented.

## Numerical parity thresholds

Use `torch.testing.assert_close` with these initial thresholds when comparing the
owned decoder with the pinned Hugging Face oracle:

| Compared values | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Intermediate tensors and final hidden state | `2e-3` | `2e-3` |
| Final logits | `5e-3` | `5e-3` |

Greedy token IDs must still match exactly. These thresholds may be changed only after
the measured discrepancy envelope and its numerical cause are recorded.

## Stable error codes

Error codes are lowercase `snake_case` strings and are part of the API contract.

| HTTP/status or terminal path | Stable code |
|---|---|
| `400` malformed or invalid request | `invalid_request` |
| `413` request body too large | `request_body_too_large` |
| `413` prompt plus output exceeds context | `context_limit_exceeded` |
| `422` maximum KV demand cannot fit an idle pool | `kv_capacity_exceeded` |
| `429` pending queue is full | `pending_queue_full` |
| `429` preprocessing capacity is exhausted | `preprocessing_capacity_exceeded` |
| `503` model or service is unavailable | `model_unavailable` |
| Stream cancelled because mailbox is full or transport write times out | `slow_consumer` |
| Stream failed because the model worker failed | `worker_failure` |
| Stream closed during graceful shutdown | `server_shutdown` |
| Internal cancellation after client disconnect | `client_disconnected` |

`client_disconnected` is recorded internally and does not require a final SSE event.

Use the JSON envelopes and validation order in [SPEC.md](SPEC.md#http-contract). These codes are future HTTP/stream contracts; the current verifier CLI is not an implementation of that API.
