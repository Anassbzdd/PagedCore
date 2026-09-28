# Resolved implementation constants

These values close the open Phase 0 contracts. They are implementation defaults and
acceptance thresholds, not evidence that the GPU memory margin or parity envelope has
passed its later T4 validation gate.

## Model and software baseline

| Constant               | Value                                      |
| ---------------------- | ------------------------------------------ |
| Model                  | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`       |
| Model revision         | `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Tokenizer revision     | `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Reference Python       | `3.11.15`                                  |
| Reference PyTorch      | `2.7.1`                                    |
| Reference Transformers | `4.52.4`                                   |

The model and tokenizer must both be loaded with the immutable revision above. The
reference Python version is a 3.11.x environment; `pyproject.toml` remains the
supported-version contract until dependency locking is completed in Phase 1.

## Memory and lifecycle defaults

| Constant                   |    Value | Contract                                                         |
| -------------------------- | -------: | ---------------------------------------------------------------- |
| `kv_pool_mib`              |   `8192` | Initial FP16 KV-pool size on the T4 target                       |
| `workspace_margin_mib`     |   `1024` | Memory kept free for model work and temporary tensors            |
| `max_request_body_bytes`   | `262144` | Limit for the complete HTTP request body; reject with `413`      |
| `shutdown_timeout_seconds` |     `10` | Maximum graceful-shutdown wait before forced cleanup is reported |

The pool size is checked against actual free device memory at startup. It is not a
guarantee that the configuration is safe before the Phase 8 worst-case workspace test.

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
| `503` model or service is unavailable | `model_unavailable` |
| Stream cancelled because token delivery is full | `slow_consumer` |
| Stream failed because the model worker failed | `worker_failure` |
| Stream closed during graceful shutdown | `server_shutdown` |
| Internal cancellation after client disconnect | `client_disconnected` |

`client_disconnected` is recorded internally and does not require a final SSE event.
