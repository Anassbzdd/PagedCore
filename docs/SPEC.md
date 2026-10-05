Purpose: Define what the MVP must do, its public interface, and what counts as finished.

# PagedCore MVP Specification

## Goal

Build a single-GPU LLM inference service that implements its own paged KV cache, continuous batching scheduler, and streaming HTTP API. Publish correctness evidence and a reproducible comparison with vLLM.

PagedCore is a learning and portfolio engine. Its result is an explainable implementation and measurement, not a claim that it is a production alternative to vLLM.

This document defines required MVP behavior. It does not describe everything as already implemented. See [implementation and evidence status](STATUS.md) for the current gate.

## Fixed assumptions

| Decision            | MVP value                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Model               | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision `af8e934848d8dd00074cc2cd8a40a9b05c3b011e` |
| Architecture        | Llama-compatible decoder only                                                             |
| GPU                 | One NVIDIA T4                                                                             |
| Weight and KV dtype | FP16                                                                                      |
| Model context limit | 2,048 tokens total (prompt plus requested output)                                          |
| KV block size       | 16 token slots (`block_tokens`)                                                           |
| API                 | Local FastAPI service; one model worker                                                   |
| Generation          | Greedy decoding; one completion per request                                               |
| Output limit        | 1–256 tokens; default 128                                                                 |
| Pending queue       | 64 requests by default; configurable and frozen at 64 for benchmarks                       |
| Active sequences    | 32 by default; configurable and frozen at 32 for benchmarks                              |
| Per-request token queue | 32 token events by default; always bounded                                             |
| Preprocessing capacity | Four concurrent body-read/validation/tokenization operations by default; no waiting queue |
| Stream write deadline | 10 seconds per transport write by default; a stalled writer is a slow consumer |
| Request body        | 262,144 bytes maximum (`max_request_body_bytes`)                                          |
| KV pool             | 8,192 MiB initial pool; 1,024 MiB workspace margin                                        |
| Deployment          | Linux with CUDA; Docker image and direct install; bind to `localhost` by default           |
| Graceful shutdown   | 10-second graceful waiting budget; process termination is the escalation path             |

The model configuration lists 22 layers, 32 attention heads, four KV heads, hidden size 2,048, and a 2,048-token context. The checkpoint and tokenizer both use revision `af8e934848d8dd00074cc2cd8a40a9b05c3b011e`. [Source: TinyLlama configuration](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/af8e934848d8dd00074cc2cd8a40a9b05c3b011e/config.json). See [resolved implementation constants](CONSTANTS.md) for the software baseline and API error codes.

**Derived capacity check:** Head dimension is `2048 / 32 = 64`.
FP16 KV storage is `2 × 22 layers × 4 KV heads × 64 values × 2 bytes = 22 KiB` per cached token, or `352 KiB` per 16-token block. This excludes model weights, CUDA workspace, allocator overhead, and temporary tensors. Measure available memory at startup; do not infer a safe block count from the T4’s nominal memory alone. These names are used throughout the source documents: `context_limit=2048` and `block_tokens=16`.

## MVP behavior

1. **Paged KV storage.** Allocate a GPU block pool and free list. Each sequence has a logical-to-physical block table. Decode attention reads the blocks through that table. It must not reconstruct the sequence’s entire KV history into a contiguous tensor each step.

2. **Continuous batching.** Each worker iteration performs cleanup, considers the FIFO head for admission, prefills at most one newly admitted request, and runs one model decode step for the active batch. Admission requires both a free active-sequence slot and enough KV credits. Newly admitted requests can join without waiting for the previous batch to finish.

3. **Prefill**. For the MVP, process one prompt at a time. During prefill, the system can use normal temporary tensors, then copy the generated KV cache into its physical KV blocks. A very long prompt may block or slow down decoding for requests that are already running.

4. **Bounded service.** Bound preprocessing, pending requests, active requests, and token delivery. Reserve maximum KV credits at worker admission, before prefill; pending requests own no KV blocks, credits, or active slots. Reject new requests when preprocessing or pending capacity is full. Disconnects and slow consumers trigger cleanup at a safe worker boundary.

5. **Streaming.** Send each generated token to the client as soon as it is produced. When generation finishes, send a final completion message. Keep token messages in a limited-size queue, but send completion or error signals through a separate path so they can still be delivered even if the token queue is full.

6. **Measurements.** Track when each request arrives, starts running, produces its first token, produces later tokens, and finishes. Do not store the prompt or generated text. Also track basic server stats such as waiting requests, active sequences, KV-cache usage, and cancellations. Use these measurements to calculate latency percentiles such as p50, p95, and p99, and always report how many requests were measured.

## HTTP contract

### `POST /v1/generate`

Request body:

```
{
  "prompt": "Explain KV caching in one sentence.",
  "max_new_tokens": 64,
  "ignore_eos": false
}
```

Require an `application/json` body containing one object with only `prompt`, `max_new_tokens`, and `ignore_eos`. Reject unknown fields and explicit `null` values. `prompt` is a required nonempty string whose decoded contents can be encoded as UTF-8; preserve whitespace, including a whitespace-only nonempty prompt. `max_new_tokens` is a strict JSON integer from 1 to 256, excluding booleans, floats, and numeric strings; it defaults to 128. `ignore_eos` is a strict JSON boolean and defaults to `false`.

Enforce the `262144`-byte limit while reading the body, including when `Content-Length` is absent or incorrect. Acquire a preprocessing slot before accumulating a body or tokenizing; do not queue additional preprocessing jobs. Release that slot on every exit. Body reading, validation, and tokenization remain bounded separately from the pending queue.

Tokenize the raw prompt with the pinned tokenizer using `add_special_tokens=true` and `truncation=false`, retaining its pinned BOS/EOS settings. Do not apply a chat template, normalize the prompt, or manually add special tokens. Count the resulting IDs, including special tokens, when rejecting `prompt tokens + requested output tokens > 2048`. The oracle and benchmark adapters must verify identical prompt IDs and effective special-token settings.

With `ignore_eos=false`, stop when EOS is selected. Use `true` for fixed-length benchmark generation. Decode emitted IDs with `skip_special_tokens=true` and `clean_up_tokenization_spaces=false`; incremental text plus the final flush must match that full-sequence decode. Ignored EOS IDs still count as emitted tokens even when their text delta is empty.

For an admitted request, reserve the maximum KV demand before prefill:

```text
required_blocks = ceil((prompt_tokens + max_new_tokens - 1) / block_tokens)
```

The `-1` accounts for the first output token selected from prefill logits; subsequent output tokens are appended during decode. Reject the request with `kv_capacity_exceeded` when this demand cannot fit in an idle pool.

The response is `text/event-stream` over the POST request. A client uses an HTTP streaming client or `fetch`; native browser `EventSource` does not issue POST requests.

```
event: token
data: {"request_id":"...","index":0,"token_id":123,"text_delta":"Hello"}

event: done
data: {"request_id":"...","output_tokens":1,"finish_reason":"eos","text_delta":""}
```

There is one `token` event for every model token that is actually emitted. Sometimes `text_delta` can be empty because the tokenizer needs to wait for more tokens before it can safely produce text.
The final `done` event sends any remaining buffered text. If you join all `text_delta` values from the token events and the final `done` event, you should get exactly the same text as decoding the full token sequence at once.
When `ignore_eos=false`, EOS stops generation but is not sent as a token event. If EOS is produced immediately, then no token event is sent, `output_tokens=0`, and TTFT is not recorded.
Token indices are zero-based and consecutive. On successful completion, drain all accepted token events in order before sending exactly one `done` event. A terminal signal must never overtake those tokens. On cancellation or failure, queued tokens may be discarded; no subsequent `done` event is permitted.
`finish_reason` is either:
- `eos` — generation stopped because EOS was reached.
- `length` — generation stopped because the requested token limit was reached.

Before streaming starts, return `400` for invalid JSON, media type, UTF-8, or fields; `413` for a body-size or context-limit violation; `422` for maximum KV demand that cannot fit an idle pool; `429` for exhausted preprocessing or pending capacity; and `503` for an unavailable model. Readiness and preprocessing-capacity checks precede body parsing. Enforce body size before parsing, then schema, context, idle-pool capacity, and atomic pending insertion. Use the stable codes in [resolved implementation constants](CONSTANTS.md).

Pre-stream failures use `application/json` and this envelope:

```json
{"error":{"code":"invalid_request","message":"Request validation failed."}}
```

Codes are stable; messages must not echo request content or credentials. After streaming starts, an error event uses `{"request_id":"...","error":{"code":"worker_failure","message":"Generation failed."}}` and closes the stream when delivery is possible. A full token mailbox or transport write exceeding the configured deadline cancels the request as `slow_consumer`. Close through the separate terminal path without requiring a final SSE event or waiting for a blocked send. Client disconnect is recorded internally and requires no final event.

### Operational endpoints

- `GET /healthz`: process is alive.

- `GET /readyz`: model, tokenizer, and KV pool are initialized.

- `GET /metrics`: Return local server statistics as JSON. Do not include prompt text or generated text. Track totals such as accepted, completed, failed, rejected, cancelled, and emitted tokens. Also report current values such as waiting requests, active sequences, KV-cache blocks in use or reserved, and cached token slots.

Bind the server to `localhost` by default, so only the same computer can access it. For the MVP, do not add user authentication, public internet access, or isolation between multiple users.

Shutdown makes readiness false, stops admission, and signals pending/active work. The ten-second budget bounds cooperative waiting, not guaranteed interruption of a running thread or CUDA operation. If safe worker termination is impossible within that budget, fail closed and report that process termination is required. A supervisor/operator terminates the process; do not reuse the live worker's resources or claim graceful cleanup succeeded. Test graceful shutdown and forced escalation separately.

## Acceptance criteria

The final MVP acceptance criteria are:

Planned implementation, test, and evidence ownership is tracked in the
[MVP acceptance traceability table](TRACEABILITY.md).

- For greedy decoding, the model should produce the same tokens as the pinned Hugging Face reference. Also compare some internal logits and make sure the differences stay within an allowed FP16 numerical tolerance.

- Tests should prove that KV blocks are allocated and reused correctly, requests stay isolated from each other, and resources are freed when a request is cancelled. They should also check that queue limits and active-request limits work, and that a slow client cannot make output memory grow forever or prevent the stream from eventually ending.

- A two-request test shows that one request can join while another is decoding.

- A forced fragmented-allocation test places logical blocks at shuffled physical IDs, poisons unrelated blocks, and still matches the reference. A trace of decode attention shows block-table reads without a full-history KV gather.

- Run tests for both CPU and GPU code. In CI, automatically run `ruff` for code quality, `mypy` for type checking, and `pytest` for tests.

- A clean Linux/CUDA setup and Docker run both start the service.

- One command reproduces the benchmark workload and emits raw per-request results plus a summary table with p50/p95/p99 fields and observation counts. A p99 field is `null` with reason `insufficient_sample` when it has fewer than 1,000 relevant observations.

- The public report states measured results, configuration, limitations, and at least one investigated performance gap.

There is no speed target. Correct behavior and honest measurement are the release gates.

### Official measurement entry

Before official comparative runs, prove decoder/paging parity, ownership and lifecycle invariants, HTTP behavior, current CPU CI and T4 checks, and worst-case memory safety. The direct Linux/T4 setup and benchmark harness must pass their smoke/synthetic checks; freeze the protocol, inputs, and settings. Docker validation, the completed comparison/report, and published artifacts belong to final delivery and do not block earlier diagnostic pilots. Pilots must remain labeled as exploratory evidence.

### Portfolio evidence and artifact release

The technical demonstration is complete after all correctness gates and the frozen measurements support an inspectable report. Artifact release additionally requires Docker/clean-install checks and the exact tested tag, distributions, and checksums. See [validation milestones](VALIDATION_PLAN.md#completion-milestones). Neither milestone is satisfied by documentation alone.

## Outside the MVP

No multi-model serving, UI, quantization, distributed inference, prompt-prefix sharing, copy-on-write, preemption, recomputation, CPU KV offload, chunked prefill, speculative decoding, sampling controls, or vLLM API compatibility. Add a stretch feature only after the MVP report is published, with its own correctness tests and before/after measurement.
