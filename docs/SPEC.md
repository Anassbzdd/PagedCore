Purpose: Define what the MVP must do, its public interface, and what counts as finished.

# PagedCore MVP Specification

## Goal

Build a single-GPU LLM inference service that implements its own paged KV cache, continuous batching scheduler, and streaming HTTP API. Publish correctness evidence and a reproducible comparison with vLLM.

PagedCore is a learning and portfolio engine. Its result is an explainable implementation and measurement, not a claim that it is a production alternative to vLLM.

## Fixed assumptions

| Decision            | MVP value                                                                    |
| ------------------- | ---------------------------------------------------------------------------- |
| Model               | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, pinned to a tested repository revision |
| Architecture        | Llama-compatible decoder only                                                |
| GPU                 | One NVIDIA T4                                                                |
| Weight and KV dtype | FP16                                                                         |
| Model context limit | 2,048 tokens                                                                 |
| KV block size       | 16 token slots                                                               |
| API                 | Local FastAPI service; one model worker                                      |
| Generation          | Greedy decoding; one completion per request                                  |
| Output limit        | 1–256 tokens; default 128                                                    |
| Pending queue       | At most 64 requests; configurable                                            |
| Active sequences    | At most 32 sequences; configurable and frozen for benchmarks                 |
| Output queue        | At most 32 token events per request                                          |
| Deployment          | Linux with CUDA; Docker image and direct install                             |

The model configuration lists 22 layers, 32 attention heads, four KV heads, hidden size 2,048, and a 2,048-token context. Pin the exact checkpoint and tokenizer revision together when implementation begins. [Source: TinyLlama configuration](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/af8e934848d8dd00074cc2cd8a40a9b05c3b011e/config.json).

**Derived capacity check:** Head dimension is `2048 / 32 = 64`. 
FP16 KV storage is `2 × 22 layers × 4 KV heads × 64 values × 2 bytes = 22 KiB` per cached token, or `352 KiB` per 16-token block. This excludes model weights, CUDA workspace, allocator overhead, and temporary tensors. Measure available memory at startup; do not infer a safe block count from the T4’s nominal memory alone.

## MVP behavior

1. **Paged KV storage.** Allocate a GPU block pool and free list. Each sequence has a logical-to-physical block table. Decode attention reads the blocks through that table. It must not reconstruct the sequence’s entire KV history into a contiguous tensor each step.

2. **Continuous batching.** Each worker iteration performs cleanup, considers the FIFO head for admission, prefills at most one newly admitted request, and runs one model decode step for the active batch. Admission requires both a free active-sequence slot and enough KV credits. Newly admitted requests can join without waiting for the previous batch to finish.

3. **Prefill**. For the MVP, process one prompt at a time. During prefill, the system can use normal temporary tensors, then copy the generated KV cache into paged memory. A very long prompt may block or slow down decoding for requests that are already running.

4. **Bounded service.** Put limits on how many requests can wait, how many can run at the same time, and how many output tokens can wait to be sent to each client. When a request is accepted, reserve enough KV-cache memory for it to finish. If the waiting queue is full, reject new requests. If a client disconnects or stops reading the output, eventually free its reserved memory and active slot.

5. **Streaming.** Send each generated token to the client as soon as it is produced. When generation finishes, send a final completion message. Keep token messages in a limited-size queue, but send completion or error signals through a separate path so they can still be delivered even if the token queue is full.

6. **Measurements.** Track when each request arrives, starts running, produces its first token, produces later tokens, and finishes. Do not store the prompt or generated text. Also track basic server stats such as waiting requests, active requests, KV-cache usage, and cancellations. Use these measurements to calculate latency percentiles such as p50, p95, and p99, and always report how many requests were measured.

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

`prompt` is nonempty UTF-8 text. The service applies the pinned tokenizer without truncation. `max_new_tokens` is an integer from 1 to 256. `ignore_eos` is `false` by default, so generation normally stops when the model produces the EOS token. Set it to `true` only for benchmarks where you want every request to generate a fixed number of tokens. Reject any request where `prompt tokens + requested output tokens > 2048`.

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
`finish_reason` is either:
- `eos` — generation stopped because EOS was reached.
- `length` — generation stopped because the requested token limit was reached.

Before streaming starts, return `400` for malformed fields, `413` for a context-limit violation, `422` for a request whose maximum KV demand cannot fit even in an idle pool, `429` when the pending queue is full, and `503` when the model is unavailable. After streaming starts, send `event: error` with a stable error code and close the stream when delivery is possible. If the 32-event token queue fills, cancel the request with reason `slow_consumer` and close its stream through the out-of-band terminal signal without requiring a final SSE event; the GPU worker must not wait for the client to read. Client disconnect is recorded internally and requires no final event.

### Operational endpoints

- `GET /healthz`: process is alive.

- `GET /readyz`: model, tokenizer, and KV pool are initialized.

- `GET /metrics**`. Return local server statistics as JSON. Do not include prompt text or generated text. Track totals such as accepted, completed, failed, rejected, cancelled, and emitted tokens. Also report current values such as waiting requests, active sequences, KV-cache blocks in use or reserved, and cached token slots.

Bind the server to `localhost` by default, so only the same computer can access it. For the MVP, do not add user authentication, public internet access, or isolation between multiple users.

## Acceptance criteria

The MVP is ready to benchmark when:

- For greedy decoding, the model should produce the same tokens as the pinned Hugging Face reference. Also compare some internal logits and make sure the differences stay within an allowed FP16 numerical tolerance.

- Tests should prove that KV blocks are allocated and reused correctly, requests stay isolated from each other, and resources are freed when a request is cancelled. They should also check that queue limits and active-request limits work, and that a slow client cannot make output memory grow forever or prevent the stream from eventually ending.

- A two-request test shows that one request can join while another is decoding.

- A forced fragmented-allocation test places logical blocks at shuffled physical IDs, poisons unrelated blocks, and still matches the reference. A trace of decode attention shows block-table reads without a full-history KV gather.

- Run tests for both CPU and GPU code. In CI, automatically run `ruff` for code quality, `mypy` for type checking, and `pytest` for tests.

- A clean Linux/CUDA setup and Docker run both start the service.

- One command reproduces the benchmark workload and emits raw per-request results plus a summary table with p50/p95/p99 fields and observation counts. A p99 field is `null` with reason `insufficient_sample` when it has fewer than 1,000 relevant observations.

- The public report states measured results, configuration, limitations, and at least one investigated performance gap.

There is no speed target. Correct behavior and honest measurement are the release gates.

## Outside the MVP

No multi-model serving, UI, quantization, distributed inference, prompt-prefix sharing, copy-on-write, preemption, recomputation, CPU KV offload, chunked prefill, speculative decoding, sampling controls, or vLLM API compatibility. Add a stretch feature only after the MVP report is published, with its own correctness tests and before/after measurement.
