Purpose: Record the architecture, memory rules, and technical decisions that implementation must preserve.

# PagedCore Technical Design

## Core design decision

The KV-cache allocator and the attention code must both use the same page table. It is not enough to track free blocks separately while Hugging Face attention still reads from a normal contiguous KV cache. In real paged inference, attention must read the KV data through the page-table mapping created by the allocator.

PagedCore loads the pinned TinyLlama weights and tokenizer, then implements the Llama decoder forward path needed for this one model: RMSNorm, Q/K/V projections, RoPE, grouped-query attention, output projection, SwiGLU MLP, final norm, and LM head. The prefill and decode paths share these weights. The Hugging Face model is a correctness oracle, not the runtime cache implementation.

PagedAttention’s defining mapping is from a sequence’s logical KV blocks to noncontiguous physical blocks read by attention. 

## Components

```
HTTP / SSE
   │
   ├── validation and tokenization
   │
   └── bounded pending queue
             │
      single model worker
             │
       admission scheduler ─── KV credit ledger + active-slot limit
             │
       prefill / batched decode
             │
      block-table attention ─── GPU KV block pool + free list
             │
       bounded per-request output queues and measurements
```

FastAPI handles HTTP connections, streaming, and request queues. A separate thread runs the GPU model and manages the scheduler, KV blocks, and page tables, so GPU work never blocks FastAPI's event loop.

When a new request arrives, the HTTP handler tries to add it to the pending FIFO queue. This operation is atomic, so the server can safely reject the request if the queue is already full.

Each request has:

- a cancellation flag,
- an output queue that can hold up to 32 token events,
- a separate terminal signal for `done` or `error`.

The GPU worker cannot directly modify FastAPI's event-loop objects, so it uses `loop.call_soon_threadsafe(...)` to safely send tokens and completion signals back to the event loop.

The terminal signal is separate from the token queue. Therefore, even if the token queue becomes full because the client is slow, the server can still mark the request as completed, failed, or cancelled.

## KV layout and invariants

Use one preallocated FP16 GPU tensor conceptually shaped:

```
[num_layers, 2, num_blocks, block_tokens, num_kv_heads, head_dim]
```

The `2` axis represents K and V. A sequence stores ordered physical block IDs in its block table and its count of cached token positions.

Invariants:

1. A free block belongs to no sequence. An allocated block belongs to exactly one sequence in the MVP.
2. Logical token position `p` maps to `block_table[p // 16]` and slot `p % 16`.
3. Allocate a new physical block before writing position `p` when `p % 16 == 0`.
4. Attention reads only valid slots, including the partially filled final block.
5. Release every block and every capacity credit exactly once on EOS, length limit, cancellation, or failure.
6. No request reads another request’s blocks.
7. One worker mutates allocator and scheduler state; HTTP handlers never mutate it directly.
8. The active sequence count never exceeds the configured `max_active_sequences`; the MVP default is 32.

The physical pool is allocated at startup. “Allocate on demand” means assigning a free physical block to a sequence when needed, not asking CUDA for a new tensor per token. Track both pool occupancy and live token-slot utilization; they answer different questions.

## Admission without preemption

Preemption is outside the MVP, so an active request must be able to finish even if every other request reaches its output limit. Admission requires an available active slot and capacity credits for:

```
ceil((prompt_tokens + max_new_tokens - 1) / 16) blocks
```

The subtraction reflects generation order: prefill caches the prompt and produces the first token; subsequent decode steps cache previously emitted tokens. The request also must satisfy `prompt_tokens + max_new_tokens <= 2048`.

Credits are a promise of future blocks, not blocks physically assigned in advance. Admit a waiting request only when its required credits fit alongside all active reservations. Physical blocks are still assigned as tokens enter the cache. The pending queue is strict FIFO: only its head is considered, and a head request that cannot fit blocks later requests rather than being bypassed. Completion and cancellation return the active slot and credits before the next admission pass.

This policy is conservative because a request may reserve more KV capacity than it actually uses. That can reduce how many requests run at the same time.
Report reserved KV credits separately from blocks that are actually allocated and tokens that are currently stored.
Preemption may improve this later, but do not add it to the MVP unless it is explicitly designed and implemented.

When the server starts, load the model first and check how much GPU memory is still free. Create the KV-cache pool using the configured `kv_pool_mib`, but keep about 1 GiB free for temporary model work.
Test this setup with 32 active sequences and the largest supported prompt. If it can still cause OOM, increase the safety margin. If you increase `max_active_sequences`, test the memory limit again.
If the requested KV pool does not fit in GPU memory, stop startup and show a clear error.

## Scheduler iteration

The GPU worker repeats these steps:

1. **Clean up requests.** Remove cancelled requests that are still waiting. For active requests that finished, failed, or were cancelled, free their KV blocks, reserved capacity, and active slot.
2. **Admit one new request.** Check the first request in the FIFO queue. If there is a free active slot and enough KV capacity, admit it and run its prefill. Otherwise, leave it waiting. Only one new request can be prefetched per loop iteration.
3. **Run decoding.** If prefill produced a normal token, send it to the client. Then put all unfinished active requests into one decode batch and run one model decode step.
4. **Send decode tokens.** Each active request can produce at most one new token from that decode step. If a request finishes, mark it for cleanup at the start of the next loop iteration.

If there is no active or pending work, the worker waits without polling. Serial prefill can delay the decode step, and strict FIFO can cause head-of-line blocking; both are measured MVP limitations.

## Forward-pass sequence

**Prefill:** Tokenize the prompt and run the whole prompt through the model with causal attention. Store each layer’s K and V values in the request’s paged KV cache. Use the final logits to choose the first output token. Temporary contiguous tensors are allowed during this step.

**First token:** Choose the highest-probability token from the prefill logits. If it is EOS and `ignore_eos=false`, stop immediately and emit nothing. Otherwise, send the token to the client. If only one output token was requested, stop without adding that token to the KV cache.

**Decode:** For every active request, take its most recently emitted token and current position. Run those tokens together as one batch through the model. Compute their new K/V values and append them to the correct KV pages.

During attention, read the request’s valid KV blocks in order. Process at most one block at a time and combine the attention results using a numerically stable online softmax. Do not rebuild the entire KV history into one large contiguous tensor.

After the decode step, emit one new token for each active request. Finished or cancelled requests are then removed.

For this model, eight query heads share each KV head. Apply RoPE using the token’s absolute position before storing K in the cache. Perform the softmax calculations in FP32 for numerical stability, then convert the attention output back to FP16.

Before optimizing performance, verify that the outputs and intermediate values closely match the Hugging Face reference.

This implementation will use many small PyTorch operations, which may create significant overhead and limit performance. Measure and report this limitation.

## Request lifecycle

```
received → validated → pending → admitted → prefilling
         → decoding → completed
```

From pending, prefilling, or decoding, a request may become cancelled or failed. Cancellation is checked before prefill, between decode iterations, and before publishing an event. An already launched GPU operation cannot be interrupted; cleanup occurs at the next worker boundary.

If a request’s token queue is full, mark that request as `slow_consumer` and cancel it. On the next worker loop, free its KV blocks, reserved capacity, and active slot, then close the request. Do not wait for space in the queue, because the client may not be reading anymore. The HTTP stream should watch both the token queue and the terminal signal so it can stop cleanly after cancellation. Benchmark clients should always read streamed tokens quickly.

If the pending queue is full, reject the request before starting the stream. If the GPU worker fails, fail the affected requests, free their resources, record the error, and mark the server as not ready if the model cannot safely continue. Never keep serving if you are unsure which request owns which KV blocks.

## Measurements

Use monotonic timestamps. Record:

- `arrival`: request accepted by the service.
- `admission`: capacity credits granted.
- `first_token`: first token event produced.
- `completion`: final token event or EOS decision completed.
- Each token event timestamp.

`TTFT = first_token - arrival`, including queue and prefill. `ITL` is each interval between consecutive token events; it is undefined for a response with fewer than two events. End-to-end latency is `completion - arrival`. Output throughput is completed emitted tokens divided by the measured wall-clock interval. Also report achieved requests per second, errors, and rejections.

If EOS happens before any token is emitted, do not record TTFT or ITL; leave them missing instead of setting them to zero. For responses with at least two output tokens, calculate `TPOT = (total latency - TTFT) / (output_tokens - 1)`. Each request record should store timestamps, token counts, finish reason, and any cancellation or error code, but never store the prompt or generated text.

KV metrics use these exact definitions:

- `block_occupancy = occupied_blocks / total_blocks`.
- `reservation_ratio = reserved_blocks / total_blocks`.
- `slot_fill = live_cached_token_slots / (occupied_blocks × 16)`, defined as zero when no blocks are occupied.
- `effective_pool_fill = live_cached_token_slots / (total_blocks × 16)`.

The benchmark table's `KV utilization` column means peak `block_occupancy`. Report the other three values separately for PagedCore and use `N/A` for another system when an equivalent measurement is unavailable.

The service records per-request data. The benchmark client measures the same events from outside the server; client measurements are the primary comparison with vLLM because they include transport effects.

## Decisions and known limits

| Decision                           | Reason                                                                       | Limit                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| One pinned Llama checkpoint        | Makes model parity testable                                                  | No architecture portability claim                      |
| FP16 on one T4                     | Matches the project’s hardware budget                                        | No cross-GPU claim                                     |
| PyTorch blockwise decode attention | Makes page-table use inspectable                                             | Kernel-launch overhead is expected                     |
| Serial prefill                     | Keeps the MVP bounded                                                        | Long prompts can raise active-request ITL              |
| Conservative KV credits            | Guarantees progress without preemption                                       | Lower possible concurrency                             |
| 32 active-sequence limit           | Bounds decode workspace and batch size                                       | Can queue work even when KV blocks remain              |
| Strict FIFO admission              | Simple and deterministic                                                     | A large head request can block smaller requests        |
| Single worker                      | Clear GPU ownership                                                          | One process and one GPU only                           |
| Bounded output queues              | Prevents slow clients from consuming unbounded memory or stalling the worker | Slow clients can receive partial output before closure |
| Local API                          | Enough to study serving behavior                                             | No public-service security claim                       |
