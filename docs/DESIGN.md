Purpose: Record the architecture, memory rules, and technical decisions that implementation must preserve.

# PagedCore Technical Design

This is the required target architecture. The decoder, paging, worker, and HTTP components are not implemented yet; [STATUS.md](STATUS.md) distinguishes delivered code from evidence and planned work.

## Core design decision

The KV-cache allocator and the attention code must both use the same block table. It is not enough to track free blocks separately while Hugging Face attention still reads from a normal contiguous KV cache. In real paged inference, attention must read the KV data through the block-table mapping created by the allocator.

PagedCore loads the pinned TinyLlama weights and tokenizer, then implements the Llama decoder forward path needed for this one model: RMSNorm, Q/K/V projections, RoPE, grouped-query attention, output projection, SwiGLU MLP, final norm, and LM head. The prefill and decode paths share these weights. The Hugging Face model is a correctness oracle, not the runtime cache implementation.

PagedAttention’s defining mapping is from a sequence’s logical KV blocks to noncontiguous physical KV blocks read by attention.

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
       bounded per-request token queues and measurements
```

FastAPI handles HTTP connections, streaming, and request validation. A separate thread runs the GPU model and owns scheduler/KV mutations. Body reading and validation hold one of four preprocessing slots; acquire without waiting and reject overflow before accumulating a body. Enforce body size during reads. Tokenization runs outside the event loop in a bounded executor, with at most one submitted job per occupied preprocessing slot. Cancellation cannot release a slot while its tokenizer job is still running. Pending insertion is nonblocking; a cancelled preprocessing job cannot later enqueue a request.

When a new request arrives, the HTTP handler tries to add it to the pending FIFO queue, whose default bound is 64 requests. This operation is atomic, so the server can safely reject the request if the queue is already full.

Each request has:

- a cancellation flag,
- one thread-safe bounded token mailbox that holds up to 32 events by default,
- a separate terminal signal for `done`, `error`, or cancellation.

The worker publishes into the mailbox with nonblocking `put_nowait`. Detect fullness there, before scheduling callbacks, and cancel that request as `slow_consumer`. Event-loop callbacks contain notifications, not token payloads. Coalesce wakeups so each request has at most one queued notification; guard the notification flag/mailbox handoff against lost wakeups. Cancellation is a thread-safe signal, and HTTP handlers never mutate allocator or scheduler state.

Use `loop.call_soon_threadsafe(...)` only to notify the loop and resolve its terminal future. Scheduling one payload callback per token would bypass the mailbox bound while the loop is stalled. A closed event loop is a publication failure: signal cancellation and retain worker-owned cleanup. Test stalled-loop behavior, closed-loop publication, and notification races.

The scheduling API supplies thread safety, not a bounded payload queue: [Python event-loop reference](https://docs.python.org/3.11/library/asyncio-eventloop.html#asyncio.loop.call_soon_threadsafe).

The terminal signal is separate from the token queue. Therefore, even if the token queue becomes full because the client is slow, the server can still mark the request as completed, failed, or cancelled.

Resolve one generation terminal outcome exactly once. Successful completion drains accepted tokens before `done` and the final text flush. Cancellation/failure can discard remaining tokens and never emits `done`. A separate stream supervisor watches disconnect/terminal cancellation while transport writes are pending; it cancels a stalled send rather than waiting for the generator to resume. Each send has the configured ten-second deadline. A delivery failure after generation completed does not rewrite the generation outcome or release resources twice; record it separately as a transport failure.

## KV layout and invariants

Use one preallocated FP16 GPU tensor conceptually shaped:

```
[num_layers, 2, num_blocks, block_tokens, num_kv_heads, head_dim]
```

The `2` axis represents K and V. A sequence stores ordered physical block IDs in its block table and its count of cached token positions. In the formulas below, `block_tokens=16` and `context_limit=2048` are the fixed MVP values from the specification.

Invariants:

1. A free block belongs to no sequence. An allocated block belongs to exactly one sequence in the MVP.
2. Logical token position `p` maps to `block_table[p // block_tokens]` and slot `p % block_tokens`.
3. Allocate a new physical block before writing position `p` when `p % block_tokens == 0`.
4. Attention reads only valid slots, including the partially filled final block.
5. Release acquired blocks, credits and slots exactly once at safe boundaries on EOS, length, cancellation and recoverable failure. Unsafe CUDA/ownership failures require fail-closed process termination as described below.
6. No request reads another request’s blocks.
7. One worker mutates allocator and scheduler state; HTTP handlers never mutate it directly.
8. The active sequence count never exceeds the configured `max_active_sequences`; the MVP default and benchmark value are 32.

At every quiescent worker boundary, assert:

```text
free_blocks + owned_blocks = total_blocks
owned_blocks <= reserved_blocks <= total_blocks
reserved_blocks = sum(active_request.maximum_block_credits)
owned_blocks = sum(active_request.assigned_blocks)
active_slots = number_of_admitted_requests_not_yet_cleaned_up
```

Reserved credits include already-owned blocks. Unassigned reserved capacity is `reserved_blocks - owned_blocks`; it overlaps the physical free list and must not be counted as a third disjoint block category. Pending requests own none of these resources. Verify per-request ownership, not just aggregate sums.

`CapacityCounts` validates nonnegative integer counts and the first two equations
when a consistent snapshot is constructed. It does not inspect free lists, block
tables, request reservations, configured active limits, or valid cached positions.
Add those concrete assertions in the allocator (P4) and worker (P6), including
uniqueness of physical block ownership and zero ownership for pending requests.
Here, active requests include admitted terminal requests awaiting cleanup.

The metrics name `occupied_blocks` means the same count as ledger `owned_blocks`; it is not an additional resource category.

Releasing a block requires all previously launched accesses to it to be ordered before any reuse. Keep model work on one worker-owned CUDA stream; order reuse on that stream and use completion events where host-side publication or shutdown requires completion. Never return pages for unsynchronized use by another stream. CUDA-context failure makes continued serving unsafe: mark readiness false and terminate rather than issuing speculative recovery operations on the broken context.

The physical pool is allocated at startup. “Allocate on demand” means assigning a free physical block to a sequence when needed, not asking CUDA for a new tensor per token. Track both pool occupancy and live token-slot utilization; they answer different questions.

## Admission without preemption

Preemption is outside the MVP, so an active request must be able to finish even if every other request reaches its output limit. Admission requires an available active slot and capacity credits for:

```
required_blocks = ceil((prompt_tokens + max_new_tokens - 1) / block_tokens)
```

The subtraction reflects generation order: prefill caches the prompt and produces the first token; subsequent decode steps cache previously emitted tokens. The request also must satisfy `prompt_tokens + max_new_tokens <= context_limit`.

Credits are a promise of future blocks, not blocks physically assigned in advance. Admit a waiting request only when its required credits fit alongside all active reservations. Physical blocks are still assigned as tokens enter the cache. The pending queue is strict FIFO: only its head is considered, and a head request that cannot fit blocks later requests rather than being bypassed. Completion and cancellation return the active slot and credits before the next admission pass.

This policy is conservative because a request may reserve more KV capacity than it actually uses. That can reduce how many requests run at the same time.
Report reserved KV credits separately from blocks that are actually allocated and tokens that are currently stored.
Preemption may improve this later, but do not add it to the MVP unless it is explicitly designed and implemented.

When the server starts, load the model first and check how much GPU memory is still free. Create the KV-cache pool using the configured `kv_pool_mib` (initial default: `8192` MiB), while keeping the `workspace_margin_mib` default of `1024` MiB free for temporary model work. These are configuration defaults, not proof that the final margin is safe.
Measure dense prefill workspace before adding the pool, paged/batched workspace when decode is implemented, and the final end-to-end maximum-context/32-active workload. Increase the margin or reduce capacity if unsafe, documenting changes before freezing benchmarks. If `max_active_sequences` changes, repeat relevant memory tests.
If the requested KV pool does not fit in GPU memory, stop startup and show a clear error.

The initial pool has 23,831 whole blocks. Thirty-two maximum-context reservations use at most 4,096 blocks (1,408 MiB), so the default active-slot limit binds before KV credits. The two fixed workloads reserve about 110 and 440 MiB at 32 active requests. These are derived upper bounds, not GPU measurements. Keep defaults provisional until measured; use separately labeled smaller-pool/mixed-length experiments to exercise credit pressure and FIFO blocking.

## Scheduler iteration

The GPU worker repeats these steps:

1. **Clean up requests.** Remove cancelled requests that are still waiting. For active requests that finished, failed, or were cancelled, free their KV blocks, reserved capacity, and active slot.
2. **Admit one new request.** Check the first request in the FIFO queue. If there is a free active slot and enough KV capacity, admit it and run its prefill. Otherwise, leave it waiting. Only one new request can be prefilled per loop iteration.
3. **Run decoding.** If prefill produced a normal token, send it to the client. Then put all unfinished active requests into one decode batch and run one model decode step.
4. **Send decode tokens.** Each active request can produce at most one new token from that decode step. If a request finishes, mark it for cleanup at the start of the next loop iteration.

If there is no active or pending work, the worker waits without polling. Serial prefill can delay the decode step, and strict FIFO can cause head-of-line blocking; both are measured MVP limitations.

## Forward-pass sequence

**Prefill:** Consume prompt IDs already validated by preprocessing and run the whole prompt with causal attention. Store each layer's K and V in the request's paged cache. Use final logits to choose the first output token. Temporary contiguous tensors are allowed here; the worker does not retokenize or apply a chat template.

**First token:** Choose the highest-probability token from the prefill logits. If it is EOS and `ignore_eos=false`, stop immediately and emit nothing. Otherwise, send the token to the client. If only one output token was requested, stop without adding that token to the KV cache.

**Decode:** For every active request, take its most recently emitted token and current position. Run those tokens together as one batch through the model. Compute their new K/V values and append them to the correct physical KV blocks.

During attention, read the request’s valid physical KV blocks in logical block-table order. Process at most one block at a time and combine the attention results using a numerically stable online softmax. Do not rebuild the entire KV history into one large contiguous tensor.

After the decode step, emit one new token for each active request. Finished or cancelled requests are then removed.

For this model, eight query heads share each KV head. Apply RoPE using the token’s absolute position before storing K in the cache. Perform the softmax calculations in FP32 for numerical stability, then convert the attention output back to FP16.

Before optimizing performance, verify that the outputs and intermediate values closely match the Hugging Face reference.

This implementation will use many small PyTorch operations, which may create significant overhead and limit performance. Measure and report this limitation.

## Request lifecycle

### Shared records

[`engine_types.py`](../src/pagedcore/engine_types.py) defines immutable records
without importing Torch, Transformers, or the HTTP layer. `RequestId` and `BlockId`
are distinct static types; `MonotonicTimestamp` is integer nanoseconds from the
server process's monotonic clock. These ID/timestamp wrappers do not perform
runtime validation.

`RequestState` names the lifecycle stages below. `GenerationTerminal` is a union
of completed, cancelled, and failed records: only completion carries `eos` or
`length`; cancellation carries its stable trigger; failure carries an `EngineError`.
The separate `TransportTerminal` records normal closure, disconnect, or delivery
failure. It does not change a resolved generation outcome or imply GPU cleanup.
Transport may terminate before generation when cancellation awaits a safe worker
boundary. Only the worker will resolve generation and release ownership.

`TokenEvent` is a transient mailbox record with a zero-based index, token ID, and
successful publication timestamp. It is not an SSE payload. The HTTP adapter will
add incremental text deltas and the final completion flush; terminal records remain
outside the bounded token mailbox. `EngineError` uses the stable codes in
[CONSTANTS.md](CONSTANTS.md#stable-error-codes) and content-free public messages.

`RequestTiming` defines the six server timestamps described under Measurements;
unobserved events are `None`. `RequestCounts` separates prompt tokens (including
special tokens), requested output limit, successfully published output tokens,
and live cached positions. These measurement records retain no text, token IDs,
or per-token history. `CapacityCounts` separates physical ownership, reserved
credits, active slots, and live slots. `KVReservation` records maximum block
credits and rejects nonpositive or noninteger values, including booleans;
`BlockTable` records physical IDs in logical order. `CapacityCounts` rejects
negative or noninteger counts, including booleans, and broken aggregate block
equations. `validate_request_transition` checks state changes without mutating
state. Request mutation, per-request accounting, cleanup, publication,
serialization, and instrumentation remain work for their owning gates.

```
received → validated → pending → admitted → prefilling → decoding → completed
                                              └────────────────→ completed
```

Only the following state changes are legal:

| Current state | Allowed next states |
|---|---|
| received | validated, cancelled, failed |
| validated | pending, cancelled, failed |
| pending | admitted, cancelled, failed |
| admitted | prefilling, cancelled, failed |
| prefilling | decoding, completed, cancelled, failed |
| decoding | completed, cancelled, failed |
| completed, cancelled, failed | none |

Prefill can complete on immediate EOS or a one-token output limit. Cancellation
and failure are allowed before admission and between admission and prefill;
pre-pending rejection remains an HTTP error, not a streamed generation outcome.
Skipped stages, backwards changes, and self-transitions raise `ValueError`.
Another decode step keeps the existing state without a transition. A terminal
generation outcome is resolved once and never rewritten by delivery failure.

### Ownership and release-once contract

Before admission, requests own no KV blocks, credits, or active slots. Admission
grants maximum credits and one active slot before prefill assigns physical blocks.
Moving to completed, cancelled, or failed does not itself release ownership:
an admitted terminal request retains its remaining resources until worker cleanup.

For EOS, length, disconnect, slow consumer, shutdown, or recoverable failure,
worker cleanup returns each acquired block, reservation, and active slot once.
Partial acquisition failure returns only resources actually acquired. Repeated
cleanup signals after successful release are no-ops, not further counter decrements
or free-list insertions. The allocator must detect a direct double-free or foreign
release; request cleanup idempotence must not hide corrupted ownership. Transport
closure neither performs GPU cleanup nor resets its release status. These are
allocator/worker implementation and regression-test requirements for P4/P6;
shared records and validators do not provide cleanup evidence.

Check cancellation before prefill, between iterations, and before publication. A launched GPU operation cannot be interrupted; release occurs at the next safe worker boundary. Shutdown first clears readiness and stops admission, then cancels work and wakes an idle worker. Wait at most `shutdown_timeout_seconds=10` for cooperative cleanup and worker exit. A timed join cannot kill a thread. If it remains alive, report failed graceful shutdown and require supervisor/operator process termination; do not free or reuse its live GPU resources. Process termination reclaims resources through the OS/driver and is not evidence of release-once graceful cleanup.

Python threads cannot be forcibly stopped by a timed join: [Python thread reference](https://docs.python.org/3.11/library/threading.html#thread-objects). The external termination path preserves the one-process/one-model-worker MVP; it adds no second serving worker.

If a request’s token queue is full, mark that request as `slow_consumer` and cancel it. On the next worker loop, free its KV blocks, reserved capacity, and active slot, then close the request. Do not wait for space in the queue, because the client may not be reading anymore. The HTTP stream should watch both the token queue and the terminal signal so it can stop cleanly after cancellation. Benchmark clients should always read streamed tokens quickly.

If the pending queue is full, reject the request before starting the stream. If the worker fails, fail affected requests and release safely recoverable resources. Mark the server not ready if ownership or CUDA state is uncertain; terminate when safe in-process cleanup is impossible. Never continue serving with uncertain ownership.

## Measurements

Use monotonic timestamps and distinguish generation from delivery:

- `arrival`: validated request accepted into pending state.
- `admission`: capacity credits and an active slot granted.
- `first_token`, `last_token`, and per-token times: successful publication into the bounded mailbox, including empty text deltas.
- `generation_terminal`: worker decides EOS, length, cancellation, or failure.
- `transport_terminal`: stream finishes or its delivery fails.

Internal `TTFT = first_token - arrival`, including pending wait and prefill. Internal ITL pools consecutive token-publication gaps. `TPOT = (last_token - first_token) / (output_tokens - 1)` for at least two emitted tokens. Generation latency is `generation_terminal - arrival`; stream latency is `transport_terminal - arrival`. EOS decisions and terminal transport overhead must not be silently included in TPOT. Track transport status separately from the generation finish reason.

For zero-token EOS, TTFT/ITL/TPOT are missing. ITL and TPOT are undefined below two emitted tokens. Export counts, times, statuses, and stable reasons; never retain prompt/generated text or token IDs in metrics or benchmark records. Keep aggregate server metrics and bounded diagnostic retention. The benchmark client owns full public-run timing records; do not retain an unbounded server-side request history.

KV metrics use these exact definitions:

- `block_occupancy = occupied_blocks / total_blocks`.
- `reservation_ratio = reserved_blocks / total_blocks`.
- `slot_fill = live_cached_token_slots / (occupied_blocks × block_tokens)`, defined as zero when no blocks are occupied.
- `effective_pool_fill = live_cached_token_slots / (total_blocks × block_tokens)`.

The benchmark table's `KV utilization` column means peak `block_occupancy`. Report the other three values separately for PagedCore and use `N/A` for another system when an equivalent measurement is unavailable.

Define timestamp types with core interfaces and add instrumentation while implementing the worker and service. Internal token-publication times and client-observed output times are different measurements. Client comparisons follow the [validation protocol](VALIDATION_PLAN.md#client-timing-and-accounting); never subtract timestamps from different clock domains.

## Decisions and known limits

| Decision                           | Reason                                                                       | Limit                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| One pinned Llama checkpoint        | Makes model parity testable                                                  | No architecture portability claim                      |
| FP16 on one T4                     | Matches the project’s hardware budget                                        | No cross-GPU claim                                     |
| PyTorch blockwise decode attention | Makes block-table use inspectable                                           | Kernel-launch overhead is expected                     |
| Serial prefill                     | Keeps the MVP bounded                                                        | Long prompts can raise active-request ITL              |
| Conservative KV credits            | Guarantees progress without preemption                                       | Lower possible concurrency                             |
| 32 active-sequence limit           | Bounds decode workspace and batch size                                       | Can queue work even when KV blocks remain              |
| Strict FIFO admission              | Simple and deterministic                                                     | A large head request can block smaller requests        |
| Single worker                      | Clear GPU ownership                                                          | One process and one GPU only                           |
| Bounded token queues               | Prevents slow clients from consuming unbounded memory or stalling the worker | Slow clients can receive partial output before closure |
| Local API                          | Enough to study serving behavior                                             | No public-service security claim                       |
