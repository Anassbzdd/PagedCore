
Purpose: Turn the design into an ordered build, test, benchmark, and publication plan.

# PagedCore Implementation and Validation Plan

## Build order

Complete each gate before starting the next substantial feature.

|Gate|Work|Evidence required|
|---|---|---|
|1. Environment|Pin model/tokenizer revision and Python, PyTorch, Transformers, CUDA-compatible packages. Record T4 driver and CUDA details.|Clean setup loads the checkpoint in FP16.|
|2. Model parity|Implement the one-model decoder without paging. Compare layer outputs, final logits, and greedy tokens with the pinned reference.|Automated GPU parity tests with documented tolerances.|
|3. Paged cache|Add GPU pool, free list, block tables, append/read/release operations.|Boundary, reuse, isolation, and capacity tests.|
|4. Paged attention|Replace decode attention with block-table traversal and blockwise online softmax.|Parity tests at lengths 1, 15, 16, 17, and multiple blocks; no full-history gather.|
|5. Scheduler|Add credits, strict FIFO pending queue, 32-slot active limit, one serial prefill per iteration, iteration-level decode batch, and cleanup. Reserve `ceil((prompt_tokens + max_new_tokens - 1) / block_tokens)` blocks per admitted request.|Interleaved-request, head-of-line, active-limit, and cancellation tests; invariants hold after each iteration.|
|6. HTTP service|Add validation, SSE, overload responses, readiness, and metrics.|Streaming client tests, disconnect tests, and concurrent requests.|
|7. Delivery|Add CI, Docker, setup instructions, benchmark runner, and report.|Clean-clone checks and reproducible T4 run.|

Keep each gate in a reviewable commit or small series of commits. Do not optimize an incorrect forward pass.

## Test strategy

**CPU unit tests:** Exercise the allocator and scheduler with a fake model. Test empty/full pools, exact block boundaries, strict FIFO and head-of-line behavior, the active-sequence limit, credit release, cancellation at every state, double-free prevention, requests too large for the pool, and recovery after a failed request. Assert invariants after each state transition.

**GPU correctness tests:** On the pinned checkpoint, compare PagedCore with Hugging Face using the same token IDs and FP16 weights. Cover one and multiple sequences, short and long prompts, block boundaries, EOS, maximum output length, and reused physical blocks. Force a multi-block sequence onto shuffled physical block IDs and poison unrelated blocks so an implementation that ignores the block table fails. Compare intermediate outputs and logits with explicit absolute/relative tolerances; also check greedy token equality on deterministic fixtures. Investigate mismatches rather than widening tolerances until a failing case passes.

**Service tests:** Verify `POST /v1/generate`, `GET /healthz`, `GET /readyz`, and `GET /metrics`, including status codes, the default 64-request pending bound, the `262144`-byte body limit, and stable error codes. Verify SSE framing, one token event per emitted token, incremental decoding with incomplete byte sequences, final text flushing, first-step EOS, queue overload, the default 32-sequence active limit, simultaneous clients, disconnect during pending/prefill/decode, and readiness after an injected worker failure. Fill a token queue deliberately and prove the out-of-band terminal signal closes the stream. Confirm cancellation eventually returns all blocks, credits, and active slots.

**Memory and lifecycle tests:** Run repeated mixed-length requests and cancellations, then assert zero allocated blocks, zero reserved credits, and zero occupied active slots. Track device VRAM to detect unexplained growth. Test the maximum supported prompt/output combination at the default 32-sequence active limit before setting the final workspace margin. Verify graceful shutdown completes within the default 10-second bound or reports forced cleanup.

Run CPU checks, `ruff`, and `mypy` on every CI push. Run GPU tests on an actual T4 before release; a CPU-only CI pass does not certify the engine.

## Benchmark protocol

The comparison uses the same pinned TinyLlama weights and tokenizer, FP16, one T4, 2,048-token context, greedy decoding, and fixed output length. Run PagedCore and vLLM as separate servers on the same otherwise-idle machine, one at a time. Use one benchmark client with backend adapters: PagedCore uses `/v1/generate`, and vLLM uses `/v1/completions` so no chat template is added. Both receive the same raw prompt strings, and server-side tokenization is included in client TTFT. The official vLLM benchmark documentation distinguishes TTFT, inter-token latency, and time per output token, so report these names precisely. [Source: vLLM benchmark documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/).

Pin and smoke-test the exact vLLM release on the T4. Its frozen command must explicitly set `--dtype half`, `--max-model-len 2048`, `--block-size 16`, `--gpu-memory-utilization 0.90`, and `--no-enable-prefix-caching`; record all resolved engine settings. Keep the pinned release's normal chunked-prefill, scheduler, kernel, and CUDA-graph defaults because vLLM is the production reference, then name those differences when analyzing the gap. Configure PagedCore to remain within the same 90% device-memory ceiling and publish both systems' configured KV bytes; observed peak VRAM is still reported rather than assumed equal. T4-class GPUs do not support BF16 in vLLM, so the explicit FP16 override is required for this checkpoint. [Source: vLLM CUDA platform dtype checks](https://docs.vllm.ai/en/latest/api/vllm/platforms/cuda/).

Create a checked-in workload file with deterministic prompts of approximately 128 and 512 tokenizer tokens. Use 32 and 128 generated tokens respectively. Enable `ignore_eos` in both servers and disable EOS stopping in the Hugging Face baseline so every system produces the fixed length. Verify actual prompt and output token counts; do not silently truncate. Hash the workload file and record its seed.

For each workload:

1. Pilot both servers, then freeze a common offered-request-rate grid that includes an uncongested point and at least one point near saturation. Use Poisson arrivals (`burstiness=1.0`) from saved per-repetition seeds and no client-side concurrency cap. Publish the chosen grid before inspecting comparative results.
    
2. Restart the server between repetitions and clear any reusable prefix cache. Warm up with 20 requests; exclude them from results.
    
3. Run at least five repetitions of at least 100 measured requests per grid point, using the same precomputed arrival schedules. Increase the total to at least 1,000 requests for any cell used to make a p99 TTFT claim.
    
4. Let admitted requests complete; record attempted, completed, rejected, and failed requests. Never compute latency percentiles from successes while hiding rejection rates.
    
5. Save raw per-request event times, token counts, device VRAM samples, configuration, and environment metadata.
    

Report client-measured TTFT p50/p95, pooled ITL p50/p95, per-request TPOT p50/p95, end-to-end latency p50/p95, `output_tok_per_s`, achieved request rate, error/rejection rate, and peak device VRAM. Use `output_tok_per_s = total completed emitted tokens / benchmark interval seconds` and `achieved_req_per_s = completed requests / benchmark interval seconds`. Emit p99 fields and sample counts for every metric, but set a p99 value to `null` with `insufficient_sample` when fewer than 1,000 relevant observations exist. Calculate a 95% bootstrap confidence interval across repetition-level throughput values. Do not treat individual tokens from one run as independent repetitions.

Sample total device-used memory through NVML every 50 ms, with no other GPU processes, and subtract the idle-device baseline recorded immediately before server startup. Device-level sampling includes vLLM child processes. A server’s internal KV utilization is valid for its own diagnosis; mark the comparison table `N/A` where an equivalent vLLM or Hugging Face measure is unavailable. For PagedCore, `KV utilization` is peak occupied blocks divided by total blocks; separately report reservation ratio, slot fill, and effective pool fill.

Run Hugging Face `generate` as a **single-request reference baseline** for latency and output throughput. Label its concurrent-QPS cells `N/A`; do not imply it is a configured online serving system.

## Analysis required for publication

Publish the raw result files and a table with these columns:

|Workload and offered rate|System|Completed / attempted|TTFT p95|ITL p95|TPOT p95|Output tok/s|Peak VRAM|KV utilization|
|---|---|---|---|---|---|---|---|---|

Follow the table with:

- A correctness summary and the largest observed reference discrepancy.
    
- Where time goes: queue, prefill, dense model operations, blockwise attention, scheduler, and streaming. Use synchronized GPU profiling only for component timings; keep ordinary client latency measurements unsynchronized.
    
- The measured gap to vLLM at matched workload and load, with uncertainty and named confounds.
    
- The effects of serial prefill and conservative capacity credits.
    
- Any overload, OOM, or failed runs, including how they were handled.
    

Do not assert that a particular optimization caused a gap unless a profile or controlled experiment supports it.

## Risks and responses

|Risk|Response|
|---|---|
|FP16 numerical differences change greedy output|Compare intermediate tensors and logits; document tolerance and any true divergence.|
|A “paged” implementation copies full KV history each step|Review decode attention’s data path and test it with a full-history-gather prohibition.|
|Pool sizing leaves too little transient workspace|Measure worst-case GPU use, raise the workspace margin, and fail startup when unsafe.|
|Cancelled streams leak blocks or credits|Exercise cancellation at each lifecycle state and assert post-run zero ownership.|
|Many short requests create an unsafe decode batch|Gate admission on the 32-slot active limit as well as KV credits; validate workspace before changing the limit.|
|A full token queue prevents stream termination|Use the separate terminal future and test forced slow-consumer closure.|
|Long prefills damage ITL|Measure mixed prompt lengths and name serial prefill as an MVP limitation.|
|Benchmark favors one server through configuration or hidden failures|Pin configurations, use one client/workload, publish raw data and rejection counts.|
|vLLM version or T4 support changes|Pin and smoke-test the exact release before comparative runs; record any unsupported configuration.|

## Release gate

Publish only after model parity, allocator invariants, service lifecycle tests, lint/type checks, clean installation, Docker smoke test, and the T4 benchmark pass. The README must link the architecture, exact reproduction commands, raw measurements, results table, and limitations. Write one technical report explaining a measured gap or failure.

After that release, consider preemption/recomputation, chunked prefill, copy-on-write and prefix sharing, then speculative decoding. Each is a separate change with its own correctness test and before/after result.
