
Purpose: Turn the design into an ordered build, test, benchmark, and publication plan.

# PagedCore Implementation and Validation Plan

This is the public evidence contract and phase map. [STATUS.md](STATUS.md) records current implementation/evidence limits. Planned tests, commands, and artifact paths below are requirements, not assertions that those surfaces exist.

## Build order

Complete each runtime gate before starting the next substantial runtime feature. Preserve dense parity -> allocator correctness -> paged attention -> lifecycle correctness -> HTTP. CI, evidence tooling, and synthetic benchmark analysis can support these gates earlier; they do not bypass them. The task IDs below define public ownership even when the detailed local checklist is not distributed.

| Phase / task IDs | Work | Evidence required |
|---|---|---|
| P0.1-P0.6: baseline | Package structure and reconciled public contracts | Parseable metadata and consistent constants/acceptance ownership |
| P1.1-P1.6: environment | Frozen CPU/CUDA setup, configuration, diagnostics, controls, reference verifier and evidence closure | Attributable fresh Linux/T4 install, reference forward, CPU checks and CUDA repeatability report |
| P9.1-P9.2: early automation | Schedule after environment implementation, before P2; retain IDs for traceability | CPU lint/format/strict typing/pytest automation and repeatable T4 job with test counts and artifacts |
| P2.1-P2.5: types/reference | Minimal interfaces, timing schema, fixtures and frozen oracle | Deterministic reference IDs/tensors; documented special-token, attention, and numerical settings |
| P3.1-P3.7: dense parity | Owned TinyLlama forward without paging | Layer/logit/greedy T4 parity and dense-prefill workspace record |
| P4.1-P4.6: allocator | One pool, free list, tables and capacity ledger | Boundary, rollback, isolation, reuse and independent property-model tests |
| P5.1-P5.7: paged attention | Block-table traversal and FP32 online softmax | Poisoned fragmentation, no-full-gather traces, T4 parity and batched workspace record |
| P6.1-P6.10: lifecycle | Credits, FIFO, active limits, bounded thread bridge, cleanup, measurements and cooperative stop | Fake-model scheduling/fault/race tests, release-once assertions and safe CUDA reuse |
| P7.1-P7.8: HTTP | Bounded preprocessing, strict validation, SSE/transport supervision and operational endpoints | CPU fake-model plus T4 black-box tests, delivery ordering and shutdown escalation |
| P8.1-P8.8: hardening | Verify existing instrumentation/shutdown; final memory/fault/soak checks | Bounded retention, privacy, worst-case T4 memory and graceful/forced-stop reports |
| P9.3-P9.7: delivery | Clean distributions, container and operator documentation | Clean wheel/sdist installs and direct/container service smoke checks |
| P10.1-P10.9: harness | Shared external client, frozen inputs and tested analysis | Synthetic replay, failure/timeout accounting and attributable pilot artifacts |
| P11.1-P11.10: measurements/report/release | Frozen comparison, profiling, report; then artifact publication | Inspectable raw results/report, followed by exact tested tag and clean artifact smoke |

Start CI/T4 procedures immediately after environment implementation. Define measurement types in P2 and instrument worker/service during P6/P7. Smoke-test candidate vLLM/T4 compatibility early; freeze its final version/configuration before official runs. Synthetic analyzer work can start in P2. Official comparisons require P3-P8 and a direct Linux/T4 service smoke, plus the validated P10 harness; final Docker/distribution delivery can finish alongside the report.

Keep each gate in a reviewable commit or small series of commits. Do not optimize an incorrect forward pass.

## Evidence attribution

For each completed gate retain a reviewed artifact under `results/validation/` identifying the gate, UTC capture time, implementation revision, source identity, lock SHA-256, actual sanitized arguments, setup record, resolved configuration, seeds, installed dependency versions, device identity and test report. Capture GPU name/UUID, selected CUDA device, compute capability, driver/runtime and relevant memory totals; distinguish CUDA-visible devices from physical inventory.

A Git revision alone does not identify dirty source. Retain sanitized differences and relevant source hashes, or rerun after establishing a clean attributable source state. Clearly separate recommended commands from commands actually executed; the current CLI's hardcoded provenance strings do not establish execution history. Never include credentials or private prompt/output data in provenance.

Test reports must state collected, selected, executed, passed, failed and skipped counts/reasons. A successful process exit with no executed required GPU tests is insufficient. Environment reference verification and CUDA RNG repeatability are separate Phase 1 checks; neither substitutes for future custom-engine parity/lifecycle suites. Label CPU results explicitly and keep failed runs rather than replacing them with only a successful summary.

## Reference and numerical policy

The P2 oracle explicitly loads the pinned HF model with `attn_implementation="eager"`, FP16 weights and evaluation/inference mode. Record the resolved backend, Torch/CUDA math settings, model configuration and tokenizer settings in reference artifacts. The existing Phase 1 verifier does not explicitly freeze that attention backend; its saved forward pass is only environment smoke evidence.

Capture all decoder-relevant values from the loaded configuration, including RMSNorm epsilon, RoPE theta/scaling, intermediate size, head mapping, biases, weight tying, BOS/EOS IDs and vocabulary. Follow the [raw-prompt/decoding policy](SPEC.md#http-contract). Compare token IDs directly across oracle, owned decoder and benchmark adapters. Use public deterministic fixtures; immediate-EOS and stopping paths may also use controlled logits, with synthetic coverage labeled separately from actual checkpoint parity.

Apply [CONSTANTS.md](CONSTANTS.md#numerical-parity-thresholds) tolerances, report maximum observed discrepancies and exact greedy equality. Diagnose deviations layer by layer. Changing a backend or tolerance requires new measured evidence, not merely updated expectations.

## Test strategy

**CPU unit tests:** Exercise the allocator and scheduler with a fake model. Test empty/full pools, exact block boundaries, strict FIFO and head-of-line behavior, the active-sequence limit, credit release, cancellation at every state, double-free prevention, requests too large for the pool, and recovery after a failed request. Assert invariants after each state transition.

Use an independent CPU ownership/accounting model for randomized operations. Cover partial failures/rollback and per-request ownership as well as aggregate counts. Test bounded mailbox publication while the event loop is stalled, coalesced wakeups/lost-wakeup races, closed-loop publication, final-token/terminal races and duplicate cleanup signals. Complete lifecycle correctness with the fake model before adding HTTP.

**GPU correctness tests:** On the pinned checkpoint, compare PagedCore with Hugging Face using the same token IDs and FP16 weights. Cover one and multiple sequences, short and long prompts, block boundaries, EOS, maximum output length, and reused physical blocks. Force a multi-block sequence onto shuffled physical block IDs and poison unrelated blocks so an implementation that ignores the block table fails. Compare intermediate outputs and logits with explicit absolute/relative tolerances; also check greedy token equality on deterministic fixtures. Investigate mismatches rather than widening tolerances until a failing case passes.

**Service tests:** Verify `POST /v1/generate`, `GET /healthz`, `GET /readyz`, and `GET /metrics`, including status codes, the default 64-request pending bound, the `262144`-byte body limit, and stable error codes. Verify SSE framing, one token event per emitted token, incremental decoding with incomplete byte sequences, final text flushing, first-step EOS, queue overload, the default 32-sequence active limit, simultaneous clients, disconnect during pending/prefill/decode, and readiness after an injected worker failure. Fill a token queue deliberately and prove the out-of-band terminal signal closes the stream. Confirm cancellation eventually returns all blocks, credits, and active slots.

Cover strict booleans/integers, unknown fields, invalid UTF-8/surrogates, whitespace preservation, special-token IDs, error envelopes and validation precedence. Bound four preprocessing operations and test bodies with missing/incorrect `Content-Length`. Cancel tokenization and verify that its slot remains held until the job exits and no cancelled request is enqueued. Test successful mailbox draining before `done`, transport-write deadlines and cancellation of an already blocked send, including failure after generation has finished.

**Memory and lifecycle tests:** Measure dense-prefill workspace in P3, pool plus batched-decode workspace in P5, and final maximum-context/32-active service memory in P8. Exercise long prefill while other requests already hold KV. Run repeated mixed requests/cancellations and assert zero owned blocks, reserved credits and active slots after graceful completion. Test CUDA ordering before page reuse and device failures that require fail-closed termination. Track process/device memory for unexplained growth.

Test normal cooperative shutdown and forced escalation separately. A stuck worker must produce a failed-graceful-shutdown outcome after the ten-second waiting budget without freeing its live resources; an external subprocess supervisor then terminates the process. Report escalation duration separately. Do not count forced termination as proof that graceful cleanup returned all ownership to zero.

Run CPU checks, `ruff`, and `mypy` on every CI push. Run GPU tests on an actual T4 before release; a CPU-only CI pass does not certify the engine.

Establish these commands now; install matching `cpu` or `cuda` extras and `dev` with the frozen lock first:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
python -m pytest -m "not gpu and not benchmark"
python -m pytest -m gpu -ra
```

The last command requires an actual T4 and an inspected test report. Add package-build, service, container and benchmark smoke commands when those surfaces exist.

## Benchmark protocol

The comparison uses the same pinned TinyLlama weights and tokenizer, FP16, one T4, 2,048-token context, greedy decoding, and fixed output length. Run PagedCore and vLLM as separate servers on the same otherwise-idle machine, one at a time. Use one benchmark client with backend adapters: PagedCore uses `/v1/generate`, and vLLM uses `/v1/completions` so no chat template is added. Both receive the same raw prompt strings, and server-side tokenization is included in client TTFT. The official vLLM benchmark documentation distinguishes TTFT, inter-token latency, and time per output token, so report these names precisely. [Source: vLLM benchmark documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/).

Pin and smoke-test the exact vLLM release on the T4. Its frozen command must explicitly set `--dtype half`, `--max-model-len 2048`, `--block-size 16`, `--gpu-memory-utilization 0.90`, and `--no-enable-prefix-caching`; record all resolved engine settings. Keep the pinned release's normal chunked-prefill, scheduler, kernel, and CUDA-graph defaults because vLLM is the production reference, then name those differences when analyzing the gap. Configure PagedCore to remain within the same 90% device-memory ceiling and publish both systems' configured KV bytes; observed peak VRAM is still reported rather than assumed equal. T4-class GPUs do not support BF16 in vLLM, so the explicit FP16 override is required for this checkpoint. [Source: vLLM CUDA platform dtype checks](https://docs.vllm.ai/en/latest/api/vllm/platforms/cuda/).

Create a checked-in workload file with deterministic prompts of approximately 128 and 512 tokenizer tokens. Use 32 and 128 generated tokens respectively. Enable `ignore_eos` in both servers and disable EOS stopping in the Hugging Face baseline so every system produces the fixed length. Verify actual prompt and output token counts; do not silently truncate. Hash the workload file and record its seed.

Before each backend's pilot, verify exact prompt IDs, special-token behavior, resolved KV dtype, context/output limits and EOS policy. Smoke-test candidate vLLM compatibility early; pin the exact release and confirm that its flags implement the frozen policy rather than assuming latest-version behavior. Exclude unsupported cells with a recorded reason instead of silently changing precision or workloads.

For each workload:

1. Pilot both servers, then freeze a common offered-request-rate grid that includes an uncongested point and at least one point near saturation. Use Poisson arrivals (`burstiness=1.0`) from saved per-repetition seeds and no client-side concurrency cap. Publish the chosen grid before inspecting comparative results.

2. Restart the server between repetitions and clear any reusable prefix cache. Warm up with 20 requests; exclude them from results.

    Freeze a balanced backend run order across repetitions, workload/order seeds, request deadline and drain policy. Record the actual order; avoid consistently running one backend first.

3. Run at least five repetitions of at least 100 measured requests per grid point, using the same precomputed arrival schedules. Increase the total to at least 1,000 requests for any cell used to make a p99 TTFT claim.

4. Let admitted requests complete; record attempted, completed, rejected, and failed requests. Never compute latency percentiles from successes while hiding rejection rates.

    Drain after the final scheduled arrival until every request reaches a terminal outcome or its frozen deadline. Timeouts are failures and remain in raw data; close the connection to signal cancellation. Retain late-result observations separately without reclassifying a timed-out attempt as a success.

5. Save raw per-request event times, token counts, device VRAM samples, configuration, and environment metadata.


Report client-measured TTFT p50/p95, pooled streamed-output ITL p50/p95, per-request TPOT p50/p95, end-to-end latency p50/p95, `output_tok_per_s`, achieved request rate, error/rejection rate, and sampled peak device VRAM, using the timing definitions below. Emit p99 fields and sample counts for every latency metric, but set a p99 to `null` with `insufficient_sample` below 1,000 relevant observations. Calculate a 95% bootstrap confidence interval across repetition-level throughput values, recording bootstrap seed/resample count. Five repetitions are a minimum and produce limited uncertainty estimates; do not treat tokens or pooled gaps as independent repetitions.

Sample total device-used memory through NVML every 50 ms, with no other GPU processes. Retain absolute samples, device memory total and the idle baseline recorded before startup; publish both absolute sampled peak and baseline-subtracted peak. Include startup/warmup/measurement phase labels, actual sample timestamps and gaps. These are sampled peaks and can miss shorter transients; use separate allocator/profiling diagnostics for workspace safety. Device-level sampling includes vLLM child processes. Internal KV metrics are diagnostic; use `N/A` where another system lacks an equivalent. For PagedCore, report peak occupancy, reservation ratio, slot fill and effective pool fill.

Run Hugging Face `generate` as a **single-request reference baseline** for latency and output throughput. Label its concurrent-QPS cells `N/A`; do not imply it is a configured online serving system.

Record the HF baseline's resolved attention backend and math settings separately from the correctness oracle. If using a different backend for latency, freeze and name that difference; do not attribute it to PagedCore paging.

### Client timing and accounting

All primary comparisons use the same external client and one monotonic clock. Record scheduled arrival, actual dispatch, first/last observed output, observed output-event gaps, terminal receipt or failure/timeout, verified token counts and statuses. Report dispatch lag; a client that cannot sustain the schedule cannot certify the offered-rate cell. Do not cap client concurrency, but report resource exhaustion as a failed attempt rather than dropping it.

For comparison, an output event is a nonempty generated text delta for both adapters. Metadata-only or empty deltas do not establish client TTFT; PagedCore's internal first-token timestamp remains a distinct diagnostic. Use `TTFT = first_output - dispatch`, `TPOT = (last_output - first_output) / (output_tokens - 1)` when at least two output tokens and observable output exist, and `end_to_end = terminal - dispatch`. Also retain `last_output - dispatch` and terminal overhead so final flushing/EOS/transport costs are visible. Undefined values are `null` with a reason, never zero.

Client ITL pools gaps between consecutive observed output events. Stream events, model tokens and network reads are distinct: one output event may carry multiple tokens and one network read may contain several events. Never manufacture per-token timestamps by retokenizing text or assigning zero gaps within a chunk. Verify counts from backend completion metadata and PagedCore event indices while retaining only counts/times, not token IDs or generated text. Report stream granularity and missing usage/count metadata; invalidate a count-dependent cell when counts cannot be established. Internal model-token ITL must be labeled separately.

The throughput interval starts at the first scheduled measured arrival and ends at the last terminal/deadline, including dispatch lag and drain. Publish interval bounds, offered-window duration and drain duration. `output_tok_per_s = completed_requests_emitted_tokens / interval_seconds`; `achieved_req_per_s = completed_requests / interval_seconds`. Report partial tokens from failed/cancelled attempts separately. Require `attempted = completed + rejected + failed`, with timeouts/cancellations/client failures included in failed and explicit subcounts. Successful latency summaries always appear beside all outcome counts.

### Mechanism experiments

The default 8,192 MiB pool exceeds the maximum KV reservation allowed by 32 active/context-bounded requests: 1,408 MiB. The fixed workloads use about 110/440 MiB at that active limit. These derived capacities cannot demonstrate credit pressure under default settings.

Alongside the official frozen comparison, run separately labeled PagedCore experiments with smaller pools, mixed prompt/output lengths, fragmentation/reuse, and long-prefill interference. Select settings that actually trigger credit limits and FIFO blocking; retain admission/queue, occupancy/reservation and cleanup measurements. Use paired public inputs/seeds and change one factor at a time for causal claims. Never replace the official cells with a favorable sensitivity setting. These experiments exercise existing MVP behavior and add no scheduling features.

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

Publish installable v0.1.0 artifacts only after model parity, allocator invariants, service lifecycle tests, lint/type checks, clean installation, Docker smoke test, and the T4 benchmark pass. The README must link architecture, exact reproduction commands, raw measurements, results and limitations. The technical report may precede artifact publication under the milestones below.

After that release, consider preemption/recomputation, chunked prefill, copy-on-write and prefix sharing, then speculative decoding. Each is a separate change with its own correctness test and before/after result.

## Completion milestones

1. **Diagnostic pilot:** label exploratory measurements and failures; never present them as an official comparison or as closing an unproven correctness gate.
2. **Official measurement entry:** P3-P8 correctness, lifecycle/service, privacy and memory gates pass on attributable source; CPU CI/T4 reports exist; direct Linux/T4 service smoke and P10 synthetic/pilot checks pass. Freeze inputs, versions, grid, run order, deadlines and analysis definitions before comparative runs.
3. **Portfolio demonstration:** publish attributable correctness evidence, official raw/summarized measurements, mechanism experiments and a profile-backed technical report explaining a gap or failure. No speed target or additional serving feature is required.
4. **Artifact release:** finish distribution/container checks, final documentation and the release checklist; tag the exact tested source and publish installable artifacts/checksums. Run the post-release smoke separately. Release completion is distinct from the earlier technical demonstration.
