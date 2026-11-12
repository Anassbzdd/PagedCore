
Purpose: Turn the design into an ordered build, test, benchmark, and publication plan.

# PagedCore Implementation and Validation Plan

This is the public evidence contract and phase map. [STATUS.md](STATUS.md) records current implementation/evidence limits. Planned tests, commands, and artifact paths below are requirements, not assertions that those surfaces exist.

## Current numerical acceptance contract

The owner-approved November 11 [numerical policy](NUMERICAL_POLICY.md) removes
required HF parity/token agreement/fidelity. It keeps independent FP64 attention,
component checks and the unchanged blocking contiguous FP32 integrated comparison.
The November 10 T4 logit/KV failures remain unresolved; P5.3 stays open.
Quality against the same checkpoint in vLLM is a separate planned held-out gate,
not a replacement numerical oracle. No new T4 validation is claimed.

## Build order

Complete each runtime gate before starting the next substantial runtime feature. Preserve independent dense checks -> allocator correctness -> paged attention -> lifecycle correctness -> HTTP. CI, evidence tooling, and synthetic benchmark analysis can support these gates earlier; they do not bypass them. The task IDs below define public ownership even when the detailed local checklist is not distributed.

| Phase / task IDs | Work | Evidence required |
|---|---|---|
| P0.1-P0.6: baseline | Package structure and reconciled public contracts | Parseable metadata and consistent constants/acceptance ownership |
| P1.1-P1.6: environment | Frozen CPU/CUDA setup, configuration, diagnostics, controls, owned checkpoint verifier and evidence closure | Attributable fresh Linux/T4 install, owned forward, CPU checks and CUDA repeatability report |
| P9.1-P9.2: early automation | Schedule after environment implementation, before P2; retain IDs for traceability | CPU lint/format/strict typing/pytest automation and repeatable T4 job with test counts and artifacts |
| P2.1-P2.5: types/fixtures | Minimal interfaces, lifecycle/accounting validators, timing schema, fixtures and independent component checks | CPU rejection of illegal transitions and invalid capacity snapshots; deterministic reference IDs/tensors; documented special-token, attention, and numerical settings |
| P3.1-P3.7: dense model | Owned TinyLlama forward without paging | Independent component coverage (gaps open), historical dense evidence and dense-prefill workspace |
| P4.1-P4.6: allocator | One pool, free list, tables and capacity ledger | Boundary, rollback, isolation, reuse and independent property-model tests |
| P5.1-P5.7: paged attention | Block-table traversal, standalone and integrated FP32 online-softmax decode | Poisoned fragmentation, no-full-gather traces, T4 integrated accuracy and batched workspace record |
| P6.1-P6.10: lifecycle | Credits, FIFO, active limits, bounded thread bridge, cleanup, measurements and cooperative stop | Fake-model scheduling/fault/race tests, release-once assertions and safe CUDA reuse |
| P7.1-P7.8: HTTP | Bounded preprocessing, strict validation, SSE/transport supervision and operational endpoints | CPU fake-model plus T4 black-box tests, delivery ordering and shutdown escalation |
| P8.1-P8.8: hardening | Verify existing instrumentation/shutdown; final memory/fault/soak checks | Bounded retention, privacy, worst-case T4 memory and graceful/forced-stop reports |
| P9.3-P9.7: delivery | Clean distributions, container and operator documentation | Clean wheel/sdist installs and direct/container service smoke checks |
| P10.1-P10.9: harness | Shared external client, frozen inputs and tested analysis | Synthetic replay, failure/timeout accounting and attributable pilot artifacts |
| P11.1-P11.10: measurements/report/release | Frozen comparison, profiling, report; then artifact publication | Inspectable raw results/report, followed by exact tested tag and clean artifact smoke |

Start CI/T4 procedures immediately after environment implementation. Define measurement types in P2 and instrument worker/service during P6/P7. Smoke-test candidate vLLM/T4 compatibility early; freeze its final version/configuration before official runs. Synthetic analyzer work can start in P2. Official comparisons require P3-P8 and a direct Linux/T4 service smoke, plus the validated P10 harness; final Docker/distribution delivery can finish alongside the report.

Keep each gate in a reviewable commit or small series of commits. Do not optimize an incorrect forward pass.

Quality/distribution evaluation is planned release work under AC-01 with separate
pilot/frozen held-out evidence in [the policy](NUMERICAL_POLICY.md). The paging
collector does not implement it. Historical phase completions remain attributed
to their original source/contracts; new independent coverage gaps remain open.

## Allocator gate status

P4 remains complete at its recorded allocator boundary. P5.1 prompt writes and
P5.2 standalone attention retain [October 30](validation/2026-10-30-p52-t4-review.md)
and [October 31](validation/2026-10-31-p52-t4-review.md) T4 evidence. P5.3 integrated
accuracy fails on T4; P5.4 integrated guard evidence, P5.6 paged generation and
P5.7 workspace/safe reuse remain open. See [STATUS.md](STATUS.md).
The [superseded plan snapshot](validation/2026-11-11-superseded-validation-plan.md)
preserves previous diagnostic instructions and conclusions. They are optional
forensic reproduction, not additional active release requirements.

## Evidence attribution

For each completed gate retain a reviewed artifact under `results/validation/` identifying the gate, UTC capture time, implementation revision, source identity, lock SHA-256, actual sanitized arguments, setup record, resolved configuration, seeds, installed dependency versions, device identity and test report. Capture GPU name/UUID, selected CUDA device, compute capability, driver/runtime and relevant memory totals; distinguish CUDA-visible devices from physical inventory.

A Git revision alone does not identify dirty source. Retain sanitized differences and relevant source hashes, or rerun after establishing a clean attributable source state. Clearly separate recommended commands from commands actually executed. Manifest version 2 records runtime/test source and install-metadata hashes and actual CLI arguments with manifest paths redacted. Programmatic calls leave command arguments unknown; setup history must be retained separately, because the verifier cannot observe the installer or an outer `uv` invocation. Missing hashes/Git metadata are unknown, not proof of a clean checkout. Never include credentials or private prompt/output data in provenance.

Test reports must state collected, selected, executed, passed, failed and skipped counts/reasons. A successful process exit with no executed required GPU tests is insufficient. Owned environment verification and CUDA RNG repeatability are separate Phase 1 checks; neither substitutes for future custom-engine parity/lifecycle suites. Label CPU results explicitly and keep failed runs rather than replacing them with only a successful summary.

Required ten GPU cases: CUDA RNG repeatability, owned paged prefill, two FP64
attention cases, two KV storage/allocator cases and four integrated decode cases
(`single`, `mixed`, `fragmented`, `held_out`). Prefill now compares with owned dense
prefill; its historical HF run remains attributed to its original revision.
Integrated cases use 1/15/16/17/49 and 37/73/111/145 prefixes, three cached steps,
row reversal, logit `5e-3` and K/V `2e-3` absolute/relative tolerances and exact owned
argmax. Fragmentation injects a staged logit failure, rolls back and retries.
Check unchanged history, poisoned unrelated pages, reservations, stable storage
and zero cleanup. Only serving decode runs under the no-full-history guard.

Required policy is `fp32-paged-v2-independent`; missing/duplicate/non-finite or
failing owned comparisons reject export. HF fields and tolerance-envelope files
are not required. Old v1 evidence stays unchanged. The notebook keeps failures and
raw logits even when export rejects the run. The shell collector additionally
requires no-pool/pool-pressure workspace, without certifying maximum-active
paged/service memory. See [export regressions](../tests/unit/test_t4_evidence.py).

## Reference and numerical policy

[The policy](NUMERICAL_POLICY.md) defines independent model coverage and its
common-mode gaps, the blocking FP32 comparison, and planned quality/fidelity
protocols: separate pilot data, justified frozen margins/counts, held-out cached
decode loss/perplexity, fixed tasks and uncertainty. Full KL needs complete aligned
probabilities on identical histories; top-k KL is unavailable. No quality evaluator
or vLLM scoring adapter is implemented yet. Historical HF comparisons are optional.

### Greedy parity

Exact greedy choices remain required against the owned reference on the frozen
paging suite. P5.6 separately covers real paged generation, EOS, maximum output and
reuse. Universal token identity with HF/vLLM is not a requirement.

### Tolerance envelope

Historical dense HF envelopes remain in original archives, including the
[October 20 review](../results/validation/2026-10-20-t4/review.json). Current
collection does not require/export them. Integrated thresholds stay unchanged.

### Dense workspace

P3.7 runs [the workspace recorder](../scripts/measure_dense_workspace.py) in a
separate process through `bash scripts/validate_t4.sh`, before integrated GPU accuracy tests.
Only the owned pinned FP16 weights are loaded; no HF model or paged pool is present.
Use an otherwise idle T4 with exactly one CUDA-visible device and the native CUDA
caching allocator. The recorder uses the same seed and strict math policy as parity.

Measure these single-sequence paths using fixed synthetic IDs:

| Path | Prompt tokens | Requested output | Decode forwards |
|---|---:|---:|---:|
| Maximum direct prefill | 2,048 | 0 | 0 |
| Maximum generation prompt | 2,047 | 1 | 0 |
| Maximum output at the context limit | 1,792 | 256 | 255 |

Each path runs three times: once after releasing unused allocator segments, then
twice reusing the allocator cache. Generation ignores EOS so it reaches the output
limit. Reset allocator peaks immediately before each path, including input creation.
Synchronize raw samples before/after the forward and after returned tensors are
released; retain [first-use library workspace growth](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cublas-workspaces)
and require stable live
allocations on repeats. Allocation retries or OOMs fail the run.

Retain `dense-workspace.json` with device-free/total bytes, live/reserved bytes,
allocator peaks and full allocator statistics at each boundary, plus model-loading
baselines, all nine runs, math settings, source/lock/recorder hashes and environment
identity. `dense-workspace-allocator.txt` retains allocator summaries. Failures leave
partial samples and diagnostics in the raw directory; the collector rejects missing
or incomplete workspace evidence before producing a successful ZIP.

The peak increment above a run's baseline includes input, KV and logits as well as
temporary tensors; it is not an isolated scratch-buffer size. Boundary device-free
samples can miss transients, and allocator peaks exclude allocations outside
PyTorch's allocator. These measurements do not certify a pool size from nominal
VRAM or close the later paged/batched/service memory gates. Review actual T4 results
before revising capacity or the provisional workspace margin. CPU tests validate
recording and evidence rejection only.

The retained [October 21 T4 archive](../results/validation/2026-10-21-t4/pagedcore-p37-t4.zip)
and [baseline review](../results/validation/2026-10-21-t4/review.json) close P3.7
at source `e9e21fb`. All nine runs and 27 samples passed, with no allocator retries
or OOMs. Values below are recomputed from raw bytes (`MiB = bytes / 2**20`):

| Path | Peak allocated MiB | Largest increment above its run baseline MiB |
|---|---:|---:|
| Maximum direct prefill | 3,228.22 | 1,128.05 |
| Maximum generation prompt | 3,228.19 | 1,096.01 |
| Maximum output | 2,979.72 | 847.54 |

Retain this no-pool record unchanged as the dense baseline. The 1,128.05 MiB
increment exceeds the former provisional 1,024 MiB margin, but includes dense KV,
inputs, logits and temporary tensors. It neither isolates workspace demand nor
certifies the configured 8,192 MiB pool or 32-active workload.

### Combined memory validation before capacity freeze

The collector now also runs the recorder with `--with-pool` in a fresh process.
After loading weights, this mode checks actual free device bytes against
`kv_pool_mib + workspace_margin_mib`, allocates and zeroes one unused FP16 tensor
of exactly the configured pool bytes, then holds it through the same nine dense
runs. It checks headroom again after allocation, rejects allocator retries/OOMs,
and retains failures with partial samples. This is an early memory-pressure probe;
the tensor does not implement the paged pool, ownership or block-table attention.
The existing dense cache remains present, so its increment still includes dense KV.

The separate `pool-dense-workspace.json` and `pool-dense-workspace-allocator.txt`
record reservation bytes, model/pool baselines, configured margin and active limit,
all raw samples/peaks and the observed margin comparison. A passed measurement
means the recorded paths ran successfully; `capacity_validated` remains `false`,
including when the dense increment fits the margin. The collector requires both
complete reports and verifies reservation/configuration agreement before packaging.
The [October 22 T4 archive](../results/validation/2026-10-22-pool-memory-cpu/pagedcore-pool-memory-t4.zip)
and [review](../results/validation/2026-10-22-pool-memory-cpu/t4-review.json)
retain this probe at clean source `6c6eec8`: all nine pressure runs passed,
with zero retries/OOMs, 464 passing CPU tests and ten passing GPU tests without
skips. The exact 8,192 MiB reservation stayed live through all 27 run boundaries.
Peak allocated memory reached 11,420.22 MiB and allocator reserved memory
11,626 MiB; the minimum boundary free sample was 3,128.81 MiB.
The dense increment remained 1,128.05 MiB. Repeated cleanup allocations stabilized
after one first-use 32 MiB increase. This is single-sequence reservation evidence.
That run used the former 1,024 MiB margin. The [October 25 T4 archive](../results/validation/2026-10-25/pagedcore-p42-t4-20261025T153708Z.zip)
and [October 26 review](validation/2026-10-26-p42-t4-review.md) at clean source
`f72f39c` now supply the same nine pressure runs and 27 samples with the revised
1,536 MiB margin. All passed with zero retries/OOMs, the same 1,128.05 MiB maximum
dense increment and stable repeated cleanup after a one-time 32 MiB increase.
Peak allocated/reserved memory was 11,420.22/11,626 MiB; minimum sampled device
free memory was 3,128.81 MiB. The archive also retains 498 passing CPU tests and
12 passing GPU tests without skips. This validates the revised setting only for
the recorded single-sequence reservation probe; `capacity_validated=false`
remains correct. Integrated model/pool and paged/batched/service capacity gates
remain pending.

Run the full attributable procedure on T4 from a clean committed checkout:

```bash
bash scripts/validate_t4.sh
```

For a focused pressure probe in an already configured CUDA environment, choose a
new output directory; it does not replace the collector's complete evidence bundle:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --no-sync python scripts/measure_dense_workspace.py results/local/pool-pressure --with-pool
```

Capacity remains provisional until all of these have reviewed T4 evidence:

1. The pool-reservation probe above, alongside the retained no-pool baseline.
2. **P5.7:** the implemented paged pool with maximum active batched decode, and
   long prefill while other sequences retain KV; include page-write/copy overlap
   and ordered page reuse. Measure actual paged code, not an unused reservation.
3. **P8.6/P8.7:** the actual maximum-context/32-active service workload and mixed
   cancellation/fault/soak runs, with allocator/device memory and cleanup checks.

If those runs expose OOMs, retries or unsafe headroom, raise the margin or lower
the pool/active limit and repeat the affected tests before freezing capacity.
Changing the pool, margin or active limit invalidates the relevant capacity result.
These later gates remain pending until integrated paged/batched and service
evidence is retained.

### Oracle harness

One next focused T4 entry point:
[paged_decode_accuracy.ipynb](../notebooks/paged_decode_accuracy.ipynb), fresh Kaggle
session, Internet/T4 enabled, Run All, download the final ZIP even on failure.
It embeds exact working source, runs owned checkpoint/tokenizer smoke and CPU/GPU
checks, and retains hashes, environment, commands, raw logits and failures. Passing
establishes only tested mathematics/paging, not P5.6/P5.7, model-coverage gaps,
held-out quality or serving/performance.

`bash scripts/validate_t4.sh` remains the broader clean-checkout collector including
dense workspace. Its checkpoint smoke uses `scripts.verify_owned_checkpoint`,
without an HF model forward. It leaves the raw directory on failure.
Optional commands remain `pagedcore verify`, `pagedcore oracle` and
`python -m pytest --run-hf-diagnostics -m hf_diagnostic`. They are not release
requirements or replacements for owned checks. Diagnostic scripts/notebooks stay
for reproducing retained reports; their old embedded source is historical.

Required CUDA runs set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python, disable
TF32/reduced-precision reductions and use deterministic algorithms with strict
math flags retained in reports. These controls do not promise cross-backend
bitwise equality. CPU checks cannot certify CUDA behavior.

### Public reference fixtures

[Public inputs](../tests/fixtures/reference_prompts.json) retain raw prompts, exact
pinned IDs, special tokens, boundaries and maximum-output cases. Tokenizer checks
verify hashes/settings and explicitly skip when the pinned files are uncached
offline. Controlled logits independently test EOS, ignored EOS, output limits and
consumed cache positions. HF stopping runs only with explicit diagnostics enabled.

### Numerical comparisons

Follow [the policy](NUMERICAL_POLICY.md), preserving thresholds and failures.
A comparator redesign needs measured causes and independent replacement evidence.

## Test strategy

**Shared-contract CPU tests (P2.2):** Cover successful lifecycle paths with and
without decode, cancellation/failure before and after admission, and rejection of
skipped stages, backwards changes, self-transitions and every change from a terminal
state. Reject nonpositive reservation credits, negative/noninteger capacity counts,
and violations of `free + owned = total` or `owned <= reserved <= total`. Accept
snapshots with reserved but unassigned capacity. These validators do not prove
release-once cleanup, per-request ownership, or CUDA ordering; implement and test
those assertions with the concrete allocator/worker in P4/P6.

**CPU unit tests:** Exercise the allocator and scheduler with a fake model. Test empty/full pools, exact block boundaries, strict FIFO and head-of-line behavior, the active-sequence limit, credit release, cancellation at every state, double-free prevention, requests too large for the pool, and recovery after a failed request. Assert invariants after each state transition.

Allocator failure-safety regressions reject impossible reservations without
acquisition and corrupt/free/foreign/swapped tables before reads, writes or
release. Inject synchronous copy failures into K and V writes across partial and
new blocks; verify unchanged valid history, other requests and accounting, then
strict release-once cleanup and complete reuse. The T4 allocator test also covers
a synchronous exception after issuing K but before V, followed by retry/reuse;
this is not evidence of recovery from an actual CUDA device failure.

Use an independent CPU ownership/accounting model for randomized operations. Cover partial failures/rollback and per-request ownership as well as aggregate counts. Test bounded mailbox publication while the event loop is stalled, coalesced wakeups/lost-wakeup races, closed-loop publication, final-token/terminal races and duplicate cleanup signals. Complete lifecycle correctness with the fake model before adding HTTP.

**GPU correctness tests:** Use independent FP64 attention on identical operands,
owned contiguous FP32 integration at unchanged thresholds, poisoned/shuffled pages,
rollback/retry, exact old slots, new writes, stable storage, zero ownership and the
no-full-history guard. Exact owned-reference greedy choices remain required on the
finite paging suite. The integrated T4 failure remains blocking. Independent model
coverage and cached-decode quality must also close before release. HF comparisons
are optional and deselected by default.

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -m gpu -k "not legacy and not paired_at_17 and not test_pinned_candidate_paged_decode_on_t4" -ra
```

The last command requires an actual T4 and an inspected test report. The six
registered markers (`unit`, `integration`, `gpu`, `service`, `slow`, `benchmark`)
and their selection commands are documented in
[Contributing](../CONTRIBUTING.md#selecting-test-suites). Marks can overlap; every
hardware-dependent test must carry `gpu` regardless of its other marks. Routine
CPU CI excludes `gpu` and `benchmark` and includes CPU integration, service and
slow checks. Empty or all-skipped selections do not satisfy a validation gate.
Add package-build, service, container and benchmark smoke commands when those surfaces exist.

## Benchmark protocol

The comparison uses the same pinned TinyLlama weights and tokenizer, FP16, one T4, 2,048-token context, greedy decoding, and fixed output length. Run PagedCore and vLLM as separate servers on the same otherwise-idle machine, one at a time. Use one benchmark client with backend adapters: PagedCore uses `/v1/generate`, and vLLM uses `/v1/completions` so no chat template is added. Both receive the same raw prompt strings, and server-side tokenization is included in client TTFT. The official vLLM benchmark documentation distinguishes TTFT, inter-token latency, and time per output token, so report these names precisely. [Source: vLLM benchmark documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/).

Pin and smoke-test the exact vLLM release on the T4. Its frozen command must explicitly set `--dtype half`, `--max-model-len 2048`, `--block-size 16`, `--gpu-memory-utilization 0.90`, and `--no-enable-prefix-caching`; record all resolved engine settings. Keep the pinned release's normal chunked-prefill, scheduler, kernel, and CUDA-graph defaults because vLLM is the production reference, then name those differences when analyzing the gap. Configure PagedCore to remain within the same 90% device-memory ceiling and publish both systems' configured KV bytes; observed peak VRAM is still reported rather than assumed equal. T4-class GPUs do not support BF16 in vLLM, so the explicit FP16 override is required for this checkpoint. [Source: vLLM CUDA platform dtype checks](https://docs.vllm.ai/en/latest/api/vllm/platforms/cuda/).

Create a checked-in workload file with deterministic prompts of approximately 128 and 512 tokenizer tokens. Use 32 and 128 generated tokens respectively. Enable `ignore_eos` in both servers and, if an optional Hugging Face baseline is reported, disable its EOS stopping so every system produces the fixed length. Verify actual prompt and output token counts; do not silently truncate. Hash the workload file and record its seed.

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

Optionally run Hugging Face `generate` as a **single-request diagnostic baseline** for latency and output throughput. Label its concurrent-QPS cells `N/A`; do not imply it is a configured online serving system.

Record the HF baseline's resolved attention backend and math settings as optional diagnostic metadata. If using a different backend for latency, freeze and name that difference; do not attribute it to PagedCore paging.

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

Publish installable v0.1.0 artifacts only after independent model correctness, paging accuracy, model quality, allocator invariants, service lifecycle tests, lint/type checks, clean installation, Docker smoke test, and the T4 benchmark pass. The README must link architecture, exact reproduction commands, raw measurements, results and limitations. The technical report may precede artifact publication under the milestones below.

After that release, consider preemption/recomputation, chunked prefill, copy-on-write and prefix sharing, then speculative decoding. Each is a separate change with its own correctness test and before/after result.

## Completion milestones

1. **Diagnostic pilot:** label exploratory measurements and failures; never present them as an official comparison or as closing an unproven correctness gate.
2. **Official measurement entry:** P3-P8 correctness, lifecycle/service, privacy and memory gates pass on attributable source; CPU CI/T4 reports exist; direct Linux/T4 service smoke and P10 synthetic/pilot checks pass. Freeze inputs, versions, grid, run order, deadlines and analysis definitions before comparative runs.
3. **Portfolio demonstration:** publish attributable correctness evidence, official raw/summarized measurements, mechanism experiments and a profile-backed technical report explaining a gap or failure. No speed target or additional serving feature is required.
4. **Artifact release:** finish distribution/container checks, final documentation and the release checklist; tag the exact tested source and publish installable artifacts/checksums. Run the post-release smoke separately. Release completion is distinct from the earlier technical demonstration.
