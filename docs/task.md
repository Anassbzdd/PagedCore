# PagedCore delivery plan

Purpose: turn the existing specification, design, and validation plan into an ordered execution checklist from an empty repository to a tested, reproducible release.

## 1. Scope and definition of done

PagedCore v0.1.0 is a learning and portfolio inference engine for one pinned TinyLlama checkpoint on one NVIDIA T4. It owns its Llama forward path, paged KV cache, block-table decode attention, continuous-batching scheduler, and FastAPI/SSE service.

For this plan, **production-ready** means reliable, testable, observable, reproducible, and safely bounded for the documented local single-user deployment. It does **not** mean public-internet hardening, multi-tenant isolation, vLLM API compatibility, or feature parity with a general-purpose serving engine.

The release is complete only when all of the following are true:

- The pinned model produces matching greedy tokens against Hugging Face within documented FP16 tensor/logit tolerances.
- Decode attention demonstrably reads noncontiguous KV pages through each sequence's block table without gathering the full history.
- Admission, cancellation, slow-consumer handling, failures, and shutdown release every block, credit, active slot, and worker resource exactly once.
- The local API enforces all documented bounds and returns stable status/error codes.
- CPU checks pass in CI; the complete GPU suite passes on a real T4.
- A direct Linux/CUDA install and the Docker image both pass clean-start smoke tests.
- One command runs a reproducible PagedCore/vLLM benchmark and saves raw data, metadata, and an honest summary.
- The README and technical report explain architecture, reproduction, measured results, limitations, and at least one investigated performance gap.
- There are no unresolved release-blocking defects or unqualified performance claims.

## 2. Current state

The repository currently contains documentation only:

- `docs/SPEC.md` defines the fixed MVP behavior and public contract.
- `docs/DESIGN.md` defines the architecture and lifecycle invariants.
- `docs/VALIDATION_PLAN.md` defines the validation and benchmark protocol.
- There is no Git repository, Python package, test suite, CI workflow, container, benchmark harness, or runtime implementation yet.

These three documents are the source of truth. If implementation reveals a conflict, change the relevant document and code in the same pull request. Do not silently weaken an invariant or present a documented feature as implemented evidence.

## 3. Fixed MVP boundaries

Do not expand these while completing v0.1.0:

| Area | v0.1.0 decision |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, exact model and tokenizer revision pinned together |
| Runtime | Linux, CUDA, one NVIDIA T4, one process, one model worker |
| Precision | FP16 weights and FP16 KV cache; FP32 online-softmax accumulation |
| Context | 2,048 tokens total; output 1-256 tokens, default 128 |
| KV pages | 16 token slots per physical block |
| Generation | Greedy, one completion, optional EOS ignore for fixed-length benchmarks |
| Scheduling | Strict FIFO, maximum 64 pending and 32 active by default, one serial prefill per iteration |
| Capacity | Reserve full possible KV demand at admission; no preemption |
| Streaming | FastAPI POST plus SSE, bounded 32-event queue per request, out-of-band terminal future |
| Deployment | Local service, `localhost` default, direct install and Docker |

Out of scope: sampling controls, quantization, multi-model or multi-GPU serving, public authentication, multi-tenant isolation, OpenAI/vLLM API compatibility, preemption, recomputation, CPU KV offload, copy-on-write, prefix sharing, chunked prefill, speculative decoding, and a web UI.

## 4. Execution rules

- Complete phases in order. A later phase may be prototyped, but it may not be declared complete before its dependencies and exit gate pass.
- Use small reviewable commits. Prefer one gate or one coherent invariant per commit.
- Every bug fix adds a regression test first or in the same change.
- Keep the reference Hugging Face path out of the serving runtime; it is an oracle for tests and a benchmark baseline only.
- Optimize only after correctness and lifecycle ownership are proven.
- Never fix parity by widening tolerances without locating and documenting the numerical source.
- Record exact commands, resolved configuration, seeds, revisions, and environment metadata for every release result.

Task status markers:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete with linked evidence
- `[!]` blocked; record the blocker and owner directly below the task

## 5. Ordered implementation plan

### Phase 0 - Repository and contract baseline

**Depends on:** nothing

- [x] **P0.1 Initialize the repository.** Create Git history, `.gitignore`, a concise `README.md`, `LICENSE`, and contribution instructions. Ignore local caches, model files, benchmark outputs that are not intended as evidence, virtual environments, and Obsidian state.
- [x] **P0.2 Choose the package layout.** Use a `src/pagedcore/` package, `tests/` split by unit/integration/GPU/service, `benchmarks/`, `scripts/`, and `artifacts/` or `results/` with a documented checked-in policy.
- [ ] **P0.3 Add project metadata.** Create `pyproject.toml` with supported Python version, runtime dependencies, development extras, console entry points, Ruff, mypy strict mode, pytest markers, and package metadata.
- [ ] **P0.4 Resolve and record open constants.** Pin the model/tokenizer revision, Python/PyTorch/Transformers versions, numeric parity tolerances, default `kv_pool_mib`, initial 1 GiB workspace margin, maximum request-body size, shutdown timeout, and stable error-code names.
- [ ] **P0.5 Reconcile documentation defects.** Check endpoint names, formulas, defaults, and terminology across the three source documents before code copies them. Document deliberate changes in a short decision log.
- [ ] **P0.6 Add a traceability table.** Map every MVP acceptance criterion to its implementation owner, test, and final evidence file. Start with `pending` links and fill them as phases complete.

**Exit gate:** a clean checkout has an obvious structure, install metadata parses, the source documents agree on constants and interfaces, and every acceptance criterion has a planned test/evidence location.

### Phase 1 - Reproducible development environment

**Depends on:** Phase 0

- [ ] **P1.1 Freeze dependencies.** Commit reproducible direct-install and development dependency locks/constraints for Linux/CUDA. Keep CPU-only developer installation possible for allocator/scheduler tests.
- [ ] **P1.2 Add configuration loading.** Implement a typed immutable configuration object with validated defaults for model revision, context/output limits, block size, pool size, queue sizes, active slots, bind address, and logging.
- [ ] **P1.3 Add environment diagnostics.** Provide one command that prints Python, package, CUDA, driver, GPU, model revision, and resolved PagedCore settings without secrets or prompt text.
- [ ] **P1.4 Add deterministic controls.** Seed Python and PyTorch where applicable, enable inference mode, document deterministic limitations on CUDA, and make test fixtures reproducible.
- [ ] **P1.5 Verify the target machine.** On a T4, load pinned weights/tokenizer in FP16, run one reference forward pass, and capture the environment manifest.

**Exit gate:** a fresh environment installs from committed metadata and the pinned checkpoint loads on the T4 with the expected architecture, dtype, tokenizer revision, and 2,048-token limit.

### Phase 2 - Core types and reference fixtures

**Depends on:** Phase 1

- [ ] **P2.1 Define engine types.** Add explicit request IDs, lifecycle states, finish/cancellation reasons, token events, terminal outcomes, timestamps, block IDs, reservations, and structured engine errors.
- [ ] **P2.2 Encode invariants near ownership.** Centralize assertions for block ownership, reserved credits, active slots, valid page-table positions, legal state transitions, and release-once behavior.
- [ ] **P2.3 Create deterministic fixtures.** Check in small raw prompt fixtures covering short input, 15/16/17-token boundaries, multiple blocks, first-step EOS if reproducible, maximum output, and mixed request lengths. Store text only when it is explicitly a public test fixture.
- [ ] **P2.4 Build the oracle harness.** Capture selected Hugging Face intermediate tensors, logits, and greedy token IDs without coupling production modules to Transformers cache behavior.
- [ ] **P2.5 Establish test commands.** Define `unit`, `integration`, `gpu`, `service`, `slow`, and `benchmark` markers so CPU CI never implies GPU certification.

**Exit gate:** types and fixtures are stable, illegal lifecycle transitions fail clearly, and one command can generate/compare reference evidence for a deterministic prompt.

### Phase 3 - Llama decoder parity without paging

**Depends on:** Phase 2

- [ ] **P3.1 Load and map weights.** Validate all expected TinyLlama tensor names, shapes, dtypes, tied weights, and configuration values; fail clearly on a mismatched revision.
- [ ] **P3.2 Implement dense components.** Add RMSNorm, Q/K/V projections, grouped-query head mapping, RoPE at absolute positions, output projection, SwiGLU MLP, final norm, and LM head.
- [ ] **P3.3 Implement contiguous prefill and decode.** Use a simple correctness-first cache only for this phase; keep the interfaces replaceable by paged storage.
- [ ] **P3.4 Compare layer by layer.** Test embeddings, norms, attention outputs, MLP outputs, final hidden state, and logits with explicit absolute/relative tolerances.
- [ ] **P3.5 Prove greedy parity.** Compare exact generated token IDs for all deterministic fixtures, including EOS and length stopping behavior.
- [ ] **P3.6 Measure the tolerance envelope.** Record the largest observed tensor and logit discrepancies and the exact hardware/software environment.

**Exit gate:** automated T4 tests pass the documented tensor/logit tolerances and exact greedy token checks. No unexplained mismatch is hidden by relaxed tolerances.

### Phase 4 - GPU paged KV cache

**Depends on:** Phase 3

- [ ] **P4.1 Allocate the pool once.** Create the FP16 device tensor with conceptual shape `[layers, 2, blocks, 16, kv_heads, head_dim]`; do not allocate a CUDA tensor per token.
- [ ] **P4.2 Implement the allocator.** Add free-list allocation, logical-to-physical block tables, partial-block tracking, append, read-one-block, and release-all operations under single-worker ownership.
- [ ] **P4.3 Implement capacity accounting.** Track total, free, occupied, reserved blocks, live cached token slots, and per-request ownership separately.
- [ ] **P4.4 Enforce failure safety.** Reject impossible reservations, make release idempotence explicit, detect double-free/foreign-read/corrupt-table errors, and preserve allocator consistency after a request failure.
- [ ] **P4.5 Unit-test all boundaries.** Cover empty/full pools, token positions 0/15/16/17, exact multiples, partial blocks, fragmented allocation, block reuse, exhaustion, cancellation, and cross-request isolation.
- [ ] **P4.6 Add invariant/property tests.** Run randomized allocate/append/release sequences against a small CPU model and assert ownership/accounting after every operation.

**Exit gate:** allocator tests prove every block has exactly one valid owner or is free, capacity counters reconcile, and all state returns to zero after mixed completions, failures, and cancellations.

### Phase 5 - Block-table decode attention

**Depends on:** Phase 4

- [ ] **P5.1 Write paged prefill output.** Allow temporary contiguous tensors for one prompt, then place rotated K/V into its assigned physical pages with correct absolute positions.
- [ ] **P5.2 Implement blockwise online softmax.** Traverse valid physical blocks in logical order, accumulate numerically stable softmax state in FP32, and return FP16 attention output.
- [ ] **P5.3 Batch one-token decode.** Project the last emitted token for each active sequence, append its K/V, and apply per-sequence causal lengths and grouped-query mapping.
- [ ] **P5.4 Prohibit dense-history shortcuts.** Add an assertion/instrumented test that fails if decode materializes each sequence's full cached K/V history.
- [ ] **P5.5 Prove the page table is used.** Force logical pages onto shuffled physical IDs, poison unrelated blocks, and match the reference at lengths 1, 15, 16, 17, and longer multi-block cases.
- [ ] **P5.6 Re-run end-to-end parity.** Compare intermediate outputs, logits, and greedy tokens for one and multiple sequences, reused pages, EOS, and maximum requested output.

**Exit gate:** the poisoned-fragmentation test passes, traces show bounded block-at-a-time reads, and all decoder parity tests still pass on the T4.

### Phase 6 - Admission scheduler and model worker

**Depends on:** Phase 5

- [ ] **P6.1 Implement request records and queues.** Add the bounded pending FIFO, maximum active count, per-request cancellation flag, bounded token queue, and separate terminal future.
- [ ] **P6.2 Implement conservative credits.** Reserve `ceil((prompt_tokens + max_new_tokens - 1) / 16)` blocks at admission and require both credits and an active slot.
- [ ] **P6.3 Implement the exact worker iteration.** Cleanup first, consider only the FIFO head, prefill at most one admitted request, then run one decode step for the active batch.
- [ ] **P6.4 Implement lifecycle cleanup.** Return blocks, credits, and slots exactly once for EOS, length, pending cancellation, active cancellation, slow consumer, model failure, and shutdown.
- [ ] **P6.5 Preserve thread ownership.** Keep GPU/scheduler mutation in one dedicated worker thread; cross into the event loop only with `loop.call_soon_threadsafe(...)`.
- [ ] **P6.6 Avoid idle polling.** Wake the worker on new work/cancellation and block efficiently when no pending or active work exists.
- [ ] **P6.7 Test scheduling behavior with a fake model.** Cover strict FIFO head-of-line blocking, 32-slot cap, credit starvation/recovery, one-prefill-per-iteration, request joining during another decode, fairness, and all cancellation states.
- [ ] **P6.8 Inject failures.** Fail prefill, decode, publication, and cleanup paths; assert terminal resolution, failed readiness when unsafe, and zero leaked ownership.

**Exit gate:** deterministic tests prove the documented iteration order, continuous admission, bounded active state, no worker wait on clients, and leak-free recovery from every injected terminal path.

### Phase 7 - FastAPI/SSE service

**Depends on:** Phase 6

- [ ] **P7.1 Implement app lifespan.** Load tokenizer/model/pool once, start the worker only after initialization succeeds, expose readiness atomically, and join/close resources during shutdown.
- [ ] **P7.2 Validate before streaming.** Enforce schema, nonempty UTF-8 prompt, body limit, output range, context limit after tokenization, impossible-pool request, pending capacity, and model availability.
- [ ] **P7.3 Implement stable response codes.** Return the documented `400`, `413`, `422`, `429`, and `503` cases before opening the stream, with machine-readable stable error codes.
- [ ] **P7.4 Implement SSE framing.** Emit exactly one token event per emitted model token, preserve safe incremental decoding, flush remaining text in `done`, and handle zero-token EOS correctly.
- [ ] **P7.5 Implement disconnect and backpressure.** Mark disconnected clients without requiring a final event. On a full 32-event queue, set `slow_consumer`, release resources at the next worker boundary, and close via the terminal future without blocking the worker.
- [ ] **P7.6 Add operational endpoints.** Implement `/healthz`, `/readyz`, and JSON `/metrics` without prompt/generated text or unbounded request labels.
- [ ] **P7.7 Add service integration tests.** Cover exact SSE bytes/events, multibyte/incomplete decoded text, simultaneous clients, overload, active limit, disconnect in each lifecycle state, slow-reader forced closure, and worker failure.
- [ ] **P7.8 Add a CLI entry point.** Expose validated server options, bind to localhost by default, print the resolved non-sensitive configuration, and exit nonzero on startup failure.

**Exit gate:** black-box tests satisfy the full HTTP contract, a slow or disconnected client cannot stall the model worker, and every terminal path eventually releases resources and ends the stream.

### Phase 8 - Observability and operational hardening

**Depends on:** Phase 7

- [ ] **P8.1 Add structured logs.** Include request ID, lifecycle transition, duration, counts, and stable failure reason; never log prompt text, generated text, tokens, authorization headers, or model-cache secrets.
- [ ] **P8.2 Add monotonic request measurements.** Record arrival, admission, token times, first token, completion, finish reason, counts, and error/cancellation code using monotonic time.
- [ ] **P8.3 Calculate metrics exactly.** Implement TTFT, pooled ITL inputs, per-request TPOT inputs, end-to-end latency, throughput counters, block occupancy, reservation ratio, slot fill, and effective pool fill with documented undefined cases.
- [ ] **P8.4 Bound memory and cardinality.** Use aggregate counters/gauges and bounded retention/export; do not keep unbounded per-request histories in the server.
- [ ] **P8.5 Add graceful shutdown.** Stop admission, signal active/pending work, wait only for the configured bound, release all state, join the worker, and report if forced termination was required.
- [ ] **P8.6 Validate worst-case workspace.** Run the maximum supported prompt/output combination and 32 active sequences on T4; raise the margin or lower configured capacity if any OOM/unsafe headroom appears.
- [ ] **P8.7 Run soak and fault tests.** Exercise mixed lengths, queue pressure, repeated cancellation, injected failures, and restart cycles while sampling process and device memory.
- [ ] **P8.8 Perform dependency/container security review.** Audit runtime dependencies, run the container as non-root where compatible with GPU runtime, minimize the image, document model-download/network behavior, and resolve release-blocking findings.

**Exit gate:** metrics agree with controlled fixtures, prompt/output content is absent from logs and metrics, worst-case T4 validation is OOM-safe, soak tests show no unexplained growth, and SIGTERM ends within the documented bound.

### Phase 9 - Automation, packaging, and user documentation

**Depends on:** Phase 8

- [ ] **P9.1 Build the CI pipeline.** On every push run formatting check, Ruff, strict mypy, CPU pytest, package build, metadata validation, and a CPU-only service/config smoke test.
- [ ] **P9.2 Add explicit GPU validation.** Provide a T4 workflow or documented repeatable manual job that runs all `gpu` tests and uploads the environment manifest and test report. Never label CPU CI as full validation.
- [ ] **P9.3 Package the application.** Build wheel and source distribution, install each into a clean environment, and verify CLI/import behavior without the source tree on `PYTHONPATH`.
- [ ] **P9.4 Create the CUDA container.** Pin a compatible base, install only runtime dependencies, add health checking, preserve the localhost-safe default outside container-specific opt-in, and document GPU/cache mounts.
- [ ] **P9.5 Test clean setup paths.** From a clean Linux checkout, run direct install, model download/load, one generation, operational endpoints, clean shutdown, Docker build, and Docker generation.
- [ ] **P9.6 Complete the README.** Include project purpose, honest scope, architecture diagram, quick start, API example, configuration, testing, benchmark reproduction, results links, privacy behavior, troubleshooting, and limitations.
- [ ] **P9.7 Add operator and developer docs.** Document architecture/invariants, configuration reference, failure codes, T4 validation, benchmark schema, contributor workflow, and how to add a regression test.

**Exit gate:** CI is green; artifacts install cleanly; direct and Docker smoke tests pass on Linux/T4; a new reader can run and understand the bounded service without reading source code.

### Phase 10 - Reproducible benchmark system

**Depends on:** Phase 9

- [ ] **P10.1 Pin comparison systems.** Freeze the vLLM release and command, Hugging Face baseline version, CUDA/driver environment, and all resolved engine settings. Smoke-test T4 compatibility before full runs.
- [ ] **P10.2 Check in workloads.** Create deterministic approximately 128-token/32-output and 512-token/128-output raw-prompt workloads. Verify exact token counts, freeze their seed and hash, and disable EOS stopping consistently.
- [ ] **P10.3 Build one external client.** Use backend adapters for PagedCore and vLLM while sharing request scheduling, parsing, timing, validation, and output schema.
- [ ] **P10.4 Precompute offered-load schedules.** Generate saved per-repetition Poisson arrival schedules with no client concurrency cap. Reuse the same schedules for both servers.
- [ ] **P10.5 Capture complete evidence.** Save raw per-request events/counts/status, server config, workload hash, seeds, environment, git revision, commands, and NVML device-used samples every 50 ms with the idle baseline.
- [ ] **P10.6 Implement analysis.** Report attempted/completed/rejected/failed counts; TTFT, pooled ITL, per-request TPOT, and end-to-end p50/p95; throughput; achieved request rate; error/rejection rate; and peak VRAM.
- [ ] **P10.7 Guard statistical claims.** Emit sample counts for every percentile, set p99 to `null` with `insufficient_sample` below 1,000 relevant observations, and bootstrap a 95% CI across repetition-level throughput values.
- [ ] **P10.8 Validate the harness.** Unit-test timestamp math and missing cases; replay synthetic known results; ensure failures/rejections cannot disappear from summaries; and compare a small run manually.
- [ ] **P10.9 Provide one reproducible command.** The command must start from frozen inputs and produce raw JSONL or an equivalently inspectable format plus a machine-readable and Markdown summary.

**Exit gate:** synthetic validation passes, a pilot run produces complete replayable evidence, both backends receive equivalent raw prompts/schedules, and no summary can hide unsuccessful requests.

### Phase 11 - T4 comparison, analysis, and release

**Depends on:** Phase 10

- [ ] **P11.1 Freeze the test matrix before comparison.** Pilot both systems, select an offered-rate grid with an uncongested point and at least one near saturation, then commit the grid before inspecting comparative results.
- [ ] **P11.2 Run controlled repetitions.** Use an otherwise-idle T4, restart servers between repetitions, record idle VRAM, perform 20 excluded warmups, and run at least five repetitions of at least 100 measured requests per cell.
- [ ] **P11.3 Collect enough data for any p99 claim.** Reach at least 1,000 relevant observations in a cell or publish `null`/`insufficient_sample`; do not extrapolate.
- [ ] **P11.4 Run the Hugging Face baseline.** Measure it only as a single-request latency/output-throughput reference and mark concurrent-QPS cells `N/A`.
- [ ] **P11.5 Investigate one meaningful gap.** Profile queue, prefill, dense operations, blockwise attention, scheduler, and streaming. Use GPU synchronization only in profiling runs, not normal client latency runs.
- [ ] **P11.6 Publish raw and summarized results.** Include completed/attempted, TTFT p95, ITL p95, TPOT p95, output tokens/s, peak VRAM, and PagedCore KV utilization, plus all required counts and intervals.
- [ ] **P11.7 Write the technical report.** Explain correctness evidence, largest observed discrepancy, bottlenecks, gap to vLLM, uncertainty, configuration differences, serial-prefill/credit tradeoffs, failures, and limitations.
- [ ] **P11.8 Execute the release checklist.** Re-run all CPU/GPU/service/lifecycle tests, clean installs, Docker smoke test, dependency audit, docs link check, and `git diff --check` from the release candidate.
- [ ] **P11.9 Publish v0.1.0.** Tag the exact tested commit, attach/installable artifacts and checksums, link immutable benchmark evidence, write factual release notes, and verify every public command from the published tag.
- [ ] **P11.10 Perform a post-release smoke test.** Install from the released artifact in a fresh environment, run one request, inspect health/readiness/metrics, and record any follow-up issue without rewriting evidence.

**Exit gate:** the tagged commit and released artifact match the fully tested code; raw evidence is available; results and limits are stated literally; and a clean consumer setup reproduces the basic service.

## 6. Dependency and critical-path summary

```text
P0 repository/contracts
  -> P1 reproducible environment
  -> P2 types/reference fixtures
  -> P3 dense decoder parity
  -> P4 paged KV allocator
  -> P5 block-table attention
  -> P6 scheduler/worker
  -> P7 HTTP/SSE
  -> P8 hardening/observability
  -> P9 CI/package/container/docs
  -> P10 benchmark harness
  -> P11 T4 runs/report/release
```

Work that can proceed in parallel without weakening the critical path:

- CPU allocator/property-test infrastructure can begin after P2 while dense GPU parity is being stabilized.
- README skeleton, architecture diagram, and CI scaffolding can begin after P0, but their final claims depend on P8-P11 evidence.
- Benchmark data schemas and synthetic analyzer tests can begin after measurement definitions are stable in P2, but live comparison waits for P9.
- Docker scaffolding can begin after P1; the release image waits for service and shutdown behavior through P8.

## 7. Required test matrix

| Suite | Runs where | Required coverage | Release requirement |
|---|---|---|---|
| Static quality | CI, local | formatting, Ruff, strict mypy, package metadata | Every push and release candidate |
| CPU unit | CI, local | config, types, allocator model, scheduler/fake model, metric math | Every push |
| Property/state | CI, local | randomized ownership and lifecycle transitions | Every push with bounded examples; extended before release |
| GPU parity | T4 | layers, logits, greedy tokens, page boundaries, fragmented pages | Pass on release candidate |
| GPU lifecycle | T4 | repeated mixed requests, cancellation/failure, zero leaked state, VRAM trend | Pass on release candidate |
| Service | CI plus T4 | validation, SSE, concurrency, slow client, disconnect, readiness | CPU fake-model cases in CI; real-model cases before release |
| Installation | clean Linux/T4 | wheel/sdist install, model load, CLI, one request | Pass for both artifacts |
| Container | clean Linux/T4 | build, GPU start, health/readiness, request, SIGTERM | Pass on release candidate |
| Soak/fault | T4 | queue pressure, mixed lengths, failures, restart, no unexplained growth | Documented pass before benchmark |
| Benchmark harness | CI plus T4 pilot | replay math, missing samples, failure accounting, metadata completeness | Pass before official runs |

Minimum commands must be documented and stable by Phase 9. Exact command names may change during scaffolding, but the final set must cover:

```bash
python -m ruff check .
python -m mypy --strict src/pagedcore
python -m pytest -m "not gpu and not benchmark"
python -m pytest -m gpu
python -m build
docker build .
```

## 8. Benchmark acceptance criteria

The official comparison is valid only if:

- PagedCore and vLLM use the same pinned TinyLlama weights/tokenizer, FP16, one T4, 2,048 context, greedy/fixed-length generation, raw prompts, and checked-in arrival schedules.
- vLLM explicitly uses `--dtype half`, `--max-model-len 2048`, `--block-size 16`, `--gpu-memory-utilization 0.90`, and disabled prefix caching; all other resolved defaults are recorded.
- PagedCore stays within the same 90% device-memory ceiling and reports configured KV bytes.
- Each server runs alone on the otherwise-idle device and is restarted between repetitions.
- Raw data includes unsuccessful requests; latency statistics never silently filter away overload behavior.
- Client measurements are primary. Internal metrics are diagnostic and are not compared where definitions differ.
- Peak VRAM is derived from device-level NVML samples after subtracting the recorded idle baseline.
- The report makes no causal performance claim without profiling or a controlled experiment.

## 9. Documentation deliverables

Before v0.1.0, the repository must contain:

- `README.md`: value proposition, truthful scope, quick start, API example, architecture overview, results summary, limitations, and links.
- `docs/SPEC.md`: final public behavior and acceptance criteria.
- `docs/DESIGN.md`: implemented architecture, invariants, threading, attention path, lifecycle, and known limits.
- `docs/VALIDATION_PLAN.md`: frozen test and benchmark protocol.
- A configuration/reference page with every option, default, validation rule, and benchmark-frozen value.
- A reproducibility page with direct install, Docker, T4 environment, exact commands, workload hashes, seeds, and artifact schema.
- A technical report with correctness evidence, result tables, analysis, failures, and limitations.
- An architecture diagram whose page-table data path and worker/event-loop boundary match the code.
- Release notes for v0.1.0 that distinguish implemented behavior, measured evidence, and future work.

## 10. Risk register

| Risk | Early warning | Mitigation / stop condition |
|---|---|---|
| FP16 differences change greedy tokens | Logit rank changes or parity failure | Compare intermediate tensors; locate RoPE/GQA/norm/softmax error; do not widen tolerance blindly |
| Paging is only bookkeeping | Decode allocates a history-sized tensor or ignores shuffled IDs | Keep poisoned-fragmentation and no-full-gather tests as release blockers |
| T4 OOM from workspace, not KV capacity | Spikes near 32 active sequences or long prefill | Measure worst case, increase margin/reduce pool or active limit, fail startup when unsafe |
| Block/credit/slot leak | Nonzero ownership after terminal state or growing VRAM | Centralize release-once cleanup and run cancellation/fault/soak tests |
| Slow client blocks progress | Worker waits on output queue or stream never ends | Nonblocking publication, cancellation flag, out-of-band terminal future, forced closure test |
| Strict FIFO causes head-of-line delay | Free capacity exists while the head cannot fit | Keep as measured MVP behavior; expose queue time and explain limitation |
| Serial prefill damages active ITL | ITL spikes on mixed prompt lengths | Measure and report; do not add chunked prefill before v0.1.0 |
| Benchmark comparison is unfair | Different prompts, schedules, EOS rules, memory ceilings, or hidden rejections | Freeze one harness/config matrix and publish raw inputs, failures, and resolved settings |
| vLLM/T4 compatibility changes | Startup failure or silent dtype/kernel fallback | Pin and smoke-test the exact version; record fallbacks; change protocol before official runs |
| CI creates false confidence | CPU checks green while GPU path is untested | Separate markers/badges and require dated T4 evidence for release |
| Model download/revision drifts | Different hashes/config on clean setup | Pin model and tokenizer revision together and record resolved commit/hash |
| Scope expansion delays release | Sampling, public auth, new models, or kernels enter critical path | Move nonessential work to stretch goals until v0.1.0 is published |

## 11. Release checklist

- [ ] All phase exit gates P0-P11 are satisfied with linked evidence.
- [ ] Source documents and implementation agree; traceability table has no `pending` release item.
- [ ] Static, CPU, property, GPU, lifecycle, service, install, container, and benchmark-harness tests pass.
- [ ] T4 environment manifest and worst-case memory evidence are saved.
- [ ] No allocator ownership, CUDA memory, thread, task, or stream leak remains in soak tests.
- [ ] API defaults to localhost and public deployment is explicitly unsupported.
- [ ] Logs, metrics, fixtures, raw results, and artifacts contain no private prompt/generated content or credentials.
- [ ] Direct install and Docker commands work from the exact release candidate.
- [ ] Benchmark raw data, workload hashes, seeds, configuration, and analysis are published together.
- [ ] Every p99 statement has at least 1,000 relevant observations; otherwise it is explicitly unavailable.
- [ ] README/report claims are supported by tests or released measurements and include limitations.
- [ ] Release artifacts have checksums and install successfully in a fresh environment.
- [ ] The tagged commit, artifacts, documentation, and benchmark evidence all identify the same version.

## 12. Stretch goals after v0.1.0

Choose only one major serving feature at a time. Each stretch goal requires a design update, correctness regression suite, controlled before/after benchmark, and separate release.

Recommended order:

1. **Preemption and recomputation:** improve admission flexibility while proving progress and exact resource recovery.
2. **Chunked prefill:** reduce decode interference from long prompts; measure TTFT/ITL tradeoffs under mixed workloads.
3. **Copy-on-write and prefix sharing:** safely share physical pages and make ownership/reference counting testable.
4. **A fused paged-attention kernel:** replace the correctness-first PyTorch block loop only after profiling identifies it as the dominant gap.
5. **Speculative decoding:** add a pinned draft model and acceptance/correction tests before measuring speedup.
6. **Quantization or additional Llama checkpoints:** treat each dtype/model as a new compatibility claim with its own parity and memory evidence.

Multi-GPU serving, public multi-tenant deployment, and broad API compatibility should remain separate projects unless the v0.1.x evidence justifies expanding PagedCore's purpose.
