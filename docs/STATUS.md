# Implementation and evidence status

Snapshot: 2026-10-26. Phase 3 dense decoder parity is complete. Phase 4 GPU paged
KV cache remains current. The [October 25 T4 archive](../results/validation/2026-10-25/pagedcore-p42-t4-20261025T153708Z.zip)
at clean source `f72f39c` records 498 passing CPU tests, 12 passing GPU tests with
zero skips and passing formatting/Ruff/strict mypy. Both small-pool storage and
basic allocator storage/reuse/stream-ownership tests executed on T4. The
[October 26 review](validation/2026-10-26-p42-t4-review.md) verified ZIP integrity,
all 33 source hashes, lock and recorder/collector identity. This supplies the
previously missing P4.1/P4.2 storage evidence at that revision. No new GPU
execution occurred locally.

The newer P4.3 reservation/accounting and P4.4 failure-safety changes remain
CPU-tested only; their runtime and GPU test differ from the archived source.
P4.5 boundary coverage is the next implementation task; P4.5-P4.6 and the
Phase 4 exit gate remain open.

The same archive supplies nine passing dense runs and nine passing pressure
runs with the 8,192 MiB unused reservation and revised 1,536 MiB margin. The
largest dense increment is 1,128.05 MiB, with no allocation retries/OOMs and stable
repeated cleanup after a one-time 32 MiB increase. This is single-sequence
reservation evidence; capacity remains provisional under the
[combined memory protocol](VALIDATION_PLAN.md#combined-memory-validation-before-capacity-freeze).
The collector's required-test list still omits both KV tests. Manual inspection
confirmed both passed here; the [review](validation/2026-10-26-p42-t4-review.md#findings-and-remaining-work)
records the open collector follow-up.

Earlier CPU reports and T4 observations remain below as historical evidence.

## Current gate

Phase 4 GPU paged KV cache is the next implementation gate. P3.1 weight loading is
implemented and CPU-validated, as are P3.2 dense components and P3.3 contiguous
prefill/decode. P3.4 layer/logit comparisons passed on the pinned checkpoint/T4
in the retained October 18 archive. P3.5 greedy selection/stopping passed all public
T4 fixtures in the October 19 and October 20 archives. P3.6 tolerance measurements
passed on T4 in the October 20 archive. P3.7 dense maximum-path workspace is
recorded in the October 21 archive, satisfying the Phase 3 exit gate. The
[pool allocation function](../src/pagedcore/kv_cache.py) now constructs one FP16
tensor with shape `[22, 2, num_blocks, 16, 4, 64]`, rounds the configured byte
budget down to whole blocks and checks workspace headroom before and after
allocation. The caller retains this storage for the worker's lifetime.
`KVAllocator` now assigns blocks from a free list, keeps per-request block tables
and valid lengths, appends K/V, reads one logical block and releases a request's
blocks for reuse. It enforces its creating thread and CUDA stream; reads borrow
only valid slots from the existing pool. These basic operations now also have
small-pool T4 evidence at `f72f39c` in the October 25 archive.
It now grants explicit per-request reservations at admission and derives total,
free, owned and reserved block counts, active slots and live cached slots from
its records. Reserved credits include owned blocks and remain held until cleanup;
pending requests stay outside the allocator. CPU tests cover these accounting
transitions and reservation limits. Failure safety now rejects impossible
reservations, validates tables/free lists against independent physical ownership
before access or mutation, and preserves valid history/accounting after
synchronous copy failure. Direct double release stays an error; duplicate worker
cleanup must be idempotent when integrated in P6. These newer P4.3/P4.4 changes
require another attributable T4 run. Worker startup/decoder integration and
boundary/property-test gates remain pending. P4.1 storage validation is satisfied
at the archived revision; the Phase 4 exit gate remains unsatisfied.
The October 13
archive and successful hosted CI below supersede the earlier Phase 2 evidence
blockers; the October 18 layer test supplies the narrower owned-forward evidence.

| Surface | Implemented and observed | Evidence boundary |
|---|---|---|
| Package/configuration/diagnostics | Frozen CPU/CUDA graph, immutable pinned settings, safe diagnostics and target verifier | October 7 clean Linux/T4 reference setup exists; the environment manifest lacks the selected GPU UUID |
| Deterministic controls | Python/Torch seeding and inference mode | CPU repeatability plus one executed October 7 CUDA RNG repeatability test; no cross-device promise |
| Shared engine contracts | Immutable records, legal transitions, positive reservation credits and aggregate capacity checks | 108 passing CPU contract cases; concrete ownership/release-once/CUDA ordering remain P4/P6 work |
| Public fixtures | Exact pinned IDs at 15/16/17 and 32/49 tokens, whitespace, mixed lengths, maximum output and synthetic EOS cases | Current cached-tokenizer integration check executed and passed; these inputs do not prove batching or an owned generation loop |
| HF oracle | Frozen eager FP16 capture/comparison, strict deterministic policy, snapshots and safe artifact persistence | October 13 T4 reference repeatability for `boundary_15`, with setup and selected-device attribution |
| Internal timing analyzer | Separate generation/delivery durations, TTFT/ITL/TPOT and missing zero/one-token values | 18 synthetic CPU cases; early transport closure is valid, impossible generation ordering and invalid timestamps still fail |
| CPU CI | Push/PR workflow for frozen setup, format/lint/types/tests and distribution smoke | Hosted run `38445324328` succeeded for prior source; owned-decoder CPU/static checks also passed in the October 21 T4 collection, without a newer hosted CI claim |
| Owned weight loading | Pinned safetensors/config validation, complete typed mapping, BF16-to-FP16 conversion, untied embedding/head | 69 synthetic contract cases, real pinned CPU load and owned-loader T4 execution in the October 18 layer test |
| Dense decoder components | RMSNorm, Q/K/V, absolute RoPE, GQA mapping/dense attention, output projection, SwiGLU, final norm/LM head | Synthetic CPU components plus the pinned-checkpoint T4 layer comparisons at fixed tolerances |
| Contiguous prefill/decode | Single-sequence owned forward, residuals, per-layer FP16 cache and one-token append at absolute positions | CPU orchestration and October 18/19 T4 prefill/three-step decode comparisons; October 21 dense workspace below, no paged/batched proof |
| Layer comparisons | All-layer snapshots during prefill and three reference-token cached decode steps, with fixed intermediate/logit tolerances and named failures | Two-layer random-weight CPU evidence and passing 22-layer pinned-checkpoint T4 test at 1/15/16/17/49 prompt tokens |
| Dense greedy generation | Local-cache argmax loop, exact selected/emitted IDs, EOS/length precedence and request bounds | Synthetic stopping checks plus all six public T4 fixture cases passed in the October 19 archive |
| Tolerance measurements | Per-tensor FP32 error analysis, combined-tolerance ratios, maxima and environment-attributed T4 export | October 20 T4 evidence at `88be0f8`: all 2,260 FP16 snapshot comparisons have zero discrepancy; limited to the recorded forwards/environment |
| Dense workspace recording | Separate owned-model process, maximum prefill/output paths, raw samples/peaks, allocator diagnostics and collector acceptance checks | October 21 T4 archive at `e9e21fb`: all nine runs passed; dense single-sequence evidence only, safe pool capacity remains unproven |
| Pool-pressure recording | Separate process holds configured pool bytes during dense paths; headroom checks, margin observation and independent artifacts | October 25 T4 archive at `f72f39c`: nine runs passed with an 8,192 MiB reservation and 1,536 MiB margin; no paged/batched/service capacity certification |
| Runtime KV pool construction | One CUDA FP16 tensor, whole-block sizing and pre/post-allocation workspace checks | Nine CPU cases plus the passing small-pool T4 storage test at `f72f39c`; integrated model/pool memory validation remains pending |
| Basic KV block allocator | Free-list assignment, immutable block tables, partial-block lengths, append, one-block views and release-all under worker ownership | October 25 T4 storage/reuse and stream-guard test passed at `f72f39c`; newer accounting/failure-safety changes still need T4 execution; boundary/property gates remain open |
| KV capacity accounting | Per-request credits, derived capacity snapshots, ownership/accounting assertions and reservation-bounded appends | Four added CPU cases cover multi-request accounting, reserved-but-unassigned capacity, per-request credit limits and pending exclusion; October 25 archive predates these changes; T4 and worker lifecycle integration remain pending |
| KV allocator failure safety | Impossible-reservation rejection, pre-access integrity checks, independent physical ownership, strict direct release and unpublished partial-copy writes | 52 added CPU cases cover corrupt/swapped tables, corrupt free lists, copy failures and cleanup/reuse; extended T4 test selected but skipped locally; duplicate worker-cleanup integration remains P6 work |
| Paged attention/worker/service | Not implemented | No decoder integration, lifecycle cleanup, concurrency, HTTP or GPU memory-safety proof |
| Container/benchmark/report | Not implemented | No container, serving performance or comparative measurement proof |

## Current CPU evidence

On 2026-10-26, all 90 [pool/allocator unit cases](../tests/unit/test_kv_cache.py)
passed in the full 554-test CPU suite, with 12 GPU cases deselected and no CPU
skips. The 52 added failure-safety cases cover impossible reservations, corrupt
and swapped tables, invalid/duplicate/free/foreign physical blocks, corrupt free
lists, and failures on each of the first four K/V copies starting at cached
lengths 0/15/16/17. They verify unchanged valid history, other requests and
accounting, strict double-release rejection, complete block reuse and zero final
ownership/credits/slots. Before the runtime fix, 32 of the initial 48 regression
cases failed; the 16 copy-failure cases already passed and retain that protection.

Formatting, Ruff, strict mypy and `git diff --check` passed. Attributable local-only
reports are [full CPU](../results/local/p44-cpu.xml),
[initial regressions before the fix](../results/local/p44-regressions-before.xml),
and [GPU selection](../results/local/p44-gpu-selection.xml). Commands used
`.venv/Scripts/python.exe -m ruff format --check .`, `-m ruff check .`,
`-m mypy --strict src/pagedcore`, and
`-m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p44-cpu.xml`
with `HF_HUB_OFFLINE=1` for pytest. Both KV GPU tests were selected with
`-m pytest tests/gpu/test_kv_pool_cuda.py -ra
--junitxml=results/local/p44-gpu-selection.xml` and skipped without CUDA. The
extended allocator case injects a synchronous failure after K is issued and
before V, then retries and reuses storage. It must execute on an actual T4;
neither the CPU tests nor the earlier archive prove these changes on GPU or
recovery from an actual CUDA/context failure. P4.5 boundary coverage is next;
P4.5-P4.6, worker cleanup integration and the Phase 4 exit gate remain open.

Previously, on 2026-10-25, all 38 [pool/allocator unit cases](../tests/unit/test_kv_cache.py)
passed as part of the full 502-test CPU suite, with 12 GPU cases deselected and
no CPU skips. Four added cases cover consistent multi-request capacity snapshots,
reserved-but-unassigned capacity, per-request credit limits and zero resources
for requests not yet admitted. Existing cases now supply explicit reservations
and still cover valid-slot contents, block boundaries, fragmentation, isolation,
release/reuse, invalid inputs and worker-thread access. Empty-request release
and full-pool checks also assert the ledger. These are CPU storage/bookkeeping
checks; pending-queue lifecycle integration and CUDA safety remain unproven.
Formatting, Ruff, strict mypy and `git diff --check` passed. The local-only
[CPU JUnit report](../results/local/p43-cpu.xml) records the run. Commands used
`.venv/Scripts/python.exe -m ruff format --check .`, `-m ruff check .`,
`-m mypy --strict src/pagedcore`, and
`-m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p43-cpu.xml`
with `HF_HUB_OFFLINE=1` for pytest.
Both [GPU pool/allocator tests](../tests/gpu/test_kv_pool_cuda.py) were separately
selected with `-m pytest tests/gpu/test_kv_pool_cuda.py -ra
--junitxml=results/local/p43-gpu-selection.xml` and skipped because CUDA is
unavailable, as recorded in the local-only
[selection report](../results/local/p43-gpu-selection.xml). The allocator test
checks mapped writes/partial-block reads, reuse, stable backing allocation,
capacity snapshots and rejection of access from another CUDA stream; this updated
version must execute on a real T4. The earlier test passed in the October 25
archive, before P4.3 changed the allocator and added capacity assertions.
This earlier run validates P4.3 capacity accounting on CPU and predates the
P4.4 failure-safety changes described above.

On 2026-10-23, all nine [pool unit cases](../tests/unit/test_kv_cache.py) passed.
They use mocked CUDA memory/allocation calls with CPU or meta storage, including
the default 8,192 MiB sizing without allocating that much host memory. The full
CPU suite passed 473 tests with 11 GPU cases deselected; formatting, Ruff,
strict mypy and `git diff --check` passed. The local-only
[JUnit report](../results/local/p41-cpu.xml) records this run. Commands used
`.venv/Scripts/python.exe -m ruff format --check .`, `-m ruff check .`,
`-m mypy --strict src/pagedcore`, and
`-m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p41-cpu.xml`
with `HF_HUB_OFFLINE=1` for pytest.
The [T4 storage test](../tests/gpu/test_kv_pool_cuda.py) was skipped locally because
Torch is `2.7.1+cpu`. The October 25 archive now supplies the actual T4 FP16
storage and stable-allocation evidence at `f72f39c`; CPU results alone do not
establish GPU memory safety.

The [margin-update CPU report](../results/validation/2026-10-22-pool-memory-cpu/margin-update-cpu.json)
and [JUnit result](../results/validation/2026-10-22-pool-memory-cpu/margin-update-cpu-tests.xml)
record 464 passing tests and ten GPU cases deselected on 2026-10-22 after the
default changed to 1,536 MiB. Ruff, formatting, strict mypy over runtime and
recorder, and `git diff --check` passed. This is CPU-only validation.

On 2026-10-22, the [pool-pressure CPU report](../results/validation/2026-10-22-pool-memory-cpu/report.json)
and [JUnit result](../results/validation/2026-10-22-pool-memory-cpu/cpu-tests.xml)
record 464 passing CPU tests with ten GPU cases deselected. Ruff, formatting,
strict mypy over runtime and recorder, Bash syntax and `git diff --check` passed.
Tests verify reservation lifetime, unchanged baseline artifacts, headroom/OOM/retry
failures, and rejection of incomplete or mismatched pool-pressure evidence.
The local Torch build remains `2.7.1+cpu`. The separate October 22 T4 archive
now supplies pool-pressure evidence for the former 1,024 MiB margin. The default
margin is raised to a provisional 1,536 MiB because the observed dense increment
is 1,128.05 MiB, exceeding the former threshold by 104.05 MiB. Peak allocated
memory with the reservation was 11,420.22 MiB; allocator reserved memory reached
11,626 MiB, with 3,128.81 MiB free at the lowest sampled boundary. These are
not device-wide transient peaks. The 8,192 MiB pool remains unchanged.
The October 25 archive now supplies a pressure run with the revised 1,536 MiB
margin. Integrated paged/batched/service memory measurements remain pending.
No new GPU execution occurred locally.

The [workspace tests](../tests/unit/test_dense_workspace.py) and
[collector tests](../tests/unit/test_t4_evidence.py) exercise boundary workloads,
transient peaks, allocator reuse, first-use workspace, stable cleanup, failure
retention and rejection of incomplete evidence with synthetic CPU fakes. On
2026-10-21, `python -m pytest -m "not gpu and not benchmark" -ra` passed 419 tests
with ten GPU cases deselected; Ruff, formatting and strict mypy passed, including
strict typing of the recorder. The [local JUnit report](../results/local/p3.7-cpu-tests.xml)
retains the CPU result. The local PyTorch build is `2.7.1+cpu` with no CUDA device;
these checks alone cannot close P3.7. The October 21 T4 archive now supplies its
required hardware-backed measurements.

The [tolerance-measurement report](../results/validation/2026-10-19-p3.6-cpu.json)
records local commands and source identity. Synthetic tests check measurement
coverage, zero and negative references, FP16-to-FP32 error arithmetic, exporter
maxima/locations, metadata retention, ZIP contents and rejection of missing
measurements. It records the original CPU-only limitation; the October 20 T4
archive now supplies the required hardware-backed measurement evidence.

Earlier CPU observations follow.

The current [greedy-generation report](../results/validation/2026-10-18-p3.5-cpu.json)
records commands, source/lock identity and CPU-only limits. All 397 selected CPU
tests passed with zero skips; ten GPU cases were deselected. Seven decoder GPU
cases were separately selected and skipped for lack of CUDA. Six random-weight
generation cases compare exact IDs and finish reasons with independent eager HF;
seven controlled stopping cases also assert consumed tokens, cache positions and
forward counts, including no extra decode after immediate EOS or the output limit.
Output/EOS type bounds and context overflow are checked. The T4 collector requires
all six named public fixture cases; its Bash syntax check passed locally.

The earlier [layer-comparison report](../results/validation/2026-10-17-p3.4-cpu.json)
records commands, source/lock identity and CPU-only limits. All 382 selected CPU
tests passed with zero skips; four GPU tests were deselected. The new T4 test was
also selected separately and skipped because CUDA is unavailable. New cases compare
embeddings, both per-layer norms, projected attention, MLPs, residual outputs,
final normalized hidden state and logits at lengths 1/15/16/17/49, including three
cached decode steps. An injected layer error checks mismatch detection. Initial
failures traced to FP16-rounded RoPE buffers in the synthetic HF fixture; restoring
FP32 frequencies fixed them at unchanged tolerances. The runtime decoder is unchanged.

The earlier [contiguous-decoder report](../results/validation/2026-10-16-p3.3-cpu.json)
records commands, source/lock identity, dependency versions and CPU-only limits.
All 376 selected tests passed with zero skips; three GPU tests were deselected.
Synthetic two-layer tests use pinned head geometry and reduced MLP/vocabulary.
They check prefill logits against eager HF, repeated decode against fresh prefixes,
causal visibility, K/V contents, request isolation, validation and cache preservation
after injected failure. A synthetic 2,047-position history tests the last decode
position and overflow; it does not prove maximum-prefill memory safety.

The earlier [dense-component report](../results/validation/2026-10-14-p3.2-cpu.json)
records commands, source/lock identity, dependency versions and CPU-only limits.
All 346 selected tests passed with zero skips; three GPU tests were deselected.
Synthetic component tests compare with pinned HF math and independently check
projection orientation, head mapping, absolute positions, causal visibility and
the untied LM head. They do not establish real-model parity.

The earlier [weight-loading report](../results/validation/2026-10-14-p3.1-cpu.json)
records commands, source/lock identity, dependency versions and CPU-only limits.
All 324 selected tests passed with zero skips; three GPU tests were deselected.
The full pinned checkpoint loaded as finite FP16 CPU tensors with 22 layer records.

The earlier [audit-fix report](../results/validation/2026-10-12-p2-audit-fixes/report.json) and
[JUnit report](../results/validation/2026-10-12-p2-audit-fixes/cpu-tests.xml) identify
source/lock hashes, actual commands, the existing local environment and test counts:
258 collected, 255 selected/executed/passed, zero failed/skipped, and three GPU
cases deselected. The new pending/active disconnect tests failed before the timing
fix and pass afterward. Formatting, Ruff lint and strict mypy over eight source
files pass. The T4 collector passes Bash syntax checking only.

Earlier CPU evidence remains historical: [P1.6](../results/validation/2026-10-05-cpu/2026-10-05-p1.6-cpu.json),
[P2.1](../results/validation/2026-10-07-p2.1-cpu.json),
[P2.2](../results/validation/2026-10-08-p2.2-cpu.json),
[P2.3](../results/validation/2026-10-08-p2.3-cpu.json),
[oracle arithmetic](../results/validation/2026-10-09-p2.4-cpu.json),
[strict reference policy](../results/validation/2026-10-10-reference-determinism-cpu.json)
and [marker selection](../results/validation/2026-10-12-p2.5-cpu.json).
Their old formatting failures do not describe the corrected current checkout.

## Saved T4 observations

The [October 25 archive](../results/validation/2026-10-25/pagedcore-p42-t4-20261025T153708Z.zip)
and [October 26 review](validation/2026-10-26-p42-t4-review.md) at clean source
`f72f39c` retain 498 passing CPU tests and 12 passing GPU cases without skips,
including the small-pool and basic allocator storage/reuse/stream tests. All
2,260 dense tensor/logit comparisons have zero observed error; both HF captures
match across 39 tensors. The selected device is one CUDA-visible Tesla T4, UUID
`cf644039-def8-fd96-25df-701f921c0215`; the bundle retains fresh frozen setup,
actual interpreter/dependencies, source/lock hashes and complete command logs.

Both workspace modes contain nine passing runs and 27 samples each. Peak
allocated/reserved memory is 3,228.22/3,434 MiB without the reservation and
11,420.22/11,626 MiB with the unused 8,192 MiB FP16 reservation. The largest
dense increment is 1,128.05 MiB against the configured 1,536 MiB margin. Counters
remain zero for retries/OOMs; repeated live cleanup is stable after a one-time
32 MiB increase. `capacity_validated=false` correctly remains set. The actual
KV tests use a separate two-block pool; neither they nor the pressure probe
establish integrated model/pool, paged/batched or service memory safety. This
archive predates the current P4.3 changes. The review records the unresolved
collector required-test gap, which did not omit tests in this actual run.

The [October 21 archive](../results/validation/2026-10-21-t4/pagedcore-p37-t4.zip)
at clean source `e9e21fb` retains `dense-workspace.json`,
`dense-workspace-allocator.txt`, command logs, environment/source attribution,
419 passing CPU tests and ten passing GPU cases without skips. The workspace
recorder used a separate owned-model process with no HF model or paged KV pool
on one CUDA-visible Tesla T4, UUID `16f24614-365b-0803-b469-e55e0ebaebdc`.
Each path ran three times: 2,048-token direct prefill, 2,047+1 generation and
1,792+256 generation with EOS ignored. All nine runs passed, all 27 samples
record zero allocation retries/OOMs, and cleanup live allocation is stable on
repeats. Peak allocated memory reached 3,228.22 MiB; the largest increment above
the corresponding pre-forward baseline was 1,128.05 MiB.

That increment includes input, dense KV, logits and temporary tensors; it is not
an isolated scratch-space bound. The initial 1,024 MiB workspace margin is below
this observed increment, and neither that margin nor the 8,192 MiB pool default
is certified safe by this run. Pool plus paged/batched measurements and final
maximum-context/32-active service checks remain separate P5.7/P8.6 gates.

The [October 20 archive](../results/validation/2026-10-20-t4/pagedcore-p36-t4.zip)
at clean source `88be0f8` retains 401 passing CPU tests, ten passing GPU cases and
the owned-decoder `tolerance-envelope.json`. The [review](../results/validation/2026-10-20-t4/review.json)
verifies its CRC, all 29 recorded source hashes, lock/collector/fixture identity,
complete measurement labels and JSON/JUnit/metadata agreement. All 22 layers are
covered at prompt lengths 1/15/16/17/49, with prefill and three reference-token
cached decode steps: 2,240 intermediate and 20 logit comparisons all have zero
absolute error and zero tolerance ratio. Displayed `bos_only` maximum locations
are the first entries in a tie, not the only measured input.

The selected device is one CUDA-visible Tesla T4 with UUID, compute capability
7.5 and 15,636,037,632 bytes of reported memory. The bundle records Linux,
Python 3.11.15, Torch 2.7.1+cu126, Transformers 4.52.4, CUDA 12.6 and driver
580.178.04, fresh frozen setup and dependencies from the actual `.venv` interpreter.
Tolerances remain unchanged. This establishes the observed FP16 envelope for
those inputs/configuration, not a bound over all inputs, cross-platform equality,
or maximum-path workspace safety.

The [October 19 archive](../results/validation/2026-10-19-t4/pagedcore-p35-t4.zip)
at clean source `7bbcd6a` retains 397 passing CPU tests and ten passing GPU cases
with zero skips, including all public greedy fixtures and the layer comparison.
Its recorded source hashes match that Git revision (local Windows line endings
differ for some files); the lock also matches. The selected device is one
CUDA-visible Tesla T4 with UUID, Torch 2.7.1+cu126, CUDA 12.6 and driver 580.178.04.
Its `dependencies.txt` was frozen from the outer environment, so it does not
identify the test interpreter's complete installed packages; the collector now
targets `.venv/bin/python` explicitly. Manifest/test output establish the pinned
Python, Torch and Transformers versions. `comparison.json` records HF-vs-HF
repeatability only. No owned-decoder envelope or workspace measurement is retained.

The [October 18 archive](../results/validation/2026-10-18-t4/t4-20261018T153034Z.zip)
at clean source `8940a1c` retains four passing GPU tests without skips, including
all-layer prefill/three-step cached-decode comparison. Its manifest source hashes
match that revision, and its selected device is one CUDA-visible Tesla T4
with UUID, Torch 2.7.1+cu126 and CUDA 12.6. The
[P3.5 CPU report](../results/validation/2026-10-18-p3.5-cpu.json) records this
archive review and hash. It predates the owned greedy loop and does not prove
greedy parity, a measured tolerance envelope, workspace, paging or serving.

The earlier [October 13 archive](../results/validation/2026-10-13-t4/t4-20261013T180545Z.zip)
at source `d6fb549` retains setup/commands, 255 passing CPU tests and three executed
passing GPU tests with zero skips, one CUDA-visible T4 and its UUID. The two
`boundary_15` oracle captures match exactly across 39 tensors. Reviewed again on
October 14, this closes the earlier environment/reference attribution gaps;
it does not prove an owned decoder, paging, memory safety or performance.

The [evidence review](../results/validation/2026-10-12-t4-evidence-review.json)
separates two runs and records hashes, actual results and unknown fields.

On October 7, the retained [manifest](../results/validation/2026-10-07-t4/manifest.json),
[setup/command log](../results/validation/2026-10-07-t4/commands.log) and
[GPU report](../results/validation/2026-10-07-t4/pytest-report.xml) identify clean
source `211bb0c`, the frozen lock, Linux, pinned Python 3.11.15, Torch 2.7.1+cu126
and Transformers 4.52.4. A reference forward and CUDA RNG repeatability both
executed and passed, with zero skips. Two physical T4s were present; one was
CUDA-visible. The manifest does not record its UUID. This is environment evidence.

The archive stored at [2026-10-07-t4/p2.4-evidence.zip](../results/validation/2026-10-07-t4/p2.4-evidence.zip)
contains a separate run captured on October 12 at source `51dcf8f`. One oracle
GPU test and 58 synthetic oracle/timing tests executed and passed with zero skips.
Two `boundary_15` captures have three emitted tokens and 39 FP16 tensors, exact
ID/outcome/text agreement and zero observed tensor differences. Eager attention,
strict deterministic settings, cuBLAS workspace, selected T4 UUID, dependencies
and source/lock identity are retained. The current oracle implementation matches;
the timing analyzer and shared fixture formatting changed afterward.

The October 12 archive has no installer/setup log and does not identify the
CUDA-visible device count. Its physical inventory lists two T4s; the selected
`cuda:0` is explicit. This proves HF reference repeatability for one fixture,
not an owned decoder, paging, memory safety or performance.

## Hosted CPU CI

The newer [successful run](https://github.com/Anassbzdd/PagedCore/actions/runs/38445324328)
for `0b94ebf` supersedes the historical formatting failure. Its completed/success
status was rechecked through GitHub's API on October 14. That run predates the
weight loader, dense components and contiguous decoder; no hosted run was triggered
for these local edits.

## Next implementation gate

Phase 3 is complete with the reviewed T4 parity, tolerance-envelope and dense
workspace evidence above. Phase 4 GPU paged KV cache remains current: pool
construction and basic allocator operations have small-pool T4 storage/reuse
evidence at `f72f39c`. Newer P4.3 capacity accounting and P4.4 failure safety have
CPU evidence and require their own attributable T4 run. The next implementation
task is P4.5 allocator boundary coverage, followed by P4.6 property tests. The
collector required-test gap also remains open. Phase 4 remains unsatisfied;
later paged/batched and service
memory gates also remain open. Continue allocator ->
paged attention -> lifecycle -> HTTP.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending.
The [public validation plan](VALIDATION_PLAN.md#build-order) defines gate ownership;
the ignored local checklist is not the only public definition of a gate.
