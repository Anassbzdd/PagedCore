# Implementation and evidence status

Snapshot: 2026-11-06. **Phase 4 GPU paged KV cache is complete at the allocator
boundary. The current gate is Phase 5: block-table decode attention.**
Standalone single-query paged attention has reviewed T4 evidence at `3401406`.
The newer batched one-token paged decode and staged attention reads have
synthetic CPU coverage. The first integrated T4 decode comparisons fail their
numerical tolerances; see the [November 1 failure review](validation/2026-11-01-p53-t4-failure.md).
T4 integrated paged-decoder acceptance, continuous
batching and the service remain outstanding.

## Current gate

The [November 4 T4 review](validation/2026-11-04-p53-t4-value-reduction-review.md)
verifies the supplied [diagnostic ZIP](../results/validation/2026-11-04/p53-diagnostics.zip)
at clean `fcc3639`, with all 38 recorded source hashes matching Git blobs.
At the single-request 17-token boundary, scores, both probability dtypes and
control K/V match HF exactly, but the striped value result differs at layer 1
even with HF probabilities forced. The experimental page-product reducer delays
the first local difference to layer 4 but still fails parity: maximum integrated
ratios are `3.502409` (striped) and `3.695339` (candidate). Both candidate T4 tests
now reach and fail logit comparison after the guard repair. This isolates the
first discrepancy to value arithmetic and final rounding. P5.3 remains open.

The [T4 scalar capture](../results/validation/2026-11-04/value-scalar-t4.json)
records the 17 probability/value operands at the first failing coordinates.
The [November 5 notebook results](../results/validation/2026-11-05/value-reduction-notebook-review.json)
show that two alternating FP32 sums match the 17-token single-request attention
replay. At 18 tokens, the first local difference moves to layer 0, head 2,
channel 39, with HF probabilities held fixed. Both two-lane variants fail the
single- and mixed-request T4 logit checks. The reference profiler records cuBLAS
GEMVX variants 5 and 6 at 17 and 18 tokens, respectively, and replaying the
captured operands with the original reference shapes reproduces both HF outputs.
The per-thread token grouping still needs verification across cache lengths.
These notebook results used runtime overrides for diagnosis. Serving uses the
four-lane reducer, and integrated T4 parity and workspace acceptance remain open.

The November 5 [integrated value probe](../scripts/diagnose_paged_decode.py)
compares the striped baseline with page-wise FP64 accumulation. If FP64 fails,
it tries the native HF value multiply where every owned V page matches the
reference, using the owned probabilities and recording fallback layers. The
guarded CPU regression reproduces the 17-token four-lane and 18-token two-lane
scalar errors on fragmented pages. FP64 matches both saved scalars, but its
pinned-checkpoint CPU diagnostic at `fa395d1` plus the probe changes fails
integrated tolerance (first logit failure at 17 tokens, maximum ratio
`2.679343`, all argmax comparisons equal) on Torch 2.7.1+cpu. The raw
`fp64-decode-probe-cpu.json` is retained under local results. On the same CPU
workload the native control matches every integrated comparison exactly, with
286 native layer calls and zero fallbacks. Its report is retained locally as
`native-value-probe-cpu.json`. The collector
now requires a [fragmented mixed decode case](../tests/gpu/test_decoder_parity.py)
that checks staged rollback and retry. That case's comparison helper passes
with the pinned CPU model. The non-GPU suite records 711 passes and 20 GPU cases
deselected. Ruff, formatting, strict mypy and collector syntax checks pass.
The [November 5 T4 probe review](validation/2026-11-05-p53-integrated-value-probe.md)
records all three integrated diagnostic arms. Page-wise FP64 fails T4 parity
with a worst integrated ratio of 6.974359. The native value control makes the
single-request 17-token case exact, but a mixed 18-token row first diverges in
scores with identical Q/K/V and native value arithmetic still active. That
row's logit ratio is 2.375193. The combined T4 control substitutes HF
probabilities in 11 layer calls with equal Q/K/V and uses the same native
value multiply. All 18 logit comparisons and all 1,628 K/V comparisons
then match exactly, with zero fallbacks. The page-local T4 replay compares
396 layer calls on equal Q/K. The tested batch-32 FP16 layout leaves 17
rounded probability mismatches, equal to the baseline count. FP32 and FP64
leave 43 and 22, respectively, and introduce discrepancies in previously
exact layers. The next control should separate K head packing and logical
output width. P5.3 and the fragmented acceptance case remain open.

The [November 6 handoff](validation/2026-11-06-p53-handoff.md) reviews two
additional T4 probes on clean `fa395d1`. The score-layout replay finds the same
34 raw-dot/17 rounded-probability mismatches for native and packed page-local
products; both full-width reference-only products match exactly. Those full-width
scratch calculations are diagnostic only and cannot be used in serving because
decode must not gather full KV history. A separate actual-reducer control at 17
attended tokens records the first attention-output differences at layer 1
(`5.96e-8`) and layer 4 (`1.91e-6`) while reference operands still match. Its
integrated maximum logit tolerance ratio is `2.804499`; greedy tokens match.
This strengthens localization to value arithmetic but does not identify a
passing reducer or satisfy the T4 gate. No production arithmetic changed.

On November 6, the [focused 17-token regression](../tests/gpu/test_attention_cuda.py)
ran on a Tesla T4 at `6ede22e`. The serving reducer passed the page-table,
no-full-history and cache-accounting checks, then differed from the captured
HF output by one FP16 rounding step (`5.96e-8`). The same mismatch reproduces
in a CPU replay of those operands. A separate profile on clean `6ede22e`
reproduced all 22 layer records from the November 4 capture. The
[handoff](validation/2026-11-06-p53-handoff.md) records the baseline and next
probe. The [addition-order sweep](VALIDATION_PLAN.md) ran on the pinned
TinyLlama checkpoint on CPU. The four-lane model matched all 4,080 synthetic
trials at the sampled coordinate. The T4 sweep, a replacement reducer and
integrated T4 acceptance are pending.

The owned dense decoder, reservation-aware KV allocator and paged prefill exist.
The [October 31 review](validation/2026-10-31-p52-t4-review.md) verifies
[two retained T4 runs](../results/validation/2026-10-31/p52-t4-review.json)
at clean `34014063bcbe1d07f59b49284522d63fe5ebec6a`: each passes 617 CPU
and 15 GPU cases, with zero failures/errors/skips. Both required CUDA attention
tests and pinned paged prefill execute and pass. All 37 source hashes, the lock
and the logged collector/workspace-recorder hashes match the commit; formatting,
Ruff and strict mypy pass. P5.2 is complete at that revision. The current
attention/decoder/allocator contain later batched/staged-append changes, so
these archives do not establish T4 correctness for the newer integration.

The [November 1 failure/diagnostic bundle](../results/validation/2026-11-1/p53-debug-t4-20261101T135931Z.zip)
at clean `b2fd3b0` retains 649 passing CPU cases and 15 passing/two failing GPU
cases, with no skips. Both integrated paged-decode cases fail existing logit
tolerances. Controlled T4 comparisons isolate attention rounding and
batch-dependent numerical drift; none of the tested arithmetic alternatives
closes parity. See the [review](validation/2026-11-01-p53-t4-failure.md).
That investigation changed neither tolerances nor engine arithmetic. It predates
the newer local P5.4 guard and is not accepted Phase 5 completion evidence.

The November 1 numerical repair now uses two-pass eager rounding for integrated
attention and single-request operation shapes within the batch transaction.
The standalone online mode, pinned eager reference, fixtures and numerical
thresholds remain unchanged. See the [repair review](validation/2026-11-01-p53-numerical-repair.md)
for attributable CPU checks and operation-level diagnostics. The arithmetic
redesign has no passing integrated T4 acceptance evidence; P5.3 and the Phase 5
release gate stay open.

The [November 2 diagnostic investigation](validation/2026-11-02-p53-numerical-investigation.md)
reviews the supplied [T4 replay](../results/validation/2026-11-2/p53-diagnostic.json)
at clean `b3a0fe1`, with matching attention/decoder/script/lock hashes. The single
request is exact at 16 cached tokens. At 17, all isolated eager-attention scores
remain exact, but layer 1's attention output first differs by `5.96e-8`; integrated
logits exceed their unchanged tolerance (maximum ratio `1.977457`). CPU-shaped
value reduction versus CUDA/cuBLAS is the leading hypothesis; probability outputs
and identical-probability V reduction still need direct T4 comparison. The local
focused attention/decoder suite passes 133 cases on Torch 2.7.1+cpu. No runtime
fix was applied, and the diagnostic is not release evidence.

The November 2 follow-up implements probability capture, direct control-page
comparisons, fixed-HF-probability replay and an opt-in page-product accumulator.
The candidate's [real-checkpoint CPU replay](../results/validation/2026-11-2/p53-paged-value-cpu.json)
fails integrated tolerance (maximum ratio `2.660326`), despite exact probabilities
and control-page K/V. It is not accepted as a fix; the serving default remains
unchanged. The [CPU JUnit report](../results/validation/2026-11-2/p53-value-replay-cpu.xml)
records 684 passes and 19 deselected cases, with zero skips; Ruff, formatting,
strict mypy and diff checks pass. A subsequently supplied Kaggle test transcript
records both candidate T4 tests aborting in the storage guard; see below.
The original reducer still needs the new probability/value replay on T4 before
choosing another numerical change.

On November 2, the supplied [Kaggle transcript](../results/validation/2026-11-2/p53-candidate-t4-guard-failure.txt)
shows both candidate cases stopping before logit comparison: the guard propagates
KV tracking through `baddbmm` and incorrectly rejects the ordinary 5,632-element
MLP output. A CPU regression reproduces the false positive before the fix and
passes after recognizing `baddbmm` as a reduction, alongside existing matmuls.
The one-page storage check remains in force. This repairs test instrumentation,
not numerical parity; candidate T4 comparisons still need a rerun.
The corrected guard passes [685 CPU tests](../results/validation/2026-11-2/p53-guard-baddbmm-cpu.xml),
with 19 cases deselected and zero skips; Ruff, formatting, strict mypy and diff
checks pass. This is CPU evidence for the instrumentation fix only.

| Surface | Implemented and observed | Evidence boundary |
|---|---|---|
| Environment and reference oracle | Frozen CPU/CUDA dependencies, pinned model/tokenizer, safe diagnostics, deterministic controls, reference capture/comparison | October 31 first run creates a fresh project environment in Kaggle; second synchronizes it. Selected-device/source attribution retained; no clean-host or service-install claim |
| Engine records and timing analysis | Immutable request/lifecycle/capacity records, synthetic timestamp analysis and public fixtures | CPU contract tests; worker instrumentation and external benchmark client remain planned |
| Owned dense decoder | FP16 weights, Llama components, contiguous prefill/decode and greedy generation | T4 all-layer comparisons, exact greedy fixtures and 2,260 zero-error tensor/logit records; dense single-sequence scope |
| Persistent KV pool and allocator | One FP16 pool, free list, immutable block tables, partial-block reads, per-request credits, capacity snapshots, ownership checks and release/reuse | T4 two-block storage, accounting, synchronous copy-failure recovery and stream rejection; CPU boundary/corruption/isolation tests |
| Prompt writes into paged storage | `decoder.prefill_paged` runs one owned prompt forward, then appends rotated K/unrotated V through the admitted request's allocator | CPU boundary/failure/retry checks plus October 30 pinned-model T4 logits/all-layer K/V comparisons at 1/15/16/17/49 tokens, shuffled/reused pages, isolation and release accounting |
| Single-query paged attention | `attention.paged_attention` reads valid allocator pages in logical order, maps eight query heads per KV head, accumulates scores/online softmax in FP32 and returns FP16 output | 25 CPU cases plus two explicit CUDA cases pass in both October 31 runs at `3401406`: boundary/maximum lengths, shuffled/poisoned pages, head mapping, large scores, autocast, read-only accounting and final release. Later staged-layer iterator changes remain CPU-only |
| Independent allocator model | Random create/append/release sequences, separate expected token history and credit/ownership model | Four CPU cases for 1/2/4/8-block pools, 50 examples each, up to 40 operations per example; includes failure injection and zero final ownership |
| Evidence collector | Requires both KV GPU tests, pinned paged prefill and two explicit CUDA attention tests; rejects missing/skipped/failed/errored cases | CPU regression coverage; both October 31 archives include all 15 required passing GPU cases and complete logs/reports |
| Batched one-token paged decode | `decoder.decode_paged` uses single-request arithmetic shapes inside each layer, eager-rounding block-table attention, per-request positions/valid lengths and one staged commit for all rows | CPU comparisons against separate dense decode calls, exact batch-composition/order checks, layer K/V at `2e-3`, page boundaries, unchanged history, input rejection and synchronous failure/retry including a later row's logits. New T4 decoder parity and workspace evidence remain pending |
| No dense-history decode guard | Shared test instrumentation follows K/V storage aliases and copies, rejecting temporary storage larger than one page during actual decode | CPU boundary/mixed/maximum-context checks and eight injected K/V shortcut cases pass; the same guard wraps T4 decode cases; the supplied candidate run stopped at a guard false positive, corrected locally, so T4 parity still requires a rerun |
| Worker and HTTP/SSE | Planned | No continuous batching, request cancellation or service validation yet |
| Container, benchmark and technical report | Planned | No serving performance or vLLM comparison results |

Direct allocator double release raises `KeyError`. The future worker must make
duplicate cleanup signals harmless by recording successful cleanup. Synchronous
copy-failure recovery does not establish recovery from asynchronous CUDA/context
failure. These lifecycle requirements remain in Phases 5–8.

## Current CPU evidence

The 2026-11-01 numerical repair passes **674 CPU tests, 17 GPU cases deselected,
zero failures/errors/skips**, plus Ruff, formatting and strict mypy. The
[repair review](validation/2026-11-01-p53-numerical-repair.md) records commands,
regressions and [retained source identity](../results/validation/2026-11-01/p53-cpu-repair-summary.json).
The pinned CPU replay is exact for logits, all-layer K/V and greedy argmax in
18 single/mixed row-step comparisons (three steps, reordered rows). This is
CPU-only evidence; no new T4 acceptance or throughput claim is made.

On 2026-11-01, P5.4 adds the [shared storage guard](../tests/conftest.py) to
existing [CPU decode tests](../tests/unit/test_decoder.py) and required
[T4 decode cases](../tests/gpu/test_decoder_parity.py). Eight injected shortcuts
verify rejection of concatenation (including converted/flattened pages),
stacking, indexed gathering and copying into a preallocated full-history buffer,
for K and V separately. Pool views and one-page FP32 copies remain permitted;
reference and prefill calculations run outside the guard. This is CPU evidence;
the guard and integrated decoder still require attributable T4 execution.
The working-tree run passes **658 CPU tests, with 17 GPU cases deselected and
zero skips** on Torch 2.7.1+cpu. The local/ignored report is
`results/local/p54-cpu.xml`, from
`uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p54-cpu.xml`
with `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
`uv run --no-sync python -m ruff check .`,
`uv run --no-sync python -m ruff format --check .`,
`uv run --no-sync python -m mypy --strict src/pagedcore`, GPU collection (17
cases) and `git diff --check` pass. No T4 execution is claimed.

On 2026-11-01, validation preparation adds two pinned-checkpoint T4 cases in
[`test_decoder_parity.py`](../tests/gpu/test_decoder_parity.py): single-request
and mixed-length batched paged decode, each with three cached steps, reordered
rows, logits/all-layer K/V and ownership checks. The collector requires both
cases, with CPU regressions rejecting missing or unsuccessful results. These
tests subsequently failed in the November 1 T4 run reviewed above; the prior
15-case archives cannot prove their results.
Local validation passes 649 CPU tests, with 17 GPU cases deselected and zero
skips (`results/local/p53-validation-prep-cpu.xml`, local/ignored). Ruff format
and lint, strict mypy, collector Bash syntax, GPU collection and
`git diff --check` pass; the collector regressions account for 77 CPU cases.
P5.3 remains pending T4 execution and retained archive review. Batched workspace
and CUDA page-reuse ordering remain separate P5.7 requirements.

On 2026-10-31, the local fix on top of `9fa2f2f` restores the missing
`KVAllocator.iter_layer_blocks` body and corrects its formatting. Before the fix,
the existing staged-layer visibility and per-page worker ownership tests both
failed because the method returned `None`. After the fix, **641 CPU tests pass,
15 GPU cases are deselected, and zero tests skip**; the local/ignored JUnit report
is `results/local/p53-iterator-fix-cpu.xml`. Ruff formatting/lint, strict mypy and
`git diff --check` pass. This is CPU-only evidence; the corrected iterator and
integrated batched decoder still require T4 validation.

The 2026-10-30 local working-tree run passed **641 CPU tests, with 15 GPU cases
deselected and zero skips**. This includes 17 new
[batched-decode cases](../tests/unit/test_decoder.py) and seven new
[allocator staging/iterator cases](../tests/unit/test_kv_cache.py), plus the
existing paged-prefill, paged-attention and ownership coverage. Commands were
`uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p53-cpu.xml`
with `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`,
`uv run --no-sync python -m ruff check .`,
`uv run --no-sync python -m ruff format --check .`,
`uv run --no-sync python -m mypy --strict src/pagedcore` and
`git diff --check`; all passed. The JUnit report
is local/ignored. This local run used Torch 2.7.1+cpu, not CUDA; its GPU cases
were deselected. The October 31 T4 evidence above applies to the earlier commit.
All 190 local Markdown links in the affected project documents resolve. The suite
includes 69 collector regressions; all GPU cases were deselected locally.

The [October 27 CPU review report](../results/validation/2026-10-27/cpu-review.json)
and [JUnit report](../results/validation/2026-10-27/cpu-tests.xml) record the
then-current working-tree checks separately from the T4 archive: **567 passed, 12 GPU
cases deselected, zero skips**. Ruff, formatting, strict mypy and lock consistency
also passed. The suite includes
the four allocator property cases and 57 collector cases. It uses the cached
pinned tokenizer with `HF_HUB_OFFLINE=1`; no local GPU execution is claimed.

The CPU CI [workflow](../.github/workflows/ci.yml) checks frozen setup,
format/lint/types, CPU tests, package build and installed-wheel diagnostics.
Current hosted CI was not rechecked during this review. Local checks and archived
T4 checks do not imply current hosted CI success.

## Saved T4 observations

The October 27 archive identifies one CUDA-visible Tesla T4 (`cuda:0`), UUID
`31d9002c-2d65-9f44-5eed-53ea68290f64`, compute capability 7.5, Python 3.11.15,
Torch 2.7.1+cu126, Transformers 4.52.4, CUDA 12.6 and driver 580.178.04.
Both KV GPU cases explicitly executed and passed. They exercise a two-block pool,
including a failed V copy at a block boundary followed by retry, release and reuse.
HF repeatability matched across all 39 retained tensors; all six public owned
greedy cases passed. See the review for checksums, commands and source limits.

Both dense workspace modes contain nine passing runs and 27 samples each:

| Measurement | Dense only | Dense with unused 8,192 MiB reservation |
|---|---:|---:|
| Peak PyTorch allocated, MiB | 3,228.22 | 11,420.22 |
| Peak PyTorch reserved, MiB | 3,434 | 11,626 |
| Largest increment above run baseline, MiB | 1,128.05 | 1,128.05 |
| Minimum sampled device-free memory, MiB | 11,320.81 | 3,128.81 |

All samples record zero allocation retries/OOMs. After a first-use 32 MiB increase,
repeated cleanup returns to the pre-forward live allocation. This is
single-sequence dense pressure evidence; the reservation tensor is separate from
the tested allocator pool. `capacity_validated=false` remains correct.
The 8,192 MiB pool, 1,536 MiB margin and 32-active configuration remain provisional
until the [combined memory gates](VALIDATION_PLAN.md#combined-memory-validation-before-capacity-freeze)
pass on the actual paged/batched decoder and service.

## Next implementation gate

Continue Phase 5 with poisoned fragmentation and
T4 no-full-history-gather checks/parity, then batched workspace and safe page
reuse. Follow the
[public phase map](VALIDATION_PLAN.md#build-order) and required
[architecture](DESIGN.md#kv-layout-and-invariants).

P5.1 prompt-to-page writes and P5.2 standalone single-query online softmax have
retained T4 evidence. The [October 31 review](validation/2026-10-31-p52-t4-review.md)
closes P5.2 at `3401406`. P5.3 batched one-token decode is implemented with CPU coverage;
its T4 evidence remains pending. P5.4 no-full-history-gather instrumentation has
CPU coverage and is wired into the required T4 decode tests; T4 evidence remains
pending. P5.5 poisoned-fragmentation coverage is the next implementation task.
Paged generation and worker cleanup remain unimplemented.

Final MVP acceptance criteria in [TRACEABILITY.md](TRACEABILITY.md) remain open.
Phase 4 completion proves the allocator foundation; later gates must establish
paged attention, lifecycle cleanup, HTTP behavior, service memory safety,
deployment and reproducible measurements.

Earlier evidence is indexed under [results](../results/README.md). Historical
reports retain their original observations and limitations; superseded blockers
are not current status.
