# Implementation and evidence status

Snapshot: 2026-10-31. **Phase 4 GPU paged KV cache is complete at the allocator
boundary. The current gate is Phase 5: block-table decode attention.**
Standalone single-query paged attention has reviewed T4 evidence at `3401406`.
The newer batched one-token paged decode and staged attention reads have
synthetic CPU coverage. T4 integrated paged-decoder validation, continuous
batching and the service remain outstanding.

## Current gate

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
| Batched one-token paged decode | `decoder.decode_paged` batches projections/model layers, uses per-request absolute positions and valid page lengths, and commits staged K/V after the full forward succeeds | Synthetic CPU comparisons against separate dense decode calls; layer K/V, mixed lengths, reordered rows, page boundaries, input rejection and synchronous failure/retry coverage. No pinned-checkpoint/T4 decoder parity or workspace evidence |
| Worker and HTTP/SSE | Planned | No continuous batching, request cancellation or service validation yet |
| Container, benchmark and technical report | Planned | No serving performance or vLLM comparison results |

Direct allocator double release raises `KeyError`. The future worker must make
duplicate cleanup signals harmless by recording successful cleanup. Synchronous
copy-failure recovery does not establish recovery from asynchronous CUDA/context
failure. These lifecycle requirements remain in Phases 5–8.

## Current CPU evidence

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
no-full-history-gather checks, then T4 parity, batched workspace and safe page
reuse. Follow the
[public phase map](VALIDATION_PLAN.md#build-order) and required
[architecture](DESIGN.md#kv-layout-and-invariants).

P5.1 prompt-to-page writes and P5.2 standalone single-query online softmax have
retained T4 evidence. The [October 31 review](validation/2026-10-31-p52-t4-review.md)
closes P5.2 at `3401406`. P5.3 batched one-token decode is implemented with CPU coverage;
its T4 evidence remains pending. P5.4 no-full-history-gather instrumentation is
the next implementation task. Paged generation and worker cleanup remain unimplemented.

Final MVP acceptance criteria in [TRACEABILITY.md](TRACEABILITY.md) remain open.
Phase 4 completion proves the allocator foundation; later gates must establish
paged attention, lifecycle cleanup, HTTP behavior, service memory safety,
deployment and reproducible measurements.

Earlier evidence is indexed under [results](../results/README.md). Historical
reports retain their original observations and limitations; superseded blockers
are not current status.
