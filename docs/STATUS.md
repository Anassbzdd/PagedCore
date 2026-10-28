# Implementation and evidence status

Snapshot: 2026-10-27. **Phase 4 GPU paged KV cache is complete at the allocator
boundary. The current gate is Phase 5: block-table decode attention.**
The service, continuous batching and paged attention are not implemented yet.

## Current gate

The owned dense decoder and reservation-aware KV allocator exist. The
[October 27 T4 review](validation/2026-10-27-phase4-review.md) verifies the
[retained archive](../results/validation/2026-10-27/t4-20261027T200933Z.zip)
at clean commit `986392e230078594714258b38786243290174fe6`: 554 CPU tests and
12 GPU tests passed, with zero skips, plus formatting, Ruff and strict mypy.
All 33 manifest source hashes and the lock match that commit. The current runtime
and GPU test match the archived source; the newer property tests and collector
required-test fix are separately CPU-validated.

| Surface | Implemented and observed | Evidence boundary |
|---|---|---|
| Environment and reference oracle | Frozen CPU/CUDA dependencies, pinned model/tokenizer, safe diagnostics, deterministic controls, reference capture/comparison | Fresh Linux/T4 setup and selected-device/source attribution retained in the October 27 archive |
| Engine records and timing analysis | Immutable request/lifecycle/capacity records, synthetic timestamp analysis and public fixtures | CPU contract tests; worker instrumentation and external benchmark client remain planned |
| Owned dense decoder | FP16 weights, Llama components, contiguous prefill/decode and greedy generation | T4 all-layer comparisons, exact greedy fixtures and 2,260 zero-error tensor/logit records; dense single-sequence scope |
| Persistent KV pool and allocator | One FP16 pool, free list, immutable block tables, partial-block reads, per-request credits, capacity snapshots, ownership checks and release/reuse | T4 two-block storage, accounting, synchronous copy-failure recovery and stream rejection; CPU boundary/corruption/isolation tests |
| Independent allocator model | Random create/append/release sequences, separate expected token history and credit/ownership model | Four CPU cases for 1/2/4/8-block pools, 50 examples each, up to 40 operations per example; includes failure injection and zero final ownership |
| Evidence collector | Requires both KV GPU test names and rejects missing/skipped/failed/errored cases | CPU regression coverage and successful export of a copy of the actual October 27 reports; the T4 run used the earlier collector |
| Paged attention, worker and HTTP/SSE | Planned | No integrated paged decoding, continuous batching, request cancellation or service validation yet |
| Container, benchmark and technical report | Planned | No serving performance or vLLM comparison results |

Direct allocator double release raises `KeyError`. The future worker must make
duplicate cleanup signals harmless by recording successful cleanup. Synchronous
copy-failure recovery does not establish recovery from asynchronous CUDA/context
failure. These lifecycle requirements remain in Phases 5–8.

## Current CPU evidence

The [October 27 CPU review report](../results/validation/2026-10-27/cpu-review.json)
and [JUnit report](../results/validation/2026-10-27/cpu-tests.xml) record the
current working-tree checks separately from the T4 archive: **567 passed, 12 GPU
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

Implement Phase 5 block-table decode attention: noncontiguous reads, FP32 online
softmax, poisoned fragmentation and no-full-history-gather checks, then T4 parity,
batched workspace and safe page reuse. Follow the
[public phase map](VALIDATION_PLAN.md#build-order) and required
[architecture](DESIGN.md#kv-layout-and-invariants).

Final MVP acceptance criteria in [TRACEABILITY.md](TRACEABILITY.md) remain open.
Phase 4 completion proves the allocator foundation; later gates must establish
paged attention, lifecycle cleanup, HTTP behavior, service memory safety,
deployment and reproducible measurements.

Earlier evidence is indexed under [results](../results/README.md). Historical
reports retain their original observations and limitations; superseded blockers
are not current status.
