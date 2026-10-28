# Phase 4 completion review

Reviewed on 2026-10-27. Phase 4 is complete at the allocator boundary: the
reservation-aware runtime has attributable T4 storage/accounting/failure checks,
and independent CPU property tests cover ownership, isolation and cleanup.
The next gate is Phase 5 block-table decode attention. This review inspected
saved T4 evidence and ran CPU checks; no new T4 execution occurred locally.

## Archive and attribution

- [Archive](../../results/validation/2026-10-27/t4-20261027T200933Z.zip):
  SHA-256 `7b6355d91a25625bf37d4f72e29b43352962c60783f06ab1f5bd64432cd50401`.
- Collection window: 2026-10-27, 20:09:33–20:15:11 UTC.
- Clean revision: `986392e230078594714258b38786243290174fe6`.
- Lock SHA-256: `46700b60b0a8d492fc673aec8a0e12c27ae43124dd8e194f8c754dbd6ca7ea48`.
- ZIP CRC checks passed. All 33 manifest source hashes match Git blobs at that
  revision; the workspace reports carry the same provenance. Collector and
  workspace-recorder hashes in the command log match the committed scripts.
- The selected `cuda:0` is a Tesla T4, UUID
  `31d9002c-2d65-9f44-5eed-53ea68290f64`, compute capability 7.5,
  15,636,037,632 bytes reported by CUDA. Two physical T4s were listed; exactly
  one was CUDA-visible under `CUDA_VISIBLE_DEVICES=0`.
- Linux, Python 3.11.15, Torch 2.7.1+cu126, Transformers 4.52.4, CUDA 12.6 and
  NVIDIA driver 580.178.04. Model/tokenizer revision remains
  `af8e934848d8dd00074cc2cd8a40a9b05c3b011e`.

The bundle root is `t4-20261027T200933Z/`. It retains fresh frozen setup
(`uv sync --frozen --extra cuda --extra dev`), dependencies from `.venv/bin/python`,
command logs, clean Git status before/after collection, target/manifest records,
CPU/GPU JUnit, both workspace reports and diagnostics, tolerance measurements
and two HF captures with comparison. The command log supplies setup history
where the manifest's `setup_command` is null.

## Executed checks

Commands below are recorded in `commands.log`:

- `uv run --no-sync python -m ruff format --check .`, `ruff check .` and
  `mypy --strict src/pagedcore`: passed.
- `pytest -m "not gpu and not benchmark" -ra`: 566 collected, 554 selected and
  passed, 12 deselected, zero failures/errors/skips. Includes all 90 pool/allocator
  unit cases; independent property tests were added after this commit.
- `pytest -m gpu -v -ra`: 566 collected, 12 selected and passed, 554 deselected,
  zero failures/errors/skips.
- Both `test_kv_pool_storage_on_t4` and
  `test_kv_allocator_storage_and_stream_ownership_on_t4` explicitly passed.
  The two-block FP16 pool retains its backing storage. The allocator test checks
  reservation snapshots, a 16-token append, synchronous failure on the next V
  copy, unchanged published length/accounting, successful retry to 17 tokens,
  valid-slot views, rejection from another CUDA stream, strict double release,
  zero cleanup counts and reuse by another request.
- All-layer dense parity and all six public greedy cases passed. The
  `tolerance-envelope.json` records 2,260 tensor/logit comparisons with zero
  absolute error and zero tolerance ratio at unchanged thresholds.
- `pagedcore verify`, two `pagedcore oracle --case boundary_15` captures and their
  comparison passed. All 39 HF reference tensors match exactly.

The runtime, GPU test, lock and workspace recorder match the current source
(after normalizing Windows line endings). The current collector additionally
requires both KV tests; its new regression cases and the new independent
allocator property module were not part of the T4 commit. Their evidence is
[separately retained CPU validation](../../results/validation/2026-10-27/cpu-review.json),
with [JUnit](../../results/validation/2026-10-27/cpu-tests.xml). The property model
uses 1/2/4/8-block pools, 50 generated examples per case and up to 40 operations
per example; it independently checks credits, physical ownership, valid K/V
history, rejected operations, partial-copy failures and complete cleanup.

The archived collector predates its required-KV-test guard fix. Both tests did
execute in this run, so the old guard does not weaken this observed result.
During review, the current exporter accepted an extracted copy of the actual
reports. This was a CPU evidence check, not another GPU run. The retained ZIP
was not rewritten.

## Memory observations and limits

Both workspace modes pass all nine expected runs and 27 samples: direct
2,048-token prefill, 2,047+1 generation and 1,792+256 generation, three repeats
each. The maximum-output path performs 255 decode forwards per run.

| Measurement, MiB | Dense only | Dense with unused reservation |
|---|---:|---:|
| Peak PyTorch allocated | 3,228.22 | 11,420.22 |
| Peak PyTorch reserved | 3,434 | 11,626 |
| Largest increment above run baseline | 1,128.05 | 1,128.05 |
| Minimum boundary device-free sample | 11,320.81 | 3,128.81 |

The pressure tensor is exactly 8,589,934,592 bytes (8,192 MiB), held throughout.
The configured margin is 1,536 MiB. All samples record zero allocation retries
and OOMs. Cleanup retains 32 MiB on the first run and zero additional live bytes
on the remaining eight runs in each mode. Values are recalculated from raw bytes.

This is single-sequence dense measurement with an unused reservation, separate
from the actual two-block allocator test. The dense increment includes KV,
inputs, logits and temporary tensors. Boundary free-memory samples can miss
transients; PyTorch peaks exclude external allocations.
`capacity_validated=false` is correct. Integrated paged/batched and service
memory gates remain open.

## Gate decision

P4.1–P4.4 have CPU boundary/integrity tests and matching-runtime T4 storage,
reservation/accounting, retry/reuse and stream-ownership checks. P4.5 boundaries
and P4.6 independent state transitions/cleanup have CPU evidence. This closes
the allocator gate without implying worker cancellation, duplicate-signal
cleanup, asynchronous CUDA failure recovery, paged attention or service safety.
Those checks belong to Phases 5–8. The
[October 26 review](2026-10-26-p42-t4-review.md) remains historical evidence for
`f72f39c`; its allocator/collector follow-ups are now resolved within this scope.
