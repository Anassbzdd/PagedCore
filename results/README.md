# Results policy and evidence index

Commit only reviewed, reproducible validation and benchmark evidence here.
Generated local runs belong under `results/local/`, which is ignored by Git.
There are no serving benchmarks or vLLM comparison results yet.

## Current evidence

| Scope | Retained evidence | Boundary |
|---|---|---|
| Standalone paged attention completion | [October 31 review](../docs/validation/2026-10-31-p52-t4-review.md), [structured audit](validation/2026-10-31/p52-t4-review.json) and both linked T4 archives | Clean `3401406`: each run passes 617 CPU / 15 GPU cases, zero failures/errors/skips; both explicit CUDA attention tests pass. Later staged-layer/batched decode changes are outside this evidence |
| Paged prefill | [October 30 review](../docs/validation/2026-10-30-p52-t4-review.md) and [T4 archive](validation/2026-10-30/t4-20261030T172045Z.zip) | Clean `922588b`: 609 CPU / 13 GPU passes, zero skips; pinned paged prefill passed on T4; this older run predates the explicit CUDA attention tests |
| Phase 4 allocator completion | [October 27 review](../docs/validation/2026-10-27-phase4-review.md) and [T4 archive](validation/2026-10-27/t4-20261027T200933Z.zip) | Clean `986392e`: 554 CPU / 12 GPU passes, zero skips; current allocator runtime, small-pool T4 accounting/failure/reuse checks, dense parity and workspace probes |
| Current working-tree CPU checks | [CPU report](validation/2026-10-27/cpu-review.json) and [JUnit](validation/2026-10-27/cpu-tests.xml) | Includes newer independent allocator property tests and collector guard regressions; no local GPU execution |
| Dense tolerance envelope | [October 20 review](validation/2026-10-20-t4/review.json) and [archive](validation/2026-10-20-t4/pagedcore-p36-t4.zip) | Original Phase 3 tolerance gate; later archives repeat these checks |
| Dense maximum-path workspace | [October 21 review](validation/2026-10-21-t4/review.json) and [archive](validation/2026-10-21-t4/pagedcore-p37-t4.zip) | Original no-pool baseline; no paged/batched/service capacity claim |

The dated CPU reports and earlier T4 archives remain as historical evidence.
In particular, the [October 26 review](../docs/validation/2026-10-26-p42-t4-review.md)
applies to `f72f39c`; its then-open allocator/collector follow-ups are resolved by
the current evidence above. Older reports describe their recorded source and
date, not the current implementation. Keep original reports and failed-run
records intact so claims can be audited.

## Retaining a new run

Use `bash scripts/validate_t4.sh` on a clean Linux/T4 checkout. The collector
retains setup/commands, source/lock identity, selected-device details, CPU/GPU
reports, dense workspace probes and reference captures. It currently requires
15 GPU cases, including both KV tests, paged prefill and two CUDA attention tests.
Review actual executed/skipped counts
and source identity before copying a bundle from `results/local/` into
`results/validation/`. Extend the required suite when new GPU surfaces exist.

Logs, metrics and benchmark records must never contain prompt text, generated
text or token IDs. Public correctness-reference artifacts may retain token IDs,
tensors and decoded text from the checked-in public fixtures, as required by the
[oracle contract](../docs/VALIDATION_PLAN.md#oracle-harness). Never publish private
request content or credentials. Retain actual commands, source/lock identity,
device/configuration details, test counts and limitations with each artifact.
