# Implementation and evidence status

Updated November 20, 2026. **The current gate is P5.6, end-to-end paged
generation.** P5.3 batched one-token decode and P5.4 guarded page traversal
passed the frozen v3 suite. P5.7 workspace and CUDA page-reuse ordering follow
generation.

## Current evidence

| Capability | Evidence |
|---|---|
| Owned TinyLlama weights, tokenizer and dense decoder | CPU loader, component and cache checks. Historical dense T4 results apply to the recorded revisions. |
| Reservation-aware KV allocator | [Phase 4 T4 review](validation/2026-10-27-phase4-review.md) at clean commit `986392e`, plus storage, ownership and cleanup checks in the November 16 run. |
| Paged prefill and one-token batched decode | [November 16 T4 review](validation/2026-11-16-v3-complete-t4-review.md). 830 CPU tests and ten required GPU tests passed. |
| Page traversal and numerical comparison | All 45 logit and KV comparisons and all 45 greedy choices passed on common reference-token histories. The suite checked shuffled pages, poisoned unrelated pages, rollback/retry, stable storage and exact cached history. |

The [raw ZIP](../results/validation/2026-11-16/pagedcore-accuracy-wt4lkx62.zip)
contains the source snapshot, test reports and raw tensors from one visible
Tesla T4 with Torch `2.7.1+cu126`. The review verified 124 source hashes and
110 evidence hashes. The corrected suite exercised actual 37/73/111/145-token
prefixes over three cached steps. The [v3 tolerance decision](validation/2026-11-16-integrated-tolerance.md)
sets integrated logit tolerances at 0.02 and accumulated KV tolerances at 0.01,
with independent attention at 0.002 and exact greedy and storage checks.

The November 20 cleanup checked the notebook builder on CPU and clarified
the reproduction instructions. The November 16 run remains the latest T4
paging evidence. See [retaining a new run](../results/README.md#retaining-a-new-run)
for the exact replay notebook and the command to package changed source.

## Next implementation gate

- **P5.6** compares full paged and contiguous-reference generation for single
  and multiple requests, reused pages, end-of-sequence stopping and the output
  limit.
- **P5.7** measures maximum-active decode workspace and long prefill pressure on
  T4. It also verifies CUDA stream ordering before pages are reused.
- The worker, FIFO admission, HTTP/SSE, independent model coverage, held-out
  quality, benchmarks and release remain open. [Validation plan](VALIDATION_PLAN.md#build-order)
  · [Acceptance traceability](TRACEABILITY.md) · [Numerical policy](NUMERICAL_POLICY.md)

## Saved T4 observations

- The [first v3 run](validation/2026-11-16-v3-accuracy-t4-review.md) exposed
  truncated long prompts. Its 37 retained comparisons passed, and the corrected
  run completed all 45.
- FlashInfer remains an [optional experiment](validation/2026-11-14-flashinfer-backend.md).
  Its [T4 run](validation/2026-11-14-flashinfer-toolchain-t4-review.md) stopped
  at native linking under NVCC 13.0. The compiler-only replay still needs T4
  execution.
- The [16-token trace](validation/2026-11-13-sixteen-token-trace-t4-review.md),
  [scalar control](validation/2026-11-14-scalar-control-t4-review.md) and
  [checkpoint trajectory](validation/2026-11-14-checkpoint-trajectory-t4-review.md)
  document the v2 rounding investigations. Their original failures retain
  their original conclusions.
- Earlier validation and diagnostic results are indexed in
  [results/README.md](../results/README.md) and the
  [historical status snapshot](validation/2026-11-11-status-before-scope-revision.md).
