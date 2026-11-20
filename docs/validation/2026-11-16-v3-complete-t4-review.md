# Batched paged decode: complete v3 T4 acceptance

Reviewed November 16, 2026. **P5.3 batched one-token decode and P5.4 guarded
block-table traversal are complete for the frozen v3 suite. All 45 comparisons,
830 CPU tests and ten required T4 tests pass, with zero failures/errors/skips.**

Retained [archive](../../results/validation/2026-11-16/pagedcore-accuracy-wt4lkx62.zip)
and [structured audit](../../results/validation/2026-11-16/v3-accuracy-complete-t4-review.json).
Archive SHA-256: `50dd25173b52c052252c4c0a195a05e1c24f0a80b113b5d97fc057e381948549`.

## Attribution and executed checks

All **110 evidence hashes and 124 source hashes** verify, with complete manifests
and no duplicate archive members. Snapshot SHA-256
`05cfd848527a91523ea537623b8f41ccb886861858a77fc31f99125ce4599423`
matches the delivered corrected notebook. All 124 bundled files match the working
tree before this review's documentation updates; all 15 runtime files are the
same implementation that ran in the earlier v3 archive. Source-after hashes are
unchanged. The base revision `5ded0a31f2182584827aecc7cfbc0d6b7da3868c`
plus the full snapshot identifies the dirty source; this is not a clean-commit claim.

The process sees one Tesla T4, capability 7.5, Torch `2.7.1+cu126`, CUDA runtime
12.6 and Python 3.11.15. Physical inventory lists two T4s; only device 0 is exposed.
The owned checkpoint/tokenizer smoke passes for the pinned TinyLlama revision
`af8e934848d8dd00074cc2cd8a40a9b05c3b011e`. Every integrated report has the frozen
deterministic/no-TF32/no-reduced-precision settings and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
The backend is PagedCore's default normalize-first implementation, not FlashInfer.

All twelve command records exit successfully: environment setup and dependency
recording, inventory, target check, owned checkpoint smoke, Ruff lint/format,
strict runtime mypy, CPU tests, GPU tests and accuracy export.

| Required execution | Passed | Failed / errors / skipped |
|---|---:|---:|
| CPU tests on Kaggle | 830 | 0 / 0 / 0 |
| Required T4 tests | 10 | 0 / 0 / 0 |
| Cached-decode step/request comparisons | 45 | 0 |

The ten T4 cases are CUDA RNG repeatability, owned paged prefill, two independent
FP64 attention cases, two KV allocator/storage cases and four integrated cases.
GPU test names exactly match the required set. The exported summary is `passed`;
its four case reports exactly match JUnit's recorded reports and pass the current
strict report validator.

## Independent raw-tensor audit

Safely load every saved logit pair and per-page K/V collection. Independently
recompute maximum error, RMS, normalized RMS, tolerance ratio, greedy match and
top-1 margin from all **45 logit pairs**: all **270 metric fields** agree, with
maximum last-bit discrepancy `8.68e-19`. Check finite FP16 K/V shapes and complete
logical-page/layer/kind coverage for each of the **45 K/V collections**. Compute
K/V ratios in FP64; their maximum discrepancy from the recorded FP32 ratios is
`3.53e-8`, with the same pass/fail decisions throughout.

| Case | Actual prompt lengths | Cached steps | Rows | Maximum logit ratio | Maximum K/V ratio |
|---|---|---:|---:|---:|---:|
| Single | 15 | 3 | 3 | 0.605478 | 0.544712 |
| Mixed | 1/15/16/17/49 | 3 | 15 | 0.689928 | 0.628801 |
| Fragmented | 1/15/16/17/49 | 3 | 15 | 0.689928 | 0.628801 |
| Long-layout (`held_out`) | 37/73/111/145 | 3 | 12 | 0.566254 | 0.742881 |

A tolerance ratio at most one passes. The logit maximum is **0.689928** and K/V
maximum is **0.742881**, below their limits. All **45 greedy choices match** on
the common reference-token histories. All 45 step/request identities are unique,
complete, and attend to their requested prompt length plus the cached-step count.
In particular, actual 73/111/145-token inputs execute; the fixture truncation
observed in the [previous run](2026-11-16-v3-accuracy-t4-review.md) is resolved.

## Gate decision and scope

The tested code checks stable pool storage, block tables and exact old cached
values, poisoned unrelated pages, accounting/reservations, staged rollback/retry
and zero cleanup. Actual serving decode executes under the no-full-history K/V
guard in every required integrated case. Together with CPU injected-shortcut
coverage and the passing independent attention tests, this closes P5.3 and P5.4
for the documented frozen suite. No runtime code or tolerance changes are made
during this review.

Acceptance uses the previously frozen `fp32-paged-v3-fp16-drift` policy:
`atol=rtol=0.02` for integrated logits and `0.01` for accumulated K/V drift;
independent attention remains `0.002`. The owner-authorized
[tolerance decision](2026-11-16-integrated-tolerance.md) remains explicit.
Historical failures remain failures under their original source and thresholds.
This pass does not claim the old v2 contract passed.

This is finite paging-integration acceptance. It does not establish arbitrary
long free-running generation, EOS/maximum-output behavior, whole-model quality,
universal token identity, maximum-active workspace, stream-ordered page reuse,
HTTP/lifecycle behavior or performance. The repeated public long-prefix fixture
is layout coverage, not held-out natural-language quality data. FlashInfer is
still an optional unvalidated alternative.

Next required work is **P5.6 end-to-end paged generation**, followed by **P5.7
batched workspace and CUDA ordering**. The broader Phase 5 and MVP release gates
remain open. No new local GPU execution was performed during this review.

Review checks: complete archive/source integrity, current-source identity,
checkpoint/math policy, all required test names and outcomes, all raw logit/KV
comparisons and versioned reports, documentation links and `git diff --check`.
