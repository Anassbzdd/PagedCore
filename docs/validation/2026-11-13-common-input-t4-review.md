# Common-input attention T4 replay review

Reviewed November 13, 2026. **Both archived attention paths pass the existing
attention tolerance on these 44 common-input cases. P5.3 remains open.** This
review changes no implementation, tolerance or integrated assertion.

## Identity and scope

Retained [raw archive](../../results/validation/2026-11-13/pagedcore-common-input-twucs6i0.zip)
and [structured audit](../../results/validation/2026-11-13/common-input-t4-review.json).
ZIP SHA-256: `d61f78fc7ae68bacfed444be4bb9791112db257dd63810cd9ef99ea0cb9044e1`.
All 55 evidence hashes and 97 embedded file hashes verify. The checksum manifest
covers every other evidence entry; source-after hashes equal the bundled identity.
The bundled source ZIP hash matches the collection record. The replay script
matches the delivered diagnostic, and its recorded hash matches the bundled file.
The input trace ZIP exactly matches the retained November 10 T4 trace.

All six commands exit zero. Target evidence identifies one CUDA-visible Tesla T4,
Torch 2.7.1+cu126 and CUDA 12.6. Strict math flags include disabled TF32 and reduced
precision reductions, deterministic algorithms, and CUBLAS workspace `:4096:8`.

**The bundle contains the older online normalize-last paged attention.** The
workspace already has separate uncommitted normalize-first changes described in
[the implementation study](2026-11-13-vllm-implementation.md). Their attention,
trace, diagnostic and test source bytes differ from the archive. Those changes
were preserved; this run does not validate them. Source hashes for both attention
versions and the differing paths are retained in the audit.

## Independent review

Safe CPU loading uses `torch.load(..., weights_only=True, map_location="cpu")`.
For each of 22 layers and each original path's input set:

- Verify saved Q/K/V exactly match the corresponding original trace inputs.
- Verify both matching native outputs, before and after FP16 conversion, exactly
  reproduce the original T4 tensors: **88 tensor-equality checks** in total.
- Independently compute dense FP64 QK/softmax/V attention. Its largest difference
  from the saved per-head FP64 calculation is `1.1102230246251565e-16`.
- Recompute all **616 metric fields** from saved tensors. Counts, maximum errors,
  tolerance ratios and booleans reproduce exactly. Three RMS fields differ in the
  last FP64 bit during local reduction/square-root evaluation; each is within
  eight FP64 machine epsilons relatively, with no absolute floor. For example,
  one archived RMS is `5.6027526354082125e-6`, locally `5.602752635408213e-6`.
  This metadata-verification bound does not alter any runtime acceptance threshold.

The script's completed assertions cover the no-full-history guard, unchanged pool
including unused poisoned pages, storage pointer, block table, length, accounting
and zero ownership after cleanup. History length is always two, within one page;
this does not add multi-page or rollback/retry coverage.

## Results

Every row feeds identical selected Q/K/V into the actual contiguous FP32 and
actual paged implementation. Both are compared against the same independent FP64
answer. After layer 5, original paged-path and contiguous-path inputs differ;
comparisons across those different input sets are not isolated arithmetic controls.

| Metric, comparing each path with FP64 | Paged lower error | Contiguous lower error | Equal |
|---|---:|---:|---:|
| FP32 attention maximum absolute error | 24 | 12 | 8 |
| FP32 attention RMS error | 27 | 17 | 0 |
| Number of disagreements with rounded FP64 output | 7 | 3 | 34 |
| FP16 output maximum absolute error vs unrounded FP64 | 0 | 0 | 44 |

Summed rounded FP16 disagreements are 30 for paged and 34 for contiguous. These
are counts across correlated layer/input sets, including duplicate inputs in the
early layers, not independent trials, a quality score or a general superiority claim.

Both paths meet the existing attention `atol=rtol=2e-3` in all 44 cases. Their worst
attention tolerance ratios are **0.027565 paged** and **0.054037 contiguous**;
one is the acceptance boundary. These are attention ratios, not final-logit ratios.

At layer 5, both original input sets are equal. Paged maximum FP32 error is
`9.92806495347942e-9`, versus contiguous `1.1790710102710378e-8`; paged has one
rounded FP16 disagreement with FP64, contiguous two. The earlier layer-5 finding
therefore reproduces in this all-layer common-input experiment.

## Interpretation and next step

On this captured two-token trajectory, both attention calculations are accurate
within the existing attention bound. Contiguous FP32 is not consistently closer
to FP64. These results support a rounding-sensitivity explanation for the observed
attention disagreement; they do not prove that all integrated logit error is harmless.

The original integrated logit/KV failures remain retained and blocking. There is
no new end-to-end decoder run here, no wider-history measurement, and no model
quality result. Neither matching native replay nor passing these local attention
checks closes P5.3 or justifies widening its thresholds.

For the separately prepared normalize-first implementation, follow the focused
T4 attention checks and integrated accuracy run in its implementation study.
The notebook used for that run must embed the changed source; an old embedded
snapshot still runs the old implementation. Retain that run's failures as well.
No additional rounding-order sweep follows from this review.

Local work was archive/source verification, tensor/FP64 metric recomputation,
documentation links and `git diff --check`. No new local GPU execution occurred.
The runtime, tests and existing uncommitted changes were not modified, so no new
CPU suite or packaging run was needed for this evidence-only update.
