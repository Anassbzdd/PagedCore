# Single-scalar T4 control review

Reviewed November 14, 2026. **The intervention is valid and improves some logit
metrics slightly, but does not resolve the failure. Fresh rounding differences
arise later; the next observed FP16 disagreement agrees with PagedCore under
independent FP64. No serving repair is selected and P5.3 remains open.**

Retained [raw archive](../../results/validation/2026-11-14/pagedcore-scalar-control-vn_hab4p.zip)
and [structured audit](../../results/validation/2026-11-14/scalar-control-t4-review.json).
ZIP SHA-256: `45ca601c47c78cf0910a99596590faf79ad35cbd668ff62d6d0d1da649c373d3`.

## Integrity and control validity

All 68 evidence hashes and 108 source hashes verify with complete manifest
coverage. Snapshot
`89082244cbcad581575f98a50a956445533112f45d8cbc7d89635371b67838b4`
matches the prepared notebook. All working source files matched before these
review writes; source-after hashes are unchanged. Base commit plus bundled
working files identifies the source, not a clean-commit claim. Target is one
CUDA-visible Tesla T4, capability 7.5, Torch 2.7.1+cu126, with the frozen checkpoint
and strict math settings. No additional local GPU execution occurred.

All eleven command records exit zero. The reproduction cell was executed twice;
its logs/raw files retain the final execution. Those duplicate command entries
are not independent robustness trials.

Both arms are complete. All 880 stage comparisons and twelve logit metric fields
were independently recomputed from safely loaded raw tensors. The sham exactly
reproduces the original paged logits and all 440 retained stage comparisons.
Both contiguous reference forwards equal the original reference logits. All 34
observed stages before the target output match between the arms.

The only changed target element is `[0, 21, 0, 60]` in layer 1, from
`-0.0028839111328125` to `-0.0028858184814453125`. Its complete attention output
and layer hidden output then equal reference exactly. Prefill, selected token,
absolute positions, old history, pool pointer, block table, ownership,
reservations, active slots, local appended projections and zero cleanup checks
all pass. This establishes the intended local intervention on this one case.

## Numerical effect

A tolerance ratio above one fails the unchanged comparison.

| Metric | Sham / real baseline | One-scalar intervention |
|---|---:|---:|
| Failing logits / 32,000 | 13 | 12 |
| Logit maximum absolute error | 0.015625 | 0.013671875 |
| Logit RMS error | 0.002958351 | 0.002719635 |
| Logit maximum tolerance ratio | 1.303875 | 1.265566 |
| K/V maximum tolerance ratio | 1.418356 | 1.455849 |
| Greedy choice equals reference | Yes | Yes |

Logit RMS error falls 8.07%, maximum absolute error falls 12.5%, and maximum logit
ratio falls 2.94%. K/V ratio increases 2.64%. This is not a uniform improvement,
and neither arm satisfies integrated acceptance. A reference-scalar override
would not count as a serving implementation even if that arm passed.

## First remaining difference

With the intervention, layer outputs 0, 1 and 2 are exact. The next FP16 attention
difference appears in layer 3 (the fourth decoder layer), head 26, channel 44.
Q/K/V, QK products and probabilities are still identical at that call. The
weighted-value FP32 calculation again places the outputs on opposite sides of
an FP16 rounding midpoint.

| Quantity at the new coordinate | Value |
|---|---:|
| Paged FP32 output | 0.00009813907672651112 |
| Contiguous FP32 output | 0.00009813872748054564 |
| Independent FP64 using the same probabilities and V | 0.00009813907978717396 |
| Paged FP16 output | 0.00009816884994506836 |
| Contiguous FP16 output | 0.00009810924530029297 |

Direct rounding of both the fixed-P/V FP64 sum and full FP64 attention chooses
the **paged** FP16 result here. The earlier first scalar favoured the contiguous
result; this new coordinate favours paged. Neither path is a universal mathematical
oracle. Copying the next reference scalar would improve agreement at that element
while choosing the less accurate rounded value for its captured operands.

The layer-3 attention disagreement is itself rounded away later in that layer:
its final hidden output is exact. Layer 4 also ends with exact hidden output.
The first remaining hidden-output difference is at layer 5. Thus rounding can
erase differences as well as preserve them. The layer-3 scalar alone is not
established as the cause of the final remaining error.

## Independent attention check on existing captures

Using CPU FP64 head-by-head mathematics on each captured call's own Q/K/V, all
88 recorded native FP16 attention outputs satisfy the existing `atol=rtol=2e-3`
component bound. Maximum measured ratio is **0.104010**; maximum FP32 attention
error is `2.35222e-7`. These calls contain 58 distinct Q/K/V operand sets and
are correlated observations from one prompt and cached step.

The native FP16 cast is taken from the captured T4 operation, before the one
diagnostic override. The overridden emitted scalar is excluded from native-output
claims. This is a mathematical audit of existing T4 captures, not new GPU
execution or common-input native replay across later paths with different
histories. It does not validate independent full-model implementation, held-out
model quality, other lengths, batching, fragmentation or the complete release suite.

## Decision

The first scalar causally affects this case's final result, but explains only
part of the observed disagreement. Later accurate attention computations can
still select different FP16 neighbors. This supports the accumulated-error and
numerical-contract review branch; it does not expose a cache indexing defect or
select a serving change.

Preserve normalize-first attention, the unmodified failing baseline and current
thresholds. Do not patch successive scalars or chase a reference's internal
addition order. A comparator redesign remains a separate decision requiring
broader common-input mathematics, independent model/component coverage and
held-out quality evidence under the numerical policy. Existing assertions and
P5.3 remain blocking until satisfied or formally replaced with justified evidence.

Review checks: complete ZIP/source integrity, 880 raw stage comparisons, all logit
metrics and failing-element counts, exact intervention/prefix controls, 88 native
attention FP64 checks and `git diff --check`. Runtime, tests and thresholds were
not changed; the CPU test suite and full T4 suite were not rerun for this review.
