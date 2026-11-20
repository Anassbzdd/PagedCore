# Sixteen-token T4 trace review

Reviewed November 13, 2026. **The first observed divergence is FP32 weighted-value
rounding on identical operands. No storage/indexing defect is exposed by this
case. Integrated logit and K/V acceptance still fail; P5.3 remains open.**

Retained [raw archive](../../results/validation/2026-11-13/pagedcore-layer-trace-iq8cr7k9.zip)
and [structured audit](../../results/validation/2026-11-13/sixteen-token-trace-t4-review.json).
ZIP SHA-256: `e76c3e2e08865816647cf043d13ce32444fc04b2d4970bc01cd245c492efd67f`.

## Attribution and controls

All 45 evidence hashes and 102 bundled source hashes verify, with complete
manifest coverage. Snapshot hash
`84f5b56dd182cbca97520ed213b779cd70d78666f4c381deff4c2d67786a0f71`
matches the prepared notebook. Every bundled working file matched the checkout
before this review's diagnostic correction and documentation edits. Base commit
`5ded0a31` plus the bundled working files identifies the source; this is not a
clean-commit claim. Source-after hashes are unchanged. All nine recorded commands
exit zero. These are diagnostic commands, not the full accuracy suite.

Target: one CUDA-visible Tesla T4, capability 7.5, Torch 2.7.1+cu126; frozen
Python 3.11.15, pinned checkpoint and strict math flags. Both raw logits exactly
reproduce the retained normalize-first failure. The run uses 15 IDs from the
original fixture prefix plus one cached step, attending to 16 tokens in one page.
Tracing leaves both paths' final logits exactly unchanged. All 440 stage
comparisons and all logit metrics were recomputed from safely loaded raw tensors.

## First differing operation

Layers below use zero-based numbering: layer 0 is the first decoder layer.

| Observation | Result |
|---|---|
| Layer 0 Q/K/V, cache reads, QK scores and probabilities | Exact between paths |
| First FP32 difference | Layer 0 `P @ V`, maximum `3.725290298461914e-9` |
| Layer 0 FP16 attention and layer output | Exact; the FP32 difference disappears on conversion |
| Layer 1 inputs, K/V, QK scores and probabilities | Still exact |
| First difference surviving FP16 | Layer 1, query head 21, channel 60; one element differs by `1.9073486328125e-6` |
| First later hidden/K/V input drift | Layer 2, following the layer-1 output difference |

Native replays are exact in all 20 recorded stage/output controls. The two paths
receive identical Q/K/V for the common-input experiments. Replaying `P @ V` with
identical probabilities and V reproduces the difference, isolating the weighted
value product. The captured paths have different product geometries; the internal
GPU reduction sequence was not profiled or established.

At the first differing FP16 coordinate:

| Quantity | Value |
|---|---:|
| Paged FP32 result, exactly the FP16 midpoint | -0.0028848648071289062 |
| Contiguous FP32 result, slightly below the midpoint | -0.00288486503995955 |
| Independent FP64 sum using the same P and V | -0.002884864910819074 |
| Paged FP16 result, tie rounded to the even neighbor | -0.0028839111328125 |
| Contiguous FP16 result | -0.0028858184814453125 |

Direct rounding of the FP64 sum selects the contiguous result at this coordinate.
Recomputing the whole attention operation in FP64 selects it here too. This is
a local observation, not proof that the contiguous path is always closer to the
mathematical model. Both attention paths have tiny FP64 errors in the two captured
layers: maximum full-attention errors range from about `5e-9` to `1.6e-8`.

## Diagnostic correction: round FP64 directly

The archived `fp16_mismatches` metric used `expected.half()`. The local captured
scalar proves that this tensor conversion can round FP64 through FP32 first.
A value just below an FP16 midpoint becomes an exact FP32 midpoint and can then
choose the other FP16 neighbor. Consequently, four of the 32 archived rounded
mismatch-count fields are misleading, although the raw FP64 tensors, FP32/FP16
outputs and maximum/RMS errors remain valid.

For the layer-1 identical-P/V replay, the archived counts were paged 0 versus
contiguous 1. Correct direct binary64-to-binary16 conversion gives **paged 1
versus contiguous 0**, for both duplicate input-source arms. Full-attention
FP64 mismatch counts remain paged 2 versus contiguous 1 at layer 1. The two
input-source arms have identical values here and are not independent samples.

The diagnostic now uses direct binary16 packing for FP64 references. Two focused
regressions cover values on either side of the midpoint and the actual captured
scalar. No serving arithmetic or acceptance threshold changes. Original archived
metrics remain retained with the correction explicitly recorded in the audit.
All 128 original metric fields reproduce as originally calculated. Independent
FP64 recomputation across CPU platforms differs by at most `1.11e-16`; this does
not affect direct FP16 rounding or the conclusions. It is separate from the
unchanged integrated acceptance thresholds.

## Cache checks and remaining failure

All 44 K/V read comparisons are measured. Every past slot matches the reference
exactly and remains unchanged from prefill. Every appended slot matches its own
rotated K or unrotated V projection, absolute position 15 is correct, and captured
views share pool storage. The guarded decode does not reconstruct full KV history.
Cleanup returns owned blocks, reserved credits and active slots to zero.

Later appended K/V values differ between paths after hidden-state drift begins.
Their maximum tolerance ratio is **1.418355889**, with the first K failure at
layer 8. This is a measured failing comparison, not evidence that old cache
history was corrupted.

Thirteen of 32,000 logits exceed the unchanged bound. Maximum absolute error is
`0.015625`, maximum tolerance ratio **1.303875407**, and RMS error
`0.002958351227`. The greedy choice agrees in this one step. Hidden-output
differences grow from maximum `3.0518e-5` at layer 1 to `0.0078125` at layer 21.
The trace shows propagation and later arithmetic differences; no counterfactual
intervention proves that the first single differing scalar alone causes the
entire final error.

## Decision and validation

This reaches the flowchart's accumulated-decoder-error branch. Preserve the
established normalize-first implementation and current integrated comparator.
The evidence does not select a serving repair, justify widening tolerances or
authorize closing P5.3. A comparator redesign still needs independent component,
broader common-input and held-out quality evidence under the numerical policy.
Do not rerun this unchanged case or resume reduction-order sweeps expecting a fix.

Local work: ZIP/source integrity, complete raw comparison/metric recomputation,
independent FP64 and direct-rounding analysis. The diagnostic-only fix passes
Ruff and [18 focused CPU tests](../../results/validation/2026-11-13/sixteen-token-rounding-cpu.xml),
including both one-token and fifteen-token prompt traces. T4 execution is the
returned archived run; no new local GPU run or complete accuracy rerun occurred.
