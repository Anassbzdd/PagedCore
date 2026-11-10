# Two-token T4 layer-trace review

Reviewed November 10, 2026. Current gate: P5.3, still open.

## Evidence identity and checks

The owner returned `results/zip/pagedcore-layer-trace-1__el9f9.zip`.
The [structured review](../../results/validation/2026-11-10/layer-trace-t4-review.json)
retains its archive SHA-256, environment, first differing tensors and rounding
investigation. All 38 evidence checksums and 85 embedded source hashes match.
The source-after manifest is unchanged; all bundled file contents matched the
local checkout before this documentation update. The diagnostic hashes match.

The notebook executed its embedded snapshot
`5e54d25d6ab5419b3ddc19c9cf0b3720335ad62746a130112e0e94d6ea39daec`.
Its recorded base is `33f2ffca21bbd66b54c0e624c01c17d376e07f71` plus bundled
working files. This is the notebook's snapshot identity, not a clean-base-commit
claim; cloning a newer commit did not change the source embedded in the notebook.

All seven recorded setup/verification/trace commands exit zero. Execution uses
one visible Tesla T4 (capability 7.5), Torch 2.7.1+cu126, Transformers 4.52.4,
Python 3.11.15 and the pinned TinyLlama revision. Deterministic settings include
TF32/reduced-precision accumulation disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The trace completes all **418 comparisons across 22 layers**, with no missing
points. Independent CPU inspection of all 22 raw layer files using
`torch.load(..., weights_only=True)` reproduces every recorded comparison,
including original layout metadata, exactness, maximum errors and K/V slices.
Both traced final logits equal their uninstrumented baselines. Baseline tensors
also match the previous two-token T4 archive exactly; recomputed logit metrics
match. This is attributable T4 evidence reviewed locally, not a new local GPU run.

## First differences on T4

All layer numbers below are zero-based.

| Point | First differing layer | Maximum absolute difference | Interpretation |
| --- | ---: | ---: | --- |
| FP32 attention result before FP16 cast | 0 | 7.450580596923828e-9 | First recorded comparable value difference |
| FP16 attention output | 5 | 1.9073486328125e-6 | One element rounds differently |
| Layer output | 5 | 0.000244140625 | Difference survives residual/MLP operations |
| Incoming hidden state and current Q/K/V | 6 | Varies by tensor | Propagated divergence after layer 5 |
| Pool K/V compared with reference | 6 | K: 0.001953125; V: 0.000244140625 | New token differs; old cached token still matches |

At layer 0, hidden input, input normalization, projected Q/K/V, rotated Q/K,
unchanged V, both actual pool token slots, QK products and scaled scores all
match exactly. The first recorded FP32 mismatch is the **normalized attention
result immediately before the final FP16 cast**, not QK multiplication.
1,013 of 2,048 FP32 elements differ. The first coordinate `[0, 0, 0, 0]` is
paged `-0.0012318509398028255`, reference `-0.0012318510562181473`.
The maximum diagnostic tolerance ratio is only `3.480121130488116e-6`.

The paged FP32 result uses shape/stride `[4, 8, 64] / [512, 64, 1]`;
the reference uses `[1, 32, 1, 64] / [2048, 64, 64, 1]`. Both are contiguous,
offset zero, with 8,192-byte FP32 storage on `cuda:0`. Comparisons align their
logical head/channel order. Original layouts remain in every trace record.

In layers 0 through 4, the FP32 differences disappear on conversion to FP16:
all recorded FP16 component outputs, final layer outputs and subsequent layer
inputs match. Layer 5 still has identical inputs, Q/K/V, pool reads, QK products
and scores. At its attention output, one FP16 element diverges: head 0, channel 3.
The reference and paged outputs have identical FP16 shape and stride here.
Output projection, residual, post-attention norm, MLP and layer output then
diverge. Layer 6 starts with that changed hidden state; its Q/K/V differences
must not be treated as the original cause.

The CPU result in the [preparation report](2026-11-10-two-token-layer-trace.md)
started at the layer-0 QK product. That CPU localization does not apply to this
T4 run. The automatic QK-specific replay correctly reports `not_applicable`;
no native QK replay or packing experiment was executed by this T4 trace.

## Layer-5 FP16 rounding investigation

At `[batch=0, head=0, token=0, channel=3]`, the captured T4 values are:

| Quantity | Value |
| --- | ---: |
| Paged FP32 attention | -0.002963065868243575 |
| Reference FP32 attention | -0.0029630661010742188 |
| Midpoint between the two FP16 neighbors | -0.0029630661010742188 |
| Paged FP16 output | -0.0029621124267578125 |
| Reference FP16 output | -0.002964019775390625 |

The reference FP32 value lands exactly at the midpoint; the paged value is
`2.3283064365386963e-10` above it. The cast therefore produces adjacent FP16
values. This localizes the first persistent divergence to a rounding boundary.
It does not prove that this single element alone causes the final logit failure;
later attention operations can introduce additional differences.

The captured paths normalize in different places. The reference forms FP32
softmax probabilities before multiplying V. The online paged path multiplies
unnormalized exponential weights by V and divides the accumulated output by
their sum. Their FP32 rounding can differ even with identical scores and V.
The archive does not yet isolate the contributions of exponential evaluation,
normalization rounding and weighted-value accumulation on T4.

Two isolated **CPU FP64** replays of the same archived layer-5 operands show
why the reference should not automatically be treated as the exact mathematical
answer at this coordinate:

- Recomputing attention entirely in FP64 from the shared FP16 Q/K/V gives
  `-0.002963065840592333`, which rounds to the **paged** FP16 value.
- Holding the already-rounded FP32 scores fixed and computing softmax/PV in
  FP64 gives `-0.0029630662002208363`, which rounds to the **reference** value.

Thus score rounding as well as normalization/PV arithmetic matters near this
midpoint. These are local attention controls, not integrated FP64 decoder
acceptance or proof that either complete decoder is more accurate. They do not
authorize a tolerance change or select a serving repair.

## K/V, final outputs and next control

All 44 K/V tensor comparisons were actually measured, with separate old/new
slot metrics. The maximum K/V ratio is **1.9796380090497736**. Every old cached
token matches the reference across all 22 layers. Every appended slot matches
its own path's projected/rotated K or projected V exactly. The first K/V
disagreement with the reference is in the new slot at layer 6, after hidden-state
divergence. This supports propagated activation differences for this fixture,
not a corrupt copy at the initial pool read. It does not establish all paging
or fragmented-layout invariants beyond the exercised single-page case.

The earlier acceptance archive's initialized K/V zero remains unmeasured;
the complete coverage here is a separate measurement. Final logit ratio remains
**2.1704749959099088**, maximum absolute error **0.015625**, with exact greedy
agreement. Blocks, reserved credits, active slots and cached slots all return
to zero. Numerical acceptance still fails.

The next focused T4 control should hold layer-5 Q/K/V fixed and separate
normalized-probability rounding, weighted-value products and final division,
checking their FP32 and FP16 results against independent FP64 attention. Keep
the first FP32 difference at layer 0 in view as well. Do not substitute the CPU
QK finding or blindly force the paged result to the FP32 reference at this scalar.
No serving code, tests, tolerances or checklist status changed in this review.

Review checks: archive/source SHA-256 verification, source-after and local-file
identity, command exit status, complete raw-trace recomputation, baseline tensor
comparison, logit metric recomputation and the two isolated CPU FP64 replays.
`git diff --check` passes. No new GPU execution or full test-suite rerun was
needed for these documentation/evidence-only changes. **P5.3 remains open.**
