# Independent accuracy T4 archive review

Reviewed November 13, 2026. The owned-checkpoint manifest records capture on
November 12 at 22:40:50 UTC. **P5.3 remains open.** This review changes no serving
arithmetic, tolerances, assertions or acceptance rules.

## Attribution

Retained [raw archive](../../results/validation/2026-11-13/pagedcore-accuracy-zh7r_l2j.zip)
and [structured review](../../results/validation/2026-11-13/accuracy-t4-review.json).
Archive SHA-256: `dedb8bb881eb95e3352a2474e0b7bd71c884b77eef4ed01a6147ad871ebc9464`.

All 24 recorded evidence hashes and all 94 embedded source hashes verify. The
evidence checksum manifest covers every other evidence file; the source manifest
covers every other embedded source entry. The source-after manifest is unchanged.
Snapshot SHA-256 is
`a71446636bf2f997bdfd0917232a07c1297af8febf6d6a03540833ff7973b9f3`, matching
the notebook-run record. All bundled files matched the local checkout at `8e15e0b`
before this review's documentation writes. Recorded base `4d03d428` plus bundled
working files is source attribution, not proof of a clean base commit.

The run uses Python 3.11.15, Torch 2.7.1+cu126, Transformers 4.52.4 and the pinned
TinyLlama checkpoint. One Tesla T4 is CUDA-visible (`CUDA_VISIBLE_DEVICES=0`);
physical inventory reports two GPUs. Selected UUID matches inventory. The manifest
records strict math, disabled TF32/reduced-precision reductions, deterministic
algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

All setup, target, owned checkpoint/tokenizer smoke, lint, format, types and CPU
commands exit zero. GPU regressions exit one; accuracy export correctly exits one
and marks the run rejected. No required HF model comparison ran.

## Results verified from raw tensors

CPU: **764 passed, zero failures/skips**. GPU: **six passed, four failed, zero
skips/errors**. Passing GPU cases are RNG repeatability, owned paged prefill, both
independent-FP64 attention cases and both KV storage/allocator cases.

All four integrated cases stop at their first logit assertion. Each saved tensor
contains only paged and owned-reference logits. Safe CPU loading uses
`torch.load(..., weights_only=True, map_location="cpu")`. Recomputing all six logit
metrics in FP64 reproduces every reported metric exactly (24 metrics total).

| Case | Attended tokens | Recorded / required rows | Maximum tolerance ratio | Logits beyond tolerance / 32,000 |
|---|---:|---:|---:|---:|
| Single | 16 | 1 / 3 | 1.475121 | 12 |
| Mixed | 2 | 1 / 15 | 2.170475 | 181 |
| Fragmented | 2 | 1 / 15 | 2.170475 | 181 |
| Held-out lengths | 38 | 1 / 12 | 2.034720 | 41 |

All four recorded argmax choices match. This is four predictions, not complete
generation agreement. Overall maximum absolute logit error is 0.015625 in each
case. PyTorch's smaller greatest-absolute-difference message describes the subset
of failing elements; the saved metric covers all elements, including large logits
whose errors are allowed by relative tolerance. These values are consistent.

The K/V ratio zero in each row is its initialized placeholder. The source places
the failed logit assertion before K/V comparison, so these zeros are **not K/V
measurements**. Only 4 of 45 required integrated rows were recorded. The fragmented
case stops before its later rollback/retry injection. These reports cannot certify
complete integrated K/V, unchanged history or rollback/retry coverage. Separate
passing allocator tests retain their own narrower scope.

## What this adds to the investigation

Both full paged and reference tensors for all four saved rows are exactly equal
to the corresponding tensors in the earlier
`results/zip/pagedcore-accuracy-kavvwb5q.zip`. Mixed and fragmented tensors also
match each other exactly. This confirms repeatability after removing mandatory
HF collection, without establishing complete trajectories or whole-model accuracy.

The earlier [two-token layer trace](2026-11-10-layer-trace-t4-review.md) and
[layer-5 replay](2026-11-10-layer5-replay.md#returned-t4-replay) therefore remain
relevant to the reproduced two-token failure. They localized rounding sensitivity
but did not establish a complete independent replacement accuracy contract.
This archive contains no per-layer Q/K/V operands and adds no new cause localization.
Passing standalone FP64 attention checks does not prove integrated model accuracy.

## Bounded next investigation

Use the already retained two-token trace operands for all 22 layers. For each
layer, choose the original contiguous-path Q/K/V and original paged-path Q/K/V as
two explicitly labeled input sets (44 layer/input sets in total). Feed both FP32
attention schedules and independent FP64 attention exactly the same selected
operands. Compare full outputs before/after FP16 casting and report maximum/RMS
error and rounded mismatch counts. Require native T4 replay equality only for the
matching original path. After layer 5, the original paths have different operands;
comparing their native outputs directly does not isolate attention arithmetic.

This is a proposed focused diagnostic, not implemented or executed by this archive
review. It needs no new checkpoint capture for the retained two-token operands.
It cannot by itself close broader lengths, generation, quality or memory gates.
Use its result to justify a repair or a separately documented comparator redesign
under [the policy](../NUMERICAL_POLICY.md). Current assertions remain blocking;
do not widen thresholds or restart a kernel-rounding reproduction sweep.

## Local review checks

ZIP/source/manifest identity, full checksum coverage, XML/command outcomes, safe
raw tensor loading, all metric recomputations, comparisons with the retained old
archive, documentation links and `git diff --check`. These are CPU review of
user-supplied T4 evidence; no new local GPU execution occurred. Runtime code/tests
are unchanged, so the CPU suite and packaging were not rerun for this evidence-only
update. P5.3, integrated P5.4, P5.6 and P5.7 remain unsatisfied.
