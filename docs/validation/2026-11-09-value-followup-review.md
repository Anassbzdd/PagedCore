# Corrected value-candidate T4 follow-up

Reviewed November 9, 2026. The [structured review](../../results/validation/2026-11-09/value-followup-review.json)
was recomputed from `results/zip/value-followup-c9r490zr.zip`, SHA-256
`88d990dcd75a101903000ddddf7fc90deccafc9f91c4f70d11ce1bac6ffbd7b6`.
The raw archive is retained locally in the ignored `results/zip/` directory;
it is not distributed with this review in Git.

All 1,610 evidence-file checksums and all 72 retained source-file hashes match.
The source base is `7d9ea61ed81b6028980373860f9a2967935a9683` plus the bundled
working source, not a clean-current-commit claim. The unavailable historical
synthetic ZIP is named with its original hash; no runtime source is omitted.
The archived runner includes the exact rank-one half-split helper, verified
equal to the local corrected helper. Its SHA-256 is
`0c3bad6b02cd27dbd4b4d64b5e271bfd0c2fff92cc3cbd78425735affa75f886`.

Each arm/case runs in a fresh process on one visible T4 with the pinned eager
checkpoint, Torch `2.7.1+cu126`, Transformers `4.52.4`, seed zero, TF32 and
reduced-precision accumulation disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
The environment inventory lists two physical T4s; the runner selects one.
The serving reducer and score path are unchanged. This is diagnostic evidence.

## Completed comparisons

| Arm | Case | Recorded request-step comparisons | Numerical failures | Collection outcome |
|---|---|---:|---:|---|
| Baseline | Single | 3/3 | 2 | Complete |
| Baseline | Mixed | 15/15 | 11 | Complete |
| Baseline | Fragmented | 15/15 | 11 | Complete |
| Candidate | Single | 3/3 | 0 | Complete |
| Candidate | Mixed | 10/15 | 3 | Aborted in step 3: unknown kernel family |
| Candidate | Fragmented | 10/15 | 3 | Aborted in step 3: unknown kernel family |

Zero process exit means completed collection, not necessarily numerical
acceptance. The candidate mixed cases have no final `result.json`; their
ten recorded comparisons cannot stand in for the planned fifteen.
All completed next-token argmax comparisons match HF. Original tolerances
remain `atol=rtol=5e-3` for logits and `2e-3` for K/V.

## What the passing single case proves

The prompt-15 trajectory completes the forwards attending 16, 17 and 18 tokens
on T4. All final logits and all compared K/V elements match exactly at every
step, and the greedy checks, guard and final release accounting pass.
Dispatch counts are 22 native one-page, 22 alternating-sum and 22 half-split
calls. This closes the previous uncertainty about whether this narrow combined
value-candidate trajectory can complete and meet parity on the T4.

It is not exact attention at every intermediate boundary: at 18 tokens there
are three differing scaled-score elements, 37 FP32 probability differences,
one FP16 probability difference and one weighted-value output difference across
the 22 layers. The first is at layer 2. These differences do not change final
logits or cached K/V for this trajectory. The observed exact final tensors do
not imply universal bitwise equivalence of the attention implementation.

## Remaining numerical failures

The mixed workload includes histories different from the single case: a
17-token prompt followed by one decode step, and a 16-token prompt followed by
two steps, both attending 18 tokens. These cases fail logit tolerance with
maximum ratios `2.342147` and `2.465692`. A passing ratio is at most one.
Their Q/K/V are equal when scores first differ at layer 2. The first FP16
probability and weighted-value changes appear later, at layer 8 in the first
history and layer 13 in the second. Thus the early FP32 probability discrepancy
alone must not be described as immediately altering the FP16 value inputs.

In the first failing history, layer 8 still has identical Q/K/V but a score
difference changes one rounded probability and 32 attention-output elements.
Q/K/V differences begin at layer 9. Its subsequent 19-token forward first
differs in cached K/V at layer 9, so that later failure includes propagated
history drift; it is not independent proof of a defective 19-token reducer.
Do not attribute these different histories' failures to batch composition alone.

Across 433 completed candidate-layer comparisons with equal Q, K, V and FP16
probabilities, there are no weighted-output mismatches: 154 native one-page,
172 alternating and 107 half-split calls. Offline half-split replay on the two
failing histories' independent HF operands also matches every saved layer output.
This supports investigating scores next; it is not a general proof of reducer
equivalence across untested lengths or kernel families.

Other recorded candidate rows at 17, 50 and 51 tokens have exact final logits
and K/V. Some 50-token scores differ while FP16 probabilities and weighted
outputs remain exact. A local unequal value is not automatically a failed
integrated comparison.

## Collection and ownership limits

The candidate mixed and fragmented cases stop during step 3 at
`Candidate NOT exercised: length=18, family=unknown`. In that reference row,
the actual eager CUDA kernel lists are present, but the separate score/value
replay kernel lists are empty for all 22 layers. This remains a profiler
collection problem; the strict runner aborts rather than silently falling back.
Each partial mixed case records 67 native, 88 alternating and 66 half-split
calls, confirming that the candidate was genuinely exercised before the abort.

All six final cleanup records show zero owned blocks, reserved credits, active
slots and cached slots. Completed candidate calls no longer encounter the prior
MLP guard false positive. The fragmented case completes its step-2 rollback and
retry and has identical recorded numerical results to the unfragmented mixed
case. This does not establish the later maximum-active workspace or CUDA page
reuse ordering gates, or complete the aborted fragmented case.

## Current gate and next evidence

P5.3 remains open. The combined reducer is a test-only candidate; there is no
general serving repair or full T4 release acceptance in this archive. The next
numerical target is the paged QK score calculation on identical operands in the
two failing 18-token histories, especially the layers where FP16 probabilities
first change. Collect complete mixed/fragmented runs without dependence on a
missing replay profile, while recording the actual reducer used. Preserve the
page-table reads, one-page K/V scratch limit, tolerances and exact greedy checks.
Longer kernel families, end-to-end paged generation, workspace and safe page
reuse remain documented separate requirements.

This review ran archive/source SHA-256 verification, runtime-helper source
comparison, result/dispatch/cleanup aggregation and layer-by-layer analysis of
the retained reports. No runtime code changed and no new local model/GPU test
execution is claimed. The numerical observations above are backed by the
supplied T4 run rather than a local CPU inference run.
