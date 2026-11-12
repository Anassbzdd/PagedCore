# Saved layer-5 attention replay

Prepared November 10, 2026. Current gate: P5.3, still open.

`scripts/replay_layer5.py` consumes the existing `raw/layer-05.pt` from the
T4 layer trace. It loads no checkpoint and executes no decoder trajectory.
Original operand shapes, strides and offsets are restored. Identical logical
scores and V are required before comparing the two layouts.

The arms compare reference and grouped softmax-before-PV, explicit exp/sum
normalization before PV, and the serving-style normalization after PV.
Additional controls promote only the final division, or both PV and division,
to FP64. Independent FP64 attention is measured both with the saved FP32 scores
held fixed and with scores recomputed from the shared FP16 Q/K/V. The report
retains full-output errors, FP16 mismatch counts and head-0/channel-3 values;
raw arm outputs are saved alongside it.

Both native FP32 arms must exactly reproduce the saved T4 precast outputs
before interpreting T4 controls. Failed replay retains evidence and exits
nonzero. A CPU smoke does not enforce equality with GPU outputs. The tested
CPU smoke on the saved operands completes all eight arms; both full-output
native matches are false on CPU, as explicitly recorded, so those results
cannot stand in for T4 execution. The FP64 Q/K/V control reproduces the prior
isolated scalar `-0.002963065840592333`.

Validation: CPU saved-operand smoke, Ruff lint/format, strict mypy. No full model
run or full pytest rerun was performed for this focused diagnostic preparation,
following the owner's request to keep this step short. The local report is
`results/local/layer5-replay-cpu/report.json`. Run on T4 with the existing frozen
trace environment, then retain the new script, input tensor, report and outputs.
No serving code, tolerance, acceptance test or numerical policy changes.

## Returned T4 replay

Reviewed November 10, 2026. The owner returned `layer5-replay.zip`; a copy is
retained at `results/zip/layer5-replay.zip`. The
[structured review](../../results/validation/2026-11-10/layer5-replay-t4-review.json)
records the archive and all six file hashes. Archived script bytes match the
local script (SHA-256 `6d4164782a6aee5da73218e7bf8349941eb1d22fd640c7698267e269e2d78d46`);
the input bytes match the previously reviewed T4 layer-5 capture
(`df137bdc4080e63f2fff7256550ebbe83d2ecb05bdd18ad21f0dc6ee75f73254`).
The runner reports T4, Torch 2.7.1+cu126, the strict math settings and exit zero.
This small replay ZIP reuses the prior trace environment and does not contain
a separate full GPU inventory or source snapshot.

Independent inspection of saved tensors verifies both native outputs exactly
reproduce the original T4 FP32 attention results. All five metrics for all eight
arms recompute exactly. All three normalize-first variants match the reference
**across the full FP32 output tensor**, not merely at head 0/channel 3. These
include the grouped layout and explicit exp/sum probabilities. Therefore changing
to the grouped matmul geometry alone does not change the reference result for
these held operands; moving normalization across the V product does.

| Arm | Maximum absolute error vs FP64 Q/K/V attention | FP16 elements differing from rounded FP64 |
| --- | ---: | ---: |
| Reference / grouped softmax first / explicit exp normalization first | 1.1790710102710378e-8 | 2 |
| Normalize last | 9.92806495347942e-9 | 1 |
| FP64 final division only | 1.04538559302092e-8 | 1 |
| FP64 PV and final division | 8.619940768117562e-9 | 1 |

Counts cover 2,048 output elements in this one saved attention call. Promoting
only final division, or both PV and division, preserves the paged FP16 value at
head 0/channel 3. Final division precision alone therefore does not explain the
reference/paged FP16 split there. Differences in normalization placement and
the rounded operands entering PV matter. This does not independently identify
every contribution from exponentials, probability rounding and product accumulation.

The full-FP64 Q/K/V result rounds to the paged scalar, while FP64 attention using
the already-rounded FP32 scores rounds to the reference scalar. In this saved
call, normalize-last has both a smaller maximum error and fewer rounded FP16
disagreements against the full FP64 attention result than normalize-first.
Forcing reference equality at this point would not be an accuracy improvement
under those two measured quantities. This does not establish accuracy for other
layers/histories or the full decoder, nor authorize a numerical-policy change.

Next evaluate both arithmetic schedules against FP64 on common Q/K/V inputs
across more captured layers/histories before selecting a serving repair.
After layer 5 the original paged and reference trajectories have different
operands, so a new isolated comparison must explicitly feed both schedules the
same chosen operands. Native-replay equality then applies only to its matching
original path. An integrated reference diagnosis remains separate.

Review checks were archive/script/input identity, safe raw tensor loading,
full-tensor native and normalize-first equality, all arm metric recomputations,
and `git diff --check`. No code changed or new local GPU execution occurred.
P5.3 stays open; the existing integrated logit ratio is still 2.1705, with no
new integrated run in this replay.
