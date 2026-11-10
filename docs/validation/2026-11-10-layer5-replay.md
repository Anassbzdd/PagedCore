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
