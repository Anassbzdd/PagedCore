# Checkpoint trajectory T4 review

Reviewed November 14, 2026. **The diagnostic completes all 32 cached steps and
402 real-weight component checks. Instrumentation preserves native logits.
All 32 integrated logit/KV comparisons still fail; P5.3 remains open.**

Retained [raw ZIP](../../results/validation/2026-11-14/pagedcore-checkpoint-fs3zypy8.zip)
and [structured audit](../../results/validation/2026-11-14/checkpoint-trajectory-t4-review.json).
ZIP SHA-256: `5cc2bc7361066075786265d8b681a29ff98debd4c644a8d6cdf648c067373c88`.

## Attribution

All 482 evidence hashes and 112 source hashes verify with complete manifest
coverage. Snapshot
`b7a3b052bbb0f4f3860ffc8a44c41718478071f4f941f2b8e15efba0af6d811d`
matches the prepared notebook. All working files matched the bundled source
before these review writes, and source-after hashes are unchanged. The base
commit plus bundled working files identifies the source, not a clean-commit claim.
One CUDA-visible T4, Torch 2.7.1+cu126 and the frozen checkpoint/math settings
are recorded. All eight command records exit zero. This is a successful diagnostic
collection, not an accuracy acceptance result.

## Cached decoding

Prefill uses 15 public fixture IDs. Both paths then consume the same 32 fixed
continuation IDs through actual KV reuse, attending to lengths 16 through 47.
Page boundaries are crossed. This is not free-running greedy generation: equal
argmax choices describe predictions on fixed common histories only.

| Observation | Result |
|---|---:|
| Cached steps per run | 32 |
| Paged/reference raw logits checked across baseline and audited runs | All 64 tensors exactly equal |
| Argmax choices equal between paths | 32 / 32 |
| Steps exceeding the current logit threshold | 32 / 32 |
| Steps exceeding the current K/V threshold, reported during execution | 32 / 32 |
| Maximum logit tolerance ratio | 2.682216 |
| Maximum reported K/V tolerance ratio | 3.990878 |

A ratio above one fails. All 384 saved logit metric fields were recomputed from
the raw pairs. K/V ratios were measured during execution, but raw K/V is not
retained in this collector, so those ratios cannot be independently regenerated
from this ZIP alone. The guard, pool-pointer and cached-length assertions execute;
both runs finish with zero ownership, reserved credits and active slots.

The instrumented run produces exactly the same native logits as the baseline at
every step for both paths. Thus the component auditing did not change this run's
observable outputs. No full accuracy suite or new local GPU execution occurred.

## Real-weight local components

At cached steps 0, 15 and 31, audit RMSNorm, SwiGLU and the final head in both
paths. Expected values use separate FP64 reductions with the documented FP16
projection/activation/normalization boundaries. They are local staged references,
not a single full-FP64 model trajectory or new acceptance contract.

| Component | Recorded calls | Maximum absolute error | Maximum diagnostic scaled error |
|---|---:|---:|---:|
| RMSNorm | 264 | 0.00048828125 | 0.156642 |
| SwiGLU | 132 | 0.001850687 | 0.171220 |
| Final head | 6 | 0.007457410 | 0.086548 |

The diagnostic scale uses `2e-3 * (1 + abs(reference))` for norm/MLP and `5e-3`
for final logits, solely to contextualize errors against existing comparison
scales. It does not define a new component acceptance rule or override integrated
failures. The sampled local deviations do not expose a large component arithmetic
error; they do not independently establish the entire model's accuracy.

All 402 saved component input/native/reference records are finite, and their
804 reported maximum/RMS metric fields were recomputed. CPU versus GPU reduction
order changes the recomputed RMS by at most `2.17e-19`. Real weights were not
redownloaded during local review, so the reference tensors themselves were not
regenerated locally; their checkpoint, producer source and raw tensor identity
are retained for reproduction.

## Decision and limits

This adds real-weight local-component evidence and a longer fixed cached
trajectory. It does not prove arbitrary residual composition, all model math,
2048-token context, 256-token generation, EOS behavior, batching, fragmentation
or quality relative to an established backend. It includes one public fixture,
not a held-out quality corpus. Perplexity, task performance and vLLM comparison
are still unmeasured.

Preserve the runtime and thresholds. No serving repair is selected. The next
quality work should establish a pinned vLLM/T4 smoke baseline and a cached-scoring
adapter that proves KV reuse on identical ground-truth histories before a pilot
or held-out study. A comparator revision still requires its documented independent
evidence; the existing integrated failures remain blocking in the meantime.

Review checks: all ZIP/source hashes, complete raw-logit cross-run controls,
384 logit metrics, 804 component metrics, finite tensor checks, cleanup and
`git diff --check`. No runtime or test changes were made in this review.
