# Normalize-before-values accuracy T4 review

Reviewed November 13, 2026. **The recorded two-token mismatch is resolved, but
P5.3 remains open: all four integrated test cases still fail.** No runtime code,
tests, tolerances or acceptance rules were changed during this review.

## Attribution

Retained [raw archive](../../results/validation/2026-11-13/pagedcore-accuracy-p6ts13rg.zip)
and [structured audit](../../results/validation/2026-11-13/normalize-first-accuracy-t4-review.json).
ZIP SHA-256: `e900806e8b8589be425d3a75ffb91b5a837638f60127e34db8d5acb7975c58e6`.
All 26 evidence hashes and 97 bundled source hashes verify, with complete manifest
coverage and unchanged source-after hashes. All bundled files matched the working
checkout before these review writes. This includes the changed attention and
strengthened GPU tests; the archive did not execute the older online algorithm.

Snapshot SHA-256 `0847965022829c254b20f2a7730f60442ca3ae3bd4fae8a93e553a2bb0ff7e0e`
matches the refreshed notebook's recorded snapshot. Base revision `8e15e0ba`
plus the bundled dirty working files identifies the run, not a clean-commit claim.
The checkpoint manifest records November 13 at 19:15:21 UTC. Target identifies
one CUDA-visible Tesla T4 (capability 7.5), Torch 2.7.1+cu126, CUDA 12.6, Python
3.11.15 and Transformers 4.52.4. Physical inventory lists two T4s; only one is
CUDA-visible. The pinned checkpoint and strict math flags are recorded.

## Results

CPU: **767 passed**, zero failures/errors/skips. GPU: **six passed, four failed**,
zero errors/skips. Setup, target/checkpoint smoke, Ruff lint/format, strict mypy
and CPU commands succeed. GPU regressions and accuracy export exit one; the
export correctly rejects this incomplete/failing run.

Both required attention tests pass, now including the no-full-history guard,
tiny probabilities and cross-page cancellation. Paged prefill, both KV storage/
allocator tests and CUDA RNG repeatability also pass. These establish their
tested component scope, not end-to-end model accuracy or maximum workspace safety.

All six raw logit pairs were safely loaded with `weights_only=True` on CPU. All
36 reported logit metric fields reproduce exactly from the full saved tensors.
A tolerance ratio of one is the acceptance boundary.

| Recorded comparison | Attended tokens | Previous ratio | New ratio | New failing logits / 32,000 |
|---|---:|---:|---:|---:|
| Mixed/fragmented request 0, first step | 2 | 2.170475 | **0, exact logits** | 0 |
| Single request 0, first step | 16 | 1.475121 | **1.303875, fails** | 13 |
| Mixed/fragmented request 1, first step | 16 | Not reached | **1.303875, fails** | 13 |
| Held-out request 0, first step | 38 | 2.034720 | **1.909388, fails** | 41 |

Previous values come from the retained [older accuracy run](2026-11-13-accuracy-t4-review.md).
The 16- and 38-token reference tensors are exactly unchanged across archives.
Within this run, single/0 and mixed/1 and fragmented/1 have identical paged and
reference tensors; both two-token rows are also identical to each other.

All six recorded greedy choices match. This is not complete generation agreement.
Maximum absolute logit error remains 0.015625 in the failing rows. The 38-token
maximum tolerance ratio decreases, but RMS error increases from 0.00355642 to
0.00415190; the change is not a uniform accuracy improvement.

The two-token rows proceed through all per-layer K/V comparisons and old-history
assertions, with measured K/V ratio zero. The four failing rows stop before K/V
comparison, so their zero K/V fields remain **unmeasured placeholders**. Only
six of 45 required integrated rows are recorded. No second/third cached step or
fragmented rollback/retry injection is reached. The original failure evidence is
retained without relabeling it as passing.

## Interpretation and boundary

Normalizing before multiplying values fixes the captured two-token mismatch at
the original thresholds. That supports the intended normalization-placement
adaptation, but it does not resolve the full integrated contract. The remaining
16-token failure occurs entirely within one page; crossing a KV page boundary is
not necessary for it. This archive has logits, not per-layer operands, so it does
not locate that remaining failure's first arithmetic difference.

Keep the established algorithm and blocking assertions. Do not widen tolerances,
start a backend-rounding sweep, or rerun this unchanged full suite expecting a
different answer. If further cause localization is undertaken, bound it to the
remaining 16-token case before choosing another implementation change. This
review does not select a new repair or redesign the comparator.

P5.3, complete integrated P5.4, P5.6 generation, P5.7 workspace/reuse, independent
model coverage and held-out quality remain open. Current standalone T4 attention
passes do not close those gates.

Local work: complete checksum/source verification, XML/command inspection, safe
raw-logit loading, all metric recomputations, same-input cross-archive comparisons,
documentation links and `git diff --check`. No new local GPU execution occurred.
No runtime/test files changed, so the CPU suite was not rerun for this review.
