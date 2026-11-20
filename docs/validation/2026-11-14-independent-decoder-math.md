# Independent decoder-component math checks

Prepared November 14, 2026. **CPU validation only. No serving behavior, comparator
or threshold changes; P5.3 remains open.**

The [scalar-control review](2026-11-14-scalar-control-t4-review.md) shows locally
accurate attention can still disagree after repeated FP16 rounding. Before any
numerical-contract review, fill independent validation gaps outside attention.
The paged and contiguous paths share helpers, so their agreement cannot alone
establish that those helpers implement the intended model.

## Added coverage

[Independent scalar tests](../../tests/unit/test_independent_decoder_math.py)
calculate expected values with Python binary64 arithmetic, `math.fsum`, explicit
SiLU and direct binary16 rounding. They do not call decoder, dense or Transformers
helpers to calculate expected outputs. Both runtime paths are tested against
those separately calculated answers using the existing logit/KV/component bounds.

| Area | Evidence |
|---|---|
| SwiGLU MLP | Distinct gate/up/down projections, positive/negative gates, zero and activation magnitudes representative of normalized inputs |
| Final normalization and LM head | Independent RMS equation, learned scales and sparse untied output mapping; zero/repeated/permuted token rows |
| Embedding lookup and residual composition | Analytic block with nonzero attention and MLP contributions, post-attention normalization of the updated hidden state, both residual additions and final head |
| Prefill/cached orchestration | One active block at layer 0 or 21 among 22 synthetic layers; 1- or 3-token prefill plus a cached step through contiguous FP32 and paged decode |
| Cache and guard | Analytically expected rotated-zero K and unrotated V, complete prefix values, correct cached length, blockwise decode guard and zero cleanup |

Uniform attention has zero Q/K and known prefix means. The two-/four-position
cached steps have exactly representable uniform probabilities; prefill keeps its
existing FP16 probability policy. Expected MLP results explicitly round projected
gate/up values, SiLU and gated products before the final projection. Norm expected
values follow the documented normalization/learned-scale conversion boundaries.

This isolates architecture, orientation and composition without trying to reproduce
an undocumented GPU reduction order. The one-active-block fixtures have a reduced
MLP width of three; they are not an independent general 22-layer checkpoint oracle,
multi-active-block proof or real T4 accuracy result. Existing tests separately
cover RoPE, attention, allocator ownership and broader layout/lifecycle properties.

## Verification and sensitivity

[CPU evidence](../../results/validation/2026-11-14/independent-decoder-math-cpu.json)
and [JUnit](../../results/validation/2026-11-14/independent-decoder-math-cpu.xml):
**20 passes** (ten new tests and ten existing independent RMSNorm/RoPE cases).
Ruff lint/format and strict source mypy pass. Local Python 3.11.15 / Torch
2.7.1+cpu has no CUDA.

A controlled local sensitivity audit temporarily injects four mistakes and
requires the independent assertions to fail: swapped gate/up projections,
output head tied to embeddings, omitted MLP residual contribution and normalization
of pre-attention rather than updated hidden state. All four are detected; patches
are restored and no runtime file is edited. The small-activation MLP tests were
strengthened before collecting this evidence so the existing absolute tolerance
cannot hide those errors. No thresholds were widened.

Commands:

```text
python -m pytest tests/unit/test_independent_decoder_math.py tests/unit/test_independent_components.py -xq
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
git diff --check
```

## Remaining work

Keep normalize-first attention and the current blocking comparison. Next validate
real-checkpoint longer cached trajectories and plan a small cached-decode quality
pilot against an established, explicitly pinned vLLM environment. The adapters
must score identical histories through actual KV reuse; prefill-only perplexity
is insufficient. Pilot and held-out data/counts/margins remain separate and must
follow the numerical policy before any new acceptance rule is frozen.

No full CPU rerun, new T4 run, quality evaluator, quality acceptance or comparator
redesign is claimed here. Synthetic component checks reduce a validation gap;
they do not convert the retained integrated failure into a pass.
