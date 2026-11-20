# Integrated FP16 drift budget

Decision: November 16, 2026. The owner explicitly requested a solution including
changing tolerance. Keep the owned decoder and default normalize-first paged
attention. Revise only the integrated numerical budget, then run the complete
T4 suite. **P5.3 is pending that run.** No runtime arithmetic fix or GPU pass is
claimed by this change.

## Why this is a reasonable next step

The [16-token trace](2026-11-13-sixteen-token-trace-t4-review.md) and
[single-scalar control](2026-11-14-scalar-control-t4-review.md) isolate equal-input
FP32 value reductions that cross FP16 rounding midpoints. Replacing the first
scalar with the reference's choice does not remove later drift. The next
independently checked disagreement favors PagedCore's rounded result. All 88
captured native attention calls satisfy the existing independent FP64 bound.
This supports allowing accumulated integration drift instead of reproducing
one backend's internal reduction order. It does not prove every case is correct.

[PyTorch 2.7's numerical-accuracy documentation](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)
explains that mathematically equivalent operations, batched products and slices
can differ in floating-point arithmetic. [FlashAttention's tests](https://github.com/Dao-AILab/flash-attention#tests)
compare error with a baseline's numerical error. These sources support measuring
error and preserving independent references. Neither source establishes a
universal whole-model logit/KV tolerance or supplies the numbers selected here.

## Frozen limits and calibration

Policy: `fp32-paged-v3-fp16-drift`. Each integrated element must satisfy
`abs(actual - reference) <= atol + rtol * abs(reference)`.

| Comparison | Prior atol and rtol | Revised atol and rtol |
|---|---:|---:|
| Integrated logits | 0.005 | 0.02 |
| Integrated K/V against the other model trajectory | 0.002 | 0.01 |
| Attention output against FP64 on identical operands | 0.002 | 0.002 |

The logit budget is four times its old value; the K/V budget is five times its
old value. These are practical margins for the finite paging suite, calibrated
on already observed failures. They are not statistical confidence bounds,
universal FP16 error bounds, or evidence of preserved language quality.

The [32-step checkpoint run](2026-11-14-checkpoint-trajectory-t4-review.md)
has a worst original logit ratio of 2.682216 and reported K/V ratio of 3.990878.
The revised budgets leave about 49% and 25% headroom over those observed maxima
respectively. They are frozen before the new complete-suite result; do not keep
raising them automatically if new cases fail.

The [retrospective audit](../../results/validation/2026-11-16/v3-budget-calibration.json)
checks the prior archive identity and hashes of its baseline report and 32 raw
logit pairs, then independently recomputes the error ratios. All 32 logit pairs
pass the revised limit, with maximum ratio **0.670554**; all 32 greedy choices
match. The maximum rescaled reported K/V ratio is **0.798176**. That archive does
not retain raw K/V, so K/V is rescaled from original recorded measurements rather
than independently regenerated. This is retrospective calibration on one fixed
trajectory, not untouched holdout evidence or new T4 execution. The old failures
retain their original conclusions.

## Checks that remain strict

- Exact greedy choices against the owned reference for every frozen row.
- Independent FP64 attention, GQA, masking, difficult scores and boundaries.
- Prefill's existing logit/KV thresholds and component tests.
- Exact old cached values, poisoned-page isolation, stable pool storage, page
  tables, credits, active slots, staged rollback/retry and zero cleanup.
- No full-history K/V gather during actual paged decode.

The larger K/V comparison applies to two independently evolving FP16 model
trajectories. It does not permit modifying stored history or writing to the
wrong page. Model quality, long free-running generation, EOS, workspace and
stream-ordered reuse retain their separate release gates.

## Complete collection and acceptance

Run `single` (15), `mixed` and `fragmented` (1/15/16/17/49), and `held_out`
(37/73/111/145), each for three reference-token cached steps: **45 rows**.
The historical `held_out` name describes layout lengths already inspected; it
is not untouched statistical holdout data or a quality corpus.

Numerical mismatches no longer stop the first row before K/V is measured.
Collect every row and apply `valid_accuracy_report` after cleanup. Structural
failures still stop their affected case. Every row must include complete finite
K/V measurements. Explicit policy/tolerances prevent silently interpreting a v2
report with the new limits. The exporter rejects missing, duplicate, unmeasured,
nonfinite, numerically failing or greedy-divergent rows and failed/skipped tests.

Retain raw logit pairs and detached per-page K/V pairs with source hashes and
target metadata. Records also retain the old logit/KV ratios for comparison.
The independent component checks remain part of the same required ten-test T4
suite. The default backend needs no FlashInfer installation or CUDA compilation.

Generate the self-contained Kaggle collector with:

```text
python -m scripts.build_numerical_notebook --accuracy --output notebooks/paged_decode_v3_validation.ipynb
```

Enable Internet and a T4 in a fresh Kaggle notebook, import the generated file,
and run all cells. Retain the ZIP even if rejected. Close P5.3 only after the
complete T4 evidence passes the frozen v3 checks and is reviewed. A successful
CPU run, this retrospective audit, or changing tolerances alone does not close it.

## Local validation

Windows, Torch `2.7.1+cpu`, CUDA unavailable: **829 CPU tests passed, 67
deselected, zero skipped** in the [full report](../../results/validation/2026-11-16/v3-clean-full-cpu.xml).
Ruff check, Ruff formatting, strict mypy for all 15 runtime source files and
`git diff --check` pass. All local links in the eight affected contract/status
documents resolve. The generated notebook's code compiles and its 123 embedded
source-file hashes verify; no GPU test was executed locally.

The earlier full run retained at `v3-full-cpu.xml` has 828 passes and one failure:
placing pytest's temporary directory inside the repository invalidated the
outside-checkout provenance fixture. The same test passes with an external
temporary directory, followed by the complete passing run. No runtime change
was needed for that environment issue. Ruff now excludes `results` so retained
evidence and temporary generated notebooks do not become lint input.

Commands (with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`):

```text
python -m pytest -m "not gpu and not benchmark" -q --basetemp <external-work-directory> --junitxml=results/validation/2026-11-16/v3-clean-full-cpu.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
git diff --check
```
