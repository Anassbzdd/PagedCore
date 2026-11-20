# Single-scalar accumulated-error control

Prepared November 14, 2026. **Diagnostic only; no new T4 result or serving repair.**
The current gate remains P5.3. The
[previous T4 trace](2026-11-13-sixteen-token-trace-t4-review.md) locates the first
FP16 difference at layer 1, query head 21, channel 60. This experiment asks how
much downstream error remains when only that one value is replaced.

Returned T4 evidence is now recorded in the
[control review](2026-11-14-scalar-control-t4-review.md): the intervention is valid,
but integrated comparisons still fail and later rounding disagreements remain.

Use the [self-contained Kaggle notebook](../../notebooks/sixteen_token_scalar_control.ipynb).
Enable Internet and T4 GPU, then Run All. No Input dataset or previous session is
required. The source and frozen dependencies are embedded, along with the original
failed logits, retained trace report and actual target-layer tensor. The builder
verifies hashes of selected retained evidence and unchanged runtime source.

## Controlled change

Keep the original 15-token fixture prefix, prefill-selected token, one decode step,
checkpoint, math settings, pool and reservation sizes. Two arms use fresh caches:

| Arm | Target attention output |
|---|---|
| Sham | Clone the output and write its original scalar back |
| Single scalar | Same clone/write, substituting only the captured reference FP16 scalar |

Cloning and writing happen in both arms to control their instrumentation. The
unmodified result at the target must equal the whole retained paged tensor before
the intervention. The replacement must change exactly one element; every other
element and output shape/stride must remain identical. All 34 observed stages
before the target FP16 output must be exact between the two arms on T4.

The sham must reproduce the original final logits exactly and match all 440
retained stage comparisons. Both reference forwards must match the original
reference logits. Verify prefill and selected token, absolute positions, unchanged
old history, appended writes against local projections, stable pool pointer and
block table, unchanged ownership/reservations/active slots and zero ownership
after cleanup. The blockwise guard remains active during both paged decodes.

## Evidence and interpretation

Retain complete layer traces and attention operands for both arms, full logits,
numerical comparisons, first remaining FP16 difference, whether the target layer's
hidden output now equals reference, cache checks, source identity and partial
failures. Cell 4 exports one ZIP even when an earlier stage fails.

- Smaller final differences show the first scalar contributes to this case's drift.
- A remaining difference shows the intervention did not explain the entire mismatch;
  its first layer/operation identifies the next unresolved point.
- Unchanged or larger error is also a valid observation. The effect need not be
  monotonic because subsequent rounding can move in either direction.
- A passing intervened comparison does not validate the real implementation: that
  arm deliberately substitutes a reference value. Neither a serving repair nor
  broader model accuracy follows from this one correlated case.

This is a bounded causal experiment, not a reduction-order sweep or a comparator
redesign. No runtime module, numerical tolerance, acceptance test or checkpoint
changes. Do not run the full T4 suite solely to collect this control.

## Local checks

The [CPU report](../../results/validation/2026-11-14/scalar-control-cpu.json)
records focused tracing/probe/notebook checks, including real synthetic 22-layer
decode controls at both prompt lengths. Tests verify sham reproducibility, exact
prefix equality, one-element changes, unchanged history and local writes, cleanup,
wrong-target rejection and restoration after exceptions. Notebook setup-failure
tests verify source/evidence checksums and export in both simulated Kaggle and
non-Kaggle environments. These CPU results do not replace the required T4 run.

Rebuild the exact-source notebook after editing its sources:

```bash
python -m scripts.build_numerical_notebook --scalar-control
```
