# Paged decode numerical repair: CPU review

Observation date: 2026-11-01. **P5.3 remains open for manual T4 validation.**

## Diagnosis and separate changes

The starting evidence is the unchanged
[T4 failure review](2026-11-01-p53-t4-failure.md) and its retained diagnostic ZIP.
Its single-request dense-attention control was exact; mixed requests still
drifted, even after independent-row `bmm` projections. Those are separate
attention-arithmetic and batch-shape problems, not evidence for relaxing tolerances.

The [new diagnostic](../../scripts/diagnose_paged_decode.py) replays eager HF's
exact inputs for norms, projections, SiLU and RoPE, and its exact Q/K/V history
for attention controls. It uses the pinned weights, eager Transformers 4.52.4,
FP16 storage and unchanged `reference_math`. Integrated decode separately starts
from owned paged prefill and consumes the same reference-selected tokens.

On CPU, the original attention first differs at **layer 0's Q/K scores**, before
softmax. Norms, Q/K/V projections and the other matched-input batch controls
are exact on this backend. The retained T4 controls establish a separate batch
effect, but do not identify its first operation. The new diagnostic includes
RoPE and component inputs so that a manual CUDA replay can locate that operation;
CPU equality cannot identify a T4-specific kernel difference.

An initial repair rounded FP32 score products to FP16, normalized globally and
accumulated FP32 page products. The retained failed CPU diagnostic shows that
this is insufficient: the first remaining attention difference is layer 3's
value product (one element, absolute error `7.62939453125e-6`), and the first
rounded-score difference is layer 6. The integrated maximum tolerance ratio is
`3.8728158473968506`, so that candidate is rejected. Matched-operand probes at
cache lengths 16 and 50 show native FP16 grouped score products match eager;
FP32 products followed by a cast do not. Four persistent FP32 value-reduction
lanes, including eager's tail handling on CPU, match all 44 saved layer probes.

The next three-step replay exposed a further shape issue that those 16/50-token
probes missed. For the request starting at length one, the second decode has a
three-token history: native grouped score matmul first differs at layer 6,
and the integrated logit tolerance ratio reaches `1.917507290840149`. Using
eight query heads as matrix rows still changes the CPU kernel/reduction selected
by eager's one-query-per-head product. The final implementation keeps query
dimension one for each head, and pads a later partial K page to 16 slots for
the product while excluding padding from softmax. Genuinely short histories
retain their actual lengths. Seed 147 reproduces the short-history bug in a
small regression: the grouped path differs in 13 attention elements, up to
`0.00048828125`; the per-head path matches exactly.

The implemented repair therefore separates two choices:

- **Attention:** native FP16 per-head page score products and scaling; one global FP32
  softmax rounded to FP16; native FP16 V multiplication for one-page histories;
  four FP32 value-reduction lanes carried across page boundaries otherwise.
- **Batch arithmetic:** single-request shapes for norms, projections, RoPE,
  activations, residuals and the final LM head, inside the existing layer-major
  batch transaction. Replacing only projections with `bmm` was not sufficient
  in the retained T4 investigation.

The [design](../DESIGN.md#blockwise-decode-attention) records the operation order,
derived buffer bounds, CPU reduction source and CUDA uncertainty. The standalone
FP32 online-attention mode remains unchanged. The new decode path adds serial
operations and per-position value products; throughput is unmeasured.

## Preserved contracts and regression coverage

K/V remain block-table page reads, with at most one page copied/converted at a
time. Only score/probability buffers span the bounded 2,048-token context. The
existing no-full-history storage guard is unchanged and still wraps actual
decode, including maximum context and injected K/V shortcuts. Its attention
matmul boundary also applies to each one-position value contribution; no guard
exemption or dense K/V reconstruction was added.

All rows remain inside one `append_decode` context. No cache length, ownership
or free-list change commits until all layers and every row's logits succeed.
The added later-row-logit fault verifies rollback after an earlier row already
computed logits, followed by retry and complete release.

The original eager oracle, dense math, public fixtures, shared parity fixtures
and required T4 tests are unchanged. Release checks remain logits
`atol=rtol=5e-3`, K/V `atol=rtol=2e-3`, and exact greedy tokens. CPU eager-attention
regressions use exact equality for deterministic controlled inputs at
1/2/3/4/5/6/7/15/16/17/49/2048 tokens. Other coverage checks global probability rounding,
all decode linear shapes, exact batch-composition/order independence, all-layer
K/V, history preservation, ownership/accounting and synchronous rollback/retry.

## CPU checks

The final working-tree run at base `8ec8b0b` passes **674 tests**, with **17 GPU
cases deselected**, zero failures/errors/skips, on Python 3.11.15 and Torch
2.7.1+cpu. This is a dirty working-tree result, not a claim about the clean base
commit. The retained source hashes and snapshots identify the actual tested code.

Commands use `.venv/Scripts/python.exe`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`
and `MKL_NUM_THREADS=1`:

```text
python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p53-repair-final-cpu.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
git diff --check
```

All pass. The eight new short-history cases plus the earlier rounding,
batch-order and later-row-logit regressions add 16 tests to the 658-test baseline.
The existing K/V comparison is tightened from its former CPU-only `5e-3` to
the required `2e-3`; the unchanged T4 comparison already used `2e-3`.

The final pinned-checkpoint CPU replay runs three decode steps for `(15,)` and
`(1, 15, 16, 17, 49)`, reversing row order between steps. All **18 row/step
comparisons** have zero absolute error in logits and every layer's paged K/V,
and exact next-token argmax equality. Matched-input norm/projection/activation/
RoPE controls and native paged-attention controls also have zero error. This
proves the listed CPU forwards, not complete EOS/256-token generation or CUDA
equivalence. Command:

```text
python -u scripts/diagnose_paged_decode.py --device cpu --output results/local/p53-repair-final-arithmetic.json
```

The [structured summary](../../results/validation/2026-11-01/p53-cpu-repair-summary.json)
and [CPU archive](../../results/validation/2026-11-01/p53-cpu-repair.zip) retain
the JUnit report, raw final measurements, both rejected candidates' measurements
and matching attention source, matched-operand tensor probes/control program,
current Python source/config/lock snapshots, math policy and environment. ZIP
integrity and all snapshot hashes were checked. Archive SHA256:
`a2e0c8ff1c9bbb4bc9de1f2b7b0b320fca590235d9cbcadbd7574985df03c0fc`.

## Manual T4 follow-up

After committing/pushing the reviewed tree and selecting that clean revision on
T4, run the diagnostic under the frozen math policy, then the existing collector:

```bash
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/diagnose_paged_decode.py --device cuda:0 \
  --output results/local/paged-decode-arithmetic-t4.json
bash scripts/validate_t4.sh
```

Retain failures and the diagnostic JSON as well as the collector bundle. A
diagnostic pass is not collector acceptance. Require the unchanged single/mixed
decode tests and storage guard to pass at the fixed thresholds; investigate the
first named operation if they do not. Maximum-active workspace, EOS/full-output
paths and CUDA page-reuse ordering remain separate later Phase 5 requirements.
No T4 code was executed during this repair.
