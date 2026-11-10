# Two-token decode reproduction

Observation date: November 10, 2026. Current gate: P5.3, still open.

The owner requested the smallest reproduction before tracing larger cases:
one prompt token and one cached decode step, alone and inside the original
mixed batch. The [previous T4 review](2026-11-10-notebook-output-directory.md)
records a mixed two-token logit tolerance ratio of 2.1705. That archive did
not run the one-token prefix alone under the revised policy.

## Diagnostic controls

[`probe_two_token_decode.py`](../../scripts/probe_two_token_decode.py) runs
`maximum_output[:1]` alone, then row 0 of lengths `(1, 15, 16, 17, 49)`.
It uses the pinned FP16 weights, original 32-block poisoned pool, original
three-step reservation sizes, `reference_math`, seed 0 and the shared
no-full-history guard. Each arm executes only one cached decode step.
The target stays within one physical page. Other mixed rows retain their
original lengths and may cross a page boundary during that step.

The owned contiguous prefill selects each row's input token, as in the failed
acceptance case. Both arms use the same loaded weights. The diagnostic checks
target prompt/input IDs, prefill logits, contiguous FP32-reference logits and
math settings for exact equality across arms. It compares paged logits at the
unchanged `atol=rtol=5e-3`, records greedy agreement, measures target K/V error
at `2e-3`, and verifies zero blocks, credits and active slots after cleanup.
Numerical mismatches do not abort collection or suppress the second arm.

`summary.json` distinguishes both logit failures, identical full signed errors,
and any change to target logits with batch membership. `same_discrepancy=true`
requires identical controls, both tolerance failures and identical signed
errors. Inspect errors and controls before interpreting a false result.
Both arms retain raw target tensors and their hashes. Environment, source,
fixture, policy and diagnostic identity accompany the report.

This deliberately narrows collection to the target row; it is not full mixed
acceptance, an HF fidelity report, or a repair. Serving code, tolerances and
the GPU acceptance suite are unchanged.
All 16 runtime-source, fixture and lock files match the embedded source in
`results/zip/pagedcore-accuracy-kavvwb5q.zip` byte for byte.

## Kaggle execution

Import [`two_token_decode.ipynb`](../../notebooks/two_token_decode.ipynb), enable
Internet and a T4, and Run All. It bundles the exact working source and frozen
setup, runs only this probe, and archives partial failures, reports and raw
tensors. No Git push or Input dataset is required. Download its final ZIP.

For an existing checkout containing these edits, run from its root:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run --frozen --extra cuda --extra dev python -m scripts.probe_two_token_decode \
  --output-dir /kaggle/working/two-token
```

Read `/kaggle/working/two-token/summary.json`. A logit ratio greater than 1
reproduces the numerical failure; exit zero means collection completed, not
that acceptance passed. A runtime/setup error produces exit 1 and a retained
error report. The bundled notebook independently requires the original
Torch 2.7.1+cu126 build and the T4 target, and retains its setup command logs.

## Local evidence

The pinned-checkpoint CPU smoke reproduces identical signed target-logit errors
alone and mixed: maximum logit tolerance ratio `2.736318407960199`, maximum
absolute error `0.01953125`, exact greedy agreement and maximum K/V tolerance
ratio `2.8939886271324125` in both arms. All blocks, credits and slots return to
zero. These are Torch 2.7.1+cpu observations, not a T4 reproduction. The original
smoke reports and tensors are retained under
`results/local/2026-11-10-two-token-cpu/` (ignored). The final smoke in
`results/local/2026-11-10-two-token-final-cpu/` reproduces those results and
verifies the current diagnostic's source hash. Every paired control passes;
`same_discrepancy=true` and `batch_membership_changes_target_logits=false`.

The full non-GPU suite passes: 770 tests, 30 deselected, no failures or skips.
Ruff lint/format and strict mypy pass. Notebook tests cover source identity and
failure archives for the new mode; the generated notebook compiles and its
embedded source hashes match. These checks do not satisfy GPU acceptance.

Validation commands:

```text
python -m scripts.probe_two_token_decode --device cpu --output-dir results/local/2026-11-10-two-token-final-cpu
python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/2026-11-10-two-token-tests.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
python -m scripts.build_numerical_notebook --two-token
```

Local pytest/probe commands use `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. These local checks establish CPU behavior. The owner's
returned T4 run is reviewed below.

## Returned T4 evidence

The owner returned `results/zip/pagedcore-two-token-irzdb020.zip` on November 10.
The [retained review](../../results/validation/2026-11-10/two-token-t4-review.json)
verifies all 17 evidence checksums and 82 embedded source hashes. Every embedded
file matched the local working source at review time, before this documentation
update. The source-after manifest is unchanged; the diagnostic, fixture and policy
hashes match the snapshot. Its SHA-256 is
`946d9fc5a8e726b736578ee31356fb4f28429cc6d1516c403a7b8a6a7308cf87`,
with base revision `33f2ffca21bbd66b54c0e624c01c17d376e07f71` plus bundled working
changes. This is not a clean-commit claim.

All seven recorded setup/verification/diagnostic commands exit zero. The selected
device is one visible Tesla T4, capability 7.5, with Torch 2.7.1+cu126,
Transformers 4.52.4 and Python 3.11.15. The inventory lists two physical T4s;
`CUDA_VISIBLE_DEVICES=0` exposes one to the run. Both arms use the documented
deterministic math settings, including `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| Target measurement | Alone | Original mixed batch, row 0 |
| --- | --- | --- |
| Attended tokens | 2 | 2 |
| Maximum logit tolerance ratio | 2.1704749959099088 | 2.1704749959099088 |
| Maximum absolute logit error | 0.015625 | 0.015625 |
| Greedy choice matches reference | Yes | Yes |
| Measured maximum K/V tolerance ratio | 1.9796380090497736 | 1.9796380090497736 |
| Owned blocks / credits / active slots after cleanup | 0 / 0 / 0 | 0 / 0 / 0 |

Independent CPU inspection of the archived tensors, using `torch.load` with
`weights_only=True`, confirms exact equality of target prompt IDs, decode input,
prefill logits, paged logits and reference logits across arms. The decode input
is the prefill argmax in both. Recomputing every logit metric from the raw tensors
matches both reports. Full signed errors are identical. Both raw paged and
reference logits also match
`pagedcore-accuracy-kavvwb5q.zip/evidence/raw-logits/mixed/step-1-request-0.pt`
exactly: this reproduces the original failure rather than only a similar ratio.

The worst tolerance coordinate is vocabulary index 7445: paged logit
`-0.13134765625`, reference `-0.11920166015625`. This coordinate maximizes the
tolerance ratio, which need not maximize the absolute error. The K/V maxima
are measured by the hash-verified diagnostic; raw K/V tensors are not archived
for independent recomputation. Both arms execute the no-full-history guard and
their cleanup assertions successfully.

**Conclusion:** the same T4 numerical discrepancy reproduces with a one-token
prompt and one decode step, independent of membership in this mixed batch.
The target occupies one page throughout. Neither another request nor a page
boundary is required for this fixture's failure. This does not identify the
first differing operation or prove batch independence for other fixtures.
Next trace the standalone two-token case against its contiguous FP32 reference,
starting at the earliest differing intermediate.

This review verifies the owner's T4 execution; it performs no new GPU run and
does not rerun acceptance. Serving code, GPU tests and tolerances remain unchanged.
The local review checks SHA-256 manifests, snapshot/source identity, command exit
codes, raw tensor equality and recomputed logit metrics. `git diff --check` passes.
P5.3 and all later gates remain open.
