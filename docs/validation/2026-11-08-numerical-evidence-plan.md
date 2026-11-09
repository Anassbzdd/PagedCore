# Reproducible T4 numerical evidence collection

Prepared November 8, 2026 against base `7d9ea61ed81b6028980373860f9a2967935a9683`
and the bundled working source. P5.3 remains open. No serving arithmetic,
numerical tolerance, or greedy-token check changes in this preparation.

Upload [the standalone notebook](../../notebooks/p53_numerical_evidence.ipynb)
to Kaggle, enable Internet and a T4 accelerator, and run all cells. It embeds
the complete relevant source and frozen lock; no push, checkout update or patch
is required. The subprocess exposes one T4 and sets the cuBLAS workspace before
importing Torch. It installs Python 3.11 and the frozen CUDA/development extras.
Download the final `pagedcore-evidence-*.zip` from `/kaggle/working`, even when
experiments or GPU tests fail. Keep the notebook unchanged for this run.

The source snapshot has per-file SHA-256 hashes, its base revision and the
builder's working-tree status. Dirty source is identified by its complete files,
not presented as a clean commit. Hashes are verified before and after execution.
Setup commands, return codes, logs, installed dependencies, GPU UUID/capability,
physical GPU inventory, math policy and model configuration are retained.
The final archive also contains checksums for each evidence file. Setup failures
and ordinary interruptions still produce an archive with partial results; a
terminated Kaggle session cannot be guaranteed to run finalization.

| Experiment | What it decides | Retained evidence |
|---|---|---|
| Rerun the retained synthetic source at 17/18/24/95/96/2048 | Whether the new environment reproduces the earlier kernel families and cancellation observations | Original hash-verified source, all trials, sampled outputs, kernel names, environment and failures in `synthetic-controls.zip` |
| Actual TinyLlama one-token attention at 34 attended lengths, all 22 layers and every head/channel | Whether synthetic family/order hypotheses describe real operands, including page edges and long-context transitions | `length-N/layer-LL.pt`, original layouts and actual eager/replay kernels in `reference.json` |
| FP32 alternating, contiguous-half, four-lane and FP64 reference-only CPU replays | Which orders match all real FP16 outputs, and where each fails | Full model outputs and exact mismatch coordinates/counts; these are hypotheses rather than GPU implementation evidence |
| Poisoned, alternating physical-page score control with the value output held fixed | Whether score or probability arithmetic differs independently of value reduction | Paged scores and FP32/FP16 probabilities for every layer, tables, accounting and guard result |
| At the first score mismatch per length, vary K packing and logical output width separately | Whether packing or width changes the raw QK product and subsequent probability rounding | Per-page operand layouts, kernel names, raw-dot/score/probability comparisons and `score-layouts.pt` |
| Guarded baseline, paired-at-17 and family-hypothesis integrated controls | Whether a value-only hypothesis improves real single/mixed/fragmented decode, and exactly where remaining failures start | Every step/request/layer, first differing attention operation, logits, all K/V comparisons, original and owned attention tensors, exact greedy equality, rollback/retry, ownership and release accounting |
| Existing component diagnostic and GPU regressions | Whether the existing reproductions still fail or pass on this exact source and environment | `existing-components.json`, complete pytest output and `gpu-regressions.xml`, including unsuccessful cases |

The attended lengths are
`4,15,16,17,18,19,24,30,31,32,33,47,48,49,63,64,65,95,96,97,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048`.
They include the five families in the [November 7 sweep](2026-11-07-value-kernel-sweep.md).
Each isolated length starts from a fixed public-fixture prefix of length N-1;
longer contexts repeat those IDs. This is a deterministic numerical workload,
not a natural-language quality or service benchmark. The integrated controls
also retain the original prompt-15/three-cached-step trajectory, so an 18-token
result can be distinguished from a prompt-17/one-step result.

Integrated arms use prompt lengths `(15,)` and `(1,15,16,17,49)` for three
cached steps, reversing row order between steps. The fragmented case inserts a
poisoned unrelated page, injects a failure after staged writes, verifies valid
history and accounting before retry, and checks untouched storage afterwards.
Every owned decode runs under the existing storage guard. Logits keep
`atol=rtol=5e-3`; K/V keep `atol=rtol=2e-3`; greedy tokens must match exactly.
Numerical failures are recorded without abandoning later rows or steps. Inputs
follow HF-selected tokens so later comparisons retain a common input trajectory;
this does not establish independent end-to-end generation acceptance.

The family hypothesis chooses alternating FP32 sums only when an actual eager
value kernel and its exact-shape replay share GEMVX variant 5. It chooses two
contiguous halves only for measured variant 6 with even length. Unsupported or
unrecognized families retain the serving baseline. Kernel names and dispatch
decisions are recorded per call. This diagnostic lookup is not a proposed
production kernel dispatcher, and a few matching real outputs do not prove an
addition order. The standalone real-operand CPU replay also reports an odd-length
half split as a counterexample control; that extension is not selected on GPU.

Raw tensor files contain Q, unexpanded K/V, raw dots, scaled scores, both
probability dtypes and weighted-value outputs. Original shape, stride and storage
offset metadata is captured before copies. For offline replay, recreate strided
operands with `torch.empty_strided(shape, stride).copy_(saved_tensor)`; loaded
CPU tensors need not preserve the original CUDA storage layout or offset.
Expanded HF K/V can be reconstructed independently using the pinned head mapping.
The full-width score controls use independent HF storage outside the guard.
They must never be moved into serving: they exceed the one-page K/V scratch bound.
Owned K/V history is never gathered by the collector.

`first_difference` means the first unequal attention boundary in layer order:
Q, K, V, scaled scores, FP32 probabilities, FP16 probabilities, then weighted V.
All measurements are retained, including exact differences within tolerance.
An upstream Q/K/V difference prevents attribution of that layer's output error
to its reducer alone. The existing component diagnostic separately checks norms,
projections, RoPE and MLP operations on common reference inputs.

To regenerate the notebook after an intentional source change:

```bash
uv run --no-sync python -m scripts.build_numerical_notebook
```

To execute just the new collector in the frozen T4 environment:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run --no-sync python -m scripts.collect_numerical_evidence \
  --output-dir results/local/numerical-evidence
```

The notebook is the deliverable because it wraps setup, independent subprocesses,
existing diagnostics and archive finalization. A zero collector exit means that
collection ran, not that parity passed. Read failed-row counts and JUnit outcomes.
Missing lengths, layers, arms or tests are incomplete evidence.

After the archive returns, verify its identity/completeness first, separate local
arithmetic from propagated drift, and use the smallest demonstrated failure as a
regression before implementing a repair. Do not infer score changes from value
matches. A repair still needs the full T4 collector, frozen tolerances, exact
greedy checks and the documented remaining gates: integrated fragmented parity,
end-to-end paged generation, maximum-active workspace and safe CUDA page reuse.
CPU preparation and this notebook cannot close those gates.

## Local preparation evidence

November 8 checks used Torch `2.7.1+cpu`; CUDA is unavailable on this machine.
The non-GPU suite passes **724 tests**, with **26 deselected and zero skipped**.
The 12 new tests cover captured 18-token arithmetic through fragmented pages,
original-layout/raw capture, bounded diagnostic measurements under the guard,
partial failure retention, unsupported-family fallback and archive creation even
when environment setup fails. This is CPU evidence for the evidence tooling.

Commands run:

```bash
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m mypy --strict src/pagedcore
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra \
  --junitxml=results/local/numerical-evidence-cpu.xml
uv run --no-sync python -m pytest -q tests/unit/test_numerical_evidence.py
uv run --no-sync python -m pytest --collect-only -q tests/gpu \
  -k "test_pinned_paged_decode_on_t4 or paired_at_17 or eager_value_sum_at_17_tokens"
git diff --check
```

The CPU tests use `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
The JUnit report is local/ignored. The GPU selection collects nine cases;
collection is not execution. A separate synthetic CPU smoke uses the existing
reduced-vocabulary/two-layer fixture repeated to 22 layers, with per-layer cache
indices. The 18-token score control completes all 22 layers on alternating
physical pages. The fragmented three-step integrated control completes 15 rows
with passing tolerance/greedy/accounting/rollback/guard checks and zero retained
resources. This smoke validates orchestration, not pinned-checkpoint or T4 parity.
The notebook's embedded source hashes and Python code are verified locally.
