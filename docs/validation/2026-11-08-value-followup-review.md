# T4 value-candidate follow-up review

Reviewed November 8, 2026. The [returned archive](../../results/zip/value-followup-xn43_mtr.zip)
has SHA-256 `48f36fbfd7cc5f91d871f312040e9f78ab67e536f7a702375e00cd7fe27cbe23`.
All 1,135 evidence-file checksums and 72 retained source-file hashes match.
The source base is `7d9ea61ed81b6028980373860f9a2967935a9683`, with the earlier
bundled working changes and the exact runtime runner retained in the archive.
Kaggle extracted one historical evidence ZIP; its original hash is explicitly
retained under `unavailable_input_archives`. No runtime source is omitted.
Each experiment runs in a fresh subprocess with Torch `2.7.1+cu126`,
Transformers `4.52.4`, the pinned eager checkpoint, strict math policy and one
visible T4. The snapshot is not presented as a clean commit.

## Observations

All three baseline experiments complete collection with their storage guard,
prefill and cleanup checks passing. Numerical parity still fails in 2/3 single
request-step comparisons and 11/15 comparisons in both mixed and fragmented
cases. A zero process exit means collection succeeded, not parity acceptance.

Candidate kernel detection now succeeds. The single case records 22 native
one-page calls, 22 alternating-sum calls, and one half-split call. Its 16- and
17-token integrated comparisons pass. At 18 tokens, layer 0 has identical Q/K/V,
scores, probabilities and weighted-value output, then the forward aborts before
final logits. Both mixed cases stop at the same kind of error after two native
calls, one alternating-sum call and one half-split call. Thus none of the three
combined candidate cases provides complete integrated acceptance evidence.

The failure is `decode materialized more than one KV page via aten.linear.default:
(1, 1, 5632)`. The diagnostic half-split reducer forms scalar products using
elementwise multiplication. The storage guard follows their V ancestry through
the accumulator into hidden states and the normal MLP projection. By contrast,
the existing striped reducer expresses each product as a rank-one matrix multiply,
which the guard recognizes as the boundary from cached V to attention output.
The 5,632-wide tensor in this traceback is the model's MLP activation, not a
gathered token history. This is a diagnostic/guard interaction, not an accepted
numerical failure or evidence that page-table addressing is wrong.

## Small diagnostic correction

Only `scripts.collect_numerical_evidence.halves_page_sum` changes: each independent
FP32 scalar product is expressed as a rank-one `torch.matmul`, using the same
pattern as the existing striped reducer. Its reduction dimension is one; the
two contiguous accumulation groups and final FP16 cast are unchanged. Each V
conversion remains at most one page. No serving arithmetic, score path, threshold
or guard rule changes, and no guard bypass or blanket `linear` exemption is added.

A new CPU regression runs the half-split output into a 5,632-wide linear layer
under the guard. It fails with the original elementwise product and passes with
the rank-one product. The focused check also retains all eight existing K/V
full-history shortcut rejections. A separate CPU replay of saved real T4 operands
checks all 22 layers at 18 tokens on fragmented pages; it is not GPU acceptance.

The next Kaggle runner reuses the restored, verified source and injects only the
exact corrected diagnostic helper in each subprocess. Its complete helper source
is retained in `runner.py` in the output archive. Kernel identification remains
strict: a missing family on a multi-page candidate call aborts instead of falling
back. Run baseline and candidate single/mixed/fragmented controls again. Preserve
every failure and dispatch count. P5.3 and all remaining documented gates stay open.

## Local validation of the diagnostic correction

Torch `2.7.1+cpu`, with CUDA unavailable: 725 non-GPU tests pass, 26 cases are
deselected and zero skip. Ruff lint/format, strict mypy and diff checks pass.
The new regression failed before the correction and passes afterwards; the
focused run passes nine checks including all eight dense-history injections.
The saved 18-token T4 operand replay matches all 22 layer outputs exactly on CPU
and finishes with zero owned blocks, credits or active slots. These observations
do not establish CUDA implementation equivalence or full decoder parity.

Commands run (CPU pytest uses `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`):

```bash
uv run --no-sync python -m pytest -q tests/unit/test_numerical_evidence.py \
  -k half_sum_can_feed
uv run --no-sync python -m pytest -q tests/unit/test_numerical_evidence.py \
  tests/unit/test_decoder.py -k "half or guard_rejects_dense_history"
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m mypy --strict src/pagedcore
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra \
  --junitxml=results/local/value-followup-guard-cpu.xml
git diff --check
```

The full CPU JUnit and `results/local/value-followup-guard-replay.json` are
local/ignored records. The next-run cell and its subprocess compile locally;
its embedded helper is verified equal to the corrected local function.
