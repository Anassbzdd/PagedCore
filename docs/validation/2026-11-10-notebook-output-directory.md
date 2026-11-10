# Notebook output-directory regression

Observation date: November 10, 2026. Current gate: P5.3, still open.

## Returned T4 evidence

The owner supplied `results/zip/pagedcore-accuracy-kavvwb5q.zip`. Review verified
24 evidence checksums and 78 source-file hashes against the embedded snapshot.
The snapshot hash is
`16dbac4c8a10ccd14b8209732c12b62ed9aaaa622a9d517e4609cf0fed895a53`;
its base revision is `946e759`, with bundled working changes. The run selected
one Tesla T4 with Torch 2.7.1+cu126. Lint, formatting and strict typing passed;
760 non-GPU tests passed and two failed; 15 GPU tests passed and four failed.

All four revised integrated decode cases failed their first recorded logit
comparison against the owned FP32 reference. Single failed at 16 attended tokens
(maximum tolerance ratio 1.4751), mixed and fragmented at two tokens (2.1705),
and held-out at 38 tokens (2.0347). Those four comparisons matched greedy choices,
but the required 45 comparisons did not complete. The K/V ratio initialized to
zero is not measured evidence: the logit assertion aborted before K/V checks.
The exporter rejected the archive. No numerical repair is established.

## Cause and correction

The two non-GPU failures concern notebook archive tests, independently of the
GPU numerical failures. Generated notebook code prefers `/kaggle/working` when
it exists; the tests changed the current directory and searched there instead.
On Kaggle the archive was produced, but the tests searched the wrong directory.

The generator now accepts `PAGEDCORE_EVIDENCE_WORKDIR` as an explicit output
directory, creates it if needed, and resolves it before starting child commands.
Without an override the existing Kaggle/current-directory defaults remain.
Both notebook modes document the override. The tests select their own temporary
output directory and simulate both presence and absence of the Kaggle folder.
They intercept temporary-directory creation to prevent writes outside the test
directory, and still verify failure retention, archive contents and hashes.

All four regression combinations failed before the generator change and passed
afterward. The full non-GPU suite passed: 764 tests, 30 deselected, no failures
or skips. Ruff lint/format and strict mypy passed. Local validation uses
Torch 2.7.1+cpu; no new T4 execution is claimed.
The full non-GPU JUnit report is retained locally at
`results/local/2026-11-10-notebook-output-cpu.xml` (ignored).

Validation commands:

```text
uv run --no-sync python -m pytest tests/unit/test_numerical_evidence.py -k notebook_source_identity -q
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/2026-11-10-notebook-output-cpu.xml
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m mypy --strict src/pagedcore
```

Pytest runs set `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
This correction changes evidence tooling, not attention arithmetic or tolerances.
The next numerical investigation must capture the first divergent intermediate
in the two-token failure using the revised FP32 serving path. Repeating the full
accuracy collector alone does not supply that missing diagnostic evidence.
