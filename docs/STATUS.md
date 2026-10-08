# Implementation and evidence status

Snapshot: 2026-10-08. Restored shared lifecycle/accounting validators and tests have fresh local CPU evidence in the [P2.2 report](../results/validation/2026-10-08-p2.2-cpu.json), attributed by source hashes against the uncommitted checkout based on `2eda7c07620e638b07270515517722bd783bab38`. The CPU CI workflow was added against commit `82102a3e55b995a0ab431a8f7a15417b29ba2421`; its first GitHub-hosted run remains unverified. Earlier environment evidence is attributed to commit `ab05602e721beb6b0fc4266a9642b09f89eee6c1` plus source hashes and a sanitized patch in the [CPU validation report](../results/validation/2026-10-05-p1.6-cpu.json). This page records observed state, not a release certificate.

## Current gate

Phase 1 implementation is substantially delivered; its evidence closure remains open and was not re-audited for this change. P2.1 shared types and P2.2 lifecycle/accounting validators are implemented and CPU-tested; release-once rules are specified, with concrete ownership/cleanup assertions deferred to P4/P6. Reference fixtures, the oracle harness and test-command work remain next. The Phase 2 exit gate remains open, followed by dense parity -> allocator correctness -> paged attention -> lifecycle correctness -> HTTP. Early CI and validation procedures support that order without implementing later runtime features.

| Surface | Observed implementation | Evidence limit |
|---|---|---|
| Package/environment | Metadata, committed lock, Python pin, CPU/CUDA extras | No retained fresh Linux/T4 clean-install report inspected |
| Configuration | Immutable typed settings and validation | CPU tests; worker/service-only settings remain planned |
| Diagnostics | Safe JSON diagnostics, hardware fallback and expected CLI configuration-error handling | CPU coverage; hardware output is not engine certification |
| Deterministic controls | Python/Torch seeds and inference mode | CPU repeatability checked; no retained CUDA repeatability report |
| Target verifier | Pinned snapshot, FP16/config/logit checks, source hashes, sanitized actual CLI arguments, manifest-write error handling | CPU regressions pass; fresh T4 evidence for the changed verifier remains open |
| Shared engine contracts | [Immutable records, legal-transition validator, positive reservation credits and nonnegative/accounted capacity snapshots](../src/pagedcore/engine_types.py); release-once contract specified | [108 CPU contract cases](../tests/unit/test_engine_types.py), including 99 added for P2.2; no request-state mutation, allocator/worker ownership checks, cleanup, instrumentation or T4 engine evidence |
| Decoder, allocator, paged attention | Not implemented | No parity, ownership, or paged-read evidence |
| Scheduler, worker, HTTP/SSE | Not implemented | No concurrency, cleanup, backpressure, or service evidence |
| CPU CI | [Workflow](../.github/workflows/ci.yml) for every push/PR: frozen CPU/dev environment, lock check, format/lint/strict types, non-GPU/non-benchmark pytest, distribution build and installed-wheel/config/CLI smoke | Local Windows CPU checks; first GitHub-hosted Linux run pending |
| Container, benchmark/report | Not implemented | No container or benchmark evidence |

## Observed CPU checks

On 2026-10-08, local Windows validation of the restored lifecycle/accounting contracts passed
Ruff lint, strict mypy over six source modules plus the contract test module, and
non-GPU/non-benchmark pytest: 175 collected, 173 selected/executed/passed, zero
failed/skipped, two GPU tests deselected. Changed-file formatting passed.
Repository-wide formatting still failed on a pre-existing missing blank line in
unchanged `tests/conftest.py`. The [P2.2 report](../results/validation/2026-10-08-p2.2-cpu.json)
retains actual commands/output, source/lock hashes, environment, test counts, and
that limitation and the failed collection before restoring the missing validator.
These checks establish shared validation rules,
not release-once cleanup, per-request ownership, CUDA safety or lifecycle execution.

The [retained report](../results/validation/2026-10-05-p1.6-cpu.json) records Windows, Python 3.11.15, Torch 2.7.1+cpu, Transformers 4.52.4 and unavailable CUDA. The existing environment was synchronized with `uv sync --frozen --extra cpu --extra dev`; this was not a fresh Linux installation. Ruff lint/format, full-source strict mypy and the offline frozen-lock check passed. Non-GPU pytest collected 67 tests, selected and executed **65**, passed all 65, skipped none, and deselected two GPU tests. Expected CLI failures were reproduced before the fix and covered by regression tests.

An explicit GPU-marked run selected two tests and skipped both with `requires CUDA`: **zero GPU tests executed**. The report retains sanitized pytest output, JUnit reports, source/lock hashes, the source patch, setup arguments, installed dependencies and seeds. CLI diagnostics passed; target verification returned the expected JSON failure without CUDA. These are local CPU observations, not CI or T4 certification.

On 2026-10-06, local Windows checks for the new workflow passed lock consistency,
frozen CPU/dev synchronization, Ruff format/lint, strict mypy (five source files)
and non-GPU/non-benchmark pytest
(67 collected, 65 selected/executed/passed, zero failed/skipped, two GPU tests
deselected). `python -m build` produced a wheel and sdist; installing the wheel
with `--no-deps --reinstall` passed isolated import, package metadata, default
configuration, CLI help and diagnostics checks. Actionlint 1.7.12 passed workflow
validation with shellcheck disabled. These command observations are local CPU
checks; they do not close the GitHub-hosted run or clean Linux/T4 installation gate.

## Saved T4 observation

An ignored local manifest dated `2026-10-04T14:37:18.785925Z` reports Tesla T4, CUDA 12.6, Torch 2.7.1+cu126, Transformers 4.52.4, the pinned TinyLlama model/tokenizer, and finite FP16 reference logits shaped `[1, 8, 32000]`. Its Git revision is the original implementation commit `157d1f3e7f8c72b23a01879850f0e9a261b8522c` and its lock hash is `ae7561ac683bfa084defe747ef8aa9380b65a42f7ed7b245ecf3fb9fe0e3df01`.

It also reports a dirty working tree without saved changes/source hashes and hardcoded command/setup provenance. No associated GPU pytest report was inspected. The older tracked validation manifest is absent from the current checkout. This page deliberately has no link to either a missing file or ignored local evidence; a fresh public checkout cannot retrieve them. This observation belongs to implementation revision `157d1f3e7f8c72b23a01879850f0e9a261b8522c`, not the changed verifier.

The saved manifest supports a reference-environment smoke observation. It does not prove the exact tested source solely from its commit ID, CUDA seed repeatability, custom decoder parity, paging, lifecycle safety, or performance.

## Required closure

1. Extend the retained CPU evidence with reviewed Linux/T4 artifacts under `results/validation/`, attributing source, lock, actual arguments, setup and installed dependencies. Retain differences/hashes for dirty source or rerun the final verifier from clean attributable source.
2. Retain a fresh Linux/T4 installation record and current GPU pytest report with collected/executed/skipped counts, including CUDA repeatability. A job in which every GPU test skips cannot satisfy the gate.
3. Re-run the changed verifier on attributable T4 source. Manifest version 2 removes assumed execution history; CLI configuration/manifest-write fixes and formatting normalization are CPU-tested but do not replace target-machine evidence.
4. Confirm the first GitHub-hosted CPU CI run and establish the repeatable T4 job before further GPU gates. Extend the existing CI workflow with service/container smoke checks when those surfaces exist. Keep target verification distinct from the later engine suites.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending. See the public [phase mapping](VALIDATION_PLAN.md#build-order) for task ownership and evidence requirements. The detailed local checklist is intentionally ignored by Git; it must not be the only definition of a public gate.
