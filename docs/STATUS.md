# Implementation and evidence status

Snapshot: 2026-10-05, based on commit `ab05602e721beb6b0fc4266a9642b09f89eee6c1` with uncommitted implementation changes identified by source hashes and a sanitized patch in the [CPU validation report](../results/validation/2026-10-05-p1.6-cpu.json). This page records observed state, not a release certificate.

## Current gate

Phase 1 implementation is substantially delivered; its evidence closure remains open. Next comes core types/reference fixtures, followed by dense parity -> allocator correctness -> paged attention -> lifecycle correctness -> HTTP. Early CI and validation procedures support that order without implementing later runtime features.

| Surface | Observed implementation | Evidence limit |
|---|---|---|
| Package/environment | Metadata, committed lock, Python pin, CPU/CUDA extras | No retained fresh Linux/T4 clean-install report inspected |
| Configuration | Immutable typed settings and validation | CPU tests; worker/service-only settings remain planned |
| Diagnostics | Safe JSON diagnostics, hardware fallback and expected CLI configuration-error handling | CPU coverage; hardware output is not engine certification |
| Deterministic controls | Python/Torch seeds and inference mode | CPU repeatability checked; no retained CUDA repeatability report |
| Target verifier | Pinned snapshot, FP16/config/logit checks, source hashes, sanitized actual CLI arguments, manifest-write error handling | CPU regressions pass; fresh T4 evidence for the changed verifier remains open |
| Decoder, allocator, paged attention | Not implemented | No parity, ownership, or paged-read evidence |
| Scheduler, worker, HTTP/SSE | Not implemented | No concurrency, cleanup, backpressure, or service evidence |
| CI, container, benchmark/report | Not implemented | Existing build outputs do not certify current artifacts |

## Observed CPU checks

The [retained report](../results/validation/2026-10-05-p1.6-cpu.json) records Windows, Python 3.11.15, Torch 2.7.1+cpu, Transformers 4.52.4 and unavailable CUDA. The existing environment was synchronized with `uv sync --frozen --extra cpu --extra dev`; this was not a fresh Linux installation. Ruff lint/format, full-source strict mypy and the offline frozen-lock check passed. Non-GPU pytest collected 67 tests, selected and executed **65**, passed all 65, skipped none, and deselected two GPU tests. Expected CLI failures were reproduced before the fix and covered by regression tests.

An explicit GPU-marked run selected two tests and skipped both with `requires CUDA`: **zero GPU tests executed**. The report retains sanitized pytest output, JUnit reports, source/lock hashes, the source patch, setup arguments, installed dependencies and seeds. CLI diagnostics passed; target verification returned the expected JSON failure without CUDA. These are local CPU observations, not CI or T4 certification.

## Saved T4 observation

An ignored local manifest dated `2026-10-04T14:37:18.785925Z` reports Tesla T4, CUDA 12.6, Torch 2.7.1+cu126, Transformers 4.52.4, the pinned TinyLlama model/tokenizer, and finite FP16 reference logits shaped `[1, 8, 32000]`. Its Git revision is the original implementation commit `157d1f3e7f8c72b23a01879850f0e9a261b8522c` and its lock hash is `ae7561ac683bfa084defe747ef8aa9380b65a42f7ed7b245ecf3fb9fe0e3df01`.

It also reports a dirty working tree without saved changes/source hashes and hardcoded command/setup provenance. No associated GPU pytest report was inspected. The older tracked validation manifest is absent from the current checkout. This page deliberately has no link to either a missing file or ignored local evidence; a fresh public checkout cannot retrieve them. This observation belongs to implementation revision `157d1f3e7f8c72b23a01879850f0e9a261b8522c`, not the changed verifier.

The saved manifest supports a reference-environment smoke observation. It does not prove the exact tested source solely from its commit ID, CUDA seed repeatability, custom decoder parity, paging, lifecycle safety, or performance.

## Required closure

1. Extend the retained CPU evidence with reviewed Linux/T4 artifacts under `results/validation/`, attributing source, lock, actual arguments, setup and installed dependencies. Retain differences/hashes for dirty source or rerun the final verifier from clean attributable source.
2. Retain a fresh Linux/T4 installation record and current GPU pytest report with collected/executed/skipped counts, including CUDA repeatability. A job in which every GPU test skips cannot satisfy the gate.
3. Re-run the changed verifier on attributable T4 source. Manifest version 2 removes assumed execution history; CLI configuration/manifest-write fixes and formatting normalization are CPU-tested but do not replace target-machine evidence.
4. Add CPU CI now and establish the repeatable T4 job before further GPU gates. Keep target verification distinct from the later engine suites.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending. See the public [phase mapping](VALIDATION_PLAN.md#build-order) for task ownership and evidence requirements. The detailed local checklist is intentionally ignored by Git; it must not be the only definition of a public gate.
