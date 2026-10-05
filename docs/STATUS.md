# Implementation and evidence status

Snapshot: 2026-10-05, implementation commit `157d1f3e7f8c72b23a01879850f0e9a261b8522c`. This page records observed state, not a release certificate. Update it only after inspecting the corresponding source and evidence.

## Current gate

Phase 1 implementation is substantially delivered; its evidence closure remains open. Next comes core types/reference fixtures, followed by dense parity -> allocator correctness -> paged attention -> lifecycle correctness -> HTTP. Early CI and validation procedures support that order without implementing later runtime features.

| Surface | Observed implementation | Evidence limit |
|---|---|---|
| Package/environment | Metadata, committed lock, Python pin, CPU/CUDA extras | No retained fresh Linux/T4 clean-install report inspected |
| Configuration | Immutable typed settings and validation | CPU tests; worker/service-only settings remain planned |
| Diagnostics | Safe JSON diagnostics and hardware fallback | CPU coverage; hardware output is not engine certification |
| Deterministic controls | Python/Torch seeds and inference mode | CPU repeatability checked; no retained CUDA repeatability report |
| Target verifier | Pinned local snapshot, FP16/config/logit checks, provenance fields | One saved Hugging Face reference forward; not custom-engine evidence |
| Decoder, allocator, paged attention | Not implemented | No parity, ownership, or paged-read evidence |
| Scheduler, worker, HTTP/SSE | Not implemented | No concurrency, cleanup, backpressure, or service evidence |
| CI, container, benchmark/report | Not implemented | Existing build outputs do not certify current artifacts |

## Observed CPU checks

The preceding audit used Python 3.11.15, Torch 2.7.1+cpu, and Transformers 4.52.4, with CUDA unavailable. Ruff lint and full-source strict mypy passed. Non-GPU pytest reported **55 passed, 2 deselected**. The two deselected tests are GPU-marked; they did not execute. Ruff formatting reported five files requiring normalization. `git diff --check` passed before this documentation change.

These were local audit observations, not a checked-in test report or CI run. No code or test remediation is claimed by the documentation update.

## Saved T4 observation

An ignored local manifest dated `2026-10-04T14:37:18.785925Z` reports Tesla T4, CUDA 12.6, Torch 2.7.1+cu126, Transformers 4.52.4, the pinned TinyLlama model/tokenizer, and finite FP16 reference logits shaped `[1, 8, 32000]`. Its Git revision matches the implementation commit above and its lock hash is `ae7561ac683bfa084defe747ef8aa9380b65a42f7ed7b245ecf3fb9fe0e3df01`.

It also reports a dirty working tree without saved changes/source hashes. No associated GPU pytest report was inspected. The older tracked validation manifest is deleted in the working tree. This page deliberately has no link to either a missing file or ignored local evidence; a fresh public checkout cannot retrieve them.

The saved manifest supports a reference-environment smoke observation. It does not prove the exact tested source solely from its commit ID, CUDA seed repeatability, custom decoder parity, paging, lifecycle safety, or performance.

## Required closure

1. Preserve reviewed evidence under tracked `results/validation/`, attributing source, lock, actual arguments, setup, and installed dependencies. Establish dirty-source identity through retained differences/hashes or rerun the final verifier from clean attributable source.
2. Retain a fresh Linux/T4 installation record and current GPU pytest report with collected/executed/skipped counts, including CUDA repeatability. A job in which every GPU test skips cannot satisfy the gate.
3. Correct hardcoded-command provenance and expected CLI manifest-write failure handling, and normalize formatting in a later code change with regression tests. Documentation edits do not repair those implementations.
4. Add CPU CI now and establish the repeatable T4 job before further GPU gates. Keep target verification distinct from the later engine suites.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending. See the public [phase mapping](VALIDATION_PLAN.md#build-order) for task ownership and evidence requirements. The detailed local checklist is intentionally ignored by Git; it must not be the only definition of a public gate.
