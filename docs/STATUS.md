# Implementation and evidence status

Snapshot: 2026-10-12. The [audit-fix CPU report](../results/validation/2026-10-12-p2-audit-fixes/report.json) records 255 passing non-GPU tests, including 18 timing cases, plus passing repository formatting, Ruff lint and strict mypy. Pending and active disconnects can now terminate transport before worker generation termination without invalidating timing analysis. These are Windows/CPU results; no new GPU execution occurred during this change.

## Current gate

Phase 2 implementation is delivered, including the timing correction. Its complete evidence gate remains open. The retained October 12 oracle run has no installer history or CUDA-visible device count, and the fixed source still needs a successful hosted CPU CI run. The next runtime gate is Phase 3 dense decoder parity; do not advance until those evidence requirements close.

| Surface | Implemented and observed | Evidence boundary |
|---|---|---|
| Package/configuration/diagnostics | Frozen CPU/CUDA graph, immutable pinned settings, safe diagnostics and target verifier | October 7 clean Linux/T4 reference setup exists; the environment manifest lacks the selected GPU UUID |
| Deterministic controls | Python/Torch seeding and inference mode | CPU repeatability plus one executed October 7 CUDA RNG repeatability test; no cross-device promise |
| Shared engine contracts | Immutable records, legal transitions, positive reservation credits and aggregate capacity checks | 108 passing CPU contract cases; concrete ownership/release-once/CUDA ordering remain P4/P6 work |
| Public fixtures | Exact pinned IDs at 15/16/17 and 32/49 tokens, whitespace, mixed lengths, maximum output and synthetic EOS cases | Current cached-tokenizer integration check executed and passed; these inputs do not prove batching or an owned generation loop |
| HF oracle | Frozen eager FP16 capture/comparison, strict deterministic policy, snapshots and safe artifact persistence | October 12 T4 reference repeatability proved for `boundary_15`; setup history and CUDA visibility for that run remain unknown |
| Internal timing analyzer | Separate generation/delivery durations, TTFT/ITL/TPOT and missing zero/one-token values | 18 synthetic CPU cases; early transport closure is valid, impossible generation ordering and invalid timestamps still fail |
| CPU CI | Push/PR workflow for frozen setup, format/lint/types/tests and distribution smoke | Latest inspected hosted run failed at formatting; the local fix passes, but no green hosted result for the fixed source exists yet |
| Decoder/paging/worker/service | Not implemented | No owned parity, paged reads, cleanup, concurrency, HTTP or GPU memory-safety proof |
| Container/benchmark/report | Not implemented | No container, serving performance or comparative measurement proof |

## Current CPU evidence

The [retained report](../results/validation/2026-10-12-p2-audit-fixes/report.json) and
[JUnit report](../results/validation/2026-10-12-p2-audit-fixes/cpu-tests.xml) identify
source/lock hashes, actual commands, the existing local environment and test counts:
258 collected, 255 selected/executed/passed, zero failed/skipped, and three GPU
cases deselected. The new pending/active disconnect tests failed before the timing
fix and pass afterward. Formatting, Ruff lint and strict mypy over eight source
files pass. The T4 collector passes Bash syntax checking only.

Earlier CPU evidence remains historical: [P1.6](../results/validation/2026-10-05-cpu/2026-10-05-p1.6-cpu.json),
[P2.1](../results/validation/2026-10-07-p2.1-cpu.json),
[P2.2](../results/validation/2026-10-08-p2.2-cpu.json),
[P2.3](../results/validation/2026-10-08-p2.3-cpu.json),
[oracle arithmetic](../results/validation/2026-10-09-p2.4-cpu.json),
[strict reference policy](../results/validation/2026-10-10-reference-determinism-cpu.json)
and [marker selection](../results/validation/2026-10-12-p2.5-cpu.json).
Their old formatting failures do not describe the corrected current checkout.

## Saved T4 observations

The [evidence review](../results/validation/2026-10-12-t4-evidence-review.json)
separates two runs and records hashes, actual results and unknown fields.

On October 7, the retained [manifest](../results/validation/2026-10-07-t4/manifest.json),
[setup/command log](../results/validation/2026-10-07-t4/commands.log) and
[GPU report](../results/validation/2026-10-07-t4/pytest-report.xml) identify clean
source `211bb0c`, the frozen lock, Linux, pinned Python 3.11.15, Torch 2.7.1+cu126
and Transformers 4.52.4. A reference forward and CUDA RNG repeatability both
executed and passed, with zero skips. Two physical T4s were present; one was
CUDA-visible. The manifest does not record its UUID. This is environment evidence.

The archive stored at [2026-10-07-t4/p2.4-evidence.zip](../results/validation/2026-10-07-t4/p2.4-evidence.zip)
contains a separate run captured on October 12 at source `51dcf8f`. One oracle
GPU test and 58 synthetic oracle/timing tests executed and passed with zero skips.
Two `boundary_15` captures have three emitted tokens and 39 FP16 tensors, exact
ID/outcome/text agreement and zero observed tensor differences. Eager attention,
strict deterministic settings, cuBLAS workspace, selected T4 UUID, dependencies
and source/lock identity are retained. The current oracle implementation matches;
the timing analyzer and shared fixture formatting changed afterward.

The October 12 archive has no installer/setup log and does not identify the
CUDA-visible device count. Its physical inventory lists two T4s; the selected
`cuda:0` is explicit. This proves HF reference repeatability for one fixture,
not an owned decoder, paging, memory safety or performance.

## Hosted CPU CI

The [latest inspected run](https://github.com/Anassbzdd/PagedCore/actions/runs/38274851925)
for `6bd0ca7` failed at **Check formatting**. Installation succeeded; lint, types,
tests, build and wheel smoke were skipped. This was verified through GitHub run
and job metadata on October 12 and is retained in the audit-fix report. A local
CPU pass does not turn that historical hosted failure into a success.

## Required closure

1. Run `bash scripts/validate_t4.sh` on the updated clean Linux/T4 checkout. Review
   and retain its ZIP, including actual setup/commands, physical inventory,
   CUDA-visible identity, reference captures and three executed passing GPU tests.
   Existing environment synchronization is labeled separately from a fresh install.
2. Retain a successful hosted CPU CI run for the fixed source, including its test
   artifact and distribution checks. No hosted run was triggered for these local edits.
3. Close P2.4 and the Phase 2 exit gate only after those reports meet the evidence
   contract. Continue dense parity -> allocator -> paged attention -> lifecycle -> HTTP.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending.
The [public validation plan](VALIDATION_PLAN.md#build-order) defines gate ownership;
the ignored local checklist is not the only public definition of a gate.
