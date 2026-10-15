# Implementation and evidence status

Snapshot: 2026-10-14. The [P3.2 CPU report](../results/validation/2026-10-14-p3.2-cpu.json)
records 346 passing non-GPU tests, including 22 dense-component cases, and passing
formatting/Ruff/strict mypy. No GPU execution occurred during this change. The
[P3.1 report](../results/validation/2026-10-14-p3.1-cpu.json) retains the earlier
weight-loading, distribution-build and real pinned-weight CPU-load evidence.

## Current gate

Phase 3 dense decoder parity is the current runtime gate. P3.1 weight loading is
implemented and CPU-validated, as are P3.2 dense components. P3.3 contiguous
prefill/decode is next. Layer/logit/greedy
T4 parity and dense workspace measurement remain unsatisfied. The October 13
archive and successful hosted CI below supersede the earlier Phase 2 evidence
blockers; they do not validate the owned loader or dense components on GPU.

| Surface | Implemented and observed | Evidence boundary |
|---|---|---|
| Package/configuration/diagnostics | Frozen CPU/CUDA graph, immutable pinned settings, safe diagnostics and target verifier | October 7 clean Linux/T4 reference setup exists; the environment manifest lacks the selected GPU UUID |
| Deterministic controls | Python/Torch seeding and inference mode | CPU repeatability plus one executed October 7 CUDA RNG repeatability test; no cross-device promise |
| Shared engine contracts | Immutable records, legal transitions, positive reservation credits and aggregate capacity checks | 108 passing CPU contract cases; concrete ownership/release-once/CUDA ordering remain P4/P6 work |
| Public fixtures | Exact pinned IDs at 15/16/17 and 32/49 tokens, whitespace, mixed lengths, maximum output and synthetic EOS cases | Current cached-tokenizer integration check executed and passed; these inputs do not prove batching or an owned generation loop |
| HF oracle | Frozen eager FP16 capture/comparison, strict deterministic policy, snapshots and safe artifact persistence | October 13 T4 reference repeatability for `boundary_15`, with setup and selected-device attribution |
| Internal timing analyzer | Separate generation/delivery durations, TTFT/ITL/TPOT and missing zero/one-token values | 18 synthetic CPU cases; early transport closure is valid, impossible generation ordering and invalid timestamps still fail |
| CPU CI | Push/PR workflow for frozen setup, format/lint/types/tests and distribution smoke | Hosted run `38445324328` succeeded for prior source; current loader/components have local CPU evidence only |
| Owned weight loading | Pinned safetensors/config validation, complete typed mapping, BF16-to-FP16 conversion, untied embedding/head | 69 synthetic contract cases plus all 201 real pinned tensors loaded on CPU; no forward pass or T4 loader validation |
| Dense decoder components | RMSNorm, Q/K/V, absolute RoPE, GQA mapping/dense attention, output projection, SwiGLU, final norm/LM head | 22 synthetic CPU cases; no complete decoder, pinned-checkpoint or T4 parity proof |
| Prefill/decode/paging/worker/service | Not implemented | No owned generation parity, paged reads, cleanup, concurrency, HTTP or GPU memory-safety proof |
| Container/benchmark/report | Not implemented | No container, serving performance or comparative measurement proof |

## Current CPU evidence

The current [dense-component report](../results/validation/2026-10-14-p3.2-cpu.json)
records commands, source/lock identity, dependency versions and CPU-only limits.
All 346 selected tests passed with zero skips; three GPU tests were deselected.
Synthetic component tests compare with pinned HF math and independently check
projection orientation, head mapping, absolute positions, causal visibility and
the untied LM head. They do not establish complete-layer or real-model parity.

The earlier [weight-loading report](../results/validation/2026-10-14-p3.1-cpu.json)
records commands, source/lock identity, dependency versions and CPU-only limits.
All 324 selected tests passed with zero skips; three GPU tests were deselected.
The full pinned checkpoint loaded as finite FP16 CPU tensors with 22 layer records.

The earlier [audit-fix report](../results/validation/2026-10-12-p2-audit-fixes/report.json) and
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

The newer [October 13 archive](../results/validation/2026-10-13-t4/t4-20261013T180545Z.zip)
at source `d6fb549` retains setup/commands, 255 passing CPU tests and three executed
passing GPU tests with zero skips, one CUDA-visible T4 and its UUID. The two
`boundary_15` oracle captures match exactly across 39 tensors. Reviewed again on
October 14, this closes the earlier environment/reference attribution gaps;
it does not prove an owned decoder, paging, memory safety or performance.

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

The newer [successful run](https://github.com/Anassbzdd/PagedCore/actions/runs/38445324328)
for `0b94ebf` supersedes the historical formatting failure. Its completed/success
status was rechecked through GitHub's API on October 14. That run predates the
weight loader and dense components; no hosted run was triggered for these local edits.

## Remaining decoder gate

Implement contiguous prefill/decode, then retain attributable
layer/logit/greedy parity and workspace measurements on T4 before advancing to the
allocator. Continue dense parity -> allocator -> paged attention -> lifecycle -> HTTP.

All final MVP acceptance rows in [TRACEABILITY.md](TRACEABILITY.md) remain pending.
The [public validation plan](VALIDATION_PLAN.md#build-order) defines gate ownership;
the ignored local checklist is not the only public definition of a gate.
