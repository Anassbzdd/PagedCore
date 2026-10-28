# MVP acceptance traceability

This table maps [MVP acceptance criteria](SPEC.md#acceptance-criteria) to their
implementation owners, current support and remaining evidence. All final
acceptance rows remain open: dense parity and allocator completion establish
prerequisites, while paged attention and the service are still planned.
The current gate is [Phase 5](STATUS.md#next-implementation-gate).
Task IDs are defined in the public [phase map](VALIDATION_PLAN.md#build-order).
A `pending` path names intended evidence, not a file that already exists.

| ID | Acceptance criterion | Owners | Current evidence and remaining checks | Final evidence |
|---|---|---|---|---|
| AC-01 | Exact greedy IDs and FP16 internal/logit parity with pinned HF | P3.4–P3.6, P5.6 | [Dense T4 comparisons](validation/2026-10-27-phase4-review.md) pass in [decoder tests](../tests/gpu/test_decoder_parity.py); paged decoder parity remains pending | `results/validation/decoder-parity.json` (pending) |
| AC-02 | KV allocation/reuse/isolation/cleanup; bounded queues/active requests; cancellation and slow-consumer safety | P4.5–P4.6, P6.4, P6.7–P6.8, P7.5, P7.7, P8.7 | [Allocator unit tests](../tests/unit/test_kv_cache.py), [property model](../tests/unit/test_kv_cache_properties.py) and [T4 storage tests](../tests/gpu/test_kv_pool_cuda.py) pass; scheduler, transport and lifecycle tests remain pending | `results/validation/resource-lifecycle.json` (pending) |
| AC-03 | A second request joins while the first is decoding | P6.3, P6.7 | `tests/integration/test_continuous_batching.py` (pending) | `results/validation/continuous-batching.json` (pending) |
| AC-04 | Poisoned fragmented KV matches HF; traces prove block-table reads without full-history gather | P5.4–P5.5 | `tests/gpu/test_paged_attention.py` (pending) | `results/validation/paged-attention.json` (pending) |
| AC-05 | CPU/GPU suites execute with counts and no implied GPU certification from CI | P2.5, P9.1–P9.2 | [CPU workflow](../.github/workflows/ci.yml), [current CPU report](../results/validation/2026-10-27/cpu-review.json) and [reviewed T4 run](validation/2026-10-27-phase4-review.md) exist; final integrated release matrix and current hosted CI remain pending | `results/validation/test-matrix.json` (pending) |
| AC-06 | Clean Linux/CUDA direct-install and Docker paths start the service | P9.3–P9.5 | `tests/integration/test_clean_install.py`, `scripts/smoke_docker.sh` and service startup (pending) | `results/validation/deployment-smoke.json` (pending) |
| AC-07 | One command produces reproducible workloads, raw records, percentile/count summaries and explicit insufficient-sample p99 | P10.3–P10.9 | Internal timing arithmetic exists; external benchmark analysis and command tests remain pending | `results/benchmark/manifest.json` (pending) |
| AC-08 | Public report includes measurements, configuration, limits and an investigated performance gap | P11.5–P11.7 | Release-doc checks and comparative measurements (pending) | `docs/REPORT.md` (pending) |

AC-01–AC-05 are correctness/validation prerequisites. Official measurement entry
also requires direct service smoke, memory/soak evidence and a validated harness.
AC-07/AC-08 close the technical demonstration; full AC-06 distribution/Docker
evidence closes artifact delivery. Follow the
[completion milestones](VALIDATION_PLAN.md#completion-milestones).

## Supporting evidence requirements

| Contract | Owners | Evidence content |
|---|---|---|
| Environment attribution | P1.5–P1.6, P9.2 | Actual setup/commands, source/lock hashes, selected GPU identity, CUDA repeatability and test counts; retained for the allocator revision |
| Bounded worker-to-loop delivery and terminal order | P6.1, P6.4–P6.5, P6.7–P6.8, P7.4–P7.7 | Stalled/closed loop, lost wakeups, last-token/done, cancellation and blocked-send tests (pending) |
| Ingress bounds and raw-prompt contract | P2.3–P2.4, P7.2–P7.3, P7.7 | Fixtures/special-token checks exist; body/tokenizer bounds, strict HTTP schema and error envelopes remain pending |
| Safe CUDA reuse and shutdown | P5.7, P6.4, P6.10, P7.1, P8.5–P8.7 | Allocator owning-stream guard passes; integrated access/reuse ordering, graceful zero ownership and forced escalation remain pending |
| Measurements and memory | P2.1, P3.7, P5.7, P6.9, P8.2–P8.6 | Synthetic clocks/timing math and dense T4 workspace exist; bounded live instrumentation and paged/batched/service memory remain pending |
| Comparable benchmarks and mechanism coverage | P10.1–P10.9, P11.1–P11.7 | Dispatch lag, granularity, terminal/deadline accounting, sampled memory and labeled mechanism experiments (pending) |
