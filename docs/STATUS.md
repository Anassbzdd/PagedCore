# Implementation and evidence status

Snapshot: 2026-11-11. **Current gate: Phase 5 block-table decode attention;
P5.3 remains open after integrated T4 logit/KV failures.** Phase 4 is complete
at the allocator boundary. P5.1 prompt writes and P5.2 standalone attention have
retained T4 evidence; integrated batched decode is implemented but has not passed
its independent FP32 contract. No serving arithmetic changed in this scope revision.

## Current gate

The November 10 [archive review and notebook correction](validation/2026-11-10-notebook-output-directory.md)
records the returned `pagedcore-accuracy-kavvwb5q.zip`: 15 GPU passes and four
integrated logit failures, including a two-token case, plus 760 non-GPU passes
and two notebook-test failures. The notebook output-directory mismatch is
corrected with four CPU regression cases; attention arithmetic is unchanged.
P5.3 remains open. The paired reproduction and first differing T4 intermediate
are now confirmed below; no serving repair is selected.

The [two-token T4 reproduction review](validation/2026-11-10-two-token-reproduction.md#returned-t4-evidence)
verifies `pagedcore-two-token-irzdb020.zip`: all 17 evidence and 82 source hashes
match. One prompt token plus one decode step fails alone and as row 0 of the
original mixed batch, with identical full signed logit errors (maximum ratio
2.1705). Both raw paged/reference logits also match the prior failed mixed
case exactly. Greedy choices match; the measured K/V ratio is 1.9796 and cleanup
returns ownership to zero in both arms. For this fixture, neither batch membership
nor a page boundary is needed to trigger the failure. The earlier CPU result
is separate (ratio 2.7363). Serving math and GPU acceptance tests are unchanged;
the standalone trace below now localizes its first differing intermediate.

The [T4 layer-trace review](validation/2026-11-10-layer-trace-t4-review.md) verifies
`pagedcore-layer-trace-1__el9f9.zip`: 38 evidence hashes, 85 source hashes and all
418 raw comparisons match. On T4, layer-0 inputs, pool reads, QK products and
scaled scores are exact; the first difference is the FP32 attention result before
casting (maximum 7.45e-9). The first FP16 difference appears at layer 5, head 0,
channel 3, across a rounding midpoint; hidden/Q/K/V drift begins at layer 6.
All old cache slots remain exact, and all new slots match their own projections.
The measured K/V ratio is 1.9796; final logit ratio stays 2.1705, greedy choices
match, instrumented logits equal baseline and cleanup returns ownership to zero.
An isolated FP64 replay from shared Q/K/V rounds to the paged value at the first
FP16 coordinate; this is not integrated accuracy evidence. Next isolate
normalization/PV/final-division rounding on fixed T4 operands before choosing a
repair. The earlier [CPU QK finding](validation/2026-11-10-two-token-layer-trace.md)
does not transfer to T4. Earlier initialized K/V zeros remain unmeasured.

The [returned layer-5 T4 replay](validation/2026-11-10-layer5-replay.md#returned-t4-replay)
verifies the script/input identity and all eight arms' raw metrics. Both native
outputs exactly reproduce the trace. All three normalize-first variants match
the full reference output; normalize-last differs. Promoting final division alone
does not change the first differing FP16 scalar. Against FP64 attention on the
shared Q/K/V, normalize-last has smaller maximum error (9.93e-9 vs 1.18e-8) and
fewer rounded FP16 mismatches (1 vs 2 of 2,048). This isolates normalization
placement sensitivity for one call and does not select a serving repair or
change the integrated reference contract. Broader common-input FP64 controls
remain needed; no integrated rerun occurred.

The November 11 owner-authorized [numerical policy](NUMERICAL_POLICY.md) removes
required HF parity/token identity/fidelity, without changing logit `5e-3`, K/V
`2e-3`, exact owned-reference greedy checks or the no-full-history guard.
Independent FP64 mathematics and paging invariants stay required. Independent
component checks have partial CPU coverage; shared-component/full-checkpoint gaps
remain explicit release work. Held-out cached-decode loss/perplexity, fixed tasks
and distribution fidelity versus vLLM are planned, with no evaluator, frozen
margins/counts, established vLLM pin or quality result yet.

Required tooling now uses owned checkpoint smoke and ten GPU cases with versioned
`fp32-paged-v2-independent` reports. HF tests require explicit
`--run-hf-diagnostics`. Old reports and their failures retain original policy IDs.
This is a scope/tooling revision with CPU verification, not a new T4 pass.
See the [revision and cleanup review](validation/2026-11-11-validation-scope-revision.md)
for executed commands, cleanup and limitations. Two unnecessary local ZIPs were
removed; recursive cache/build cleanup was blocked by automatic approval review
and remains incomplete. The working environment and raw evidence are retained.

## Implemented capabilities and evidence

| Capability | Evidence boundary |
|---|---|
| Pinned weights/tokenizer, owned dense decoder | Loader/component/cache CPU tests; historical dense HF T4 evidence at recorded revisions, now optional to rerun |
| Reservation-aware allocator | [October 27 review](validation/2026-10-27-phase4-review.md): 554 CPU / 12 GPU passes at clean `986392e`; newer property checks separate |
| Paged prefill | [October 30 review](validation/2026-10-30-p52-t4-review.md) at `922588b`; current test uses owned dense prefill and needs new T4 evidence |
| Standalone online attention | [October 31 review](validation/2026-10-31-p52-t4-review.md): two runs, each 617 CPU / 15 GPU passes at `3401406` |
| Staged batched decode and full-history guard | Synthetic CPU coverage; real integrated T4 failures above, no completed P5.3/P5.4 acceptance |
| Worker, FIFO admission, HTTP/SSE | Planned; allocator tests do not prove worker/transport lifecycle |
| Batched/service memory, safe reuse, quality and benchmarks | Open; dense workspace does not certify these |

## Current CPU evidence

November 11 final non-GPU verification: **764 passed, 59 deselected, zero skipped**.
Ruff lint/format, strict mypy and packaging/smoke checks pass on Windows CPU.
Exact commands are retained in the
[revision review](validation/2026-11-11-validation-scope-revision.md) and its
[CPU report](../results/validation/2026-11-11/cpu-review.json).
CPU results establish code/schema/collector regressions and synthetic math only.
They do not validate GPU correctness, memory safety, quality or performance.

## Saved T4 observations

The [historical status snapshot](validation/2026-11-11-status-before-scope-revision.md)
preserves the previous detailed results and limitations without keeping obsolete
instructions in this current snapshot. Original raw archives and dated reviews,
including November 10 layer trace/replay and the user's structured replay review,
are retained. HF failures remain HF failures; the independent FP32 failures remain
unresolved. No new local GPU execution occurred.

## Next implementation gate

Open [paged_decode_accuracy.ipynb](../notebooks/paged_decode_accuracy.ipynb) in a
fresh Kaggle session, enable **Internet** and **T4 GPU**, **Run All**, and return
the final ZIP even on failure. It records the revised required math/paging suite
without an HF model. Passing establishes only those comparisons; known failure
may persist. A comparator redesign first needs measured cause and independent
replacement evidence under the policy, rather than another rounding imitation.

P5.6 end-to-end paged generation and P5.7 batched workspace/CUDA reuse remain open.
Independent model coverage and planned quality evaluation also block release.
Worker lifecycle, HTTP, deployment and the frozen T4 comparison follow their gates
in [the phase map](VALIDATION_PLAN.md#build-order). Final acceptance in
[TRACEABILITY.md](TRACEABILITY.md) remains open.
