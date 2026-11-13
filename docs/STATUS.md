# Implementation and evidence status

Snapshot: 2026-11-13. **Current gate: Phase 5 block-table decode attention;
P5.3 remains open after integrated T4 logit/KV failures.** Phase 4 is complete
at the allocator boundary. P5.1 prompt writes and P5.2 standalone attention have
retained T4 evidence; integrated batched decode is implemented but has not passed
its independent FP32 contract. The November 13 normalize-before-values adaptation
passes standalone T4 attention and fixes the recorded two-token mismatch, but
integrated T4 acceptance still fails at 16 and 38 attended tokens.

## Current gate

The [normalize-before-values T4 review](validation/2026-11-13-normalize-first-accuracy-t4-review.md)
verifies `pagedcore-accuracy-p6ts13rg.zip`: all 26 evidence and 97 source hashes
match, including the changed attention/tests. CPU: 767 passes. GPU: six passes,
four integrated failures. Both focused attention tests pass on T4. Mixed and
fragmented two-token logits now match exactly, with measured K/V ratio zero.
The remaining 16-token rows fail at ratio 1.303875; the first 38-token row fails
at 1.909388. Only six of 45 rows are recorded; the four failed rows' K/V zeros
are unmeasured placeholders. All six recorded greedy choices match, not complete
generation. The correctly rejected archive and prior failures are retained.

The [vLLM implementation study](validation/2026-11-13-vllm-implementation.md)
pins v0.8.5 source and maps its V0/XFormers CUDA decode path to PagedCore.
`attention.paged_attention` now stores bounded FP32 scores, normalizes once, then
rereads valid V pages for FP32 accumulation and one final FP16 cast. The decoder,
allocator, dense reference, limits and thresholds are unchanged. Focused CPU
attention/decoder checks pass (127 tests); the full CPU checks are recorded below.
The T4 result above adds component evidence and a partial improvement, without
superseding the failed integrated evidence.

The November 13 [common-input T4 review](validation/2026-11-13-common-input-t4-review.md)
verifies all 44 layer/input cases and 88 matching native tensors from the retained
two-token trace. Both archived attention paths satisfy the existing `2e-3`
attention tolerance. Paged FP32 maximum error is lower in 24 cases, contiguous in
12, equal in eight; neither is universally closer to FP64. These correlated cases
do not establish model quality or integrated acceptance. The archive runs the
older online normalize-last implementation; it does not validate the separately
prepared uncommitted normalize-first implementation. Existing integrated failures
and P5.3 remain open; see the review for source identity and limits.

The November 13 [accuracy archive review](validation/2026-11-13-accuracy-t4-review.md)
verifies the November 12 T4 run `pagedcore-accuracy-zh7r_l2j.zip`: all 24 evidence
and 94 embedded source hashes verify; 764 CPU cases and six GPU cases pass, while
all four integrated cases fail. Recomputed raw logit ratios are 1.4751 (single),
2.1705 (mixed/fragmented) and 2.0347 (held-out lengths). Only four of 45 integrated
rows were collected; K/V zeros are unfinished placeholders. All four saved
paged/reference tensor pairs exactly match the earlier failed archive, preserving
the relevance of the existing two-token trace. That review changed no math;
P5.3 stays open after the separately implemented adaptation above.

The November 10 [archive review and notebook correction](validation/2026-11-10-notebook-output-directory.md)
records the returned `pagedcore-accuracy-kavvwb5q.zip`: 15 GPU passes and four
integrated logit failures, including a two-token case, plus 760 non-GPU passes
and two notebook-test failures. The notebook output-directory mismatch is
corrected with four CPU regression cases; attention arithmetic is unchanged.
P5.3 remains open. The paired reproduction and first differing T4 intermediate
are confirmed below; the new adaptation is separately described above.

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
FP16 coordinate; this is not integrated accuracy evidence. The retained replay
below isolates normalization placement; the new adaptation has not run on T4.
The earlier [CPU QK finding](validation/2026-11-10-two-token-layer-trace.md)
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
| Historical online attention | [October 31 review](validation/2026-10-31-p52-t4-review.md): two runs, each 617 CPU / 15 GPU passes at `3401406`; current two-pass adaptation needs T4 evidence |
| Bounded normalize-before-values attention | Both strengthened standalone tests pass in the [new T4 review](validation/2026-11-13-normalize-first-accuracy-t4-review.md); integrated acceptance still fails |
| Staged batched decode and full-history guard | Synthetic CPU coverage; real integrated T4 failures above, no completed P5.3/P5.4 acceptance |
| Worker, FIFO admission, HTTP/SSE | Planned; allocator tests do not prove worker/transport lifecycle |
| Batched/service memory, safe reuse, quality and benchmarks | Open; dense workspace does not certify these |

## Current CPU evidence

November 13 adaptation verification on Windows, Torch `2.7.1+cpu`, CUDA
unavailable: **767 passed, 59 deselected, zero skipped**. Focused attention/decoder
checks: **127 passed, 19 deselected**. Final commands (using `.venv/Scripts/python.exe`):

```text
-m ruff check .
-m ruff format --check .
-m mypy --strict src/pagedcore
-m pytest tests/unit/test_attention.py tests/unit/test_decoder.py -q
-m pytest -m "not gpu and not benchmark" -q
```

All pass; `git diff --check` passes. The
[CPU report](../results/validation/2026-11-13/vllm-adaptation-cpu.json) retains
source hashes, commands and the expected pre-adaptation regression failure.
The [implementation study](validation/2026-11-13-vllm-implementation.md) explains
coverage and limits. No new T4 execution or release-gate acceptance is claimed.

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

The changed implementation now has reviewed T4 evidence above. Do not repeat the
unchanged full collection or resume rounding-order sweeps. The remaining first
16-token failure occurs within one page; its first differing intermediate is not
captured here. Any next cause-localization work should be bounded to that case
before selecting another repair. Keep current thresholds and the comparator;
no new repair or acceptance redesign is selected by this review. The existing
[accuracy notebook](../notebooks/paged_decode_accuracy.ipynb) remains the acceptance
entry point after a justified change, with failures retained.

P5.6 end-to-end paged generation and P5.7 batched workspace/CUDA reuse remain open.
Independent model coverage and planned quality evaluation also block release.
Worker lifecycle, HTTP, deployment and the frozen T4 comparison follow their gates
in [the phase map](VALIDATION_PLAN.md#build-order). Final acceptance in
[TRACEABILITY.md](TRACEABILITY.md) remains open.
