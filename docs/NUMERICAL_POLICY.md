# Numerical correctness and HF fidelity

Decision: November 9, 2026. The owner authorized replacing the integrated
HF-eager reproduction objective with independently validated paging accuracy.
This is a deliberate contract revision, not a passing validation result.

## Required checks

1. **Architecture and weights.** Retain the existing pinned eager HF dense
   layer, prefill and public greedy tests. Their `2e-3` tensor and `5e-3`
   logit absolute/relative tolerances and exact greedy checks are unchanged.
   These establish the model implementation independently of paging.
2. **Attention mathematics.** Validate the paged primitive against independently
   assembled FP64 attention, at `atol=rtol=2e-3`. Exercise boundaries, GQA,
   poisoned unrelated pages, noncontiguous tables, long context and large scores.
   The FP64 reference may gather history in tests; serving may not.
3. **Integrated paging accuracy.** Compare real-checkpoint paged decode against
   independent contiguous decode using FP32 scores, scaling, softmax and weighted
   values, with one final attention-output cast to FP16. The contiguous reference
   uses full matrix products and shares neither page traversal nor online-softmax
   implementation. It shares weights, orchestration and non-attention components;
   the unchanged HF architectural tests cover mistakes in those shared components.
   Keep `atol=rtol=5e-3` for logits and `2e-3` for K/V. Require
   exact greedy choices against this reference on the frozen validation suite.
   Check rollback/retry, untouched history, stable pool storage and zero ownership
   after cleanup under the existing no-full-history guard.
4. **HF fidelity report.** For every integrated comparison, also run eager HF on
   the same input-token history. Retain maximum absolute error, RMS and normalized
   RMS logit errors, the old tolerance ratio, top-one agreement and the reference
   top-one/top-two margin. Differences are reported rather than rejected solely
   for crossing the old per-element HF tolerance or choosing a different token.
   Missing, malformed or non-finite fidelity evidence rejects the collection.

Normalized RMS error is `RMS(paged - reference) / max(RMS(reference), 1)`;
the unit floor keeps nearly zero reference logits well-defined. Metrics are
computed in FP64 from the saved runtime logits. No fidelity aggregate is a
replacement for the required per-case paging comparisons.

Single, mixed and fragmented cases retain the public prompt lengths 1/15/16/17/49
and three cached steps. An additional held-out layout case uses new prefixes
37/73/111/145 and three cached steps; these were not used to fit the 17/18-token
reduction hypotheses. This is extra numerical/layout coverage, not a held-out
natural-language quality benchmark. The source, fixture and policy identity must accompany
the complete T4 reports. Old HF-only reports cannot satisfy this revised gate.

## Arithmetic policy

Weights, K/V cache and surrounding model activations remain FP16. Prefill and
the original dense HF-comparison path retain their validated eager arithmetic.
Paged decode uses the existing FP32 online-softmax primitive. Its scores,
normalization and V accumulation stay FP32 until the final attention cast.
This policy is uniform across lengths and kernel families. The independent
contiguous decode reference uses FP32 dense attention with the same mathematical
policy, without reproducing the online reduction or reading a block table.

HF eager remains an architectural and behavioral reference, not a promise of
bitwise identity for a different attention implementation. The HF fidelity
numbers are measurements, not a universal accuracy guarantee. All token inputs
in these integrated forward tests follow the owned contiguous reference; HF is
fed those same tokens, so differences are compared on a common history. This
does not establish free-running generation fidelity or natural-language quality.
P5.6 must separately retain free-running behavior and stopping/EOS comparisons,
report divergence positions, and preserve exact paged-versus-owned-reference
checks on its frozen suite. No HF mismatch may be presented as an exact match.

## Evidence boundary

P5.3 closes only after all required revised tests execute and pass on a T4 and
the complete, versioned accuracy and HF-fidelity reports are retained and reviewed.
Changing tests or passing CPU cases cannot close it. Kernel-family sweeps and
legacy HF-rounding candidate tests remain forensic experiments, separate from
release acceptance. They must not become serving dispatch rules.

Allocator ownership, queue bounds, cancellation, page-table reads and one-page
K/V scratch restrictions remain exact requirements. End-to-end generation,
maximum-active workspace, safe CUDA reuse, service lifecycle and benchmarks
remain separate documented gates. This revision adds no performance claim.
