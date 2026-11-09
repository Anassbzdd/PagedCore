# Independent paging accuracy instead of HF eager reproduction

On November 9, 2026, the owner explicitly authorized changing the project goal:
validate a correct owned paged engine without reproducing every HF eager rounding
decision. The [numerical policy](../NUMERICAL_POLICY.md) is the revised acceptance
contract. This review records an implementation change and CPU preparation;
it does not close P5.3.

## Why the contract changes

The [corrected T4 candidate review](2026-11-09-value-followup-review.md) shows a
test-only reducer passing one 16/17/18-token trajectory, while other histories
still differ in score arithmetic with identical Q/K/V. The observations isolate
HF arithmetic/layout fidelity but do not establish a universal reduction rule or
a serving fix. The project will preserve those results without extending
kernel-family hypotheses into the runtime.

## Implementation

- `decode_paged` selects the existing FP32 online-softmax attention primitive.
  Score products, normalization and weighted values stay FP32 until the final
  attention-output cast. FP16 weights and KV storage remain unchanged.
- `decode(..., fp32_attention=True)` supplies the contiguous validation reference.
  It computes dense FP32 QK, softmax and PV, independently of page traversal and
  online normalization. Default dense decode and prefill preserve the historical
  HF policy. The reference shares non-attention components, so existing HF
  architectural checks remain required.
- Real-checkpoint tests require logits at `atol=rtol=5e-3`, K/V at
  `atol=rtol=2e-3`, exact owned-reference greedy choices, unchanged history,
  rollback/retry, storage stability and zero ownership after cleanup. The
  no-full-history guard is unchanged. Independent FP64 attention tests remain.
- HF runs on the same input-token histories. Every comparison retains logit
  errors, top-one agreement and margins, including disagreement. Missing,
  incomplete or non-finite reports cannot pass. Historical reducer experiments
  remain available through explicitly selected forensic tests.

## The next T4 run

Import [paged_decode_accuracy.ipynb](../../notebooks/paged_decode_accuracy.ipynb)
into a fresh Kaggle notebook, enable Internet and T4, and run all cells. It embeds
the exact source and frozen dependencies; no Git push, source-restoration cell
or Input dataset is needed. If Kaggle exposes two GPUs, child processes select
only GPU 0.

The four integrated cases contain 45 request/step comparisons:

| Case | Prompt lengths | What it decides |
| --- | --- | --- |
| Single | 15 | Whether independent accuracy holds through 16/17/18 cached tokens |
| Mixed | 1/15/16/17/49 | Whether row order and distinct cache positions preserve the result |
| Fragmented | 1/15/16/17/49 | Whether nonadjacent pages, poison and rollback/retry preserve correctness |
| Held-out layout | 37/73/111/145 | Whether the policy works beyond the lengths used to fit reducer hypotheses |

Each case runs three cached steps and reverses row order on the second step.
The extra prefixes are layout/numerical coverage, not a natural-language quality
benchmark. Inputs follow the independent owned reference; this is not a
free-running HF generation comparison.

The notebook verifies the pinned checkpoint/tokenizer before CPU tests, so the
offline tokenizer check can execute in a fresh Kaggle session. It then runs lint,
format, strict typing, the non-GPU suite and 19 required GPU cases. It writes one
`pagedcore-accuracy-*.zip` containing the source snapshot
and hashes, dependency/setup logs, commands and return codes, physical GPU
inventory and selected GPU identity, CPU/GPU JUnit reports, raw paged/reference/HF
logits and the versioned `accuracy-summary.json`. Partial failures are archived.
Interrupted or unsuccessful required commands reject the summary even if their
partial JUnit entries appear to pass.

After the archive returns, verify its source/checksums and complete reports,
review both accuracy and HF fidelity, and diagnose any remaining owned-reference
failure before selecting a repair. Only a successful reviewed T4 run can close
P5.3. End-to-end generation, maximum-active workspace, safe CUDA reuse, lifecycle
and service validation remain open under their own gates.

## Local verification

The [CPU preparation report](../../results/validation/2026-11-09/fp32-policy-cpu.json)
records commands, counts, source hashes and limitations. Ruff lint/format, strict
mypy and the full non-GPU suite pass. A separate synthetic 22-layer CPU smoke
exercises single and fragmented test orchestration; it shares small random-weight
layers and is not pinned-checkpoint or GPU evidence. GPU collection is not execution.
