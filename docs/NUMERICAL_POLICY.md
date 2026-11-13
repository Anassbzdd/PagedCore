# Independent correctness and model evaluation

Decision: November 11, 2026, authorized by the owner. HF parity, exact HF token
agreement and integrated HF fidelity collection are no longer release requirements.
The pinned TinyLlama checkpoint/tokenizer and Hub loading dependencies stay.
Historical HF results retain their original conclusions; optional comparisons use
`--run-hf-diagnostics`. This revision is not a passing validation result.

## Separate evidence contracts

| Area | Required evidence | Current boundary |
|---|---|---|
| Attention mathematics | Independent FP64 on identical Q/K/V: masks, GQA, difficult scores, boundaries and long context | Standalone T4 evidence at `3401406`; revised integrated evidence still fails |
| Paging | Shuffled pages, poisoned unrelated pages, exact old slots, correct writes, isolation, rollback/retry, stable pool, cleanup and no full-history gather | Allocator/standalone evidence retained; integrated gate open |
| Model implementation | Independent normalization, RoPE, projections, masking, positions and cache semantics | Partial CPU coverage; gaps below remain release work |
| Model quality | Held-out next-token loss/perplexity and fixed tasks against the same checkpoint in established vLLM | Planned: no evaluator, frozen data, margins or result yet |
| Distribution fidelity | KL and top-1 agreement on identical histories with complete aligned probabilities | Planned; existing logit metrics are not KL or quality evidence |
| Serving/performance | Lifecycle, bounded queues, cancellation, slow consumers, safe reuse, memory and frozen T4 benchmarks | Worker/service and comparative measurements remain planned |

## Mathematical and paging acceptance

Assemble independent FP64 attention only in tests. Compute scaled QK, allowed-key
softmax and weighted V independently of the page traversal, using identical
operands. Keep `atol=rtol=2e-3` for FP16 attention output. Cover all eight query
heads per KV head, causal exclusion, 1/15/16/17/49/2048-token lengths, nonadjacent
pages, large/cancelling scores and masked poison. Broader common-input controls
across captured layers/histories remain necessary for new integrated claims.

Serving reads one physical page at a time through the block table. Never gather,
concatenate, stack or copy full history during decode. Full-history reference math
stays outside the guard. Check old slots exactly, new writes against their own
projections and integrated reference, isolation, staged rollback/retry, pool pointer,
ownership, credits and active slots after transitions and cleanup. GPU reuse also
needs stream-ordering evidence; synchronous retry does not prove device-fault recovery.

## Integrated comparison remains blocking

Compare paged decode with contiguous FP32 attention using full matrix products,
softmax and weighted values with one final FP16 cast, without shared page traversal
or pagewise reduction. Weights, K/V and surrounding activations remain FP16. Paged
scores, normalization and V accumulation remain FP32 uniformly across lengths.
Prefill keeps existing arithmetic. The November 13 implementation adaptation
normalizes bounded FP32 scores once before pagewise FP32 value accumulation,
replacing online normalize-last arithmetic. It changes no limits, comparator or
thresholds. [Source study and rationale](validation/2026-11-13-vllm-implementation.md)
distinguish this adaptation from vLLM's FP16 probability conversion and epsilon.

Keep `atol=rtol=5e-3` for logits, `2e-3` for K/V and exact greedy choices against
the owned reference on the frozen paging suite. This finite-suite check is not
universal token identity with vLLM or across arbitrary inputs. The reference shares
non-attention components and orchestration; it cannot independently prove the model.

The November 10 integrated T4 failure remains real: two-token logit tolerance ratio
2.1705 and measured K/V ratio 1.9796. The
[layer-5 replay](validation/2026-11-10-layer5-replay.md#returned-t4-replay) isolates
normalization-placement sensitivity. Normalize-last is closer to full FP64 under
that one call's measured maximum-error and rounded-mismatch metrics. This does
not prove the whole decoder accurate or invalidate failed assertions. P5.3 stays
open. Do not start an indefinite search for one implementation's addition order.

A comparator redesign requires a separate documented justification: retain raw
failures, measure the numerical cause on shared operands across layers/histories,
explain the replacement's scope, and provide independent FP64/component and
held-out quality evidence before freezing a new rule. Quality alone cannot override
a paging defect. Until then the current integrated assertions remain blocking.

Required cases are `single` (15), `mixed` and `fragmented` (1/15/16/17/49), and
`held_out` (37/73/111/145), each with three cached steps following owned-reference
tokens. Held-out lengths are layout coverage, not natural-language quality data.
Report policy `fp32-paged-v2-independent` requires complete finite owned-reference
metrics and K/V ratios; no HF field is required. Keep old v1 evidence unchanged.
Maximum error, RMS, normalized RMS (`RMS(error)/max(RMS(reference),1)`), tolerance
ratio and top-1 margin are diagnostics alongside the existing hard assertions.

## Model implementation coverage and gaps

Required CPU checks include scalar/FP64 RMSNorm, independent half-pair RoPE,
basis-vector projection orientation/output-head merging, analytic GQA/causal
masks, cached/full-prefix positions, context-overflow rejection, rotated K and
unrotated V, isolation and atomic failures. The loader validates 201 tensor
names/shapes, config fields, BF16 storage and finite FP16 conversion. See
[components](../tests/unit/test_independent_components.py), [dense checks](../tests/unit/test_dense.py),
[cache checks](../tests/unit/test_decoder.py) and [loader checks](../tests/unit/test_weights.py).

Removing required HF comparisons leaves no independent full 22-layer checkpoint
oracle. Some synthetic fixtures are constructed with Transformers; that is not
correctness evidence. Shared components can hide common errors in MLP/residual
composition, embedding/LM mapping and prefill arithmetic. Independent full-component
and MLP coverage, real-checkpoint long cached trajectories, paged EOS/maximum-output
generation and quality evaluation remain release gaps. Historical dense HF passes
support only their recorded revisions and do not close these independent gaps.

## Model quality protocol (planned, required before release)

Use identical pinned checkpoint/tokenizer, FP16 weights/KV, one T4, context at most
2048, the same special-token policy and greedy tasks. Smoke-test and pin an exact
vLLM release and resolved configuration, using the frozen benchmark configuration
where applicable. No supported vLLM release has yet been established here. vLLM
is a quality comparator, not a numerical oracle; do not reproduce its rounding or
require universal token agreement. TensorRT-LLM execution is not required.

1. Freeze public corpus/version/license, document IDs, split seed, tokenizer
   hashes, BOS policy, token windows and scored-position mask. Deduplicate and
   separate pilot and held-out documents. Never tune on held-out results.
2. Teacher-force identical corpus histories: prefill a prefix, score the next
   ground-truth token, then feed that token through actual cached decode. Score
   up to 256 continuations/window within 2048 total positions; stratify short/long
   histories and page boundaries, including shuffled/reused pages. Full-prompt
   prefill logprobs alone cannot establish cached-decode quality. The vLLM adapter
   must demonstrate KV reuse at scored steps. If its public API cannot expose
   ground-truth scores on that path, implement/test an instrumented offline
   adapter first; do not silently substitute prefill-only evaluation.
3. Compute token-weighted mean negative log likelihood in FP64 from normalized
   ground-truth logprobs; perplexity is `exp(mean loss)`. Report document/window/
   scored-token counts and length strata, paired loss difference and perplexity
   ratio. Never average window perplexities. Retain missing scores and failures
   explicitly rather than excluding them from denominators.
4. Freeze a small labeled suite (factual multiple choice, short extractive QA,
   deterministic instruction following), datasets, prompt templates, exact scorers,
   EOS/output limits and separate pilot/held-out item IDs. Hash the manifest.
   Generate multi-token answers to exercise decode. Compare degradation against
   the same checkpoint and disclose TinyLlama's absolute baseline weakness.
5. Retain separate pilot counts, variance, repeatability and controlled injected-error
   sensitivity. Choose practical noninferiority margins (maximum acceptable loss
   increase or task-score decrease) with a written rationale tied to variability
   and useful quality. Choose final sample counts from pilot variance and desired
   confidence-interval width/power. Before seeing holdout results, freeze exact
   integer document/window/token/task counts, margins, exclusions, multiplicity
   rule, confidence method and analysis code. This revision establishes no numeric
   margin or final sample count, and therefore cannot declare quality accepted.
6. Use paired document-level bootstrap intervals for loss and paired task-item
   intervals for accuracy, at 95% confidence with frozen seed/resample count.
   Correlated tokens are not independent trials. Accept only if every predeclared
   degradation bound meets its margin. Wide intervals are inconclusive; changing
   margins/counts after inspection requires a new study and untouched holdout.

Retain versioned manifests, per-public-item scores, counts/failures, scorer code,
source/lock/device identity and uncertainty. Text remains in frozen public inputs;
reports use IDs/hashes/scores. Operational logs and performance results still contain
no prompt/generated text. Evaluators, adapters, manifests and decision export are
planned, not implemented by the current accuracy notebook.

## Distribution fidelity (planned)

On identical histories and vocabulary order, with complete finite logits from both
systems, compute FP64 log-softmax and
`KL(p_vLLM || p_PagedCore) = sum(p_vLLM * (log p_vLLM - log p_PagedCore))`.
Report direction, mean/quantiles, position counts, document-level uncertainty and
top-1 agreement/margins. Retain enough complete public-input logits to recompute.
Top-k-only probabilities cannot establish full-distribution KL: mark it unavailable.
Complete ground-truth-token logprobs can still support loss. Top-1 agreement is
descriptive, not universal token identity. Any fidelity gate needs pilot-justified
margins frozen in advance; no universal perplexity, KL or accuracy limit is adopted.

## Evaluation practice sources

Reviewed November 11, 2026. These inform the protocol, not PagedCore thresholds.

- [llama.cpp perplexity](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
  reports corpus perplexity, KL, same-top-token statistics and uncertainty; tokenizer
  and evaluation details affect comparisons.
- [vLLM sampling parameters](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)
  documents output/prompt logprobs. Verify full-vocabulary support on the pinned
  release; requested top-k output is not a complete distribution.
- [TensorRT-LLM evaluation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-eval.rst)
  uses task-specific datasets/scorers such as QA accuracy and summarization ROUGE.
- [MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
  specify quality targets, datasets, preprocessing and performance evidence.
  This project adopts explicit reproducibility and separate quality checks; it
  does not claim MLPerf compliance or borrow another model's targets.

## Next T4 action and limits

Open [paged_decode_accuracy.ipynb](../notebooks/paged_decode_accuracy.ipynb) in a
fresh Kaggle session, enable Internet and T4 GPU, and Run All. Download the final
ZIP even on failure. It embeds exact working source and records setup/environment,
CPU checks, ten required GPU cases, raw logits and versioned reports. Passing can
establish only those mathematical/paging comparisons. The known failure may persist.
It cannot close P5.6/P5.7, component gaps, quality, service or performance gates.
No new GPU validation accompanies this revision.
