# vLLM implementation study and bounded attention adaptation

Reviewed November 13, 2026. This is a source-code study and a local CPU-validated
implementation change. The subsequently returned
[T4 adaptation review](2026-11-13-normalize-first-accuracy-t4-review.md) confirms
standalone attention passes and exact two-token logits, but integrated failures
remain at 16/38 tokens. P5.3 remains blocking. All previous
raw archives, tensors, reviews and failed assertions are retained.

## Source selection

Study **vLLM v0.8.5**, resolved by `git ls-remote` to commit
`ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`. Links below pin that commit rather than
the moving default branch. This historical release provides an understandable
CUDA path for T4; this study does not establish the latest release's behavior or
select a validated benchmark/quality version.

The source configuration is V0 engine (`VLLM_USE_V1=0`), XFormers attention backend
(`VLLM_ATTENTION_BACKEND=XFORMERS`), one GPU/tensor-parallel rank, no quantization,
`dtype=half`, `kv_cache_dtype=auto`, block size 16, context 2048, no prefix caching,
and the pinned TinyLlama checkpoint. `auto` here means unquantized KV in the model
dtype, so FP16. T4 is Turing, compute capability 7.5. The
[CUDA backend selector](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/platforms/cuda.py#L224-L290)
routes pre-Ampere devices away from FlashAttention-2 to XFormers. Explicit V0
avoids confusing this with a different V1 engine backend.

These are source-compatible settings, **not an executed installation recipe**.
The vLLM release and its dependencies must be smoke-tested separately on Linux/T4;
PagedCore's Torch lock is unchanged. Benchmark defaults and the frozen protocol
remain governed by `VALIDATION_PLAN.md`, not this implementation study.

## Decoder and cache write path

The actual call chain is:

1. [LlamaAttention](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/llama.py#L130-L208)
   computes a packed QKV projection, splits Q/K/V, rotates Q and K using absolute
   positions, calls attention and applies the output projection. The scale is
   `head_dim**-0.5`: `1/8` for TinyLlama's 64 channels per head. Four KV heads
   serve 32 query heads. V is not rotated.
2. [LlamaDecoderLayer](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/llama.py#L262-L287)
   composes RMSNorm, attention, residual addition, another RMSNorm and SwiGLU.
   vLLM carries a separate residual tensor and fuses additions with normalization;
   PagedCore spells out those operations. The
   [RMSNorm kernel](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/layernorm_kernels.cu#L17-L43)
   reduces squares in FP32 and casts the normalized activation before multiplying
   the weight. Its
   [RoPE kernel](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/pos_encoding_kernels.cu#L10-L91)
   uses half-pair indexing for the selected NeoX-style Llama path. The
   [wrapper](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/layers/rotary_embedding.py#L167-L193)
   supplies a trigonometric cache in the query dtype and rotates in place.
3. [XFormersImpl.forward](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/backends/xformers.py#L469-L620)
   splits the preallocated KV storage, writes the new rotated K/unrotated V, then
   dispatches decode to `PagedAttention.forward_decode`. Normal prompt prefill
   uses xFormers memory-efficient attention; **decode uses vLLM's CUDA paged kernel**.
   The backend's name does not mean xFormers performs every attention operation.

[PagedAttention.split_kv_cache](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/ops/paged_attn.py#L41-L86)
views K as `[blocks, kv_heads, head_dim/x, block_tokens, x]`, with `x=8` for FP16,
and V as `[blocks, kv_heads, head_dim, block_tokens]`. Different layouts support
the vector loads used by the two reductions; they do not change logical attention.

The [cache write kernel](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/cache_kernels.cu#L210-L261)
takes a physical slot `s`, computes `physical_block=s//16` and `offset=s%16`,
then writes K and V directly in those layouts. Negative slots represent padding
and are skipped. With `auto` KV dtype there is no FP8 conversion or K/V scale.
PagedCore instead writes `[layer, kind, block, slot, kv_head, channel]` through
its allocator. Keep that simpler pool layout and its staged commit/rollback.

## What the decode kernel actually does

The Python dispatcher chooses
[PagedAttention V1 or V2](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/ops/paged_attn.py#L116-L191).
These names are kernel versions, separate from engine V0/V1. V1 runs when
`max_seq_len <= 8192` and either there is only one 512-token partition or
`num_sequences*num_query_heads > 512`. Thus TinyLlama uses V1 through 512 tokens;
at 2048 tokens, batches of 1–16 use V2, while 17–32 use V1. Do not describe all
T4 decode as one fixed kernel.

For unpartitioned V1, for query head `h` and cached length `L`:

```text
g = h // 8
for logical block b:
    physical = block_table[sequence, b]
    for valid token slot o:
        t = 16*b + o
        score[t] = FP32_dot(Q[h], K[physical, g, o]) / 8
m = max(score[0:L])
e[t] = exp(score[t] - m)
p[t] = e[t] / (sum(e[0:L]) + 1e-6)
read V again through the same block table
accumulate weighted values; convert completed output to FP16
```

This pseudocode shows normalization placement; the following precision details
matter:

- **Addressing and GQA.** The kernel computes `kv_head=h//(num_heads/num_kv_heads)`.
  Each CUDA warp handles logical blocks, fetching their physical IDs from the
  table. It uses 64-bit physical offsets to avoid stride multiplication overflow.
  No contiguous K/V history is reconstructed.
  [Head mapping](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L150-L168),
  [block reads and scores](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L227-L306).
- **QK and scale.** FP16 vectors feed FP32 product/accumulator vectors, reduced
  across lanes and thread groups to a `float` score. Scale is applied to that
  score in FP32, with no FP16 score conversion.
  [QK helper](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_utils.cuh#L29-L54),
  [FP32 multiplication overloads](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/dtype_float16.cuh#L273-L317),
  [FP32 multiply-add overloads](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/dtype_float16.cuh#L383-L420).
- **Stable softmax.** Scores occupy FP32 shared memory. Warp/block reductions
  find a maximum, `__expf` evaluates shifted scores, and a FP32 reduction obtains
  the exponential sum. `__fdividef` computes `1/(sum+1e-6)`. Probabilities are
  normalized **before** reading V. Only valid `L` scores participate. This is
  global normalization in V1, not page-local softmax or online numerator division.
  [Softmax implementation](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L310-L357).
- **Value product has intermediate FP16 rounding.** `L_vec` is a vector of
  `scalar_t`, so `from_float(logits_vec, ...)` converts probabilities to FP16.
  `dot(logits_vec,v_vec)` uses the generic `mul<T,T,T>` overload: for eight FP16
  values this includes packed FP16 multiplication and packed FP16 partial sums,
  then conversion to float. The returned dot is added to a FP32 accumulator;
  warp/block output reductions also use float. Therefore it is inaccurate to
  claim this path keeps all PV arithmetic FP32 until the final cast.
  [Conversion and value loop](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L359-L445),
  [generic dot](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_generic.cuh#L38-L50),
  [packed FP16 multiplication](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/dtype_float16.cuh#L207-L270),
  [partial sums and conversions](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/dtype_float16.cuh#L433-L477).
- **Masking and output.** Decode includes the just-written token and excludes
  unfilled tail slots. The kernel explicitly zeros invalid V elements because
  multiplying a zero weight by a poisoned NaN still produces NaN. PagedCore
  avoids loading them by slicing each page to its valid prefix. After float
  output reductions, `from_float` writes FP16 with round-to-nearest conversion.
  [Tail masking](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L420-L431),
  [output reduction/cast](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L447-L494),
  [round-to-nearest helper](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/dtype_float16.cuh#L115-L146).

V2 applies the same kernel within 512-token partitions. It stores partition
maxima and exponential sums in FP32, **partition outputs in FP16**, then merges
them using weights proportional to `exp_sum[j]*exp(max[j]-global_max)`. The merge
accumulator is FP32 and its result is cast again to FP16; the merge denominator
also has `1e-6`. A single partition is simply copied.
[V2 storage and launch](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/attention/ops/paged_attn.py#L154-L191),
[V2 merge](https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/csrc/attention/attention_kernels.cuh#L580-L668).

## Mapping and adaptation

| Operation | PagedCore before this change | Decision |
|---|---|---|
| Norm, QKV, RoPE, output projection, MLP and residuals | Owned `dense.py` / `decoder.py`; separate projections, explicit residuals, per-request operation shapes | Preserve. Packed projections, fused normalization, and simultaneous batched GEMMs are performance changes with different arithmetic. |
| Physical KV access | `KVAllocator.iter_layer_blocks` returns valid page views; staged writes expose only completed layers | Preserve pool, block table, ownership and transaction. This already implements the essential paged mapping. |
| Scaled QK | FP16 Q/K converted pagewise to FP32, grouped `[4,8,64]` Q and `1/8` scale | Preserve, including disabled autocast. Do not introduce an FP16 score stage. |
| Softmax and PV | Online max/denominator/unnormalized weighted sum, division after V accumulation | Adapt vLLM's normalize-before-values structure. Store scores, normalize once, reread V pages. |
| Probability and value precision | FP32 until final output | Preserve FP32 probabilities, products and accumulator; do not copy vLLM's half-vector arithmetic. |
| Tail masking | Iterator slices valid slots; unrelated pages excluded by block table | Preserve. Two passes must read the same staged history. |
| Output | One FP16 conversion of complete attention result | Preserve; no per-page or partition casts. |

The production change is confined to [attention.py](../../src/pagedcore/attention.py).
For each request/layer it writes FP32 scaled scores into `[4,8,2048]`, calls stable
softmax on only `[..., :valid_length]`, then computes:

```text
output_fp32 = 0
for each valid V page through the allocator:
    output_fp32 += probabilities_for_that_page @ FP32(V_page)
return FP16(output_fp32)
```

There is no post-PV normalization. Both passes use the existing allocator;
neither concatenates, stacks, gathers or copies full K/V history. Only reduced
scores/probabilities span the context. Each uses at most 256 KiB (512 KiB combined)
for one request/layer, plus page conversions, output and backend workspace.
The decoder processes rows/layers sequentially, so these buffers are not retained
for every row/layer. Actual allocator peaks and throughput still require T4
measurement. This trades bounded score memory and a second traversal for a
simple, recognizable algorithm.

Why this change: the retained
[layer-5 replay](2026-11-10-layer5-replay.md#returned-t4-replay) localized a
normalize-first/normalize-last difference on common inputs. The chosen structure
removes that difference by construction and follows established inference code.
It does **not** establish that normalize-first is universally more accurate:
normalize-last was closer to FP64 under that isolated call's measured metrics.
Nor does it guarantee whole-decoder acceptance: pagewise QK/PV products can still
round differently from full matrix products. We make no kernel-family guesses,
length-specific arithmetic switches or attempts to reproduce a backend's lanes.

A separately retained [44-case common-input T4 review](2026-11-13-common-input-t4-review.md)
reports both archived paths within the attention tolerance, with neither
consistently closer to FP64. Its source snapshot contains the older online
implementation; it does not validate this adaptation. This further limits any
claim that changing normalization placement improves mathematical accuracy.

Deliberate differences from vLLM:

- Retain FP32 probabilities/PV and one final cast to preserve the documented
  precision contract and avoid half underflow or premature cancellation loss.
- Use ordinary stable softmax without vLLM's denominator epsilon. At least one
  finite allowed score exists, so subtracting the maximum gives an exponential
  sum at least one; no zero denominator needs a guard. Inputs with nonfinite
  Q/K are outside this numerical assumption, not repaired by epsilon.
- Use one global normalization through 2048; no V2 partition outputs or merge.
  Splitting/fusing kernels can follow only with a measured performance need.
- Preserve owned decoder, dense prefill, independent contiguous FP32 reference,
  numerical thresholds, exact finite-suite greedy checks, and all MVP boundaries.
  Keep historical eager-rounding helpers for forensic replay, outside serving.

## Verification and remaining gate

New focused CPU tests check globally normalized FP32 weights before each V
product on nonadjacent pages `(0,2)` with a poisoned blocker/tail, tiny probabilities
that would underflow if cast to FP16, and cancellation that would lose information
if page outputs were half-rounded. The normalization test was run against the
previous implementation and failed because it supplied weights of one instead
of `1/17`; the other two precision controls already passed and remain guards.
Existing FP64 boundary/large-score, batch-order, staged-write, rollback/retry and
no-full-history-gather tests remain in force.

The two existing required T4 attention tests now also run under the storage guard;
their boundary case includes tiny probabilities and cross-page cancellation.
Their required test names and all integrated assertions remain unchanged. No new
GPU execution occurred locally (`torch=2.7.1+cpu`, CUDA unavailable). The live
two-token trace handles repeated page reads and observes the final FP32 addition;
its original read-once assumption failed after this change and was corrected.
New diagnostic output calls the active path `fp32_paged_attention`, leaving old
`online_attention` evidence untouched.

Executed commands/results are recorded in [STATUS.md](../STATUS.md). P5.3 and
integrated P5.4 stay open. Next execute the two focused attention tests on T4,
then the existing accuracy notebook once to evaluate this changed implementation
at unchanged thresholds and retain its source-attributed archive even on failure.
The accuracy notebook was rebuilt with `python -m scripts.build_numerical_notebook
--accuracy`; the embedded source/identity hashes and working runtime/test/tooling
files were verified without executing GPU cells. Upload this refreshed artifact:
an older copy still executes its older embedded source. The all-layer replay's
separate review covers the older implementation. If integrated assertions still fail, report that result;
do not widen tolerances or launch another backend-rounding sweep.
P5.6/P5.7, independent model coverage, held-out quality, lifecycle/service and
benchmark gates remain open.
