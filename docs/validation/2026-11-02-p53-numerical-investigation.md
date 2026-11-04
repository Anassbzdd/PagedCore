# Decode parity at the 16-to-17-token boundary

Observation date: 2026-11-02. Investigation only; no runtime fix or T4 acceptance
is claimed. The current gate remains P5.3 integrated paged decode parity.

## Evidence and source identity

The supplied [diagnostic report](../../results/validation/2026-11-2/p53-diagnostic.json)
contains 6,422 measurements, captured at `2026-11-02T13:12:43.310231+00:00` on
`cuda:0`, with Torch `2.7.1+cu126`, Transformers `4.52.4`, and a Tesla T4 inventory.
It identifies clean revision `b3a0fe144823e50aca2483ca3eede166b6d163c5`.
Local SHA-256 checks match its attention source, decoder source, diagnostic script
and lock. Its setup command and command arguments are null, and it explicitly
says it is not release evidence. Do not infer a passing collector run from it.

## Proven observations

For the single request, the initial prompt has 15 tokens. Each step appends the
previously selected token before attention, so steps 1/2/3 attend over 16/17/18
tokens respectively. Layer numbers below are zero-based.

| Attended tokens | Eager replay: scaled scores | Eager replay: attention output | Integrated logits |
|---|---|---|---|
| 16 | All 22 layers exact | All 22 layers exact | Exact |
| 17 | All 22 layers exact | First difference at layer 1: one element, `5.960464477539063e-8` | Maximum absolute error `0.0234375`; maximum tolerance ratio `1.977457` |
| 18 | Three unequal scores in total | Eleven unequal output elements in total | Maximum absolute error `0.015625`; maximum tolerance ratio `2.363211` |

The tolerance ratio is `abs(actual - expected) / (atol + rtol * abs(expected))`.
A value greater than one fails. Attention/K/V use `atol=rtol=2e-3`; logits use
`atol=rtol=5e-3`. No threshold was changed.

At 17 tokens, the isolated attention replay has only six unequal output elements
across all layers, with a maximum absolute error of `6.103515625e-5`. None fails
the attention tolerance. Nevertheless, the complete decoder fails its logit gate.
Its first two layers' K/V remain exact. The first unequal newly appended K/V is
at layer 2; the earliest failing K/V measurement in forward-layer order is layer
7's key, with ratio `1.105115`. Historical and newly appended slots are reported
as separate block records. The report writes logits before K/V, so its summary's
`first_failure=logits` does not establish the earliest failing forward operation.

All isolated component replays are exact on their HF operands. Batched component
replays have small discrepancies, but integrated decode already executes each
row with single-request shapes. Batch arithmetic cannot explain the single-row
failure. Every integrated greedy-token comparison still matches; that does not
satisfy tensor/logit parity.

## Where the code changes arithmetic

[`_eager_attention`](../../src/pagedcore/attention.py) computes page-local FP16
Q/K products, scales them in FP16, then normalizes scores globally and rounds
probabilities to FP16. It never gathers full-history K/V.

For at most 16 tokens, the value calculation uses native FP16 `bmm`. Above 16,
it switches to four persistent FP32 partial sums. Positions are assigned to
`position % 4`, with the final zero-to-three positions added to lane zero, then
the lanes are folded and rounded to FP16. At 17 tokens, positions 0/4/8/12/16
share lane zero. This reproduces a CPU fallback, not a documented CUDA order.

The allocator iterator reads valid slots through logical-to-physical block IDs
and yields K and V from the matching physical page. No off-by-one or head-mapping
error was found in that path. Exact Q/K scores do not independently prove that V
was read correctly, however; a direct page-to-oracle V comparison remains useful.

## Official implementation references and hypotheses

The pinned [HF eager implementation](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/llama/modeling_llama.py#L162-L183)
uses FP16 score products, FP32 softmax, an FP16 probability cast, and a native
probability/value matmul. The PagedCore rounding boundaries reflect that policy.

PyTorch's [CPU reduction](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cpu/BlasKernel.cpp#L74-L94)
has the four-lane order copied here. Its [CUDA bmm path](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Blas.cpp#L545-L619)
instead dispatches to CUDA BLAS; the [FP16 batched implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDABlas.cpp#L513-L570)
calls cuBLAS. Those sources do not establish the internal cuBLAS summation order
for the diagnostic's operands. PyTorch explicitly documents that
[CPU/GPU and batched/sliced floating-point results can differ](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
Deterministic settings make a chosen computation repeatable; they do not make
different computations identical.

**Leading hypothesis:** the switch from native CUDA value matmul to the emulated
CPU reduction produces different rounding. Subsequent layers amplify that small
change. The control replay and layer-2 K/V drift support this explanation; they
do not prove the entire causal chain.

**Remaining alternative:** softmax/probability rounding differs before the value
sum. The report's observer captures softmax input, not its output. HF supplies
FP16 scores with an FP32 softmax output; PagedCore supplies FP32 scores. At 17
tokens, both enter the same persistent-softmax algorithm family with FP32
accumulators in the pinned [CUDA source](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/SoftMax.cu#L855-L989),
so this is less strongly supported than the value-reduction hypothesis. It still
requires an explicit probability comparison. Also, CUDA's
[softmax reduction width](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/PersistentSoftmax.cuh#L283-L320)
changes at a power-of-two boundary such as 16-to-17. The page boundary alone is
not proof of a page-table defect.

The first error equals one FP16 subnormal step. That magnitude alone does not
establish that CUDA flushes subnormals or that fused multiply-add caused it.

## Minimal fix recommendation and regression

Do not change the allocator, RoPE, admission, tolerances, or reference. First
extend the existing replay to compare FP32 softmax outputs and their rounded
FP16 probabilities, then run the value calculation with HF's exact probabilities
and independently checked V pages. This separates normalization from weighted
summation without changing the serving path. Capture the first differing
coordinate and operand shapes/strides; the current JSON contains only aggregates.

If that replay confirms a value-reduction mismatch, replace only the multi-page
value reducer. A small first candidate is native page-local FP32 `baddbmm` into
one persistent FP32 output accumulator, with one FP16 cast after the final page:

```python
output = torch.zeros_like(query)  # grouped FP32 [4, 8, 64]
start = 0
for _, value in allocator.iter_layer_blocks(request_id, layer_index=layer_index):
    stop = start + value.shape[0]
    page = value.transpose(0, 1).float()  # at most one V page
    weights = probabilities[..., start:stop].float()
    for head in range(4):
        output[head] = torch.baddbmm(
            output[head, :, None],
            weights[head, :, None],
            page[head][None].expand(8, -1, -1),
        )[:, 0]
    start = stop
return output.reshape(1, 32, 1, 64).half()
```

This is an experiment, not a verified fix. It preserves global FP16
probabilities, FP32 accumulation, block-table traversal and page-sized V storage.
Page-local FP32 reduction can still differ from HF's full CUDA matmul. Do not
accept it unless the unchanged integrated T4 checks pass; native per-page FP16
outputs summed afterward would introduce additional rounding and are unsuitable.
If it fails on identical probabilities/V, capture/profile the reference reduction
before implementing a CUDA reducer; do not guess a lane count or padding rule.

The existing `test_pinned_paged_decode_on_t4[single]` already reproduces the
reported trigger: the first 15 IDs of `maximum_output`, followed by three cached
reference-token steps. Keep it as the acceptance regression: exact greedy IDs,
logits at `5e-3`, all-layer K/V at `2e-3`, unchanged old slots, ownership/accounting,
stable pool storage, final release, and the no-full-history-gather guard. The
mixed case also covers reordered requests and longer histories. Add a focused
T4 replay of the layer-1 17-token operands to diagnose the numerical stage;
retain 16 tokens as a control. Preserve the original failed report.

Run the candidate against 15/16/17, 31/32/33, longer fragmented histories and
2,048 tokens, with unrelated/unused slots poisoned. Full-history reference math
is permitted outside the owned-forward guard. Exact score/probability checks
are diagnostic assertions on identical operands; they do not replace or relax
the existing tensor/logit tolerances. A regression must fail on the old source
and pass on the candidate on an actual T4 before calling the defect fixed.

## Local validation and remaining work

Executed locally with `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

```text
uv run --no-sync python -m pytest tests/unit/test_attention.py tests/unit/test_decoder.py -q
133 passed in 93.44s
git diff --check
```

The runtime is Torch `2.7.1+cpu`, with CUDA unavailable. A synthetic 17-token
arithmetic control also demonstrated that serial FP32 and four-lane FP32 sums of
the same FP16 weighted values can round differently; it was not a reproduction
of the T4 defect. The proposed snippet passed CPU shape/numerical sanity checks
at 17/33/49/2,048 tokens against an FP64 weighted-value reference at `2e-3`;
these checks used synthetic operands and do not establish HF decoder parity.
Local links and `git diff --check` also pass.
The initial investigation changed no production code or tests; the follow-up
implementation below adds the experiment without selecting it for serving.
No new full CPU suite, Ruff, mypy, GPU execution, workspace measurement or
performance claim is made. The recommendation still needs the probability/V
replay, a passing attributable T4 regression and full collector review. Phase 5
acceptance remains open.

## Implemented diagnostic follow-up

The value reducer is now separate from score/softmax computation. The default
still uses the original single-page/native and multi-page/striped arithmetic.
The experimental `_paged_value_sum` is selected only by the diagnostic's explicit
`--value-reduction paged` option and candidate tests. No tolerance changed.

The replay captures the unchanged HF softmax output before its FP16 cast,
compares both probability dtypes, independently checks each copied K/V page,
and repeats eager attention with HF FP32 probabilities substituted at softmax
output. Its `fixed_probabilities` records therefore exclude probability drift.
Measurements include tensor shapes, strides, and the first differing coordinate
and values. The report records the selected reducer; keep output files separate.

New CPU cases check observer transparency/substitution, first-difference metadata,
candidate storage/accounting through 2,048 tokens, and cancellation requiring FP32
page products until the final cast. The candidate has separate single/mixed T4
tests that reuse the existing decoder acceptance checks and storage guard.

After the documented CUDA environment setup, run on an actual T4:

```bash
uv run --no-sync python scripts/diagnose_paged_decode.py --device cuda:0 --steps 3 --value-reduction striped --output results/local/p53-striped-t4.json
uv run --no-sync python scripts/diagnose_paged_decode.py --device cuda:0 --steps 3 --value-reduction paged --output results/local/p53-paged-t4.json
uv run --no-sync python -m pytest tests/gpu/test_decoder_parity.py -k pinned_candidate_paged_decode_on_t4 -ra --junitxml=results/local/p53-candidate-t4.xml
```

A failed replay writes its JSON before exiting unsuccessfully. These are proposed
T4 commands, not commands executed locally. Neither candidate success nor a CPU
pass closes the full collector, workspace, or Phase 5 release gates.

### Follow-up results: candidate rejected on CPU

Executed with `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

```text
uv run --no-sync python scripts/diagnose_paged_decode.py --device cpu --steps 2 --value-reduction paged --output results/local/p53-paged-value-cpu.json
exit 1: Integrated parity still fails; retain this diagnostic report.
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p53-value-replay-cpu.xml
684 passed, 19 deselected, zero skipped, in 184.82s
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m mypy --strict src/pagedcore
git diff --check
```

The [failed replay](../../results/validation/2026-11-2/p53-paged-value-cpu.json)
and [CPU JUnit report](../../results/validation/2026-11-2/p53-value-replay-cpu.xml)
are retained separately from the supplied T4 report. All 264 FP32 probability
comparisons, all 264 FP16 probability comparisons and all control-page K/V
comparisons are exact on CPU. With identical HF probabilities, the candidate
first differs in layer 2 at 17 tokens: coordinate `[0, 0, 28, 14]`,
`0.0003209114074707031` versus `0.00032067298889160156`. Its full decoder first
exceeds tolerance there (logit ratio `1.419406`); the maximum integrated ratio
across the replay is `2.660326`. All next-token argmax comparisons match.

This rejects the proposed page-product reducer as a generally parity-preserving
fix. Passing synthetic value tests did not predict real-model numerical
propagation. The experiment remains opt-in for comparison, and is not promoted
to serving. Do not infer the original T4 failure's cause from this CPU result.
Run the enhanced striped replay on T4 next to locate its first probability or
value-sum divergence; the candidate command/test is an additional controlled
experiment, not a recommended default. No alternative lane order is guessed.

GPU-directory collection finds 18 cases, including both new candidate cases;
one additional GPU-marked unit case accounts for the full CPU run's 19
deselections. Collection is not GPU execution. Ruff, formatting, strict mypy,
and diff checks pass. P5.3 remains unsatisfied.

## Supplied Kaggle failure: storage-guard instrumentation

The later [user-supplied test transcript](../../results/validation/2026-11-2/p53-candidate-t4-guard-failure.txt)
reports both candidate cases aborting at the MLP projection, before parity
assertions. It does not include a source manifest or the two diagnostic JSONs,
so it is a reported T4 failure, not attributable release evidence or a numerical
measurement of the candidate.

The guard stops tracking K/V after `matmul`, `bmm` and `mm`, because their reduced
results are attention activations. It omitted the candidate's `baddbmm`. That
omission tracks the reduced output through subsequent operations and eventually
rejects a normal `[1, 1, 5632]` FP16 MLP activation (11,264 bytes versus the
8,192-byte page limit). This is a false positive, not evidence of a KV gather.

The new CPU regression performs a page-sized weighted-value `baddbmm`, then a
5,632-element linear projection. Before the fix it fails the same storage-size
assertion (CPU dispatch exposes `mm` rather than CUDA's `linear`). After adding
only `aten.baddbmm.default` to the reduction boundary, it passes. The guard still
checks the reduction's output size before ending tracking; direct gathers and
copies continue to be checked. No blanket exemption for linear layers and no
larger page limit was introduced. Engine arithmetic and serving selection are
unchanged.

The JSON replay runs outside this test guard, so the existing `striped.json` and
`paged.json` from Kaggle remain useful. The T4 candidate tests need rerunning
with the corrected guard. This instrumentation repair does not change the
candidate's known failed CPU numerical replay or close P5.3.

Local validation after the guard fix passes 685 CPU tests, with 19 deselected,
zero skipped, in 159.03s. The retained
[JUnit report](../../results/validation/2026-11-2/p53-guard-baddbmm-cpu.xml)
comes from `uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/p53-guard-baddbmm-cpu.xml`
with `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` on
Torch 2.7.1+cpu. Ruff lint/format checks, strict mypy and diff checks pass.
The existing eight injected full-history shortcuts are still rejected. The new
regression failed before the fix and passed afterward. No T4 rerun is claimed.
