# Two-token decoder layer trace

Observation date: November 10, 2026. Current gate: P5.3, still open.

The [paired T4 reproduction](2026-11-10-two-token-reproduction.md#returned-t4-evidence)
establishes identical final failures alone and mixed. That archive contains no
layer trace, so it cannot identify the first differing T4 operation. The owner
requested the actual intermediate comparisons before another numerical change.

## Collection

[`trace_two_token_decode.py`](../../scripts/trace_two_token_decode.py) traces the
standalone one-token prefix and one decode step using the same pinned weights,
input selection, poisoned 32-block pool, reservation and deterministic math.
It wraps the existing decoder functions and returns their original outputs;
it does not implement a replacement decoder. The actual paged forward remains
under the shared no-full-history guard. An independent uninstrumented baseline
must produce exactly the same reference and paged final logits as the trace.

For all 22 layers, numbered **0 through 21**, the trace records:

- Hidden input and input RMSNorm output.
- Projected Q/K/V, rotated Q/K and unchanged V. RoPE does not rotate V.
- Actual K/V views yielded to attention by `iter_layer_blocks`, including the
  staged second token. The trace retains physical-page identity and proves the
  borrowed views share pool storage. History and appended-slot comparisons are
  separate; the appended slot is also checked against that path's own projected K/V.
- Actual attention QK product, scaled scores and FP32 output before the FP16 cast.
  Raw attention operation outputs and matmul operands are retained for replay.
- FP16 attention output, output projection, attention residual, post-attention
  RMSNorm, MLP result and final layer output.

This gives **418 comparisons** (19 per layer), ordered by layer and operation.
Each reports exact value equality, unequal-element count, maximum absolute error,
unchanged tolerance ratio and first differing coordinate/values. Original shape,
stride, storage offset, dtype, device, contiguity and backing-storage size accompany
each side. CPU snapshots do not replace original layout metadata. K/V and attention
products are compared in a shared logical head/token order outside the forward;
different runtime layouts are recorded separately from differences in values.

The first mismatch is selected by exact equality, even far inside tolerance.
It is valid only when all trace points exist and the instrumentation controls
match. Numerical tolerance failures do not abort collection. An incomplete
trace retains its raw operands and missing-point list.

If the first mismatch is QK multiplication, the diagnostic immediately tests:

1. Whether the logical Q/K operand values match exactly.
2. Native replay with original operand shapes, strides and offsets.
3. Replay with contiguous operands at each original shape.
4. Dense geometry with the same paged operand values.

These small controls run after the guarded serving forward, with the same math
settings. They are diagnostic scratch calculations, not a serving fallback.
Exact native replay is reported rather than assumed. Shape and layout controls
do not independently identify a particular hardware kernel or reduction order.

## K/V evidence correction

In the earlier `pagedcore-accuracy-kavvwb5q.zip`, `max_kv_tolerance_ratio: 0`
was initialized before the failing logit assertion. Those K/V checks never ran;
that zero is **unmeasured**, not an exact match. The later two-token reproduction
diagnostic completed its own K/V loop and measured a T4 maximum ratio of 1.9796.
These are different reports and must not be conflated.

This trace records `kv_measurement.status`, `measured_pairs` and `expected_pairs`.
Complete coverage is 44 paged-versus-reference tensor pairs (K and V across
22 layers), each with separate
history/appended measurements (88 token-slice comparisons). No measurements means
status `not_measured` and a null maximum; incomplete coverage is `partial` with
its actual count. A partial maximum never implies full K/V acceptance. The GPU
acceptance suite, assertions, thresholds and serving arithmetic are unchanged.

## CPU investigation

The [retained CPU review](../../results/validation/2026-11-10/layer-trace-cpu-review.json)
summarizes the pinned-checkpoint run in `results/local/2026-11-10-layer-trace-cpu-v2/`.
The run
completes all 418 points. Both instrumented final logits match the uninstrumented
baseline exactly. All layer-0 inputs through projected/rotated Q/K/V and the
actual pool reads, including history and the new slot, match the reference exactly.

The first exact difference is **layer 0, QK product**, before score scaling.
47 of 64 products differ; maximum absolute FP32 error is
`4.76837158203125e-6` and maximum tolerance ratio is `0.0001783312545853424`.
The first differing coordinate in grouped `[4, 8, 2]` order is `[0, 0, 0]`:
paged `0.014506280422210693`, reference `0.014506278559565544`.
The paged product's original shape/stride is `[4, 8, 2] / [16, 2, 1]`;
the reference is `[1, 32, 1, 2] / [64, 2, 2, 1]`.

Native replay reproduces both products exactly with identical logical Q/K values.
Packing the reference operands preserves its output; packing the paged operands
changes its output by up to `1.9073486328125e-6`. Dense geometry with the paged
values matches the reference product exactly. This localizes the CPU discrepancy
to matrix-product arithmetic sensitive to shape/layout, rather than a different
input value or corrupt pool read. It does not select a repair or identify a T4 kernel.

The first differing FP16 component output is also layer 0, attention output,
with maximum error `3.814697265625e-6`. Later K/V drift reaches ratio
`2.8939886271324125` with all 44 read pairs actually measured. Cleanup returns
all ownership to zero. These observations use Torch 2.7.1+cpu and cannot establish
the first T4 difference. The T4 trace remains pending.

## Kaggle execution and local checks

Import [`two_token_layer_trace.ipynb`](../../notebooks/two_token_layer_trace.ipynb),
enable Internet and T4, then Run All. Exact working source and frozen setup are
embedded. Download the final ZIP, including on failure. Inspect
`evidence/layer-trace/trace.json`, `summary.json`, all `raw/layer-XX.pt` files and
`first-product-replays.pt` when applicable. Verify `complete`,
`first_difference_valid` and `instrumentation_controls` before interpreting the
first difference. The notebook retains raw evidence, source hashes and commands.

From a checkout containing these edits:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --frozen --extra cuda --extra dev python -m scripts.trace_two_token_decode \
  --output-dir /kaggle/working/layer-trace
```

Local commands (using the frozen CPU environment):

```text
python -m scripts.trace_two_token_decode --device cpu --output-dir results/local/2026-11-10-layer-trace-cpu-v2
python -m pytest tests/unit/test_layer_trace.py tests/unit/test_numerical_evidence.py -q
python -m pytest -m "not gpu and not benchmark" -ra --junitxml=results/local/2026-11-10-layer-trace-final-tests.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
python -m scripts.build_numerical_notebook --layer-trace
```

Pytest and CPU probes set `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. The final non-GPU suite passes **777 tests**, with 30 GPU
cases deselected and no failures or skips; the focused trace/notebook suite
passes 25 tests. Ruff lint/format, strict mypy and `git diff --check` pass.
Regression coverage verifies actual 22-layer reads under the
guard, preserved forward outputs, a first mismatch inside tolerance, explicit
unmeasured/partial K/V status, restored hooks on failure, original-layout replay
and standalone notebook failure retention. No new T4 execution or release pass
is claimed. P5.3 and all subsequent gates remain open.
