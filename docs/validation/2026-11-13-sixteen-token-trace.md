# Sixteen-token cause-localization notebook

Prepared November 13, 2026. Current gate remains P5.3. This is diagnostic tooling,
not a serving repair or new T4 acceptance result.

The [four-cell notebook](../../notebooks/sixteen_token_layer_trace.ipynb) embeds
the current source, frozen lock and the raw `single/step-1` tensors from the
[normalize-first T4 failure](2026-11-13-normalize-first-accuracy-t4-review.md).
Its builder rejects a source snapshot whose runtime hashes differ from that
archive. The notebook records the base commit and exact working-file hashes;
it does not claim the working changes were committed.

## Exact case and controls

Use the original first 15 IDs of `maximum_output`, including BOS, rather than
the separate `boundary_15` prompt. Prefill selects the decode input. One cached
step appends it at absolute position 15 and attends to positions 0 through 15.
This is not an API request for only one generated token, which would stop after
prefill. Preserve the 32-block poisoned pool and original reservation size.

Cell 1 installs the frozen Python/CUDA environment and verifies the T4 and pinned
checkpoint. Cell 2 reproduces the standalone failure and requires both paged and
contiguous FP32 raw logits to equal the retained failed tensors exactly before
tracing. All source runtime files must still match the failed run's hashes.

Cell 3 observes the actual decoder calls. It records all 22 layers, original
tensor layouts, Q/K/V projections, rotated Q/K, unrotated V, absolute positions,
actual borrowed pool views, QK products, scores, probabilities, FP32 attention
before conversion, FP16 attention and subsequent layer outputs. It separately
measures old slots and appended slots, checks old history against its own prefill
snapshot, verifies writes against local projections, and preserves the pool
pointer and no-full-history guard. Traced final logits must equal the untraced
baseline for both paths. Cleanup must return all ownership and reservations to zero.

The first exact FP32 and FP16 differences are reported independently in decoder
operation order. Only their layers are replayed. Each path's own native stages
must reproduce before interpreting controls. For each captured input history,
both actual attention implementations receive identical Q/K/V; CPU FP64
head-by-head reductions provide independent attention mathematics. A separate
identical-P/V replay isolates value multiplication using the original product
layouts. Native attention/replay runs on T4; the FP64 reference runs on CPU.
Report maximum/RMS error and rounded FP16 mismatches, retaining full tensors.
These observations do not select a repair or justify changing the comparator.

Cell 4 always exports source identity, commands, environment, raw tensors,
checksums and partial failures to one ZIP. If a cell fails, run through the
export cell and inspect the retained failure. Download from Kaggle Output if
the notebook link is unavailable. A completed collection is not an accuracy pass.

## Local validation and remaining work

Synthetic CPU regressions cover both two-token and sixteen-token traces,
history/position/write checks, exact baseline controls and native replay. Analytic
FP64 tests check all eight query heads per KV head. Notebook regressions execute
the real extraction and export code under a simulated setup failure and verify
source/evidence hashes, including the new four-cell mode.

Local CPU evidence: Ruff lint/format and strict mypy pass. The full non-GPU run
records 775 passes and one environment-related failure: its temporary directory
was inside this checkout, while the affected provenance test requires a location
outside any checkout. With `TEMP`/`TMP` set outside the repository, that case and
the notebook regressions pass (23 tests). Thus all 776 selected CPU cases have
passing execution evidence across these runs; the initial failed run is retained
and is not relabeled as a complete passing run. See the
[structured CPU report](../../results/validation/2026-11-13/sixteen-token-trace-cpu.json)
and its linked raw JUnit files. Local Python 3.11.15 / Torch 2.7.1+cpu has no CUDA.
The earlier focused trace/probe run passes 16 tests, including both prompt lengths.

The two-token default remains supported. Diagnostic prompt IDs and attended
lengths are now saved from the selected case rather than hardcoded to one/two.
Runtime attention, decoder, allocator, numerical thresholds and acceptance tests
are unchanged. The focused T4 run and any resulting repair/regression, followed
by the complete required T4 accuracy suite, remain outstanding.

Rebuild the embedded snapshot after source changes:

```bash
python -m scripts.build_numerical_notebook --layer-trace --prompt-tokens 15
```
