# V3 T4 accuracy review and long-fixture correction

Reviewed November 16, 2026. **Single, mixed and fragmented decode pass the frozen
v3 numerical rules. The long-layout case stops on a truncated test fixture, so
the full gate remains open. No further tolerance or serving change is selected.**

Retained [archive](../../results/validation/2026-11-16/pagedcore-accuracy-036a7n6u.zip)
and [structured audit](../../results/validation/2026-11-16/v3-accuracy-t4-review.json).
Archive SHA-256: `2626325110387b72dc71dc963f99d61f1a8620bcc03ea282c0cbe54c71d83bc7`.

## Source and execution identity

All **94 evidence hashes and 123 source hashes** verify with complete manifest
coverage. The source snapshot hash
`ef89b70612999b616d07894cdff2f67b568d99c77e6e568f167a507c800c8b1e`
matches the delivered notebook. Source-after hashes are unchanged, and all 15
runtime source files match the working tree. The dirty base revision plus full
snapshot identifies this source; it is not a clean-commit claim.

The target identifies one CUDA-visible Tesla T4, capability 7.5, Torch
`2.7.1+cu126`, Python 3.11.15 and the pinned TinyLlama checkpoint. Two physical
T4s are listed by inventory, with only GPU 0 exposed to the process. The math
settings retain deterministic execution, no TF32 and the frozen workspace policy.
The default PagedCore attention backend runs; no FlashInfer kernel is involved.

Setup, checkpoint verification, Ruff and strict typing succeed. **829 CPU tests
pass with zero skips. Nine of ten GPU tests pass**, with zero GPU errors/skips.
The only GPU failure is `held_out`; the exporter correctly rejects the bundle.

## Independent recomputation of retained tensors

All **37 raw logit pairs** and **37 per-page K/V collections** are finite and
recomputed locally using safely loaded tensors. All 222 reported logit metric
fields agree, with maximum last-bit RMS discrepancy `8.68e-19`. Each K/V
collection has exactly the expected logical-page/layer/kind coverage and valid
shapes for its actual attended length. Independent FP64 K/V ratios differ from
the recorded FP32 ratios by at most `3.25e-8`. Every retained numerical comparison
passes the frozen logit `0.02` and K/V `0.01` absolute/relative limits; all 37
retained greedy choices agree.

| Case | Retained rows / required | Maximum logit ratio | Maximum K/V ratio | Test result |
|---|---:|---:|---:|---|
| Single, 15-token prefix | 3 / 3 | 0.605478 | 0.544712 | Passed |
| Mixed, 1/15/16/17/49 | 15 / 15 | 0.689928 | 0.628801 | Passed |
| Fragmented, 1/15/16/17/49 | 15 / 15 | 0.689928 | 0.628801 | Passed |
| Intended 37/73/111/145 | 4 / 12 | 0.477347 | 0.328947 | Invalid fixture; incomplete |

A ratio at most one passes. The last row contains actual prompt lengths
37/49/49/49 and cannot support the requested longer-length coverage. The 33 rows
from the three successful cases are valid complete evidence for those cases.
Assertions for exact cached history, poisoning/isolation, staged rollback/retry,
pool identity, accounting, cleanup and no full-history gather execute in these
passing tests. They do not certify long-layout execution or maximum-active memory.

## Cause of the remaining failure

The shared test helper takes `maximum_output.input_ids[:length]`. That public
fixture has **49 IDs**. Slicing it at 73, 111 or 145 silently returns only 49.
Actual prompt lengths are therefore `37, 49, 49, 49`, and after the first decode
step the allocator correctly reports `38 + 50 + 50 + 50 = 188` cached slots.
The test expects `37 + 73 + 111 + 145 + 4 = 370` and fails its unchanged accounting
assertion. Raw K/V shapes and the four recorded attended lengths confirm the
truncation. This is not evidence of an allocator counting error.

The numerical comparisons finish before this assertion, so four rows are saved;
steps two and three never run. The archive cannot be called a complete pass.

## Minimal correction and regression

For requests longer than the base fixture, repeat its 48-token body while keeping
one initial BOS. Preserve every existing prefix through 49 IDs exactly. Use the
requested lengths 37/73/111/145 unchanged and assert actual input and prefilled
cache lengths before decode. The tokenizer, original fixture, serving decoder,
attention, allocator, comparator, thresholds and accounting assertions are unchanged.
These repeated public IDs provide layout coverage, not a quality corpus.

Extend the fake-decoder collector regression to the same long lengths. Before
the fix it reproduces the cached-slot assertion failure. After the fix it verifies
all four actual prompt lengths, all 12 step/request rows, exact first-step
attended lengths, unchanged short prefixes, raw K/V coverage and cleanup. Its
deliberately excessive logit error must still be rejected after complete collection.
The focused accuracy/collection checks pass **27 tests** on CPU.

Run the corrected `notebooks/paged_decode_v3_fixture_fix.ipynb` on T4. It uses the
same default backend and frozen thresholds and reruns the full required suite.
Close P5.3 only after genuine long-prefix coverage passes and the returned archive
is reviewed. The first three passing cases are retained; no rounding sweep,
new backend or larger tolerance is needed to address this fixture failure.

## Local checks after the correction

The full corrected Windows CPU suite passes **830 tests, 67 deselected, zero
skipped** in 281.67 seconds. Retained [JUnit report](../../results/validation/2026-11-16/v3-fixture-full-cpu.xml).
Ruff lint/format, strict mypy for all 15 runtime files and `git diff --check` pass.
Local links in the seven affected status/review/index documents resolve. The
returned archive supplies the T4 observations above; local correction validation
is CPU-only and cannot establish the genuine long-prefix GPU result.

Executed commands use the repository virtual environment with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`:

```text
python -m pytest tests/unit/test_decode_accuracy_collection.py tests/unit/test_accuracy.py -q
python -m pytest -m "not gpu and not benchmark" -q --basetemp <external-work-directory> --junitxml=results/validation/2026-11-16/v3-fixture-full-cpu.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
git diff --check
```
