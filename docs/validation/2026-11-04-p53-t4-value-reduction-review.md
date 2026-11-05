# T4 value-reduction diagnostic review

Reviewed 2026-11-04. The current gate remains P5.3 integrated paged decode
parity. Neither reducer satisfies it. This review changes no engine arithmetic.

## Retained evidence

The supplied notebook `C:\Users\user\OneDrive\Desktop\notebooke629accf3b.ipynb`
shows the setup, commands, diagnostic summaries and two failing candidate tests.
The accompanying [ZIP](../../results/validation/2026-11-04/p53-diagnostics.zip)
contains `striped.json`, `paged.json` and `candidate-tests.xml`; its integrity
check passes. SHA-256:
`37c943309a047a4cc9b9a95743a0dbaf5cc5eb1371a652d6e107bbcb89186881`.
The [structured review](../../results/validation/2026-11-04/p53-diagnostic-review.json)
retains summaries, environment, math policy, identity checks and test failures.

Both JSON reports identify clean source
`fcc3639a200e28325cc11a7fb2180633fccd66be`. All 38 recorded source hashes match
that revision's Git blobs, and the script, fixture and lock hashes match locally.
Seven local byte-hash differences are solely Windows CRLF versus Git LF endings.
Both reports contain 9,238 records. Capture times are
`2026-11-04T22:15:23.397575+00:00` (striped) and
`2026-11-04T22:15:51.898151+00:00` (paged).

Actual setup output reports Python 3.11.15, Torch 2.7.1+cu126, Transformers
4.52.4, one visible Tesla T4, and CUDA 12.6. The notebook's stored metadata says
GPU/internet disabled, but its executed CUDA output and tests establish the
runtime used; the metadata is not treated as execution evidence. The JSON setup
command/argument fields are null, and this is not a complete release collector
archive. Successful diagnostic execution is distinct from successful parity.

## Proven result at 17 attended tokens

The single request starts with 15 prompt tokens. Decode step 2 attends over 17
tokens. For that case, every layer's eager-replay scaled scores, FP32 probabilities
and rounded FP16 probabilities match HF exactly. Direct control-page K/V checks
are exact across both complete reports.

The original striped reducer first differs at zero-based layer 1, output
coordinate `[0, 0, 2, 41]` (layout `[batch, query, head, dimension]`):

| Value | Result |
|---|---|
| PagedCore | `-0.000060677528381347656` |
| HF | `-0.00006073713302612305` |
| Absolute difference | `5.960464477539063e-8` |

The identical difference remains when the diagnostic replaces softmax output
with HF's exact FP32 probabilities before the FP16 cast. This isolates the
first divergence to the weighted-value calculation or its final FP16 rounding;
it excludes score generation, normalization and copied-page contents for this
specific replay. The exact CUDA accumulation instructions/order remain unknown.

In integrated striped decode, new K/V first differ at layer 2, the earliest
failing K/V measurement is layer 7's key, and the single-step logit tolerance
ratio is `1.977457`. The maximum integrated ratio across all cases is
`3.502409`. This is consistent with propagation from the early attention
difference, without recording every residual/MLP boundary of that propagation.

## Candidate result

The page-product candidate removes that first layer-1 discrepancy but first
differs at layer 4, coordinate `[0, 0, 20, 19]`, also by one FP16 subnormal step:
`0.000055909156799316406` versus HF `0.000055849552154541016`.
This too survives fixed-HF-probability replay. New integrated K/V first differ
at layer 9, and the earliest failing K/V measurement is layer 13's key.

| Measurement | Original striped | Candidate paged |
|---|---:|---:|
| First isolated difference at 17 tokens, zero-based layer | 1 | 4 |
| Single-request step-2 logit tolerance ratio | 1.977457 | 1.553538 |
| Maximum integrated tolerance ratio across all cases | 3.502409 | 3.695339 |

The ratio is `abs(actual - expected) / (atol + rtol * abs(expected))`; values
above one fail. Both diagnostic runs retain matching next-token argmax values.
That does not satisfy logit/K/V acceptance. Delaying the first local discrepancy
does not establish better end-to-end parity across the workload.

The candidate pytest cases now pass through the corrected storage guard to the
logit assertions. Both fail numerical comparison at `test_decoder_parity.py:198`:

- Single: 10 mismatched logit elements out of 32,000; largest absolute difference
  reported among failing elements `0.0091552734375`.
- Mixed: 38 mismatched logit elements out of 32,000; largest reported absolute
  difference among failing elements `0.01025390625`.

These are actual numerical failures rather than the earlier guard false
positive. Assertions after the failing comparison (including later ownership
and final cleanup checks) are not established by these failed test executions.
No tolerance is changed and the candidate remains opt-in.

## Limits and next experiment

The exact-probability observation is specific to the first 17-token case. Across
the full diagnostic, each reducer records 687 unequal FP32 probability elements
and 17 unequal FP16 probability elements. The first such case is at 18 attended
tokens. Do not generalize score/probability equality to all lengths. The forced
probability replay separates value arithmetic even when upstream scores differ.

Both existing value reducers are rejected as a complete numerical fix. The
next targeted experiment should retain the 17 HF probabilities and 17 V values
for the first failing scalar (query head 2, KV head 0, dimension 41), together
with HF output and a profiled reference CUDA kernel. Compare FP64 scalar math,
the existing FP32 reduction orders, and the actual CUDA result to determine the
rounding boundary before choosing another reducer. The saved JSON contains
aggregate/scalar errors, not these operands or the kernel implementation.

No further arithmetic change is justified solely by trying another lane count,
padding size or tolerance. The decoder must continue reading pages through the
block table, with no full-history K/V materialization. T4 parity and the remaining
Phase 5 workspace/lifecycle evidence remain open.

This review used ZIP integrity/hash checks, JSON and JUnit inspection, Git-blob
source comparison and notebook-output inspection. No new CPU or T4 execution
was performed locally, and no performance claim is made.
