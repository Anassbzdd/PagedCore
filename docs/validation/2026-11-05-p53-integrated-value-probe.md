# T4 integrated value and score controls

The [November 5 diagnostic ZIP](../../results/validation/2026-11-05/pagedcore-parity-probe.zip)
contains the striped baseline, page-wise FP64 probe, and native value multiply
with exact V checks. All three reports have 9,238 records. The ZIP integrity check
passes and its SHA-256 is
`bb53b06b7c7e7024a1c7594f153b52aae186118a842116b30110fa24c326e8f1`.

All 39 source hashes in each report match clean base
`fa395d1b6d836ea1080444db3b2b47bdc3b252d7`. The notebook ran in-process
overrides on that base. Use `experiment_arm` to identify the override because
the legacy `value_reduction` field says `striped` in all three reports.
In the native arm only `integrated` and the integrated layer trace reflect the
intervention. The isolated attention and fixed-probability measurements in that
arm still use the striped reducer.

The archive reports Python 3.11.17, Torch 2.7.1+cu126, Transformers 4.52.4,
cuBLAS 12.6.4.1 and driver 580.178.04. The Kaggle setup confirmed one
CUDA-visible T4 and the archive lists two physical T4s. The archive retains
physical UUIDs, with no selected-device UUID or inline probe hash.

## Integrated results

| Experiment arm | Logit comparisons above tolerance, out of 18 | Worst logit ratio | Worst K ratio | Worst V ratio |
|---|---:|---:|---:|---:|
| Striped | 13 | 3.502409 | 2.768614 | 1.251979 |
| Page-wise FP64 | 13 | 3.883116 | 6.974359 | 1.301965 |
| Native value with equal V | 6 | 2.375193 | 2.893124 | 0.989389 |

Ratios above one exceed the frozen tolerance. All 18 next-token argmax
comparisons agree in each arm. At 17 attended tokens the striped single request
first differs at layer 1 with equal Q/K/V and probabilities. FP64 moves the
first difference to layer 2, head 16, channel 58, by one FP16 step even with
the same probabilities. Its single-request logits still exceed tolerance by a
factor of 1.248103. FP64 is rejected as a parity fix.

## Independent score divergence

The native value control makes the single-request 17-token case exact. In its
18-token step, layer 2 has equal Q/K/V but different probabilities and attention
output. That request's final logits are exact. In the first mixed step, the
17-token prompt also reaches 18 attended tokens. It is exact through layer 7.
At layer 8 Q/K/V still match, but head 13's score for zero-based logical token
14 differs (`-1.3681640625` versus HF `-1.369140625`). Token 14 is on the
first full page. Its FP16 probability changes from HF's `0.020477294921875`
to `0.0205078125`. Attention output differs while the native value multiply is
still active. At layer 9 Q/K/V start differing and the native arm falls back.
This request's logit ratio is 2.375193. The 16-token prompt in the same batch
exhibits the same sequence of events at 18 attended tokens, starting at
layer 13 with fallback at layer 14.

## Combined score and value control

The [score-control ZIP](../../results/validation/2026-11-05/pagedcore-score-control.zip)
has SHA-256 `8d150906c700824d461ca169c7e26fc104889be5dd7456df2dc039c134067770`.
ZIP integrity and all 39 recorded base source hashes verify. The runtime and
math settings match the previous archive. Its `native_same_v_hf_p` arm
supplies HF probabilities only when Q/K/V match exactly and rounded
probabilities differ. It uses native value arithmetic for the multi-page calls.

It substitutes probabilities in 11 of 396 layer calls, across six request
steps at 18, 20, and 52 attended tokens. Every Q/K/V and attention-output check
is exact, with zero fallbacks. All 18 logit comparisons, 814 key comparisons,
and 814 value comparisons are exact. All next-token argmax comparisons agree.
This establishes that the two interventions together remove all observed
integrated drift in these fixtures. The probe uses oracle tensors as controls.
The independent score errors include both full and short pages, so the next
comparison should hold Q/K fixed, measure raw dots and rounded probabilities,
and check every case for newly introduced differences. Integrated serving parity
and the fragmented T4 acceptance case remain open.
