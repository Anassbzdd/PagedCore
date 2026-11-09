# T4 weighted-value kernel sweep

Reviewed November 7, 2026. The user supplied a synthetic dense P@V sweep from
checkout `7d9ea61` with a clean working tree reported in metadata. It contains
54 completed lengths, no error records, embedded source and environment data.
The embedded source SHA-256 matches metadata. This diagnostic uses independent
dense tensors outside serving. It does not validate paged decoder acceptance.

The [retained ZIP](../../results/validation/2026-11-07/value-kernel-sweep-t4.zip)
has SHA-256 `39379b6c874ad9f25ec2bff798a85788016763f834a9fbc79fb411fb4664bf6e`.
Metadata records one CUDA-visible device and an inventory of two physical T4s.
Synthetic cancellation and random-input profiles share a GEMV kernel at every
tested length. Six output coordinates were sampled. Lengths through 20 use
every ordered placement of three terms; longer lengths use 512 placements.

## Observations

- GEMVX variant 5 appears at 24 tested lengths, including 17, 19, 24, 32 and
  65. Two alternating FP32 sums match every recorded cancellation observation
  for these lengths.
- Variant 6 appears at 13 lengths: 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 26,
  28 and 30. None of the original striped models fits all cancellation trials.
- Variant 7 appears at 95, 97, 127, 129, 255, 257, 511, 513, 1023, 1025 and
  2047. GEMV NSP appears at 96, 128, 256, 512 and 1024. Variant 8 appears at
  2048. None of the original striped models fits every cancellation trial in
  these families. A rule based only on odd/even length would miss the observed
  variant-5 cases at 24, 32, 48 and 64.

## Inferred variant-6 addition order

Offline FP32 replay of the saved inputs found a matching candidate for every
variant-6 length: sum the first L/2 terms sequentially into one FP32 accumulator,
sum the second L/2 sequentially into another, add those accumulators, and cast
once to FP16. At 18 tokens, these are positions 0 through 8 and 9 through 17.
This matches 131,088 recorded coordinate observations across the 13 lengths,
plus all 78 sampled random-input outputs. These counts were recomputed locally
with NumPy from the supplied ZIP. The real execution evidence is the T4 archive.

For reproduction, form each FP32 product as the product of the saved FP16
operands. Initialize two FP32 zero arrays across trials. For i from 0 through
L/2 minus one, add product i to the first array and product L/2+i to the second.
Add the two arrays in FP32, cast to FP16, and compare with each saved output.

This is an inferred candidate addition order for these shapes and sampled
coordinates. The cancellation sweep uses restricted inputs, and the random
sample has limited discriminating power: most original models also match its
outputs. Validation on actual TinyLlama layer operands and guarded integrated
decode is required. The score path has a separate known 18-token discrepancy.
The next experiment should test this candidate on real operands and the
existing single/mixed/fragmented cases at the frozen tolerances. P5.3 remains open.
