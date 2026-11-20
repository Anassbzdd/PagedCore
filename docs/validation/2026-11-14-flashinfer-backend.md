# Experimental FlashInfer decode backend

The owner requested implementation changes after the retained rounding studies.
This experiment reuses a native paged attention kernel, keeping PagedCore's
decoder, allocator, absolute positions and transactional cache writes. It does
not promise exact agreement between different floating-point reductions.

## Implementation

- `FlashInferAttention` is an optional, lazy-loaded callable passed explicitly
  to `decode_paged(attention_backend=...)`. Existing callers keep the baseline.
- `KVAllocator.paged_layer_view` borrows two contiguous per-layer pools shaped
  `[physical_pages, 16, 4, 64]`, the logical block tuple and readable length.
  It includes the staged token only after that layer is written. Committed
  lengths/ownership remain unchanged until every layer and final logits finish.
- The wrapper uses NHD layout, 32 query heads / 4 KV heads, dimension 64, scale
  1/8, CUDA-core decode and no CUDA graph. RoPE is disabled in the wrapper because
  the owned decoder already rotates Q/K; V remains unrotated.
- Small int32 metadata is rebuilt per request/layer. There is no dense history
  conversion, layout repacking or reference-value injection. Unrelated and invalid
  pool slots remain inaccessible according to the passed indices/last-page length.
- A 128 MiB workspace and the wrapper are reused on their construction worker
  and CUDA stream. This is correctness-first per-request invocation, without a
  throughput or maximum-active-memory claim. All backend errors propagate;
  synchronous errors leave committed cache state unchanged. Device-fault recovery
  is outside the rollback claim.

## Candidate pin and source attribution

Candidate: `flashinfer-python==0.2.5`, PyPI JIT source distribution,
SHA-256 `990aa090ef781783e76b836696ece4efd23956f72b5696d622fc619a61162aef`.
The frozen environment remains Python 3.11.15 / Torch 2.7.1+cu126;
Ninja 1.11.1.4 is an explicitly installed optional build tool. `uv.lock` is not
re-resolved. The notebook checks dependencies and Torch identity after installing
the optional overlay and saves its complete freeze, SDK/compiler logs and source.

The official [0.2.5 PyPI release](https://pypi.org/project/flashinfer-python/0.2.5/)
documents JIT installation requiring NVCC. Its source archive was inspected for
`flashinfer/decode.py` wrapper construction, `plan` arguments, tuple K/V layout
and `run` signature; these match the adapter. Its `setup.py` declares NumPy,
Torch and Ninja dependencies and SM75+ for CUDA compilation. No third-party
kernel source is copied into PagedCore. FlashInfer is Apache-2.0 licensed;
PagedCore's own source license remains unchanged.

This pin is an experiment, not a proven Kaggle compatibility matrix. CUDA JIT
compilation, runtime dispatch on T4 and native numerical behavior require execution.
Missing NVCC, an incompatible package or compilation failure stops native testing
and exports logs. There is no guessed ABI wheel or automatic fallback.

## Run and evidence boundary

Import `notebooks/flashinfer_backend_validation.ipynb` into a fresh Kaggle session,
enable Internet + T4, Run All and retain the final ZIP even if any cell fails.
It embeds the exact current source with hashes and records a dirty working tree
as such; it does not claim to be the old frozen failing source or a clean commit.

1. Install the frozen base and optional pinned candidate; check one visible T4,
   owned checkpoint, compiler, source hashes and dependencies.
2. Run full CPU checks, strict types and formatting.
3. Run eight native tests: independent FP64 on identical operands at lengths
   1/15/16/17/49/2048, large scores, nonadjacent pages with poisoned unused slots,
   pool/ownership stability, staged rollback/retry and thread/stream rejection.
   Python dispatch checks page materialization outside the opaque native kernel;
   numerical poison/isolation tests exercise the kernel's actual page selection.
   Neither constitutes a general device memory-safety proof.
4. Compare baseline and candidate at exactly 15 prefill tokens + 1 decode token,
   saving raw logits and K/V error diagnostics. A numerical failure does not stop
   collection of the frozen accuracy suite.
5. Run all required T4 paging cases with the candidate: single, mixed, fragmented,
   held-out, including the existing final-logit failure/retry. Keep logits/KV
   thresholds 5e-3 / 2e-3 and exact greedy checks. The existing strict exporter
   rejects incomplete/failed/skipped evidence. Each accuracy report labels the
   backend. A rejected report does not close P5.3.
6. Collect the candidate's 32-step teacher-forced real-checkpoint trajectory and
   exact instrumentation controls, then export hashes, raw outputs and partial
   failures. This is not a held-out language-quality evaluation.

Local validation results are recorded after execution below. CPU-only mocks prove
adapter orchestration and rollback, not FlashInfer compilation or T4 accuracy.
Whole-model quality, generation/service lifecycle, long-context integrated decode,
maximum-active workspace and performance are separate remaining gates.
