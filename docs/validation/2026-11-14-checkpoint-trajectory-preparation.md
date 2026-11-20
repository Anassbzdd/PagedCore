# Checkpoint component and cached-trajectory diagnostic

Prepared November 14, 2026. T4 execution is pending; P5.3 remains open.

Update: [returned T4 evidence](2026-11-14-checkpoint-trajectory-t4-review.md) completes
all 32 steps and 402 component checks with exact instrumentation controls, while
the integrated comparison still fails. The text below records the preparation scope.

[Kaggle notebook](../../notebooks/checkpoint_trajectory_validation.ipynb): enable
Internet and T4, then Run All. Frozen source/setup, checkpoint verification, CPU
checks and partial-failure ZIP export are embedded. No Input dataset is needed.

The collector prefills 15 IDs from the public maximum-output fixture and feeds
the next 32 fixture IDs through actual cached decode, reaching lengths 16-47.
Both paths receive identical histories. Logit/KV errors are measured at every
step without aborting for numerical mismatches. This is one fixed teacher-forced
trajectory, not greedy generation, perplexity, held-out quality or full acceptance.

At decode steps 0, 15 and 31, both paths' RMSNorm, SwiGLU and final-head calls are
audited using separate FP64 reductions with documented FP16 conversion boundaries.
Save small input/native/reference tensors; reload weights from the pinned checkpoint
rather than duplicating them in evidence. Errors are diagnostics with no new
acceptance thresholds. This is not a general independent 22-layer model oracle;
real-weight residual composition and broader long-context coverage remain open.

Run an uninstrumented trajectory followed by the audited trajectory. Native logits
from both paths must be bit-equal between runs at all steps before interpreting
component results. Record pool-pointer/cached-length checks, the blockwise guard,
full raw logits and zero ownership/reservation cleanup. Complete execution should
produce 32 rows per arm and 402 local component records. Partial results and logs
are exported even if setup, auditing or exact-output controls fail.

Local validation: 22 focused CPU tests pass, including native-output preservation,
instrumentation restoration and the existing independent scalar component checks.
Ruff lint/format, strict source mypy, CLI help and syntax checks pass. The GPU-only
collector has not run end to end locally; its T4 memory, runtime and behavior must
be inspected from the returned archive. No full CPU or GPU acceptance suite rerun
is claimed. [CPU JUnit](../../results/validation/2026-11-14/checkpoint-diagnostic-cpu.xml).

Rebuild with `python -m scripts.build_checkpoint_notebook`. This changes no serving
module, comparator or tolerance. A pinned vLLM cached-quality pilot remains later
work and must prove actual KV reuse at scored ground-truth-token steps.
