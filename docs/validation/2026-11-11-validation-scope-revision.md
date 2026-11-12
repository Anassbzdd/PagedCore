# Validation scope revision and cleanup

Observation date: November 11, 2026. Owner-authorized scope revision; no commit or
push. The starting tree already modified STATUS and the November 10 layer-5 review,
and contained its untracked structured review. Those evidence additions remain.

## Contract and executable changes

HF parity, exact HF tokens and integrated HF fidelity are optional diagnostics.
The pinned checkpoint, tokenizer, model links and loading dependencies stay.
Default pytest deselects `hf_diagnostic`; explicit `--run-hf-diagnostics` restores
forensic comparisons. Required prefill/decode cases load only owned weights.
Owned checkpoint smoke verifies tokenizer file hashes/settings/public IDs, weights
and finite prefill. The v2 report/exporter has no HF fields or dense HF envelope
requirement; original integrated logit/KV/greedy assertions remain unchanged.
The required suite has ten GPU cases. Existing raw v1 reports are not relabeled.

[The numerical policy](../NUMERICAL_POLICY.md) separates mathematics, paging,
model components, quality, distribution fidelity and serving/performance. It cites
primary llama.cpp, vLLM, TensorRT-LLM and MLCommons practices. Quality evaluation
is planned: implement the cached-decode scoring adapter/evaluators, establish a
supported exact vLLM pin, collect separate pilot data, justify/freeze margins and
sample counts, then run untouched held-out loss/perplexity and fixed tasks with
uncertainty. Full KL requires complete aligned distributions, never top-k only.
No universal quality threshold or vLLM numerical reproduction requirement is added.

Independent scalar/FP64 RMSNorm and full half-pair RoPE CPU checks supplement
existing projection, mask, position and cache checks. Shared-component and full
checkpoint gaps remain explicit; old HF evidence does not close them.

## Evidence preserved

The integrated T4 two-token logit ratio 2.1705 and measured K/V ratio 1.9796 remain
failed under unchanged thresholds. The owner's November 10 layer-5 replay review
and all its raw evidence remain. It explains normalization sensitivity for one
call and provides no integrated pass. P5.3 stays open. A comparator redesign needs
measured causes and independent replacement evidence, rather than another
open-ended effort to reproduce a backend's rounding.

All original completed-gate/failure archives are retained. The historical
[plan snapshot](2026-11-11-superseded-validation-plan.md) and
[status snapshot](2026-11-11-status-before-scope-revision.md) preserve previous
wording/conclusions while keeping active requirements clear.

## Cleanup

The [inventory](../../results/validation/2026-11-11/cleanup-inventory.json) records
32 pre-cleanup ZIPs, archive hashes and member hashes after full CRC-checked reads.
The actionlint ZIP is redundant: all 11 members match retained extracted files,
including the executable/license/docs. The tiny value-probe smoke ZIP contains
synthetic CPU summary stubs, no T4 data, and has no source/document dependency.
The two local ZIPs were removed. Recursive cache/build deletion was rejected
by automatic approval review with “blocked by policy”, including a second attempt
using exact preflight-verified absolute paths. No alternate deletion mechanism was
used. These regenerable folders remain; cleanup is partial. See
[cleanup outcome](../../results/validation/2026-11-11/cache-cleanup.json).
Retain `.venv`, active install metadata, every unique/uncertain evidence archive,
and diagnostic scripts with a linked report/test or forensic reproduction role.

In particular, `results/local/status-review/t4-20261027T200933Z.zip` has the same
member names as the retained October 27 ZIP but a different tolerance envelope;
it is not proven equivalent and stays. Different dated T4 runs are separate
observations, not duplicate archives. Raw failed experiments stay even when a
newer report supersedes their acceptance contract. No entire results directory,
model cache or unrelated path is deleted. The ZIP deletions used native PowerShell after resolved-path containment checks.
The blocked cache/build targets and file counts are retained in the CPU review.

## Verification

Final full non-GPU suite: **764 passed, 59 deselected, zero skipped**. The earlier
759-test run and focused 36-/26-test runs are also retained. No optional T4 HF
comparison was executed locally.

Executed command results and CPU counts are recorded in
[cpu-review.json](../../results/validation/2026-11-11/cpu-review.json), with retained
[final JUnit](../../results/validation/2026-11-11/cpu-tests-final.xml). Verification covers Ruff,
formatting, strict mypy, non-GPU pytest, frozen-lock consistency, wheel/sdist build,
installed-wheel smoke, changed notebook JSON/code/source hashes, documentation
paths/anchors and `git diff --check`. No local GPU validation occurred.

## Exact next T4 action

Open [paged_decode_accuracy.ipynb](../../notebooks/paged_decode_accuracy.ipynb) in a
fresh Kaggle session, enable Internet and T4 GPU, Run All, then return the final
ZIP even on failure. The notebook embeds exact working source without a push.
It can establish only its tested mathematical/paging comparisons if they pass;
known integrated failure may remain. It cannot close paged free-running generation,
batched workspace/reuse, independent component gaps, model quality or service gates.
