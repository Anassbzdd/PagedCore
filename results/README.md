# Results policy

Commit only reviewed, reproducible validation and benchmark evidence in this directory.
Keep generated local runs under `results/local/`; that path is ignored by Git.

Logs, metrics and benchmark records must never contain prompt text, generated text
or token IDs. Public correctness-reference artifacts may retain token IDs, tensors
and decoded text from the checked-in public fixtures, as required by the oracle
contract in `docs/VALIDATION_PLAN.md`. Never publish private request content or
credentials. Record source/lock identity, actual setup and commands, device details,
test counts and limitations alongside retained artifacts.

The [review of retained T4 evidence](validation/2026-10-12-t4-evidence-review.json)
separates the October 7 environment checks from the October 12 oracle capture.
The latter proves reference repeatability but has no retained installer history.
Use `bash scripts/validate_t4.sh` on a clean Linux/T4 checkout to collect a new
bundle with setup logs and all three required GPU tests. Review the generated ZIP
under `results/local/` before copying it into `results/validation/`.
