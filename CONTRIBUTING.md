# Contributing to PagedCore

PagedCore is built in ordered validation gates. Before making a change, read `docs/SPEC.md`, `docs/DESIGN.md`, the relevant section of `docs/VALIDATION_PLAN.md`.

## Contribution rules

1. Keep each change within the current gate and focused on one behavior or invariant.
2. Preserve the v0.1.0 boundaries. Do not add sampling, quantization, additional architectures, multi-GPU support, prefix sharing, preemption, chunked prefill, speculative decoding, public-service security, or vLLM API compatibility.
3. Keep code, tests, and documentation aligned. A design document is not implementation evidence.
4. Add a regression test for every bug in paging, scheduling, streaming, or cleanup.
5. Run the smallest relevant checks while iterating and report whether validation was CPU-only or T4-backed. GPU correctness, memory safety, and performance claims require an actual NVIDIA T4.
6. Never include prompt text or generated text in logs, metrics, fixtures derived from private data, or benchmark results.

## CPU checks

The [CPU CI workflow](.github/workflows/ci.yml) runs on every push and pull request
on Ubuntu 24.04. It uses `.python-version` and the frozen `cpu`/`dev` extras,
checks lock consistency, formatting, lint and strict types, and runs all tests
except `gpu` and `benchmark`. This includes future CPU service tests. It retains
the pytest JUnit report even when tests fail.

Run the same checks locally after `uv sync --frozen --extra cpu --extra dev`:

```bash
uv lock --check
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m ruff check .
uv run --no-sync python -m mypy --strict src/pagedcore
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra
uv run --no-sync python -m build
```

CI also installs the built wheel without changing locked dependencies and checks
the installed package, default configuration, CLI help and environment diagnostics.
The build backend uses the existing isolated build requirements in `pyproject.toml`.
Extend this workflow with HTTP and container smoke checks when those surfaces exist.
CPU CI does not certify T4 execution or the later clean wheel/sdist installation gate.

On a clean Linux/T4 checkout with Git and `uv` available, run
`bash scripts/validate_t4.sh` to retain setup/command logs, device identity, CPU and
GPU reports, and reference captures in one local ZIP. Review the bundle before
publishing it under `results/validation/`; see the [evidence contract](docs/VALIDATION_PLAN.md#oracle-harness).

## Selecting test suites

The six registered markers are `unit` (isolated components and synthetic contracts),
`integration` (cross-component checks), `gpu` (actual T4/CUDA execution), `service`
(HTTP/SSE), `slow` (long-running checks), and `benchmark` (benchmark-harness checks).
Use explicit module or test marks. Mark every hardware-dependent test `gpu`,
including integration, service, slow, and benchmark tests that require CUDA.
Markers can overlap: the cached-tokenizer integration check is in a unit-marked
module, and `slow` describes duration rather than hardware requirements.

Run these commands from the repository root in the frozen environment:

| Suite | Command |
|---|---|
| Routine CPU CI (includes CPU integration, service and slow checks) | `uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra` |
| Isolated CPU unit checks | `uv run --no-sync python -m pytest -m "unit and not integration and not gpu and not benchmark" -ra` |
| CPU integration checks | `uv run --no-sync python -m pytest -m "integration and not gpu and not benchmark" -ra` |
| CPU service checks | `uv run --no-sync python -m pytest -m "service and not gpu and not benchmark" -ra` |
| Extended CPU checks | `uv run --no-sync python -m pytest -m "slow and not gpu and not benchmark" -ra` |
| CPU benchmark-harness checks | `uv run --no-sync python -m pytest -m "benchmark and not gpu" -ra` |
| T4 checks | `uv run --no-sync python -m pytest -m gpu -ra` |

For T4 checks install the frozen `cuda`/`dev` extras and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts, as described in the
[validation plan](docs/VALIDATION_PLAN.md#oracle-harness). Inspect executed and
skipped counts; an all-skipped run is not GPU certification. Service, slow and
benchmark suites currently have no marked tests; selection returns pytest exit
code 5 until those suites exist. Use `--collect-only` to inspect selection without
executing tests. Do not treat empty selections as completed validation gates.

## Proposing a change

Open an issue or pull request that states the roadmap task, explains why the change preserves the documented contracts, lists the checks run, and names any validation gate that remains unsatisfied.

By contributing, you agree that your contributions are licensed under the MIT License.
