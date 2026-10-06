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

## Proposing a change

Open an issue or pull request that states the roadmap task, explains why the change preserves the documented contracts, lists the checks run, and names any validation gate that remains unsatisfied.

By contributing, you agree that your contributions are licensed under the MIT License.
