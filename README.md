# PagedCore

PagedCore is a learning-focused, single-GPU LLM inference engine for studying paged KV-cache management, continuous batching, and streaming generation on one NVIDIA T4.

The v0.1.0 scope is intentionally narrow: a pinned TinyLlama checkpoint, a Llama-compatible FP16 decoder, greedy generation, a 2,048-token context, one model worker, and a local FastAPI/SSE interface. Hugging Face is the correctness oracle; the serving path will own its decoder forward pass and KV-cache behavior.

## Status

PagedCore is in the reproducible-environment gate. Package metadata, dependency locks, typed configuration loading, and safe environment diagnostics are present, but the inference engine and service are not implemented yet.

## Development setup

Use Python 3.11.15 and the committed `uv.lock` file. For the x86-64 Linux/CUDA 12.6 target (glibc 2.28 or newer):

```bash
uv sync --frozen --extra cuda
```

For CPU-only development and the allocator/scheduler test toolchain:

```bash
uv sync --frozen --extra cpu --extra dev
```

The `cpu` and `cuda` extras are mutually exclusive. CPU-only checks do not validate GPU correctness, memory safety, or performance.

Print the resolved, non-sensitive environment and PagedCore settings with:

```bash
uv run pagedcore env
```

## Project documents

- [MVP specification](docs/SPEC.md)
- [Technical design](docs/DESIGN.md)
- [Validation plan](docs/VALIDATION_PLAN.md)
- [Resolved implementation constants](docs/CONSTANTS.md)
- [MVP acceptance traceability](docs/TRACEABILITY.md)

## Repository layout

```text
src/pagedcore/       importable runtime package
tests/unit/           CPU unit tests
tests/integration/    cross-component tests
tests/gpu/            T4-backed correctness and lifecycle tests
tests/service/        HTTP/SSE service tests
benchmarks/           benchmark workloads and runners
scripts/              developer and validation utilities
results/              reviewed benchmark evidence
```

Commit only reviewed, reproducible evidence under `results/`. Generated local benchmark output belongs under `results/local/` and is ignored by Git; see [results/README.md](results/README.md) for the policy.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Contributions must preserve the documented MVP boundaries and include evidence appropriate to the current implementation gate.

## License

PagedCore is available under the [MIT License](LICENSE).
