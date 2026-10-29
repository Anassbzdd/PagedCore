# PagedCore

PagedCore is a learning-focused, single-GPU LLM inference engine for studying paged KV-cache management, continuous batching, and streaming generation on one NVIDIA T4.

The v0.1.0 scope is intentionally narrow: a pinned TinyLlama checkpoint, a Llama-compatible FP16 decoder, greedy generation, a 2,048-token context, one model worker, and a local FastAPI/SSE interface. Hugging Face is the correctness oracle; the serving path will own its decoder forward pass and KV-cache behavior.

## Status

As of 2026-10-27, Phase 4 is complete: the owned dense TinyLlama decoder and
reservation-aware GPU KV allocator have reviewed T4 correctness evidence. The
[October 27 review](docs/validation/2026-10-27-phase4-review.md) records 554 CPU
and 12 GPU passes with zero skips at clean commit `986392e`; newer allocator
property tests and collector guards are validated separately on CPU.

The current gate is **Phase 5: block-table decode attention**. Prompt writes into
paged storage have synthetic CPU coverage as of October 28; decode still uses
contiguous KV. Paged attention, continuous batching, the worker and HTTP/SSE
service remain planned. Pool/batched/service memory safety and performance
comparisons are also pending. See [current status](docs/STATUS.md) for evidence
boundaries and [the phase map](docs/VALIDATION_PLAN.md#build-order) for next steps.

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

On the target T4, verify the pinned model/tokenizer and capture a prompt-free
environment manifest with:

```bash
uv run pagedcore verify
```

The manifest is written to `results/local/environment-manifest.json` by default. It
records the resolved snapshot revision and, when run from a Git checkout, the commit,
working-tree status, lock SHA-256, and hashes of runtime/test sources and install metadata.
Manifest version 2 captures the actual CLI arguments with manifest paths redacted;
programmatic verification leaves arguments unknown. Setup history is also unknown
unless retained separately by the validation operator. Recommended commands above
are not recorded as executed commands.
This command requires CUDA device 0 to be an NVIDIA T4 and performs one reference
forward pass; a CPU-only run does not satisfy the target-machine gate.

Expected configuration or manifest-write failures return exit code 1 and JSON on
stderr. Retain setup output and CPU/GPU test reports alongside the manifest, following
the [evidence attribution contract](docs/VALIDATION_PLAN.md#evidence-attribution).

## Project documents

- [Implementation and evidence status](docs/STATUS.md)
- [MVP specification](docs/SPEC.md)
- [Technical design](docs/DESIGN.md)
- [Validation plan](docs/VALIDATION_PLAN.md)
- [Resolved implementation constants](docs/CONSTANTS.md)
- [MVP acceptance traceability](docs/TRACEABILITY.md)

## Repository layout

```text
src/pagedcore/       importable runtime package
tests/unit/           CPU unit and synthetic integration checks
tests/gpu/            T4 reference, dense decoder and KV storage checks
benchmarks/           placeholder for future workloads and runners
scripts/              developer and validation utilities
results/              retained validation evidence; benchmarks are pending
```

Commit only reviewed, reproducible evidence under `results/`. Generated local benchmark output belongs under `results/local/` and is ignored by Git; see [results/README.md](results/README.md) for the policy.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Contributions must preserve the documented MVP boundaries and include evidence appropriate to the current implementation gate.

## License

PagedCore is available under the [MIT License](LICENSE).
