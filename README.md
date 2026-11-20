# PagedCore

PagedCore is a learning-focused, single-GPU LLM inference engine for studying paged KV-cache management, continuous batching, and streaming generation on one NVIDIA T4.

The v0.1.0 scope is intentionally narrow: a pinned TinyLlama checkpoint, a Llama-compatible FP16 decoder, greedy generation, a 2,048-token context, one model worker, and a local FastAPI/SSE interface. Correctness uses independent mathematics, model-component checks and paging invariants. Quality evaluation against the same checkpoint in vLLM is planned; HF comparisons are optional diagnostics. Paged decode is checked against independent FP64 attention and contiguous FP32-attention decode under [the numerical policy](docs/NUMERICAL_POLICY.md); the serving path owns its decoder and KV-cache behavior.

## Status

As of November 16, 2026, **P5.3 batched paged decode and P5.4 guarded block-table
traversal are complete** under the frozen v3 numerical policy. The
[reviewed T4 run](docs/validation/2026-11-16-v3-complete-t4-review.md) passes
830 CPU tests, all ten required GPU tests and all 45 cached-decode comparisons,
with matching greedy choices and zero failures/errors/skips. Actual prompt
lengths include 37/73/111/145; source hashes and raw logit/KV tensors are retained.
Historical v2 failures keep their original conclusions.

The current gate is **P5.6 end-to-end paged generation**, then **P5.7 batched
workspace and CUDA page-reuse ordering**. Continuous batching, the worker,
HTTP/SSE, deployment, model-quality evaluation and performance comparisons remain
planned. This finite paging-suite pass does not establish those capabilities.
See [current status](docs/STATUS.md) and [the phase map](docs/VALIDATION_PLAN.md#build-order).

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

On the target T4, verify the owned checkpoint/tokenizer and capture a prompt-free
environment manifest with:

```bash
uv run --no-sync python -m scripts.verify_owned_checkpoint --output results/local/environment-manifest.json
```

Create `results/local/` first. The version-3 manifest records owned-forward smoke,
pinned model/tokenizer identity, environment and source/lock provenance. It requires
one CUDA-visible T4 and does not compare against an external model. Retain actual
setup/commands and reports under the [evidence contract](docs/VALIDATION_PLAN.md#evidence-attribution).
The `pagedcore verify` and `pagedcore oracle` commands are optional HF diagnostics.

## Reproduce T4 paging accuracy

The [corrected v3 notebook](notebooks/paged_decode_v3_fixture_fix.ipynb) embeds
the source from the reviewed November 16 T4 run. For a run on changed source,
build a fresh notebook from your checkout. See the
[T4 run and evidence instructions](results/README.md#retaining-a-new-run).

## Project documents

- [Implementation and evidence status](docs/STATUS.md)
- [MVP specification](docs/SPEC.md)
- [Technical design](docs/DESIGN.md)
- [Validation plan](docs/VALIDATION_PLAN.md)
- [Numerical acceptance policy](docs/NUMERICAL_POLICY.md)
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
