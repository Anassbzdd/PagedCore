# PagedCore

PagedCore is a learning-focused, single-GPU LLM inference engine for studying paged KV-cache management, continuous batching, and streaming generation on one NVIDIA T4.

The v0.1.0 scope is intentionally narrow: a pinned TinyLlama checkpoint, a Llama-compatible FP16 decoder, greedy generation, a 2,048-token context, one model worker, and a local FastAPI/SSE interface. Correctness uses independent mathematics, model-component checks and paging invariants. Quality evaluation against the same checkpoint in vLLM is planned; HF comparisons are optional diagnostics. Paged decode is checked against independent FP64 attention and contiguous FP32-attention decode under [the numerical policy](docs/NUMERICAL_POLICY.md); the serving path owns its decoder and KV-cache behavior.

## Status

As of 2026-10-27, Phase 4 is complete: the owned dense TinyLlama decoder and
reservation-aware GPU KV allocator have reviewed T4 correctness evidence. The
[October 27 review](docs/validation/2026-10-27-phase4-review.md) records 554 CPU
and 12 GPU passes with zero skips at clean commit `986392e`; newer allocator
property tests and collector guards are validated separately on CPU.

The current gate is **Phase 5: block-table decode attention**. Prompt writes into
paged storage have retained T4 evidence. The
[October 31 review](docs/validation/2026-10-31-p52-t4-review.md) closes standalone
paged attention at clean `3401406`: two runs each pass 617 CPU and 15 GPU tests
with zero skips. The newer batched one-token paged decode and staged attention
reads have synthetic CPU coverage; the revised independent-accuracy contract
still requires a new T4 run. The [self-contained Kaggle notebook](notebooks/paged_decode_accuracy.ipynb)
collects owned-reference accuracy without required HF execution. The November 10 integrated T4 comparison still fails; P5.3 remains open. Independent component coverage and held-out model quality are additional release gaps. Continuous batching, the worker and HTTP/SSE service
remain planned. Pool/batched/service memory safety and performance
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

On the target T4, verify the owned checkpoint/tokenizer and capture a prompt-free
environment manifest with:

```bash
uv run --no-sync python -m scripts.verify_owned_checkpoint --output results/local/environment-manifest.json
```

Create `results/local/` first. The version-3 manifest records owned-forward smoke,
pinned model/tokenizer identity, environment and source/lock provenance. It requires
one CUDA-visible T4 and does not compare against an external model. Retain actual
setup/commands and reports under the [evidence contract](docs/VALIDATION_PLAN.md#evidence-attribution).
The older `pagedcore verify` and `pagedcore oracle` commands remain optional HF
diagnostics. For the next required T4 run use the self-contained
[paging accuracy notebook](notebooks/paged_decode_accuracy.ipynb), enable Internet
and T4 GPU, Run All and retain the ZIP even on failure. It does not evaluate model
quality, end-to-end paged generation, workspace or the service.

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
