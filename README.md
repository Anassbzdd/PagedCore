# PagedCore

PagedCore is a learning-focused, single-GPU LLM inference engine for studying paged KV-cache management, continuous batching, and streaming generation on one NVIDIA T4.

The v0.1.0 scope is intentionally narrow: a pinned TinyLlama checkpoint, a Llama-compatible FP16 decoder, greedy generation, a 2,048-token context, one model worker, and a local FastAPI/SSE interface. Hugging Face is the correctness oracle; the serving path will own its decoder forward pass and KV-cache behavior.

## Status

PagedCore is in Phase 0 (repository and contract baseline). The implementation has not started, so this repository does not yet provide an installable package or runnable service.

## Project documents

- [MVP specification](docs/SPEC.md)
- [Technical design](docs/DESIGN.md)
- [Validation plan](docs/VALIDATION_PLAN.md)
- [Implementation roadmap](docs/task.md)

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Contributions must preserve the documented MVP boundaries and include evidence appropriate to the current implementation gate.

## License

PagedCore is available under the [MIT License](LICENSE).
