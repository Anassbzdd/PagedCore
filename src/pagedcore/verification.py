"""Target-machine verification for the pinned reference checkpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pagedcore.config import PagedCoreConfig, load_config
from pagedcore.determinism import inference_mode


class TargetVerificationError(RuntimeError):
    """Raised when the configured target cannot satisfy the Phase 1 gate."""


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch = cast(Any, import_module("torch"))
        transformers = cast(Any, import_module("transformers"))
        hub = cast(Any, import_module("huggingface_hub"))
    except (ImportError, OSError) as error:
        raise TargetVerificationError(
            "target verification requires CUDA, Transformers, and Hugging Face Hub dependencies"
        ) from error
    return torch, transformers, hub


def _download_pinned_snapshot(hub: Any, config: PagedCoreConfig) -> Path:
    snapshot = Path(hub.snapshot_download(repo_id=config.model_id, revision=config.model_revision))
    if snapshot.name != config.model_revision or not snapshot.is_dir():
        raise TargetVerificationError(
            "Hugging Face Hub did not return a local snapshot for the pinned revision"
        )
    return snapshot


def _resolved_revision(resource: Any) -> str | None:
    direct_revision = getattr(resource, "_commit_hash", None)
    if direct_revision:
        return str(direct_revision)
    init_kwargs = getattr(resource, "init_kwargs", {})
    if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])
    return None


def _validate_checkpoint(
    model: Any,
    tokenizer: Any,
    torch: Any,
    config: PagedCoreConfig,
    *,
    resolved_revision: str,
) -> dict[str, Any]:
    model_config = model.config
    expected_values = {
        "model_type": "llama",
        "num_hidden_layers": 22,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
    }
    mismatches = {
        name: {"expected": expected, "actual": getattr(model_config, name, None)}
        for name, expected in expected_values.items()
        if getattr(model_config, name, None) != expected
    }
    model_class = type(model).__name__
    if model_class != "LlamaForCausalLM":
        mismatches["model_class"] = {"expected": "LlamaForCausalLM", "actual": model_class}

    if resolved_revision != config.model_revision:
        mismatches["snapshot_revision"] = {
            "expected": config.model_revision,
            "actual": resolved_revision,
        }

    resolved_model_revision = _resolved_revision(model_config)
    if resolved_model_revision is not None and resolved_model_revision != resolved_revision:
        mismatches["model_revision"] = {
            "expected": resolved_revision,
            "actual": resolved_model_revision,
        }
    resolved_tokenizer_revision = _resolved_revision(tokenizer)
    if resolved_tokenizer_revision is not None and resolved_tokenizer_revision != resolved_revision:
        mismatches["tokenizer_revision"] = {
            "expected": resolved_revision,
            "actual": resolved_tokenizer_revision,
        }

    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if tokenizer_vocab_size != getattr(model_config, "vocab_size", None):
        mismatches["tokenizer_vocab_size"] = {
            "expected": getattr(model_config, "vocab_size", None),
            "actual": tokenizer_vocab_size,
        }

    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    expected_dtype = str(torch.float16)
    if parameter_dtypes != {expected_dtype}:
        mismatches["parameter_dtype"] = {
            "expected": [expected_dtype],
            "actual": sorted(parameter_dtypes),
        }
    if mismatches:
        raise TargetVerificationError(f"checkpoint validation failed: {mismatches}")

    observed_values = {name: getattr(model_config, name) for name in expected_values}
    return {
        "id": config.model_id,
        "revision": config.model_revision,
        "resolved_revision": resolved_revision,
        "architecture": model_class,
        "model_type": model_config.model_type,
        "dtype": expected_dtype,
        "config": observed_values,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "revision": config.model_revision,
            "resolved_revision": resolved_tokenizer_revision or resolved_revision,
            "vocab_size": tokenizer_vocab_size,
        },
    }


def _validate_forward_logits(logits: Any, input_ids: Any, model_config: Any, torch: Any) -> None:
    expected_shape = (
        int(input_ids.shape[0]),
        int(input_ids.shape[1]),
        int(model_config.vocab_size),
    )
    if tuple(logits.shape) != expected_shape:
        raise TargetVerificationError(
            f"reference forward pass returned logits with shape {tuple(logits.shape)}, "
            f"expected {expected_shape}"
        )
    if logits.dtype != torch.float16:
        raise TargetVerificationError(
            f"reference forward pass returned logits with dtype {logits.dtype}, expected {torch.float16}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise TargetVerificationError("reference forward pass returned non-finite logits")


def verify_target(config: PagedCoreConfig | None = None) -> dict[str, Any]:
    """Load the pinned reference model, run one forward pass, and return a manifest."""
    resolved_config = load_config() if config is None else config
    torch, transformers, hub = _load_runtime()

    if not bool(torch.cuda.is_available()):
        raise TargetVerificationError("target verification requires an available CUDA device")
    device_count = int(torch.cuda.device_count())
    if device_count < 1:
        raise TargetVerificationError("target verification found no CUDA devices")

    device = torch.device("cuda:0")
    device_name = str(torch.cuda.get_device_name(0))
    if "t4" not in device_name.casefold():
        raise TargetVerificationError(
            f"target verification requires an NVIDIA T4, found {device_name!r}"
        )

    try:
        snapshot = _download_pinned_snapshot(hub, resolved_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(snapshot),
            use_fast=True,
            local_files_only=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        model = model.to(device)
        model.eval()
        checkpoint = _validate_checkpoint(
            model,
            tokenizer,
            torch,
            resolved_config,
            resolved_revision=snapshot.name,
        )

        encoded = tokenizer("PagedCore target verification.", return_tensors="pt")
        inputs = {name: value.to(device) for name, value in encoded.items()}
        torch.cuda.synchronize(device)
        with inference_mode():
            outputs = model(**inputs, use_cache=True)
        torch.cuda.synchronize(device)
    except TargetVerificationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise TargetVerificationError(
            f"reference checkpoint verification failed: {error}"
        ) from error

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise TargetVerificationError("reference forward pass did not return logits")

    _validate_forward_logits(logits, inputs["input_ids"], model.config, torch)

    properties = torch.cuda.get_device_properties(0)
    from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance

    diagnostics = collect_environment_diagnostics(resolved_config)
    provenance = collect_validation_provenance()
    return {
        "manifest_version": 2,
        "status": "passed",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": diagnostics,
        "provenance": provenance,
        "target": {
            "device": "cuda:0",
            "name": device_name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "memory_mib": properties.total_memory // (1024 * 1024),
            "device_count": device_count,
        },
        "checkpoint": checkpoint,
        "forward_pass": {
            "status": "passed",
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "use_cache": True,
        },
    }
