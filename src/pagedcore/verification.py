"""Target-machine verification for the pinned reference checkpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast

from pagedcore.config import PagedCoreConfig, load_config
from pagedcore.determinism import inference_mode


class TargetVerificationError(RuntimeError):
    """Raised when the configured target cannot satisfy the Phase 1 gate."""


def _load_runtime() -> tuple[Any, Any]:
    try:
        torch = cast(Any, import_module("torch"))
        transformers = cast(Any, import_module("transformers"))
    except (ImportError, OSError) as error:
        raise TargetVerificationError(
            "target verification requires the CUDA and Transformers dependencies"
        ) from error
    return torch, transformers


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
) -> dict[str, Any]:
    model_config = model.config
    expected_values = {
        "model_type": "llama",
        "num_hidden_layers": 22,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
    }
    mismatches = {
        name: {"expected": expected, "actual": getattr(model_config, name, None)}
        for name, expected in expected_values.items()
        if getattr(model_config, name, None) != expected
    }
    model_class = type(model).__name__
    if model_class != "LlamaForCausalLM":
        mismatches["model_class"] = {"expected": "LlamaForCausalLM", "actual": model_class}

    resolved_model_revision = _resolved_revision(model_config)
    if resolved_model_revision is not None and resolved_model_revision != config.model_revision:
        mismatches["model_revision"] = {
            "expected": config.model_revision,
            "actual": resolved_model_revision,
        }
    resolved_tokenizer_revision = _resolved_revision(tokenizer)
    if (
        resolved_tokenizer_revision is not None
        and resolved_tokenizer_revision != config.model_revision
    ):
        mismatches["tokenizer_revision"] = {
            "expected": config.model_revision,
            "actual": resolved_tokenizer_revision,
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

    return {
        "id": config.model_id,
        "revision": config.model_revision,
        "architecture": model_class,
        "model_type": model_config.model_type,
        "dtype": expected_dtype,
        "config": expected_values,
        "resolved_revision": resolved_model_revision,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "revision": config.model_revision,
            "resolved_revision": resolved_tokenizer_revision,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        },
    }


def verify_target(config: PagedCoreConfig | None = None) -> dict[str, Any]:
    """Load the pinned reference model, run one forward pass, and return a manifest."""
    resolved_config = load_config() if config is None else config
    torch, transformers = _load_runtime()

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
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            resolved_config.model_id,
            revision=resolved_config.model_revision,
            use_fast=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            resolved_config.model_id,
            revision=resolved_config.model_revision,
            torch_dtype=torch.float16,
        )
        model = model.to(device)
        model.eval()
        checkpoint = _validate_checkpoint(model, tokenizer, torch, resolved_config)

        encoded = tokenizer("PagedCore target verification.", return_tensors="pt")
        inputs = {name: value.to(device) for name, value in encoded.items()}
        torch.cuda.synchronize(device)
        with inference_mode():
            outputs = model(**inputs, use_cache=True)
        torch.cuda.synchronize(device)
    except TargetVerificationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise TargetVerificationError(f"reference checkpoint verification failed: {error}") from error

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise TargetVerificationError("reference forward pass did not return logits")

    properties = torch.cuda.get_device_properties(0)
    from pagedcore.cli import collect_environment_diagnostics

    diagnostics = collect_environment_diagnostics(resolved_config)
    return {
        "manifest_version": 1,
        "status": "passed",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": diagnostics,
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
