"""Hugging Face reference capture; no Transformers cache escapes this module."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import torch
from torch import Tensor

from pagedcore.config import load_config
from pagedcore.determinism import inference_mode, seed_everything
from pagedcore.engine_types import FinishReason
from pagedcore.verification import (
    TargetVerificationError,
    _download_pinned_snapshot,
    _validate_checkpoint,
)


@dataclass(frozen=True)
class ReferenceTrace:
    input_ids: tuple[int, ...]
    greedy_ids: tuple[int, ...]
    emitted_ids: tuple[int, ...]
    finish_reason: FinishReason
    decoded_text: str
    tensors: dict[str, Tensor]


def _reference_workspace_config() -> str | None:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_available() and workspace not in (":4096:8", ":16:8"):
        raise TargetVerificationError(
            "CUDA reference capture requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 "
            "set before starting Python"
        )
    return workspace


@contextmanager
def reference_math() -> Iterator[dict[str, object]]:
    """Restore caller math settings even when capture fails."""
    workspace = _reference_workspace_config()
    flags: tuple[tuple[Any, str, bool], ...] = (
        (torch.backends.cuda.matmul, "allow_tf32", False),
        (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
        (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False),
        (torch.backends.cudnn, "allow_tf32", False),
        (torch.backends.cudnn, "benchmark", False),
        (torch.backends.cudnn, "deterministic", True),
    )
    previous = [getattr(owner, name) for owner, name, _ in flags]
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for owner, name, value in flags:
            setattr(owner, name, value)
        torch.set_float32_matmul_precision("highest")
        # Reject nondeterministic operations rather than silently weakening the reference.
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": workspace,
        }
    finally:
        torch.set_float32_matmul_precision(precision)
        for (owner, name, _), value in zip(flags, previous, strict=True):
            setattr(owner, name, value)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def capture_reference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    ignore_eos: bool,
    *,
    expected_input_ids: tuple[int, ...] | None = None,
    layers: tuple[int, ...] | None = None,
) -> ReferenceTrace:
    """Capture independent CPU snapshots of prefill and each one-token decode."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty UTF-8 string")
    prompt.encode("utf-8")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be an integer from 1 to 256")
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    if model.config._attn_implementation != "eager":
        raise ValueError("reference attention must be eager")
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    ids = tuple(cast(list[int], encoded["input_ids"][0].tolist()))
    if not ids or len(ids) + max_new_tokens > 2048:
        raise ValueError("reference input exceeds the context contract")
    if expected_input_ids is not None and ids != expected_input_ids:
        raise ValueError("prompt IDs differ from the public fixture")
    selected_layers = (
        tuple(dict.fromkeys((0, len(model.model.layers) - 1))) if layers is None else layers
    )
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("capture layers must be nonempty and unique")
    if any(
        type(index) is not int or not 0 <= index < len(model.model.layers)
        for index in selected_layers
    ):
        raise ValueError("capture layer index is invalid")

    tensors: dict[str, Tensor] = {}
    step = -1

    def begin_step(module: Any, args: Any) -> None:
        nonlocal step
        step += 1

    def hook(name: str) -> Any:
        def snapshot(module: Any, args: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError("reference capture requires finite tensors")
            prefix = "prefill" if step == 0 else f"decode.{step}"
            tensors[f"{prefix}.{name}"] = value.detach().to("cpu", copy=True)

        return snapshot

    modules = {
        "embedding": model.model.embed_tokens,
        "final_norm": model.model.norm,
        "logits": model.lm_head,
    }
    for index in selected_layers:
        layer = model.model.layers[index]
        for name, module in (
            ("input_norm", layer.input_layernorm),
            ("attention", layer.self_attn),
            ("post_attention_norm", layer.post_attention_layernorm),
            ("mlp", layer.mlp),
            ("hidden", layer),
        ):
            modules[f"layer.{index}.{name}"] = module

    handles = []
    was_training = model.training
    try:
        handles.append(model.register_forward_pre_hook(begin_step))
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        model.eval()
        transformers = import_module("transformers")
        # Construct fresh defaults: the checkpoint's chat/sampling policy is not the oracle policy.
        generation = transformers.GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=None if ignore_eos else tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        device = next(model.parameters()).device
        with reference_math(), inference_mode():
            result = model.generate(
                **{name: value.to(device) for name, value in encoded.items()},
                generation_config=generation,
                use_model_defaults=False,
                eos_token_id=None if ignore_eos else tokenizer.eos_token_id,
                # HF otherwise keeps only the final prompt logit; retain the whole prefill.
                logits_to_keep=0,
            )
        greedy = tuple(cast(list[int], result[0, len(ids) :].tolist()))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not greedy or step + 1 != len(greedy):
        raise ValueError("reference generation and capture steps disagree")
    stopped = not ignore_eos and greedy[-1] == tokenizer.eos_token_id
    emitted = greedy[:-1] if stopped else greedy
    text = str(
        tokenizer.decode(emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    )
    return ReferenceTrace(
        ids, greedy, emitted, FinishReason.EOS if stopped else FinishReason.LENGTH, text, tensors
    )


def compare_traces(actual: ReferenceTrace, expected: ReferenceTrace) -> dict[str, float]:
    """Compare plain tensors/IDs; callers need no HF model or cache objects."""
    for name in ("input_ids", "greedy_ids", "emitted_ids", "finish_reason", "decoded_text"):
        if getattr(actual, name) != getattr(expected, name):
            raise AssertionError(f"reference mismatch: {name}")
    if not expected.tensors or actual.tensors.keys() != expected.tensors.keys():
        raise AssertionError("reference tensor names differ or are empty")
    discrepancies = {}
    for name, reference in expected.tensors.items():
        value = actual.tensors[name]
        if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"non-finite reference comparison: {name}")
        tolerance = 5e-3 if name.endswith(".logits") else 2e-3
        try:
            torch.testing.assert_close(value, reference, atol=tolerance, rtol=tolerance)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error
        discrepancies[name] = float((value.float() - reference.float()).abs().max().item())
    return discrepancies


def save_reference(path: Path, trace: ReferenceTrace, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "metadata": metadata,
        "input_ids": trace.input_ids,
        "greedy_ids": trace.greedy_ids,
        "emitted_ids": trace.emitted_ids,
        "finish_reason": trace.finish_reason.value,
        "decoded_text": trace.decoded_text,
        "tensors": trace.tensors,
    }
    # A failed write must not destroy an earlier usable reference.
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_reference(path: Path) -> tuple[ReferenceTrace, dict[str, Any]]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError("invalid reference artifact") from error
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("unsupported reference artifact")
    for name in ("input_ids", "greedy_ids", "emitted_ids"):
        ids = data.get(name)
        if not isinstance(ids, (tuple, list)) or any(
            type(item) is not int or item < 0 for item in ids
        ):
            raise ValueError("invalid reference token IDs")
    if not data["input_ids"] or not data["greedy_ids"]:
        raise ValueError("reference input and decisions must be nonempty")
    tensors = data.get("tensors")
    if (
        not isinstance(tensors, dict)
        or not tensors
        or any(
            not isinstance(name, str)
            or not isinstance(value, Tensor)
            or not value.numel()
            or not value.is_floating_point()
            for name, value in tensors.items()
        )
    ):
        raise ValueError("invalid reference tensors")
    if not isinstance(data.get("metadata"), dict) or not isinstance(data.get("decoded_text"), str):
        raise ValueError("invalid reference metadata or decoded text")
    if data.get("finish_reason") not in ("eos", "length"):
        raise ValueError("invalid reference finish reason")
    trace = ReferenceTrace(
        tuple(data["input_ids"]),
        tuple(data["greedy_ids"]),
        tuple(data["emitted_ids"]),
        FinishReason(data["finish_reason"]),
        data["decoded_text"],
        data["tensors"],
    )
    return trace, data["metadata"]


def run_oracle(
    fixture_path: Path,
    case: str,
    output_path: Path,
    compare_path: Path | None = None,
    *,
    command_arguments: list[str] | None = None,
) -> dict[str, Any]:
    """Generate reference evidence only for a checked public fixture on the target T4."""
    config = load_config()
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    matches = [item for item in fixtures["prompts"] if item["name"] == case]
    if len(matches) != 1:
        raise ValueError("reference case must identify one public fixture")
    fixture = matches[0]
    if not torch.cuda.is_available() or "t4" not in torch.cuda.get_device_name(0).casefold():
        raise TargetVerificationError("oracle reference capture requires an NVIDIA T4")
    _reference_workspace_config()
    transformers = import_module("transformers")
    if transformers.__version__ != "4.52.4" or torch.__version__.split("+")[0] != "2.7.1":
        raise ValueError("oracle requires the pinned Torch and Transformers versions")
    snapshot = _download_pinned_snapshot(import_module("huggingface_hub"), config)
    settings = fixtures["tokenizer"]
    if settings["model_id"] != config.model_id or settings["revision"] != config.model_revision:
        raise ValueError("fixture model/tokenizer revision differs from the pinned target")
    for name, expected_hash in settings["source_sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("pinned tokenizer file hash mismatch")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if type(tokenizer).__name__ != settings["tokenizer_class"]:
        raise ValueError("pinned tokenizer class mismatch")
    for name in ("add_bos_token", "add_eos_token", "bos_token_id", "eos_token_id", "vocab_size"):
        if getattr(tokenizer, name) != settings[name]:
            raise ValueError(f"pinned tokenizer setting mismatch: {name}")
    seed_everything(0)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        .to("cuda:0")
        .eval()
    )
    checkpoint = _validate_checkpoint(
        model, tokenizer, torch, config, resolved_revision=snapshot.name
    )
    from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance

    with reference_math() as math_settings:
        trace = capture_reference(
            model,
            tokenizer,
            fixture["prompt"],
            fixture["max_new_tokens"],
            fixture["ignore_eos"],
            expected_input_ids=tuple(fixture["input_ids"]),
        )
    metadata: dict[str, Any] = {
        "scope": "HF reference only; no owned-decoder parity or performance claim",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "case": case,
        "seed": 0,
        "checkpoint": checkpoint,
        "model_config": model.config.to_dict(),
        "attention_backend": model.config._attn_implementation,
        "math_settings": math_settings,
        "tokenizer": settings,
        "decoding": {
            "add_special_tokens": True,
            "truncation": False,
            "chat_template": False,
            "do_sample": False,
            "max_new_tokens": fixture["max_new_tokens"],
            "ignore_eos": fixture["ignore_eos"],
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        },
        "environment": collect_environment_diagnostics(config),
        "provenance": collect_validation_provenance(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    }
    metadata["provenance"]["command_arguments"] = command_arguments
    properties = torch.cuda.get_device_properties(0)
    metadata["target"] = {
        "device": "cuda:0",
        "name": properties.name,
        "uuid": str(properties.uuid),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "memory_bytes": properties.total_memory,
    }
    discrepancies: dict[str, float] | None = None
    if compare_path is not None:
        expected, previous = load_reference(compare_path)
        for name in (
            "case",
            "checkpoint",
            "tokenizer",
            "decoding",
            "math_settings",
            "fixture_sha256",
            "attention_backend",
            "model_config",
        ):
            if metadata[name] != previous[name]:
                raise ValueError(f"reference artifact policy mismatch: {name}")
        discrepancies = compare_traces(trace, expected)
        metadata["comparison"] = {
            "scope": "HF reference repeatability only",
            "reference_sha256": hashlib.sha256(compare_path.read_bytes()).hexdigest(),
            "maximum_absolute_errors": discrepancies,
        }
    metadata["tolerances"] = {
        "intermediates": {"atol": 2e-3, "rtol": 2e-3},
        "logits": {"atol": 5e-3, "rtol": 5e-3},
        "greedy_ids": "exact",
    }
    save_reference(output_path, trace, metadata)
    return {
        "status": "passed",
        "scope": metadata["scope"],
        "case": case,
        "output_tokens": len(trace.emitted_ids),
        "finish_reason": trace.finish_reason.value,
        "tensor_count": len(trace.tensors),
        "maximum_absolute_errors": discrepancies,
    }
