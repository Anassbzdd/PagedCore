from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pagedcore.config import MODEL_REVISION, PagedCoreConfig
from pagedcore.verification import (
    TargetVerificationError,
    _download_pinned_snapshot,
    _validate_checkpoint,
    _validate_forward_logits,
    verify_target,
)


def _valid_model(
    logits: torch.Tensor | None = None,
    *,
    model_class: str = "LlamaForCausalLM",
    parameter_dtype: torch.dtype = torch.float16,
) -> object:
    output_logits = (
        logits if logits is not None else torch.zeros((1, 2, 32000), dtype=torch.float16)
    )
    model_type = type(
        model_class,
        (),
        {
            "parameters": lambda self: iter([torch.zeros(1, dtype=parameter_dtype)]),
            "to": lambda self, device: self,
            "eval": lambda self: self,
            "__call__": lambda self, **kwargs: SimpleNamespace(logits=output_logits),
        },
    )
    model = model_type()
    model.config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=22,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        vocab_size=32000,
        _commit_hash=MODEL_REVISION,
    )
    return model


def _valid_tokenizer() -> object:
    tokenizer_type = type(
        "LlamaTokenizerFast",
        (),
        {
            "__call__": lambda self, *args, **kwargs: {
                "input_ids": torch.zeros((1, 2), dtype=torch.long)
            }
        },
    )
    tokenizer = tokenizer_type()
    tokenizer.vocab_size = 32000
    tokenizer._commit_hash = MODEL_REVISION
    tokenizer.init_kwargs = {"_commit_hash": MODEL_REVISION}
    return tokenizer


def test_checkpoint_validation_records_the_resolved_snapshot_revision() -> None:
    checkpoint = _validate_checkpoint(
        _valid_model(),
        _valid_tokenizer(),
        torch,
        PagedCoreConfig(),
        resolved_revision=MODEL_REVISION,
    )

    assert checkpoint["resolved_revision"] == MODEL_REVISION
    assert checkpoint["tokenizer"]["resolved_revision"] == MODEL_REVISION


def test_snapshot_download_uses_the_pinned_model_and_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    snapshot_download = Mock(return_value=str(snapshot))

    result = _download_pinned_snapshot(
        SimpleNamespace(snapshot_download=snapshot_download), PagedCoreConfig()
    )

    assert result == snapshot
    snapshot_download.assert_called_once_with(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision=MODEL_REVISION,
    )


def test_snapshot_download_rejects_a_different_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / ("0" * 40)
    snapshot.mkdir()

    with pytest.raises(TargetVerificationError, match="pinned revision"):
        _download_pinned_snapshot(
            SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)),
            PagedCoreConfig(),
        )


def test_checkpoint_validation_rejects_a_different_snapshot_revision() -> None:
    with pytest.raises(TargetVerificationError, match="snapshot_revision"):
        _validate_checkpoint(
            _valid_model(),
            _valid_tokenizer(),
            torch,
            PagedCoreConfig(),
            resolved_revision="0" * 40,
        )


def test_checkpoint_validation_rejects_a_tokenizer_with_the_wrong_vocabulary() -> None:
    tokenizer = _valid_tokenizer()
    tokenizer.vocab_size = 1

    with pytest.raises(TargetVerificationError, match="tokenizer_vocab_size"):
        _validate_checkpoint(
            _valid_model(),
            tokenizer,
            torch,
            PagedCoreConfig(),
            resolved_revision=MODEL_REVISION,
        )


@pytest.mark.parametrize(
    ("model_class", "parameter_dtype", "expected_error"),
    [
        ("GPT2LMHeadModel", torch.float16, "model_class"),
        ("LlamaForCausalLM", torch.float32, "parameter_dtype"),
    ],
)
def test_checkpoint_validation_rejects_architecture_or_dtype_mismatches(
    model_class: str,
    parameter_dtype: torch.dtype,
    expected_error: str,
) -> None:
    with pytest.raises(TargetVerificationError, match=expected_error):
        _validate_checkpoint(
            _valid_model(model_class=model_class, parameter_dtype=parameter_dtype),
            _valid_tokenizer(),
            torch,
            PagedCoreConfig(),
            resolved_revision=MODEL_REVISION,
        )


def test_checkpoint_validation_rejects_a_tokenizer_from_another_revision() -> None:
    tokenizer = _valid_tokenizer()
    tokenizer._commit_hash = "0" * 40

    with pytest.raises(TargetVerificationError, match="tokenizer_revision"):
        _validate_checkpoint(
            _valid_model(),
            tokenizer,
            torch,
            PagedCoreConfig(),
            resolved_revision=MODEL_REVISION,
        )


def test_verify_target_rejects_a_machine_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from pagedcore import verification

    runtime = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(verification, "_load_runtime", lambda: (runtime, None, None))

    with pytest.raises(TargetVerificationError, match="available CUDA device"):
        verify_target(PagedCoreConfig())


def test_verify_target_rejects_a_non_t4_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from pagedcore import verification

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "NVIDIA A100",
    )
    runtime = SimpleNamespace(cuda=cuda, device=lambda name: name)
    monkeypatch.setattr(verification, "_load_runtime", lambda: (runtime, None, None))

    with pytest.raises(TargetVerificationError, match="requires an NVIDIA T4"):
        verify_target(PagedCoreConfig())


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros((1, 2, 32000), dtype=torch.float32),
        torch.zeros((1, 1, 32000), dtype=torch.float16),
        torch.full((1, 2, 32000), float("nan"), dtype=torch.float16),
    ],
    ids=["wrong-dtype", "wrong-shape", "non-finite"],
)
def test_forward_validation_rejects_bad_logits(logits: torch.Tensor) -> None:
    input_ids = torch.zeros((1, 2), dtype=torch.long)
    model_config = SimpleNamespace(vocab_size=32000)

    with pytest.raises(TargetVerificationError):
        _validate_forward_logits(logits, input_ids, model_config, torch)


def test_forward_validation_accepts_finite_fp16_logits_with_expected_shape() -> None:
    logits = torch.zeros((1, 2, 32000), dtype=torch.float16)
    input_ids = torch.zeros((1, 2), dtype=torch.long)

    _validate_forward_logits(logits, input_ids, SimpleNamespace(vocab_size=32000), torch)


def test_verify_target_loads_both_files_from_the_pinned_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pagedcore import cli, verification

    snapshot_path = tmp_path / MODEL_REVISION
    snapshot_path.mkdir()
    tokenizer = _valid_tokenizer()
    model = _valid_model()

    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Tesla T4",
        synchronize=lambda device: None,
        get_device_properties=lambda index: SimpleNamespace(
            major=7, minor=5, total_memory=15 * 1024**3
        ),
    )
    runtime = SimpleNamespace(
        cuda=cuda,
        device=lambda name: torch.device("cpu"),
        float16=torch.float16,
        Tensor=torch.Tensor,
        isfinite=torch.isfinite,
    )
    hub = SimpleNamespace(snapshot_download=Mock(return_value=str(snapshot_path)))
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
    )

    monkeypatch.setattr(verification, "_load_runtime", lambda: (runtime, transformers, hub))
    monkeypatch.setattr(cli, "collect_environment_diagnostics", lambda config: {})
    monkeypatch.setattr(
        cli,
        "collect_validation_provenance",
        lambda: {"git_revision": "test-commit", "uv_lock_sha256": "test-lock-hash"},
    )

    manifest = verify_target(PagedCoreConfig())

    assert manifest["manifest_version"] == 2
    assert hub.snapshot_download.call_args.kwargs == {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": MODEL_REVISION,
    }
    tokenizer_loader.assert_called_once_with(
        str(snapshot_path), use_fast=True, local_files_only=True
    )
    model_loader.assert_called_once_with(
        str(snapshot_path), torch_dtype=torch.float16, local_files_only=True
    )
    assert manifest["checkpoint"]["tokenizer"]["resolved_revision"] == MODEL_REVISION
    assert manifest["provenance"]["git_revision"] == "test-commit"
    assert manifest["forward_pass"]["logits_shape"] == [1, 2, 32000]
