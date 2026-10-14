import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from pagedcore import weights
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.weights import WeightLoadingError, load_weights

pytestmark = pytest.mark.unit

# Independent checkpoint schema: projections are [output features, input features].
LAYER_SHAPES = {
    "input_layernorm": (2048,),
    "self_attn.q_proj": (2048, 2048),
    "self_attn.k_proj": (256, 2048),
    "self_attn.v_proj": (256, 2048),
    "self_attn.o_proj": (2048, 2048),
    "post_attention_layernorm": (2048,),
    "mlp.gate_proj": (5632, 2048),
    "mlp.up_proj": (5632, 2048),
    "mlp.down_proj": (2048, 5632),
}
PINNED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 22,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "max_position_embeddings": 2048,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "pretraining_tp": 1,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "bfloat16",
    "use_cache": True,
}


def test_expected_shapes_match_the_complete_tinyllama_schema() -> None:
    expected = {
        "model.embed_tokens.weight": (32000, 2048),
        "model.norm.weight": (2048,),
        "lm_head.weight": (32000, 2048),
    }
    for index in range(22):
        for suffix, shape in LAYER_SHAPES.items():
            expected[f"model.layers.{index}.{suffix}.weight"] = shape
    assert weights._tensor_shapes() == expected
    assert len(expected) == 201


@pytest.mark.parametrize("name", PINNED_CONFIG)
def test_config_rejects_each_missing_required_value(name: str) -> None:
    config = PINNED_CONFIG.copy()
    del config[name]
    with pytest.raises(WeightLoadingError, match=name):
        weights._validate_config(config)


@pytest.mark.parametrize("name", PINNED_CONFIG)
def test_config_rejects_each_changed_required_value(name: str) -> None:
    config = PINNED_CONFIG | {name: "mismatch"}
    with pytest.raises(WeightLoadingError, match=name):
        weights._validate_config(config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mlp_bias", True),
        ("attention_dropout", 0.1),
        ("head_dim", 128),
        ("pad_token_id", 0),
        ("bos_token_id", True),
        ("pretraining_tp", 1.0),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("tie_word_embeddings", True),
    ],
)
def test_config_rejects_unsupported_options_and_invalid_types(name: str, value: object) -> None:
    with pytest.raises(WeightLoadingError, match=name):
        weights._validate_config(PINNED_CONFIG | {name: value})


def test_config_accepts_pinned_values_and_explicit_defaults() -> None:
    weights._validate_config(PINNED_CONFIG)
    weights._validate_config(
        PINNED_CONFIG
        | {"mlp_bias": False, "attention_dropout": 0.0, "head_dim": 64, "pad_token_id": None}
        | {"initializer_range": 0.02, "transformers_version": "4.35.0"}
    )


@pytest.mark.parametrize("config", [None, [], "llama"])
def test_config_requires_an_object(config: object) -> None:
    with pytest.raises(WeightLoadingError, match="JSON object"):
        weights._validate_config(config)


@pytest.fixture
def checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, torch.Tensor]]:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(PINNED_CONFIG), encoding="utf-8")
    monkeypatch.setattr(weights, "snapshot_download", Mock(return_value=str(snapshot)))
    # Exercise real safetensors I/O and conversion without a 2 GiB test allocation.
    shapes = {
        name: (2,) if len(shape) == 1 else (2, 2)
        for name, shape in weights._tensor_shapes().items()
    }
    monkeypatch.setattr(weights, "_tensor_shapes", lambda: shapes)
    tensors = {
        name: (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) + index * 4)
        .to(torch.bfloat16)
        .div(256)
        for index, (name, shape) in enumerate(shapes.items())
    }
    save_file(tensors, snapshot / "model.safetensors")
    return snapshot, tensors


def test_load_maps_every_tensor_without_transposing_or_tying(
    checkpoint: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    _, tensors = checkpoint
    result = load_weights(device="cpu", local_files_only=True)
    weights.snapshot_download.assert_called_once_with(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["config.json", "model.safetensors"],
        local_files_only=True,
    )
    mapped = {
        "model.embed_tokens.weight": result.embedding,
        "model.norm.weight": result.final_norm,
        "lm_head.weight": result.lm_head,
    }
    assert len(result.layers) == 22
    for index, layer in enumerate(result.layers):
        for field, suffix in zip(fields(layer), LAYER_SHAPES, strict=True):
            mapped[f"model.layers.{index}.{suffix}.weight"] = getattr(layer, field.name)
    assert mapped.keys() == tensors.keys()
    for name, tensor in mapped.items():
        assert tensor.dtype == torch.float16
        assert tensor.device == torch.device("cpu")
        assert not tensor.requires_grad
        torch.testing.assert_close(tensor, tensors[name].half(), rtol=0, atol=0)
    assert (
        result.embedding.untyped_storage().data_ptr() != result.lm_head.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype"])
def test_bad_metadata_fails_before_tensor_loading(
    checkpoint: tuple[Path, dict[str, torch.Tensor]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    snapshot, tensors = checkpoint
    name = "model.layers.21.self_attn.k_proj.weight"
    if change == "missing":
        del tensors[name]
    elif change == "extra":
        tensors["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(2, dtype=torch.bfloat16)
    elif change == "shape":
        tensors[name] = torch.zeros((3, 2), dtype=torch.bfloat16)
    else:
        tensors[name] = tensors[name].float()
    save_file(tensors, snapshot / "model.safetensors")
    transfer = Mock(side_effect=AssertionError("weights transferred before metadata validation"))
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    with pytest.raises(WeightLoadingError, match="tensor"):
        load_weights(device="cpu")
    transfer.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e10])
def test_load_rejects_nonfinite_values_and_fp16_overflow(
    checkpoint: tuple[Path, dict[str, torch.Tensor]], value: float
) -> None:
    snapshot, tensors = checkpoint
    tensors["lm_head.weight"][0, 0] = value
    save_file(tensors, snapshot / "model.safetensors")
    with pytest.raises(WeightLoadingError, match=r"lm_head\.weight: non-finite FP16"):
        load_weights(device="cpu")


def test_mapping_rejects_tied_embedding_and_head(
    checkpoint: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    _, tensors = checkpoint
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    with pytest.raises(WeightLoadingError, match="must be untied"):
        weights._map_weights(tensors)


def test_load_rejects_mismatched_revision_before_reading_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / ("0" * 40)
    snapshot.mkdir()
    monkeypatch.setattr(weights, "snapshot_download", lambda **kwargs: str(snapshot))
    with pytest.raises(WeightLoadingError, match="pinned revision"):
        load_weights(device="cpu")


@pytest.mark.parametrize(
    "failure",
    ["missing-config", "invalid-json", "bad-config", "missing-weights", "corrupt-weights"],
)
def test_load_reports_invalid_snapshot_files(
    checkpoint: tuple[Path, dict[str, torch.Tensor]], failure: str
) -> None:
    snapshot, _ = checkpoint
    config = snapshot / "config.json"
    tensor_file = snapshot / "model.safetensors"
    if failure == "missing-config":
        config.unlink()
    elif failure == "invalid-json":
        config.write_text("{", encoding="utf-8")
    elif failure == "bad-config":
        config.write_text(json.dumps(PINNED_CONFIG | {"hidden_size": 1}), encoding="utf-8")
    elif failure == "missing-weights":
        tensor_file.unlink()
    else:
        tensor_file.write_bytes(b"invalid")
    with pytest.raises(WeightLoadingError):
        load_weights(device="cpu")


def test_load_wraps_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weights, "snapshot_download", Mock(side_effect=OSError("offline")))
    with pytest.raises(WeightLoadingError, match="weight loading failed: offline"):
        load_weights(device="cpu")
