"""Pinned TinyLlama weights for the owned decoder, without an HF model instance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import SafetensorError, safe_open
from torch import Tensor

from pagedcore.config import MODEL_ID, MODEL_REVISION

_EXPECTED_CONFIG: dict[str, object] = {
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
# These fields are absent in the pinned JSON; freeze their Llama defaults too.
_CONFIG_DEFAULTS: dict[str, object] = {
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "head_dim": 64,
    "pad_token_id": None,
}


class WeightLoadingError(RuntimeError):
    """The checkpoint cannot satisfy the owned decoder's weight contract."""


@dataclass(frozen=True, slots=True)
class LayerWeights:
    input_norm: Tensor
    q_proj: Tensor
    k_proj: Tensor
    v_proj: Tensor
    o_proj: Tensor
    post_attention_norm: Tensor
    gate_proj: Tensor
    up_proj: Tensor
    down_proj: Tensor


@dataclass(frozen=True, slots=True)
class TinyLlamaWeights:
    embedding: Tensor
    layers: tuple[LayerWeights, ...]
    final_norm: Tensor
    lm_head: Tensor


def _validate_config(config: object) -> None:
    if not isinstance(config, dict):
        raise WeightLoadingError("checkpoint config must be a JSON object")
    for name, expected in (_EXPECTED_CONFIG | _CONFIG_DEFAULTS).items():
        actual = config.get(name, _CONFIG_DEFAULTS.get(name))
        if name not in config and name not in _CONFIG_DEFAULTS:
            raise WeightLoadingError(f"checkpoint config is missing {name}")
        if type(actual) is not type(expected) or actual != expected:
            raise WeightLoadingError(
                f"checkpoint config {name}: expected {expected!r}, found {actual!r}"
            )


def _tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (32000, 2048),
        "model.norm.weight": (2048,),
        "lm_head.weight": (32000, 2048),
    }
    layer_shapes = {
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
    for index in range(22):
        for name, shape in layer_shapes.items():
            shapes[f"model.layers.{index}.{name}.weight"] = shape
    return shapes


def _map_weights(tensors: dict[str, Tensor]) -> TinyLlamaWeights:
    embedding = tensors["model.embed_tokens.weight"]
    lm_head = tensors["lm_head.weight"]
    if embedding.untyped_storage().data_ptr() == lm_head.untyped_storage().data_ptr():
        raise WeightLoadingError("checkpoint embedding and LM head must be untied")
    layers = []
    for index in range(22):
        prefix = f"model.layers.{index}."
        layers.append(
            LayerWeights(
                input_norm=tensors[prefix + "input_layernorm.weight"],
                q_proj=tensors[prefix + "self_attn.q_proj.weight"],
                k_proj=tensors[prefix + "self_attn.k_proj.weight"],
                v_proj=tensors[prefix + "self_attn.v_proj.weight"],
                o_proj=tensors[prefix + "self_attn.o_proj.weight"],
                post_attention_norm=tensors[prefix + "post_attention_layernorm.weight"],
                gate_proj=tensors[prefix + "mlp.gate_proj.weight"],
                up_proj=tensors[prefix + "mlp.up_proj.weight"],
                down_proj=tensors[prefix + "mlp.down_proj.weight"],
            )
        )
    return TinyLlamaWeights(
        embedding=embedding,
        layers=tuple(layers),
        final_norm=tensors["model.norm.weight"],
        lm_head=lm_head,
    )


def load_weights(
    *, device: str | torch.device = "cuda:0", local_files_only: bool = False
) -> TinyLlamaWeights:
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=["config.json", "model.safetensors"],
                local_files_only=local_files_only,
            )
        )
        if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
            raise WeightLoadingError("checkpoint snapshot does not match the pinned revision")
        _validate_config(json.loads((snapshot / "config.json").read_text(encoding="utf-8")))
        shapes = _tensor_shapes()
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            names = set(checkpoint.keys())
            if names != shapes.keys():
                raise WeightLoadingError(
                    f"checkpoint tensor names: missing {sorted(shapes.keys() - names)}, "
                    f"unexpected {sorted(names - shapes.keys())}"
                )
            for name, shape in shapes.items():
                metadata = checkpoint.get_slice(name)
                if tuple(metadata.get_shape()) != shape:
                    raise WeightLoadingError(f"checkpoint tensor {name}: expected shape {shape}")
                if metadata.get_dtype() != "BF16":
                    raise WeightLoadingError(
                        f"checkpoint tensor {name}: expected storage dtype BF16"
                    )
            tensors = {}
            for name in shapes:
                tensor = checkpoint.get_tensor(name).to(device=device, dtype=torch.float16)
                if not bool(torch.isfinite(tensor).all().item()):
                    raise WeightLoadingError(f"checkpoint tensor {name}: non-finite FP16 weights")
                tensors[name] = tensor
        return _map_weights(tensors)
    except WeightLoadingError:
        raise
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        raise WeightLoadingError(f"TinyLlama weight loading failed: {error}") from error
