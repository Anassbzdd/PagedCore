"""Single-sequence dense forwards for correctness validation before paged storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from pagedcore.dense import (
    apply_rope,
    dense_attention,
    final_logits,
    output_projection,
    project_qkv,
    rms_norm,
    swiglu,
)
from pagedcore.weights import TinyLlamaWeights


@dataclass(frozen=True, slots=True)
class ContiguousCache:
    """Per-layer rotated K and unrotated V, each [1, 4, cached tokens, 64].

    Treat tensors as read-only and reuse a cache only with the weights that made
    it. Decode returns a new cache, so a failed forward cannot partially append
    to the caller's history. This full-history storage is for dense validation;
    paged decode must replace its append and attention operations together.
    """

    layers: tuple[tuple[Tensor, Tensor], ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("cache must contain at least one layer")
        first = self.layers[0][0]
        if first.ndim != 4 or not 1 <= first.shape[2] <= 2048:
            raise ValueError("cache must contain between 1 and 2048 token positions")
        shape = (1, 4, first.shape[2], 64)
        for key, value in self.layers:
            for tensor in (key, value):
                if (
                    tensor.shape != shape
                    or tensor.dtype != torch.float16
                    or tensor.device != first.device
                ):
                    raise ValueError("cache layers must share FP16 [1, 4, tokens, 64] storage")

    @property
    def length(self) -> int:
        return int(self.layers[0][0].shape[2])


@torch.inference_mode()
def prefill(input_ids: Tensor, weights: TinyLlamaWeights) -> tuple[Tensor, ContiguousCache]:
    """Consume int64 [1, prompt tokens] IDs; return all logits and the prompt cache.

    The final position's logits select the first output token. No output token
    is cached until the caller feeds it to decode.
    """
    return _forward(input_ids, weights, cache=None)


@torch.inference_mode()
def decode(
    input_ids: Tensor, cache: ContiguousCache, weights: TinyLlamaWeights
) -> tuple[Tensor, ContiguousCache]:
    """Consume the last emitted token as int64 [1, 1] at position cache.length."""
    if input_ids.shape != (1, 1):
        raise ValueError("decode requires exactly one token with shape [1, 1]")
    return _forward(input_ids, weights, cache)


def _forward(
    input_ids: Tensor, weights: TinyLlamaWeights, cache: ContiguousCache | None
) -> tuple[Tensor, ContiguousCache]:
    if input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be int64 [1, tokens]")
    if input_ids.device != weights.embedding.device:
        raise ValueError("input_ids and weights must be on the same device")
    offset = 0 if cache is None else cache.length
    tokens = input_ids.shape[1]
    if tokens < 1 or offset + tokens > 2048:
        raise ValueError("forward requires 1 to 2048 total cached token positions")
    if cache is not None:
        if len(cache.layers) != len(weights.layers):
            raise ValueError("cache layer count must match the weights")
        if cache.layers[0][0].device != input_ids.device:
            raise ValueError("cache and input_ids must be on the same device")

    positions = torch.arange(offset, offset + tokens, device=input_ids.device)[None]
    key_positions = torch.arange(offset + tokens, device=input_ids.device)
    causal_mask = (key_positions[None, :] <= positions[0, :, None])[None, None]
    hidden = F.embedding(input_ids, weights.embedding)
    layers = []
    for index, layer in enumerate(weights.layers):
        query, key, value = project_qkv(rms_norm(hidden, layer.input_norm), layer)
        query, key = apply_rope(query, key, positions)
        if cache is not None:
            previous_key, previous_value = cache.layers[index]
            key = torch.cat((previous_key, key), dim=2)
            value = torch.cat((previous_value, value), dim=2)
        layers.append((key, value))
        attention = dense_attention(query, key, value, causal_mask)
        hidden = hidden + output_projection(attention, layer)
        hidden = hidden + swiglu(rms_norm(hidden, layer.post_attention_norm), layer)
    return final_logits(hidden, weights), ContiguousCache(tuple(layers))
