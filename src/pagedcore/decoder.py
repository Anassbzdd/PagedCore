"""Single-sequence dense forwards and prompt writes into paged KV storage."""

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
from pagedcore.engine_types import FinishReason, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.weights import TinyLlamaWeights


@dataclass(frozen=True, slots=True)
class ContiguousCache:
    """Per-layer rotated K and unrotated V, each [1, 4, cached tokens, 64].

    Treat tensors as read-only and reuse a cache only with the weights that made
    it. Decode returns a new cache, so a failed forward cannot partially append
    to the caller's history. This full-history storage supports dense validation
    and temporary prompt prefill; paged decode must replace its append and
    attention operations together.
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


@dataclass(frozen=True, slots=True)
class GreedyResult:
    """A stopping EOS is selected but excluded from emitted IDs."""

    greedy_ids: tuple[int, ...]
    emitted_ids: tuple[int, ...]
    finish_reason: FinishReason


@torch.inference_mode()
def generate(
    input_ids: Tensor,
    weights: TinyLlamaWeights,
    *,
    max_new_tokens: int = 128,
    ignore_eos: bool = False,
) -> GreedyResult:
    """Run single-sequence greedy generation with the dense correctness cache."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be an integer from 1 to 256")
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be int64 [1, tokens]")
    if input_ids.shape[1] + max_new_tokens > 2048:
        raise ValueError("prompt tokens plus requested output must not exceed 2048")

    logits, cache = prefill(input_ids, weights)
    selected: list[int] = []
    for step in range(max_new_tokens):
        token = logits[:, -1:].argmax(dim=-1)
        token_id = int(token.item())
        selected.append(token_id)
        # EOS takes precedence even when selected at the output limit.
        if token_id == 2 and not ignore_eos:
            return GreedyResult(tuple(selected), tuple(selected[:-1]), FinishReason.EOS)
        if step + 1 < max_new_tokens:
            logits, cache = decode(token, cache, weights)
    ids = tuple(selected)
    return GreedyResult(ids, ids, FinishReason.LENGTH)


@torch.inference_mode()
def prefill(input_ids: Tensor, weights: TinyLlamaWeights) -> tuple[Tensor, ContiguousCache]:
    """Consume int64 [1, prompt tokens] IDs; return all logits and the prompt cache.

    The final position's logits select the first output token. No output token
    is cached until the caller feeds it to decode.
    """
    return _forward(input_ids, weights, cache=None)


@torch.inference_mode()
def prefill_paged(
    input_ids: Tensor,
    weights: TinyLlamaWeights,
    allocator: KVAllocator,
    request_id: RequestId,
) -> Tensor:
    """Prefill one admitted, empty request; return all prompt logits.

    Temporary dense K/V use absolute prompt positions starting at zero. Only K
    is RoPE-rotated. Publish the complete prompt through the allocator after the
    forward succeeds; the caller retains reservation and cleanup ownership on
    failure. No contiguous cache escapes this call.
    """
    if allocator.cached_length(request_id) != 0:
        raise ValueError("paged prefill requires an empty request cache")
    if len(weights.layers) != 22:
        raise ValueError("paged prefill requires 22 decoder layers")
    logits, cache = prefill(input_ids, weights)
    key = torch.stack([key[0].transpose(0, 1) for key, _ in cache.layers])
    value = torch.stack([value[0].transpose(0, 1) for _, value in cache.layers])
    allocator.append(request_id, key, value)
    return logits


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
