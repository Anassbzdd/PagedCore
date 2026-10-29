"""Stateless TinyLlama math; cache ownership and generation live outside this module."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from pagedcore.weights import LayerWeights, TinyLlamaWeights

_QUERY_HEADS = 32
_KV_HEADS = 4
_HEAD_DIM = 64
_KV_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS


def rms_norm(hidden: Tensor, weight: Tensor) -> Tensor:
    """Normalize in FP32, then round before multiplying the FP16 scale, like Llama."""
    values = hidden.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
    return weight * normalized.to(hidden.dtype)


def project_qkv(hidden: Tensor, weights: LayerWeights) -> tuple[Tensor, Tensor, Tensor]:
    """Map [batch, tokens, hidden] to [batch, heads, tokens, 64], without rotation."""
    batch, tokens, _ = hidden.shape
    query = F.linear(hidden, weights.q_proj).view(batch, tokens, _QUERY_HEADS, _HEAD_DIM)
    key = F.linear(hidden, weights.k_proj).view(batch, tokens, _KV_HEADS, _HEAD_DIM)
    value = F.linear(hidden, weights.v_proj).view(batch, tokens, _KV_HEADS, _HEAD_DIM)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def apply_rope(query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate Q/K at absolute [batch, tokens] positions; K is rotated before caching.

    Positions must be valid context positions on the same device as Q/K. A single
    row can supply shared positions for the batch. Values are never rotated.
    """
    if (
        position_ids.dtype != torch.int64
        or position_ids.ndim != 2
        or position_ids.shape[0] not in (1, query.shape[0])
        or position_ids.shape[1] != query.shape[2]
    ):
        raise ValueError("position_ids must be int64 [batch or 1, query tokens]")
    # Keep frequencies and trigonometry in FP32 even inside an autocast context.
    with torch.autocast(device_type=query.device.type, enabled=False):
        inverse_frequencies = 1.0 / (
            10000.0 ** (torch.arange(0, _HEAD_DIM, 2, device=query.device).float() / _HEAD_DIM)
        )
        angles = position_ids.float().unsqueeze(-1) * inverse_frequencies
        angles = torch.cat((angles, angles), dim=-1)
        cosine = angles.cos().to(query.dtype).unsqueeze(1)
        sine = angles.sin().to(query.dtype).unsqueeze(1)

    def rotate(tensor: Tensor) -> Tensor:
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return tensor * cosine + rotated * sine

    return rotate(query), rotate(key)


def repeat_kv(tensor: Tensor) -> Tensor:
    """Expand each KV head to eight consecutive query heads, for dense attention only."""
    return tensor.repeat_interleave(_KV_GROUP_SIZE, dim=1)


def dense_attention(query: Tensor, key: Tensor, value: Tensor, causal_mask: Tensor) -> Tensor:
    """Attend over dense K/V with a boolean [batch, 1, query, key] allowed mask.

    Every query must have at least one allowed key. The caller supplies causality
    from absolute positions; a single decode query must not use a top-left mask.
    Full-history expansion is allowed for prompt prefill and the dense baseline.
    """
    if causal_mask.dtype != torch.bool:
        raise ValueError("causal_mask must be boolean with True for allowed keys")
    scores = torch.matmul(query, repeat_kv(key).transpose(-2, -1)) * (_HEAD_DIM**-0.5)
    scores = scores.masked_fill(~causal_mask, torch.finfo(query.dtype).min)
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, repeat_kv(value))


def output_projection(attention: Tensor, weights: LayerWeights) -> Tensor:
    """Merge [batch, heads, tokens, 64] in head order before the output projection."""
    batch, _, tokens, _ = attention.shape
    merged = attention.transpose(1, 2).reshape(batch, tokens, _QUERY_HEADS * _HEAD_DIM)
    return F.linear(merged, weights.o_proj)


def swiglu(hidden: Tensor, weights: LayerWeights) -> Tensor:
    gate = F.silu(F.linear(hidden, weights.gate_proj))
    return F.linear(gate * F.linear(hidden, weights.up_proj), weights.down_proj)


def final_logits(hidden: Tensor, weights: TinyLlamaWeights) -> Tensor:
    """Apply the final RMSNorm and untied LM head, retaining the runtime dtype."""
    return F.linear(rms_norm(hidden, weights.final_norm), weights.lm_head)
