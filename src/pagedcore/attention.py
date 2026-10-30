"""Block-table attention over borrowed KV pages, without full-history copies."""

from __future__ import annotations

import torch
from torch import Tensor

from pagedcore.engine_types import RequestId
from pagedcore.kv_cache import KVAllocator


@torch.inference_mode()
def paged_attention(
    query: Tensor, allocator: KVAllocator, request_id: RequestId, *, layer_index: int
) -> Tensor:
    if query.shape != (1, 32, 1, 64) or query.dtype != torch.float16:
        raise ValueError("query must be FP16 [1, 32, 1, 64]")
    if type(layer_index) is not int or not 0 <= layer_index < 22:
        raise ValueError("layer_index must identify one of the 22 decoder layers")
    blocks = allocator.block_table(request_id)
    if not blocks:
        raise ValueError("paged attention requires a nonempty request cache")
    key, value = allocator.read_block(request_id, 0)
    if key.device != query.device:
        raise ValueError("query and KV cache must be on the same device")

    # Autocast must not round the FP32 score or weighted-value matmuls to FP16.
    with torch.autocast(device_type=query.device.type, enabled=False):
        grouped_query = query.reshape(4, 8, 64).float()
        maximum = torch.full((4, 8, 1), -torch.inf, dtype=torch.float32, device=query.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(grouped_query)
        for logical_block in range(len(blocks)):
            if logical_block:
                key, value = allocator.read_block(request_id, logical_block)
            scores = torch.matmul(grouped_query, key[layer_index].permute(1, 2, 0).float())
            scores = scores * (64**-0.5)
            next_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            rescale = torch.exp(maximum - next_maximum)
            probabilities = torch.exp(scores - next_maximum)
            denominator = rescale * denominator + probabilities.sum(dim=-1, keepdim=True)
            numerator = rescale * numerator + torch.matmul(
                probabilities, value[layer_index].transpose(0, 1).float()
            )
            maximum = next_maximum
        return (numerator / denominator).reshape(1, 32, 1, 64).half()
