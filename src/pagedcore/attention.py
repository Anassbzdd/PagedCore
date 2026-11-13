"""Block-table attention over borrowed KV pages, without full-history copies."""

from __future__ import annotations

from itertools import chain

import torch
from torch import Tensor

from pagedcore.engine_types import RequestId
from pagedcore.kv_cache import KVAllocator


@torch.inference_mode()
def paged_attention(
    query: Tensor,
    allocator: KVAllocator,
    request_id: RequestId,
    *,
    layer_index: int,
    eager_rounding: bool = False,
) -> Tensor:
    if query.shape != (1, 32, 1, 64) or query.dtype != torch.float16:
        raise ValueError("query must be FP16 [1, 32, 1, 64]")
    if type(layer_index) is not int or not 0 <= layer_index < 22:
        raise ValueError("layer_index must identify one of the 22 decoder layers")
    blocks = allocator.iter_layer_blocks(request_id, layer_index=layer_index)
    first = next(blocks)
    if first[0].device != query.device:
        raise ValueError("query and KV cache must be on the same device")
    # Autocast must not round the FP32 score or weighted-value matmuls to FP16.
    with torch.autocast(device_type=query.device.type, enabled=False):
        grouped_query = query.reshape(4, 8, 64).float()
        if eager_rounding:
            return _eager_attention(grouped_query, allocator, request_id, layer_index)
        # Only scores span the bounded context; K/V conversions stay page-sized.
        scores = torch.empty((4, 8, 2048), dtype=torch.float32, device=query.device)
        length = 0
        for key, _ in chain((first,), blocks):
            stop = length + key.shape[0]
            scores[..., length:stop] = torch.matmul(grouped_query, key.permute(1, 2, 0).float()) * (
                64**-0.5
            )
            length = stop
        # Normalize globally before multiplying V; never round page outputs to FP16.
        probabilities = scores[..., :length].softmax(dim=-1)
        output = torch.zeros_like(grouped_query)
        start = 0
        for _, value in allocator.iter_layer_blocks(request_id, layer_index=layer_index):
            stop = start + value.shape[0]
            output.add_(torch.matmul(probabilities[..., start:stop], value.transpose(0, 1).float()))
            start = stop
        return output.reshape(1, 32, 1, 64).half()


def _eager_attention(
    query: Tensor, allocator: KVAllocator, request_id: RequestId, layer_index: int
) -> Tensor:
    # Scores may span the bounded context; K/V storage must stay page-sized.
    scores = torch.empty((4, 8, 2048), dtype=torch.float32, device=query.device)
    heads = query.half().unsqueeze(2)
    length = 0
    for key, _ in allocator.iter_layer_blocks(request_id, layer_index=layer_index):
        stop = length + key.shape[0]
        # Native FP16 products matter: FP32 matmul followed by half() can
        # select a different reduction and cross an FP16 rounding boundary.
        if length and key.shape[0] < 16:
            # A short final page must not select the tiny-history reduction.
            page = torch.zeros((4, 16, 64), dtype=torch.float16, device=query.device)
            page[:, : key.shape[0]].copy_(key.transpose(0, 1))
        else:
            page = key.transpose(0, 1).contiguous()
        for head in range(4):
            products = torch.bmm(heads[head], page[head].T[None].expand(8, -1, -1))
            scores[head, :, length:stop] = (products[:, 0, : key.shape[0]] * (64**-0.5)).float()
        length = stop
    # Normalize once, before rounding probabilities. Page-local rounding or an
    # online weighted sum changes eager's FP16 probability boundaries.
    probabilities = scores[..., :length].softmax(dim=-1).half()
    return _eager_value_sum(probabilities, allocator, request_id, layer_index)


def _eager_value_sum(
    probabilities: Tensor, allocator: KVAllocator, request_id: RequestId, layer_index: int
) -> Tensor:
    length = probabilities.shape[-1]
    if length <= 16:
        _, value = next(allocator.iter_layer_blocks(request_id, layer_index=layer_index))
        page = value.transpose(0, 1).contiguous()
        output = torch.empty((4, 8, 64), dtype=torch.float16, device=probabilities.device)
        for head in range(4):
            output[head] = torch.bmm(
                probabilities[head, :, None], page[head][None].expand(8, -1, -1)
            )[:, 0]
        return output.reshape(1, 32, 1, 64)
    return _multi_page_value_sum(probabilities, allocator, request_id, layer_index)


def _multi_page_value_sum(
    probabilities: Tensor, allocator: KVAllocator, request_id: RequestId, layer_index: int
) -> Tensor:
    return _striped_value_sum(probabilities, allocator, request_id, layer_index, lanes=4)


def _striped_value_sum(
    probabilities: Tensor,
    allocator: KVAllocator,
    request_id: RequestId,
    layer_index: int,
    *,
    lanes: int,
) -> Tensor:
    length = probabilities.shape[-1]
    # Carry reduction lanes across page boundaries instead of summing rounded
    # page products. Keep separate storages so each temporary fits one KV page.
    partials = tuple(
        torch.zeros((4, 8, 64), dtype=torch.float32, device=probabilities.device)
        for _ in range(lanes)
    )
    tail = length - length % lanes
    start = 0
    for _, value in allocator.iter_layer_blocks(request_id, layer_index=layer_index):
        stop = start + value.shape[0]
        for slot in range(value.shape[0]):
            position = start + slot
            lane = position % lanes if position < tail else 0
            partials[lane].add_(
                torch.matmul(
                    probabilities[..., position, None].float(), value[slot, :, None, :].float()
                )
            )
        start = stop
    output = partials[0]
    for partial in partials[1:]:
        output = output + partial
    return output.reshape(1, 32, 1, 64).half()


def _paged_value_sum(
    probabilities: Tensor, allocator: KVAllocator, request_id: RequestId, layer_index: int
) -> Tensor:
    """Experimental page-local products with FP32 accumulation until the final cast."""
    output = torch.zeros((4, 8, 64), dtype=torch.float32, device=probabilities.device)
    start = 0
    for _, value in allocator.iter_layer_blocks(request_id, layer_index=layer_index):
        stop = start + value.shape[0]
        page = value.transpose(0, 1).float()
        weights = probabilities[..., start:stop].float()
        for head in range(4):
            output[head] = torch.baddbmm(
                output[head, :, None], weights[head, :, None], page[head][None].expand(8, -1, -1)
            )[:, 0]
        start = stop
    return output.reshape(1, 32, 1, 64).half()


def _paired_value_sum(
    probabilities: Tensor, allocator: KVAllocator, request_id: RequestId, layer_index: int
) -> Tensor:
    return _striped_value_sum(probabilities, allocator, request_id, layer_index, lanes=2)
