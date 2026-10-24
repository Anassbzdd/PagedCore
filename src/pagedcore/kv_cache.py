"""Shared FP16 storage for the pinned TinyLlama KV cache."""

from __future__ import annotations

import torch
from torch import Tensor

from pagedcore.config import PagedCoreConfig


def allocate_kv_pool(config: PagedCoreConfig, *, device: str | torch.device = "cuda:0") -> Tensor:
    """Allocate once after model loading; retain the tensor for the worker's lifetime.

    Axes are [layers, K/V, blocks, token slots, KV heads, head dimension].
    Storage is uninitialized: only written, valid slots may be read. Blocks
    are reused within this tensor rather than allocated as tokens arrive.
    """
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("KV pool requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("KV pool requires an available CUDA runtime")

    pool_bytes = config.kv_pool_mib * 2**20
    margin_bytes = config.workspace_margin_mib * 2**20
    block_bytes = 22 * 2 * config.block_tokens * 4 * 64 * 2
    num_blocks = pool_bytes // block_bytes
    free_bytes, _ = torch.cuda.mem_get_info(target)
    if free_bytes < pool_bytes + margin_bytes:
        raise RuntimeError("configured KV pool and workspace margin do not fit in GPU memory")

    # Round down to whole blocks; never exceed the configured byte budget.
    pool = torch.empty(
        (22, 2, num_blocks, config.block_tokens, 4, 64),
        dtype=torch.float16,
        device=target,
    )
    if torch.cuda.mem_get_info(target)[0] < margin_bytes:
        del pool
        raise RuntimeError("KV pool allocation leaves less than the configured workspace margin")
    return pool
