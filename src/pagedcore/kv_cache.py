"""Shared FP16 storage for the pinned TinyLlama KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from threading import current_thread

import torch
from torch import Tensor

from pagedcore.config import PagedCoreConfig
from pagedcore.engine_types import BlockId, BlockTable, RequestId


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


@dataclass(frozen=True, slots=True)
class _SequenceCache:
    blocks: BlockTable = ()
    length: int = 0


class KVAllocator:
    """Assign blocks within an existing pool; construct and use on the model worker.

    CUDA operations must stay on the stream current at construction. Block reads
    borrow pool storage: treat them as read-only and do not retain them after
    appending or releasing the request. CPU storage supports allocator tests.
    """

    def __init__(self, pool: Tensor) -> None:
        if (
            pool.ndim != 6
            or pool.shape[:2] != (22, 2)
            or pool.shape[2] < 1
            or pool.shape[3:] != (16, 4, 64)
            or pool.dtype != torch.float16
            or pool.device.type not in ("cpu", "cuda")
            or not pool.is_contiguous()
            or pool.requires_grad
        ):
            raise ValueError("KV pool must be contiguous FP16 [22, 2, blocks, 16, 4, 64] storage")
        self._pool = pool
        self._worker = current_thread()
        self._stream = torch.cuda.current_stream(pool.device) if pool.is_cuda else None
        self._free_blocks = [BlockId(index) for index in reversed(range(pool.shape[2]))]
        self._sequences: dict[RequestId, _SequenceCache] = {}

    def create(self, request_id: RequestId) -> None:
        """Register an admitted request before its first cache write."""
        self._check_worker()
        if request_id in self._sequences:
            raise ValueError("request already has a KV cache")
        self._sequences[request_id] = _SequenceCache()
        self._assert_ownership()

    def block_table(self, request_id: RequestId) -> BlockTable:
        self._check_worker()
        return self._sequences[request_id].blocks

    def cached_length(self, request_id: RequestId) -> int:
        self._check_worker()
        return self._sequences[request_id].length

    @torch.inference_mode()
    def append(self, request_id: RequestId, key: Tensor, value: Tensor) -> None:
        """Append rotated K and unrotated V, each FP16 [22, tokens, 4, 64]."""
        self._check_worker()
        sequence = self._sequences[request_id]
        if (
            key.ndim != 4
            or key.shape[0] != 22
            or key.shape[1] < 1
            or key.shape[2:] != (4, 64)
            or value.shape != key.shape
            or key.dtype != torch.float16
            or value.dtype != torch.float16
            or key.device != self._pool.device
            or value.device != self._pool.device
        ):
            raise ValueError("K and V must share FP16 [22, tokens, 4, 64] on the pool device")
        length = sequence.length + key.shape[1]
        if length > 2048:
            raise ValueError("KV cache must not exceed 2048 token positions")
        additional = (length + 15) // 16 - len(sequence.blocks)
        if additional > len(self._free_blocks):
            raise RuntimeError("KV pool has insufficient free blocks")
        new_blocks = tuple(reversed(self._free_blocks[-additional:])) if additional else ()
        blocks = sequence.blocks + new_blocks

        offset = 0
        while offset < key.shape[1]:
            logical_block, slot = divmod(sequence.length + offset, 16)
            count = min(16 - slot, key.shape[1] - offset)
            physical_block = blocks[logical_block]
            self._pool[:, 0, physical_block, slot : slot + count].copy_(
                key[:, offset : offset + count]
            )
            self._pool[:, 1, physical_block, slot : slot + count].copy_(
                value[:, offset : offset + count]
            )
            offset += count

        # Publish valid positions only after both K and V writes have been issued.
        if additional:
            del self._free_blocks[-additional:]
        self._sequences[request_id] = _SequenceCache(blocks, length)
        self._assert_ownership()

    def read_block(self, request_id: RequestId, logical_block: int) -> tuple[Tensor, Tensor]:
        """Borrow K/V views [22, valid slots, 4, 64] for one logical block."""
        self._check_worker()
        sequence = self._sequences[request_id]
        if type(logical_block) is not int or not 0 <= logical_block < len(sequence.blocks):
            raise ValueError("logical block must identify a cached block")
        physical_block = sequence.blocks[logical_block]
        valid = min(16, sequence.length - logical_block * 16)
        return (
            self._pool[:, 0, physical_block, :valid],
            self._pool[:, 1, physical_block, :valid],
        )

    def release_all(self, request_id: RequestId) -> None:
        """Return physical blocks on the worker stream; unknown requests raise KeyError."""
        self._check_worker()
        sequence = self._sequences.pop(request_id)
        self._free_blocks.extend(reversed(sequence.blocks))
        self._assert_ownership()

    def _check_worker(self) -> None:
        if current_thread() is not self._worker:
            raise RuntimeError("KV allocator must be used by its owning model worker")
        if (
            self._stream is not None
            and torch.cuda.current_stream(self._pool.device) != self._stream
        ):
            raise RuntimeError("KV allocator must use its owning CUDA stream")

    def _assert_ownership(self) -> None:
        assigned = [block for sequence in self._sequences.values() for block in sequence.blocks]
        assert len(assigned) == len(set(assigned)), "physical KV blocks must have unique owners"
        assert len(self._free_blocks) + len(assigned) == self._pool.shape[2]
        for sequence in self._sequences.values():
            assert len(sequence.blocks) == (sequence.length + 15) // 16
