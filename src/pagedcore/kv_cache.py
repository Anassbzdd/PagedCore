"""Shared FP16 storage for the pinned TinyLlama KV cache."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import current_thread

import torch
from torch import Tensor

from pagedcore.config import PagedCoreConfig
from pagedcore.engine_types import BlockId, BlockTable, CapacityCounts, KVReservation, RequestId


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
    reservation: KVReservation
    blocks: BlockTable = ()
    length: int = 0


@dataclass(slots=True)
class _TokenAppend:
    sequence: _SequenceCache
    written_layers: int = 0


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
        # Check tables against ownership recorded when physical blocks are assigned.
        self._block_owners: dict[BlockId, RequestId] = {}
        self._decode_appends: dict[RequestId, _TokenAppend] | None = None

    def create(self, reservation: KVReservation) -> None:
        """Grant credits at admission without assigning physical blocks.

        Pending requests stay outside the allocator until admission. Credits
        include owned blocks and remain held until release_all, even after EOS.
        """
        self._check_mutation()
        request_id = reservation.request_id
        if request_id in self._sequences:
            raise ValueError("request already has a KV cache")
        counts = self.capacity()
        if reservation.maximum_block_credits > counts.total_blocks:
            raise ValueError("reservation exceeds total KV pool capacity")
        if reservation.maximum_block_credits > 128:
            raise ValueError("reservation exceeds KV context capacity")
        if reservation.maximum_block_credits > counts.total_blocks - counts.reserved_blocks:
            raise RuntimeError("KV pool has insufficient unreserved blocks")
        self._sequences[request_id] = _SequenceCache(reservation)
        self._assert_accounting()

    def capacity(self) -> CapacityCounts:
        """Snapshot ownership and credits, including requests awaiting cleanup."""
        self._check_worker()
        return self._assert_accounting()

    def block_table(self, request_id: RequestId) -> BlockTable:
        self._check_worker()
        self._assert_accounting()
        return self._sequences[request_id].blocks

    def cached_length(self, request_id: RequestId) -> int:
        self._check_worker()
        self._assert_accounting()
        return self._sequences[request_id].length

    @torch.inference_mode()
    def append(self, request_id: RequestId, key: Tensor, value: Tensor) -> None:
        """Append rotated K and unrotated V, each FP16 [22, tokens, 4, 64].

        A synchronous copy failure leaves ownership and valid positions unchanged.
        Unpublished slots may be dirty; they must be overwritten before use.
        """
        self._check_mutation()
        self._assert_accounting()
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
        required = (length + 15) // 16
        if required > sequence.reservation.maximum_block_credits:
            raise RuntimeError("KV append exceeds request reservation")
        additional = required - len(sequence.blocks)
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
        self._block_owners.update((block, request_id) for block in new_blocks)
        self._sequences[request_id] = _SequenceCache(sequence.reservation, blocks, length)
        self._assert_accounting()

    @contextmanager
    def append_decode(self, request_ids: tuple[RequestId, ...]) -> Iterator[None]:
        self._check_mutation()
        self._assert_accounting()
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError("decode requires nonempty, distinct request IDs")
        appends = {}
        additional = 0
        for request_id in request_ids:
            sequence = self._sequences[request_id]
            if not 1 <= sequence.length < 2048:
                raise ValueError("decode requires 1 to 2047 cached token positions")
            length = sequence.length + 1
            blocks = sequence.blocks
            if (length + 15) // 16 > sequence.reservation.maximum_block_credits:
                raise RuntimeError("KV append exceeds request reservation")
            if sequence.length % 16 == 0:
                additional += 1
                if additional > len(self._free_blocks):
                    raise RuntimeError("KV pool has insufficient free blocks")
                blocks += (self._free_blocks[-additional],)
            appends[request_id] = _TokenAppend(_SequenceCache(sequence.reservation, blocks, length))

        # Block other mutations while free-list slots are borrowed for staged writes.
        self._decode_appends = appends
        try:
            yield
            if any(append.written_layers != 22 for append in appends.values()):
                raise RuntimeError("decode must write every cache layer before committing")
            if additional:
                del self._free_blocks[-additional:]
            for request_id, append in appends.items():
                original = self._sequences[request_id]
                for block in append.sequence.blocks[len(original.blocks) :]:
                    self._block_owners[block] = request_id
                self._sequences[request_id] = append.sequence
            self._assert_accounting()
        finally:
            self._decode_appends = None

    @torch.inference_mode()
    def write_decode_layer(
        self, request_id: RequestId, layer_index: int, key: Tensor, value: Tensor
    ) -> None:
        """Write one token's rotated K/unrotated V as FP16 [4, 64], in layer order."""
        self._check_worker()
        self._assert_accounting()
        if self._decode_appends is None:
            raise RuntimeError("decode writes require a staged append")
        append = self._decode_appends[request_id]
        if type(layer_index) is not int or not 0 <= layer_index < 22:
            raise ValueError("layer_index must identify one of the 22 decoder layers")
        if layer_index != append.written_layers:
            raise ValueError("decode cache layers must be written once in order")
        if any(
            tensor.shape != (4, 64)
            or tensor.dtype != torch.float16
            or tensor.device != self._pool.device
            for tensor in (key, value)
        ):
            raise ValueError("K and V must share FP16 [4, 64] on the pool device")
        logical_block, slot = divmod(append.sequence.length - 1, 16)
        block = append.sequence.blocks[logical_block]
        self._pool[layer_index, 0, block, slot].copy_(key)
        self._pool[layer_index, 1, block, slot].copy_(value)
        append.written_layers += 1

    def iter_layer_blocks(
        self, request_id: RequestId, *, layer_index: int
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Borrow valid [slots, 4, 64] pages, including a completed staged layer."""
        self._check_worker()
        self._assert_accounting()
        if type(layer_index) is not int or not 0 <= layer_index < 22:
            raise ValueError("layer_index must identify one of the 22 decoder layers")
        sequence = self._sequences[request_id]
        if self._decode_appends is not None and request_id in self._decode_appends:
            append = self._decode_appends[request_id]
            if layer_index >= append.written_layers:
                raise RuntimeError("cannot read an unwritten decode layer")
            sequence = append.sequence
        if sequence.length == 0:
            raise ValueError("attention requires a nonempty cache")
        for logical_block, physical_block in enumerate(sequence.blocks):
            self._check_worker()
            valid = min(16, sequence.length - logical_block * 16)
            yield (
                self._pool[layer_index, 0, physical_block, :valid],
                self._pool[layer_index, 1, physical_block, :valid],
            )

    def read_block(self, request_id: RequestId, logical_block: int) -> tuple[Tensor, Tensor]:
        """Borrow K/V views [22, valid slots, 4, 64] for one logical block."""
        self._check_worker()
        self._assert_accounting()
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
        """Release once on the worker stream; unknown/released requests raise KeyError.

        Request cleanup must remember successful release to make duplicate cleanup
        signals no-ops. Silently accepting a direct double release would hide bugs.
        """
        self._check_mutation()
        self._assert_accounting()
        sequence = self._sequences.pop(request_id)
        for block in sequence.blocks:
            del self._block_owners[block]
        self._free_blocks.extend(reversed(sequence.blocks))
        self._assert_accounting()

    def _check_mutation(self) -> None:
        self._check_worker()
        if self._decode_appends is not None:
            raise RuntimeError("KV mutation is not allowed during a staged decode append")

    def _check_worker(self) -> None:
        if current_thread() is not self._worker:
            raise RuntimeError("KV allocator must be used by its owning model worker")
        if (
            self._stream is not None
            and torch.cuda.current_stream(self._pool.device) != self._stream
        ):
            raise RuntimeError("KV allocator must use its owning CUDA stream")

    def _assert_accounting(self) -> CapacityCounts:
        for request_id, sequence in self._sequences.items():
            if (
                sequence.reservation.request_id != request_id
                or type(sequence.length) is not int
                or not 0 <= sequence.length <= 2048
                or type(sequence.blocks) is not tuple
                or len(sequence.blocks) != (sequence.length + 15) // 16
                or not len(sequence.blocks) <= sequence.reservation.maximum_block_credits <= 128
            ):
                raise RuntimeError("corrupt KV block table or reservation")
        assigned = [block for sequence in self._sequences.values() for block in sequence.blocks]
        if any(
            type(block) is not int or not 0 <= block < self._pool.shape[2]
            for block in self._free_blocks + assigned
        ):
            raise RuntimeError("invalid physical KV block")
        owned = set(assigned)
        free = set(self._free_blocks)
        if len(assigned) != len(owned):
            raise RuntimeError("physical KV blocks must have unique owners")
        if len(self._free_blocks) != len(free):
            raise RuntimeError("free KV blocks must be unique")
        if not free.isdisjoint(owned):
            raise RuntimeError("owned KV blocks must not be free")
        if free | owned != set(range(self._pool.shape[2])):
            raise RuntimeError("KV blocks must cover the pool")
        owners = {
            block: request_id
            for request_id, sequence in self._sequences.items()
            for block in sequence.blocks
        }
        if owners != self._block_owners:
            raise RuntimeError("KV blocks are not owned by the recorded requests")
        return CapacityCounts(
            total_blocks=self._pool.shape[2],
            free_blocks=len(self._free_blocks),
            owned_blocks=len(assigned),
            reserved_blocks=sum(
                sequence.reservation.maximum_block_credits for sequence in self._sequences.values()
            ),
            active_slots=len(self._sequences),
            live_cached_token_slots=sum(sequence.length for sequence in self._sequences.values()),
        )
