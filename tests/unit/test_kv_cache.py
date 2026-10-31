from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from pagedcore.config import PagedCoreConfig
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator, allocate_kv_pool

pytestmark = pytest.mark.unit


@pytest.fixture
def cuda_memory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    memory = Mock(return_value=(16 * 2**30, 16 * 2**30))
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory)
    return memory


@pytest.mark.parametrize("pool_mib", [1, 8192])
def test_pool_uses_one_fp16_allocation_with_whole_blocks(
    monkeypatch: pytest.MonkeyPatch, cuda_memory: Mock, pool_mib: int
) -> None:
    config = PagedCoreConfig(kv_pool_mib=pool_mib)
    empty = torch.empty
    # Meta storage checks the default pool shape without allocating eight GiB on CPU.
    allocation = Mock(
        side_effect=lambda shape, **kwargs: empty(shape, dtype=kwargs["dtype"], device="meta")
    )
    monkeypatch.setattr(torch, "empty", allocation)

    pool = allocate_kv_pool(config)

    blocks = 2 if pool_mib == 1 else 23831
    allocation.assert_called_once_with(
        (22, 2, blocks, 16, 4, 64), dtype=torch.float16, device=torch.device("cuda:0")
    )
    assert pool.shape == (22, 2, blocks, 16, 4, 64)
    assert pool.dtype == torch.float16
    assert not pool.requires_grad
    assert pool.is_contiguous()
    used_bytes = pool.numel() * pool.element_size()
    assert used_bytes <= pool_mib * 2**20 < used_bytes + 352 * 1024
    assert cuda_memory.call_count == 2
    cuda_memory.assert_called_with(torch.device("cuda:0"))


def test_pool_views_reuse_backing_storage(
    monkeypatch: pytest.MonkeyPatch, cuda_memory: Mock
) -> None:
    empty = torch.empty
    allocation = Mock(side_effect=lambda shape, **kwargs: empty(shape, dtype=kwargs["dtype"]))
    monkeypatch.setattr(torch, "empty", allocation)
    config = PagedCoreConfig(kv_pool_mib=1, workspace_margin_mib=1)
    cuda_memory.return_value = (2 * 2**20, 16 * 2**30)
    pool = allocate_kv_pool(config, device=torch.device("cuda:0"))
    pointer = pool.untyped_storage().data_ptr()

    for slot in range(16):
        pool[:, 0, 0, slot].fill_(slot)
        pool[:, 1, 1, slot].fill_(slot + 16)

    assert pool.untyped_storage().data_ptr() == pointer
    assert torch.all(pool[:, 0, 0, 15] == 15)
    assert torch.all(pool[:, 1, 1, 0] == 16)
    assert torch.all(pool[:, 1, 1, 15] == 31)
    allocation.assert_called_once()


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_pool_rejects_non_cuda_devices(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    allocation = Mock()
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        allocate_kv_pool(PagedCoreConfig(), device=device)
    allocation.assert_not_called()


def test_pool_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    allocation = Mock()
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(RuntimeError, match="available CUDA runtime"):
        allocate_kv_pool(PagedCoreConfig())
    allocation.assert_not_called()


def test_pool_rejects_insufficient_memory_before_allocating(
    monkeypatch: pytest.MonkeyPatch, cuda_memory: Mock
) -> None:
    config = PagedCoreConfig(kv_pool_mib=1, workspace_margin_mib=1)
    cuda_memory.return_value = (2 * 2**20 - 1, 16 * 2**30)
    allocation = Mock()
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(RuntimeError, match="do not fit"):
        allocate_kv_pool(config)
    allocation.assert_not_called()


def test_pool_rejects_insufficient_margin_after_allocating(
    monkeypatch: pytest.MonkeyPatch, cuda_memory: Mock
) -> None:
    config = PagedCoreConfig(kv_pool_mib=1, workspace_margin_mib=1)
    cuda_memory.side_effect = [(2 * 2**20, 16 * 2**30), (2**20 - 1, 16 * 2**30)]
    allocation = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(RuntimeError, match="leaves less than"):
        allocate_kv_pool(config)
    allocation.assert_called_once()


def test_pool_propagates_cuda_allocation_failure(
    monkeypatch: pytest.MonkeyPatch, cuda_memory: Mock
) -> None:
    allocation = Mock(side_effect=torch.OutOfMemoryError("synthetic CUDA exhaustion"))
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(torch.OutOfMemoryError, match="synthetic CUDA exhaustion"):
        allocate_kv_pool(PagedCoreConfig())
    allocation.assert_called_once()
    cuda_memory.assert_called_once()


def _kv_values(tokens: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(22 * tokens * 4 * 64, dtype=torch.float32).reshape(22, tokens, 4, 64)
    key = (key / 256 + start).half()
    return key, -key - 1


@pytest.mark.parametrize("invalid", ["shape", "dtype", "empty", "strides", "grad"])
def test_allocator_rejects_incompatible_pool_storage(invalid: str) -> None:
    pool = torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16)
    if invalid == "shape":
        pool = pool[:1]
    elif invalid == "dtype":
        pool = pool.float()
    elif invalid == "empty":
        pool = pool[:, :, :0]
    elif invalid == "strides":
        pool = torch.empty((22, 2, 1, 16, 4, 128), dtype=torch.float16)[..., ::2]
    else:
        pool.requires_grad_(True)
    with pytest.raises(ValueError, match="KV pool must be contiguous FP16"):
        KVAllocator(pool)


@pytest.mark.parametrize(
    "tokens,blocks,last_slots",
    [(1, 1, 1), (15, 1, 15), (16, 1, 16), (17, 2, 1), (32, 2, 16), (33, 3, 1)],
)
def test_allocator_maps_tokens_and_limits_reads_to_valid_slots(
    tokens: int, blocks: int, last_slots: int
) -> None:
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    request = RequestId("a")
    allocator.create(KVReservation(request, blocks))
    assert allocator.block_table(request) == ()
    assert allocator.cached_length(request) == 0
    key, value = _kv_values(tokens)

    allocator.append(request, key, value)

    table = allocator.block_table(request)
    assert len(table) == blocks
    assert len(set(table)) == blocks
    assert allocator.cached_length(request) == tokens
    for index, physical in enumerate(table):
        slots = last_slots if index == blocks - 1 else 16
        actual_key, actual_value = allocator.read_block(request, index)
        assert actual_key.shape == actual_value.shape == (22, slots, 4, 64)
        assert actual_key.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert actual_value.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert not actual_key.requires_grad and not actual_value.requires_grad
        expected = slice(index * 16, index * 16 + slots)
        torch.testing.assert_close(actual_key, key[:, expected], atol=0, rtol=0)
        torch.testing.assert_close(actual_value, value[:, expected], atol=0, rtol=0)
        torch.testing.assert_close(pool[:, 0, physical, :slots], key[:, expected], atol=0, rtol=0)
        if slots < 16:
            assert torch.isnan(pool[:, :, physical, slots:]).all()


def test_allocator_appends_across_fragmented_blocks_without_touching_another_request() -> None:
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 3))
    allocator.create(KVReservation(b, 1))
    key, value = _kv_values(33)
    other_key, other_value = _kv_values(16, start=2000)

    allocator.append(a, key[:, :15], value[:, :15])
    first_block = allocator.block_table(a)
    allocator.append(b, other_key, other_value)
    allocator.append(a, key[:, 15:16], value[:, 15:16])
    assert allocator.block_table(a) == first_block
    allocator.append(a, key[:, 16:17], value[:, 16:17])
    assert allocator.read_block(a, 1)[0].shape[1] == 1
    allocator.append(a, key[:, 17:], value[:, 17:])

    assert tuple(allocator.block_table(a)) == (0, 2, 3)
    assert tuple(allocator.block_table(b)) == (1,)
    assert allocator.cached_length(a) == 33
    for index, expected in enumerate((slice(0, 16), slice(16, 32), slice(32, 33))):
        actual_key, actual_value = allocator.read_block(a, index)
        torch.testing.assert_close(actual_key, key[:, expected], atol=0, rtol=0)
        torch.testing.assert_close(actual_value, value[:, expected], atol=0, rtol=0)
    actual_key, actual_value = allocator.read_block(b, 0)
    torch.testing.assert_close(actual_key, other_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, other_value, atol=0, rtol=0)


def test_allocator_releases_and_reuses_blocks_without_exposing_stale_slots() -> None:
    pool = torch.empty((22, 2, 3, 16, 4, 64), dtype=torch.float16)
    pointer = pool.untyped_storage().data_ptr()
    allocator = KVAllocator(pool)
    a, b, c = RequestId("a"), RequestId("b"), RequestId("c")
    allocator.create(KVReservation(a, 2))
    allocator.create(KVReservation(b, 1))
    allocator.append(a, *_kv_values(17))
    allocator.append(b, *_kv_values(16, start=2000))
    released = allocator.block_table(a)
    allocator.release_all(a)
    with pytest.raises(KeyError):
        allocator.read_block(a, 0)
    with pytest.raises(KeyError):
        allocator.release_all(a)

    allocator.create(KVReservation(c, 1))
    key, value = _kv_values(1, start=-100)
    allocator.append(c, key, value)
    assert allocator.block_table(c)[0] in released
    actual_key, actual_value = allocator.read_block(c, 0)
    assert actual_key.shape == (22, 1, 4, 64)
    torch.testing.assert_close(actual_key, key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, value, atol=0, rtol=0)
    expected_key, expected_value = _kv_values(16, start=2000)
    actual_key, actual_value = allocator.read_block(b, 0)
    torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)

    allocator.release_all(b)
    allocator.release_all(c)
    allocator.create(KVReservation(a, 3))
    allocator.append(a, *_kv_values(48))
    assert set(allocator.block_table(a)) == {0, 1, 2}
    assert pool.untyped_storage().data_ptr() == pointer


def test_allocator_reservation_limit_preserves_history_and_allows_filling_a_partial_block() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 2, 16, 4, 64), dtype=torch.float16))
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 1))
    allocator.create(KVReservation(b, 1))
    key, value = _kv_values(17)
    allocator.append(a, key[:, :15], value[:, :15])
    allocator.append(b, *_kv_values(16))
    table = allocator.block_table(a)
    before = allocator.capacity()
    with pytest.raises(RuntimeError, match="exceeds request reservation"):
        allocator.append(a, key[:, 15:], value[:, 15:])
    assert allocator.capacity() == before
    assert allocator.block_table(a) == table
    assert allocator.cached_length(a) == 15
    torch.testing.assert_close(allocator.read_block(a, 0)[0], key[:, :15], atol=0, rtol=0)

    allocator.append(a, key[:, 15:16], value[:, 15:16])
    assert allocator.block_table(a) == table
    allocator.release_all(b)
    with pytest.raises(RuntimeError, match="exceeds request reservation"):
        allocator.append(a, key[:, 16:], value[:, 16:])
    allocator.release_all(a)
    allocator.create(KVReservation(a, 2))
    allocator.append(a, key, value)
    assert allocator.cached_length(a) == 17
    assert len(allocator.block_table(a)) == 2


def test_allocator_releases_an_empty_request_and_rejects_duplicate_registration() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 1))
    before = allocator.capacity()
    with pytest.raises(ValueError, match="already has a KV cache"):
        allocator.create(KVReservation(request, 1))
    assert allocator.capacity() == before
    allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(1, 1, 0, 0, 0, 0)
    allocator.create(KVReservation(request, 1))
    allocator.append(request, *_kv_values(16))
    assert allocator.cached_length(request) == 16


@pytest.mark.parametrize("index", [-1, 1, True, 0.0])
def test_allocator_rejects_invalid_logical_blocks(index: Any) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 1))
    with pytest.raises(ValueError, match="logical block"):
        allocator.read_block(request, 0)
    allocator.append(request, *_kv_values(1))
    with pytest.raises(ValueError, match="logical block"):
        allocator.read_block(request, index)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "empty"])
def test_allocator_rejects_invalid_appends_before_assigning_blocks(invalid: str) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 1))
    key, value = _kv_values(1)
    if invalid == "shape":
        value = value[:1]
    elif invalid == "dtype":
        value = value.float()
    elif invalid == "device":
        value = value.to("meta")
    else:
        key, value = key[:, :0], value[:, :0]
    with pytest.raises(ValueError, match="K and V must share"):
        allocator.append(request, key, value)
    assert allocator.block_table(request) == ()
    assert allocator.cached_length(request) == 0
    allocator.append(request, *_kv_values(16))


def test_allocator_rejects_context_overflow_before_writing() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 128, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 128))
    allocator.append(request, *_kv_values(2048))
    last_key, last_value = (tensor.clone() for tensor in allocator.read_block(request, 127))
    with pytest.raises(ValueError, match="2048"):
        allocator.append(request, *_kv_values(1))
    assert allocator.cached_length(request) == 2048
    actual_key, actual_value = allocator.read_block(request, 127)
    torch.testing.assert_close(actual_key, last_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, last_value, atol=0, rtol=0)


def test_decode_append_exposes_only_written_layers_and_commits_together() -> None:
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    requests = (RequestId("first"), RequestId("second"))
    for request in requests:
        allocator.create(KVReservation(request, 2))
        allocator.append(request, *_kv_values(16))
    counts = allocator.capacity()
    tables = [allocator.block_table(request) for request in requests]
    key, value = _kv_values(1, start=100)
    with allocator.append_decode(requests):
        with pytest.raises(RuntimeError, match="unwritten"):
            list(allocator.iter_layer_blocks(requests[0], layer_index=0))
        for operation in (
            lambda: allocator.create(KVReservation(RequestId("other"), 1)),
            lambda: allocator.append(requests[0], key, value),
            lambda: allocator.release_all(requests[0]),
            lambda: allocator.append_decode(requests).__enter__(),
        ):
            with pytest.raises(RuntimeError, match="staged decode"):
                operation()
        for layer in range(22):
            for request in requests:
                allocator.write_decode_layer(request, layer, key[layer, 0], value[layer, 0])
                pages = list(allocator.iter_layer_blocks(request, layer_index=layer))
                assert len(pages) == 2
                torch.testing.assert_close(pages[1][0], key[layer], atol=0, rtol=0)
                torch.testing.assert_close(pages[1][1], value[layer], atol=0, rtol=0)
                # Public length/table/accounting remain at the committed boundary.
                assert allocator.cached_length(request) == 16
                assert allocator.capacity() == counts
        for request, table in zip(requests, tables, strict=True):
            assert allocator.block_table(request) == table
            assert allocator.read_block(request, 0)[0].shape[1] == 16
    assert allocator.capacity() == CapacityCounts(4, 0, 4, 4, 2, 34)
    for request in requests:
        assert allocator.cached_length(request) == 17
        allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)


@pytest.mark.parametrize("fault", ["incomplete", "order", "shape", "dtype", "device"])
def test_decode_append_rejects_incomplete_or_invalid_layer_writes(fault: str) -> None:
    allocator = KVAllocator(torch.zeros(22, 2, 2, 16, 4, 64, dtype=torch.float16))
    request = RequestId("request")
    allocator.create(KVReservation(request, 2))
    allocator.append(request, *_kv_values(16))
    counts = allocator.capacity()
    key = torch.zeros(4, 64, dtype=torch.float16)
    value = key.clone()
    if fault == "shape":
        key = key[0]
    elif fault == "dtype":
        value = value.float()
    elif fault == "device":
        key = key.to("meta")
    with (
        pytest.raises(RuntimeError if fault == "incomplete" else ValueError),
        allocator.append_decode((request,)),
    ):
        allocator.write_decode_layer(request, 1 if fault == "order" else 0, key, value)
    assert allocator.capacity() == counts
    assert allocator.cached_length(request) == 16
    # Clearing staged state permits normal ownership cleanup after any rejection.
    allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)


def test_layer_block_iterator_checks_worker_on_every_page() -> None:
    allocator = KVAllocator(torch.empty(22, 2, 2, 16, 4, 64, dtype=torch.float16))
    request = RequestId("request")
    allocator.create(KVReservation(request, 2))
    allocator.append(request, *_kv_values(17))
    pages = allocator.iter_layer_blocks(request, layer_index=0)
    next(pages)
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="owning model worker"),
    ):
        executor.submit(next, pages).result()
    allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)


def test_allocator_rejects_access_from_another_thread() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 1))
    key, value = _kv_values(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for operation, args in (
            (allocator.create, (KVReservation(RequestId("b"), 1),)),
            (allocator.capacity, ()),
            (allocator.append, (request, key, value)),
            (allocator.read_block, (request, 0)),
            (allocator.block_table, (request,)),
            (allocator.cached_length, (request,)),
            (allocator.release_all, (request,)),
            (allocator.write_decode_layer, (request, 0, key[0, 0], value[0, 0])),
            (lambda: allocator.append_decode((request,)).__enter__(), ()),
            (lambda: next(allocator.iter_layer_blocks(request, layer_index=0)), ()),
        ):
            future = executor.submit(operation, *args)
            with pytest.raises(RuntimeError, match="owning model worker"):
                future.result()
    assert allocator.block_table(request) == ()
    assert allocator.cached_length(request) == 0


def test_allocator_accounts_for_reserved_owned_and_live_capacity_across_requests() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 8, 16, 4, 64), dtype=torch.float16))
    a, b = RequestId("a"), RequestId("b")
    empty = allocator.capacity()
    assert empty == CapacityCounts(8, 8, 0, 0, 0, 0)

    allocator.create(KVReservation(a, 3))
    assert allocator.capacity() == CapacityCounts(8, 8, 0, 3, 1, 0)
    allocator.create(KVReservation(b, 2))
    assert allocator.capacity() == CapacityCounts(8, 8, 0, 5, 2, 0)
    key, value = _kv_values(33)

    allocator.append(a, key[:, :15], value[:, :15])
    assert allocator.capacity() == CapacityCounts(8, 7, 1, 5, 2, 15)
    allocator.append(b, *_kv_values(16, start=2000))
    assert allocator.capacity() == CapacityCounts(8, 6, 2, 5, 2, 31)
    allocator.append(a, key[:, 15:16], value[:, 15:16])
    assert allocator.capacity() == CapacityCounts(8, 6, 2, 5, 2, 32)
    allocator.append(a, key[:, 16:17], value[:, 16:17])
    assert allocator.capacity() == CapacityCounts(8, 5, 3, 5, 2, 33)
    allocator.append(a, key[:, 17:], value[:, 17:])
    assert allocator.capacity() == CapacityCounts(8, 4, 4, 5, 2, 49)
    assert set(allocator.block_table(a)).isdisjoint(allocator.block_table(b))
    assert allocator.cached_length(a) == 33
    assert allocator.cached_length(b) == 16

    allocator.release_all(a)
    assert allocator.capacity() == CapacityCounts(8, 7, 1, 2, 1, 16)
    allocator.release_all(b)
    assert allocator.capacity() == empty
    assert empty == CapacityCounts(8, 8, 0, 0, 0, 0)


def test_allocator_reservations_use_unpromised_capacity_instead_of_free_blocks() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 3, 16, 4, 64), dtype=torch.float16))
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 3))
    before = allocator.capacity()
    assert before == CapacityCounts(3, 3, 0, 3, 1, 0)
    with pytest.raises(RuntimeError, match="insufficient unreserved blocks"):
        allocator.create(KVReservation(b, 1))
    assert allocator.capacity() == before
    with pytest.raises(KeyError):
        allocator.block_table(b)

    allocator.append(a, *_kv_values(17))
    before = allocator.capacity()
    assert before == CapacityCounts(3, 1, 2, 3, 1, 17)
    with pytest.raises(RuntimeError, match="insufficient unreserved blocks"):
        allocator.create(KVReservation(b, 1))
    assert allocator.capacity() == before

    allocator.release_all(a)
    allocator.create(KVReservation(b, 1))
    assert allocator.capacity() == CapacityCounts(3, 3, 0, 1, 1, 0)


def test_allocator_cannot_spend_another_requests_reserved_credits() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 4, 16, 4, 64), dtype=torch.float16))
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 1))
    allocator.create(KVReservation(b, 3))
    key, value = _kv_values(17)
    allocator.append(a, key[:, :16], value[:, :16])
    before = allocator.capacity()
    assert before == CapacityCounts(4, 3, 1, 4, 2, 16)

    with pytest.raises(RuntimeError, match="exceeds request reservation"):
        allocator.append(a, key[:, 16:], value[:, 16:])

    assert allocator.capacity() == before
    assert allocator.cached_length(a) == 16
    assert allocator.block_table(b) == ()
    torch.testing.assert_close(allocator.read_block(a, 0)[0], key[:, :16], atol=0, rtol=0)
    allocator.append(b, *_kv_values(33))
    assert allocator.capacity() == CapacityCounts(4, 0, 4, 4, 2, 49)


def test_allocator_pending_requests_hold_no_resources() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 3, 16, 4, 64), dtype=torch.float16))
    pending = KVReservation(RequestId("pending"), 3)
    before = allocator.capacity()
    key, value = _kv_values(1)

    with pytest.raises(KeyError):
        allocator.block_table(pending.request_id)
    with pytest.raises(KeyError):
        allocator.cached_length(pending.request_id)
    with pytest.raises(KeyError):
        allocator.append(pending.request_id, key, value)
    with pytest.raises(KeyError):
        allocator.release_all(pending.request_id)
    assert allocator.capacity() == before == CapacityCounts(3, 3, 0, 0, 0, 0)

    allocator.create(pending)
    assert allocator.capacity() == CapacityCounts(3, 3, 0, 3, 1, 0)
    allocator.release_all(pending.request_id)
    assert allocator.capacity() == before


@pytest.mark.parametrize(
    ("blocks", "credits", "message"),
    [(2, 3, "total KV pool capacity"), (129, 129, "KV context capacity")],
)
def test_allocator_rejects_impossible_reservations_without_acquiring_resources(
    blocks: int, credits: int, message: str
) -> None:
    allocator = KVAllocator(torch.empty((22, 2, blocks, 16, 4, 64), dtype=torch.float16))
    request = RequestId("impossible")
    before = allocator.capacity()

    with pytest.raises(ValueError, match=message):
        allocator.create(KVReservation(request, credits))

    assert allocator.capacity() == before
    with pytest.raises(KeyError):
        allocator.release_all(request)


@pytest.mark.parametrize("operation", ["read", "append", "release"])
@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "foreign", "free", "negative", "outside", "boolean", "float", "short", "length"],
)
def test_allocator_rejects_corrupt_tables_before_access_or_mutation(
    corruption: str, operation: str
) -> None:
    pool = torch.full((22, 2, 4, 16, 4, 64), -999, dtype=torch.float16)
    allocator = KVAllocator(pool)
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 3))
    allocator.create(KVReservation(b, 1))
    allocator.append(a, *_kv_values(17))
    allocator.append(b, *_kv_values(1, start=2000))
    before = allocator.capacity()
    original = allocator._sequences[a]
    first, last = original.blocks
    tables = {
        "duplicate": (first, first),
        "foreign": (allocator.block_table(b)[0], last),
        "free": (allocator._free_blocks[-1], last),
        "negative": (-1, last),
        "outside": (4, last),
        "boolean": (True, last),
        "float": (float(first), last),
        "short": (first,),
        "length": original.blocks,
    }
    allocator._sequences[a] = replace(
        original, blocks=tables[corruption], length=16 if corruption == "length" else 17
    )
    sequences = allocator._sequences.copy()
    free = allocator._free_blocks.copy()
    contents = pool.clone()

    with pytest.raises(RuntimeError, match="KV"):
        if operation == "read":
            allocator.read_block(a, 0)
        elif operation == "append":
            allocator.append(a, *_kv_values(1))
        else:
            allocator.release_all(a)

    assert allocator._sequences == sequences
    assert allocator._free_blocks == free
    torch.testing.assert_close(pool, contents, atol=0, rtol=0)
    allocator._sequences[a] = original
    assert allocator.capacity() == before
    allocator.release_all(a)
    allocator.release_all(b)
    assert allocator.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)


@pytest.mark.parametrize("operation", ["read", "append", "release"])
def test_allocator_detects_foreign_ownership_even_when_tables_are_swapped(operation: str) -> None:
    pool = torch.full((22, 2, 4, 16, 4, 64), -999, dtype=torch.float16)
    allocator = KVAllocator(pool)
    a, b = RequestId("a"), RequestId("b")
    allocator.create(KVReservation(a, 2))
    allocator.create(KVReservation(b, 2))
    allocator.append(a, *_kv_values(1))
    allocator.append(b, *_kv_values(1, start=2000))
    first, second = allocator._sequences[a], allocator._sequences[b]
    allocator._sequences[a] = replace(first, blocks=second.blocks)
    allocator._sequences[b] = replace(second, blocks=first.blocks)
    sequences = allocator._sequences.copy()
    free = allocator._free_blocks.copy()
    contents = pool.clone()

    with pytest.raises(RuntimeError, match="owned"):
        if operation == "read":
            allocator.read_block(a, 0)
        elif operation == "append":
            allocator.append(a, *_kv_values(1))
        else:
            allocator.release_all(a)

    assert allocator._sequences == sequences
    assert allocator._free_blocks == free
    torch.testing.assert_close(pool, contents, atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "owned", "invalid"])
def test_allocator_rejects_corrupt_free_lists_before_release(corruption: str) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 3, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(KVReservation(request, 1))
    allocator.append(request, *_kv_values(1))
    if corruption == "duplicate":
        allocator._free_blocks.append(allocator._free_blocks[-1])
    elif corruption == "missing":
        allocator._free_blocks.pop()
    elif corruption == "owned":
        allocator._free_blocks[-1] = allocator.block_table(request)[0]
    else:
        allocator._free_blocks[-1] = -1
    sequences = allocator._sequences.copy()
    free = allocator._free_blocks.copy()

    with pytest.raises(RuntimeError, match="KV"):
        allocator.release_all(request)

    assert allocator._sequences == sequences
    assert allocator._free_blocks == free


@pytest.mark.parametrize("initial_tokens", [0, 15, 16, 17])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_allocator_copy_failure_preserves_valid_history_and_allows_cleanup_and_reuse(
    monkeypatch: pytest.MonkeyPatch, initial_tokens: int, fail_at: int
) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 4, 16, 4, 64), dtype=torch.float16))
    a, b, c = RequestId("a"), RequestId("b"), RequestId("c")
    allocator.create(KVReservation(a, 3))
    allocator.create(KVReservation(b, 1))
    key, value = _kv_values(initial_tokens)
    if initial_tokens:
        allocator.append(a, key, value)
    other_key, other_value = _kv_values(1, start=2000)
    allocator.append(b, other_key, other_value)
    before = allocator.capacity()
    table = allocator.block_table(a)
    copy = torch.Tensor.copy_
    calls = 0

    def failing_copy(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("injected copy failure")
        return copy(destination, source)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", failing_copy)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            allocator.append(a, *_kv_values(18, start=100))

    assert calls == fail_at
    assert allocator.capacity() == before
    assert allocator.block_table(a) == table
    assert allocator.cached_length(a) == initial_tokens
    for logical_block in range(len(table)):
        start = logical_block * 16
        actual_key, actual_value = allocator.read_block(a, logical_block)
        torch.testing.assert_close(actual_key, key[:, start : start + 16], atol=0, rtol=0)
        torch.testing.assert_close(actual_value, value[:, start : start + 16], atol=0, rtol=0)
    actual_key, actual_value = allocator.read_block(b, 0)
    torch.testing.assert_close(actual_key, other_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, other_value, atol=0, rtol=0)

    allocator.release_all(a)
    after_release = allocator.capacity()
    with pytest.raises(KeyError):
        allocator.release_all(a)
    assert allocator.capacity() == after_release == CapacityCounts(4, 3, 1, 1, 1, 1)
    allocator.create(KVReservation(c, 3))
    fresh_key, fresh_value = _kv_values(48, start=-100)
    allocator.append(c, fresh_key, fresh_value)
    for logical_block in range(3):
        start = logical_block * 16
        actual_key, actual_value = allocator.read_block(c, logical_block)
        torch.testing.assert_close(actual_key, fresh_key[:, start : start + 16], atol=0, rtol=0)
        torch.testing.assert_close(actual_value, fresh_value[:, start : start + 16], atol=0, rtol=0)
    allocator.release_all(b)
    allocator.release_all(c)
    assert allocator.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)
