from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from pagedcore.config import PagedCoreConfig
from pagedcore.engine_types import RequestId
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
    allocator.create(request)
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
    allocator.create(a)
    allocator.create(b)
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
    allocator.create(a)
    allocator.create(b)
    allocator.append(a, *_kv_values(17))
    allocator.append(b, *_kv_values(16, start=2000))
    released = allocator.block_table(a)
    allocator.release_all(a)
    with pytest.raises(KeyError):
        allocator.read_block(a, 0)
    with pytest.raises(KeyError):
        allocator.release_all(a)

    allocator.create(c)
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
    allocator.create(a)
    allocator.append(a, *_kv_values(48))
    assert set(allocator.block_table(a)) == {0, 1, 2}
    assert pool.untyped_storage().data_ptr() == pointer


def test_allocator_exhaustion_preserves_history_and_allows_filling_a_partial_block() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 2, 16, 4, 64), dtype=torch.float16))
    a, b = RequestId("a"), RequestId("b")
    allocator.create(a)
    allocator.create(b)
    key, value = _kv_values(17)
    allocator.append(a, key[:, :15], value[:, :15])
    allocator.append(b, *_kv_values(16))
    table = allocator.block_table(a)
    with pytest.raises(RuntimeError, match="insufficient free blocks"):
        allocator.append(a, key[:, 15:], value[:, 15:])
    assert allocator.block_table(a) == table
    assert allocator.cached_length(a) == 15
    torch.testing.assert_close(allocator.read_block(a, 0)[0], key[:, :15], atol=0, rtol=0)

    allocator.append(a, key[:, 15:16], value[:, 15:16])
    assert allocator.block_table(a) == table
    allocator.release_all(b)
    allocator.append(a, key[:, 16:], value[:, 16:])
    assert allocator.cached_length(a) == 17
    assert len(allocator.block_table(a)) == 2


def test_allocator_releases_an_empty_request_and_rejects_duplicate_registration() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(request)
    with pytest.raises(ValueError, match="already has a KV cache"):
        allocator.create(request)
    allocator.release_all(request)
    allocator.create(request)
    allocator.append(request, *_kv_values(16))
    assert allocator.cached_length(request) == 16


@pytest.mark.parametrize("index", [-1, 1, True, 0.0])
def test_allocator_rejects_invalid_logical_blocks(index: Any) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(request)
    with pytest.raises(ValueError, match="logical block"):
        allocator.read_block(request, 0)
    allocator.append(request, *_kv_values(1))
    with pytest.raises(ValueError, match="logical block"):
        allocator.read_block(request, index)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "empty"])
def test_allocator_rejects_invalid_appends_before_assigning_blocks(invalid: str) -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(request)
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
    allocator.create(request)
    allocator.append(request, *_kv_values(2048))
    last_key, last_value = (tensor.clone() for tensor in allocator.read_block(request, 127))
    with pytest.raises(ValueError, match="2048"):
        allocator.append(request, *_kv_values(1))
    assert allocator.cached_length(request) == 2048
    actual_key, actual_value = allocator.read_block(request, 127)
    torch.testing.assert_close(actual_key, last_key, atol=0, rtol=0)
    torch.testing.assert_close(actual_value, last_value, atol=0, rtol=0)


def test_allocator_rejects_access_from_another_thread() -> None:
    allocator = KVAllocator(torch.empty((22, 2, 1, 16, 4, 64), dtype=torch.float16))
    request = RequestId("a")
    allocator.create(request)
    key, value = _kv_values(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for operation, args in (
            (allocator.create, (RequestId("b"),)),
            (allocator.append, (request, key, value)),
            (allocator.read_block, (request, 0)),
            (allocator.block_table, (request,)),
            (allocator.cached_length, (request,)),
            (allocator.release_all, (request,)),
        ):
            future = executor.submit(operation, *args)
            with pytest.raises(RuntimeError, match="owning model worker"):
                future.result()
    assert allocator.block_table(request) == ()
    assert allocator.cached_length(request) == 0
