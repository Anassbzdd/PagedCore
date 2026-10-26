import pytest
import torch

from pagedcore.config import PagedCoreConfig
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator, allocate_kv_pool

pytestmark = pytest.mark.gpu


def test_kv_pool_storage_on_t4() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert "t4" in torch.cuda.get_device_name(0).casefold(), "requires an NVIDIA T4"

    pool = allocate_kv_pool(PagedCoreConfig(kv_pool_mib=1))
    assert pool.shape == (22, 2, 2, 16, 4, 64)
    assert pool.dtype == torch.float16
    assert pool.device == torch.device("cuda:0")
    assert pool.is_contiguous()
    assert not pool.requires_grad
    pointer = pool.untyped_storage().data_ptr()
    allocated_bytes = torch.cuda.memory_allocated(0)

    for slot in range(16):
        pool[:, 0, 0, slot].fill_(slot)
        pool[:, 1, 1, slot].fill_(slot + 16)
    torch.cuda.synchronize(0)

    assert pool.untyped_storage().data_ptr() == pointer
    assert torch.cuda.memory_allocated(0) == allocated_bytes
    assert torch.all(pool[:, 0, 0, 15] == 15).item()
    assert torch.all(pool[:, 1, 1, 0] == 16).item()
    assert torch.all(pool[:, 1, 1, 15] == 31).item()


def test_kv_allocator_storage_and_stream_ownership_on_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert "t4" in torch.cuda.get_device_name(0).casefold(), "requires an NVIDIA T4"

    pool = allocate_kv_pool(PagedCoreConfig(kv_pool_mib=1))
    allocator = KVAllocator(pool)
    a, b = RequestId("a"), RequestId("b")
    key = torch.arange(17, dtype=torch.float16, device="cuda:0")
    key = key[None, :, None, None].expand(22, 17, 4, 64)
    value = -key - 1
    other_stream = torch.cuda.Stream(device=pool.device)
    pointer = pool.untyped_storage().data_ptr()
    allocated_bytes = torch.cuda.memory_allocated(0)

    allocator.create(KVReservation(a, 2))
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 2, 1, 0)
    allocator.append(a, key[:, :16], value[:, :16])
    before = allocator.capacity()
    copy = torch.Tensor.copy_
    calls = 0

    def failing_copy(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return copy(destination, source)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", failing_copy)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            allocator.append(a, key[:, 16:], value[:, 16:])
    assert allocator.capacity() == before
    assert allocator.cached_length(a) == 16
    assert len(allocator.block_table(a)) == 1
    allocator.append(a, key[:, 16:], value[:, 16:])
    assert allocator.cached_length(a) == 17
    assert len(allocator.block_table(a)) == 2
    assert allocator.capacity() == CapacityCounts(2, 0, 2, 2, 1, 17)
    actual_key, actual_value = allocator.read_block(a, 1)
    assert actual_key.shape == actual_value.shape == (22, 1, 4, 64)
    assert actual_key.untyped_storage().data_ptr() == pointer
    assert actual_value.untyped_storage().data_ptr() == pointer
    assert torch.cuda.memory_allocated(0) == allocated_bytes
    torch.testing.assert_close(actual_key, key[:, 16:], atol=0, rtol=0)
    torch.testing.assert_close(actual_value, value[:, 16:], atol=0, rtol=0)

    with torch.cuda.stream(other_stream):
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            allocator.capacity()
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            allocator.append(a, key[:, :1], value[:, :1])
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            allocator.read_block(a, 0)
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            allocator.release_all(a)

    allocator.release_all(a)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)
    with pytest.raises(KeyError):
        allocator.release_all(a)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)
    allocator.create(KVReservation(b, 1))
    allocator.append(b, key[:, :1], value[:, :1])
    actual_key, actual_value = allocator.read_block(b, 0)
    assert actual_key.shape == actual_value.shape == (22, 1, 4, 64)
    assert pool.untyped_storage().data_ptr() == pointer
    assert torch.cuda.memory_allocated(0) == allocated_bytes
    torch.testing.assert_close(actual_key, key[:, :1], atol=0, rtol=0)
    torch.testing.assert_close(actual_value, value[:, :1], atol=0, rtol=0)
    allocator.release_all(b)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)
    torch.cuda.synchronize(0)
