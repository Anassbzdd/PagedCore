from unittest.mock import Mock

import pytest
import torch

from pagedcore.config import PagedCoreConfig
from pagedcore.kv_cache import allocate_kv_pool

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
