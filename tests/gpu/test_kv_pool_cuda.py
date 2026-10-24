import pytest
import torch

from pagedcore.config import PagedCoreConfig
from pagedcore.kv_cache import allocate_kv_pool

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
