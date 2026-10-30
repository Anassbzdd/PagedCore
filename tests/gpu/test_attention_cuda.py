import pytest
import torch
from torch import Tensor

from pagedcore.attention import paged_attention
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math

pytestmark = pytest.mark.gpu


@pytest.fixture
def t4_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert "t4" in torch.cuda.get_device_name(0).casefold(), "requires an NVIDIA T4"
    return torch.device("cuda:0")


def _reference(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    # Full-history FP64 CPU math is an independent oracle, permitted only in tests.
    query, key, value = (tensor.detach().cpu().double() for tensor in (query, key, value))
    heads = []
    for head in range(32):
        scores = query[0, head, 0] @ key[:, head // 8].T / 8
        heads.append(scores.softmax(dim=-1) @ value[:, head // 8])
    return torch.stack(heads).reshape(1, 32, 1, 64).half()


def _check_attention(query: Tensor, key: Tensor, value: Tensor, *, layer: int) -> None:
    assert query.is_cuda and key.is_cuda and value.is_cuda
    tokens = key.shape[1]
    blocks = (tokens + 15) // 16
    pool = torch.full(
        (22, 2, blocks + 3, 16, 4, 64), torch.nan, dtype=torch.float16, device=query.device
    )
    allocator = KVAllocator(pool)
    pointer = pool.untyped_storage().data_ptr()
    for name in ("first", "retained", "last"):
        request = RequestId(name)
        allocator.create(KVReservation(request, 1))
        sentinel = torch.full((22, 16, 4, 64), -100.0, dtype=torch.float16, device=pool.device)
        allocator.append(request, sentinel, -sentinel)
    allocator.release_all(RequestId("first"))
    allocator.release_all(RequestId("last"))
    request = RequestId("attention")
    allocator.create(KVReservation(request, blocks))
    allocator.append(request, key, value)
    table = allocator.block_table(request)
    if blocks > 1:
        assert table != tuple(range(table[0], table[0] + blocks))
    before = pool.clone()
    counts = allocator.capacity()
    expected = _reference(query, key[layer], value[layer])

    with reference_math():
        actual = paged_attention(query, allocator, request, layer_index=layer)
        with torch.autocast("cuda", dtype=torch.float16):
            autocast_actual = paged_attention(query, allocator, request, layer_index=layer)
    torch.cuda.synchronize(pool.device)

    assert actual.device == query.device and actual.dtype == torch.float16
    assert actual.shape == query.shape and not actual.requires_grad
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(autocast_actual, actual, atol=0, rtol=0)
    torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
    assert allocator.block_table(request) == table
    assert allocator.cached_length(request) == tokens
    assert allocator.capacity() == counts
    allocator.release_all(request)
    allocator.release_all(RequestId("retained"))
    assert allocator.capacity() == CapacityCounts(blocks + 3, blocks + 3, 0, 0, 0, 0)
    assert pool.untyped_storage().data_ptr() == pointer


def test_paged_attention_boundaries_on_t4(t4_device: torch.device) -> None:
    for tokens, layer in ((1, 0), (15, 21), (16, 7), (17, 0), (49, 21), (2048, 11)):
        key = torch.randn(22, tokens, 4, 64, dtype=torch.float16, device=t4_device)
        value = torch.randn_like(key)
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16, device=t4_device)
        _check_attention(query[..., ::2].requires_grad_(), key, value, layer=layer)

    key = torch.zeros(22, 33, 4, 64, dtype=torch.float16, device=t4_device)
    values = (
        100 * torch.arange(4, device=t4_device)[None, :]
        + torch.arange(33, device=t4_device)[:, None]
    )
    value = values[None, :, :, None].expand(22, 33, 4, 64).half()
    _check_attention(
        torch.zeros(1, 32, 1, 64, dtype=torch.float16, device=t4_device), key, value, layer=21
    )


def test_paged_attention_large_scores_on_t4(t4_device: torch.device) -> None:
    for reverse in (False, True):
        for query_scale in (1.0, 1000.0):
            key = torch.full((22, 49, 4, 64), -1000.0, dtype=torch.float16, device=t4_device)
            key[:, 16:32] = 0
            key[:, 32:] = 1000
            value = torch.randn_like(key)
            if reverse:
                key, value = key.flip(1), value.flip(1)
            query = torch.full((1, 32, 1, 64), query_scale, dtype=torch.float16, device=t4_device)
            _check_attention(query, key, value, layer=0)
