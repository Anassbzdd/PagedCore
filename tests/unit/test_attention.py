import pytest
import torch
from torch import Tensor

from pagedcore.attention import paged_attention
from pagedcore.engine_types import KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit


def _cache(key: Tensor, value: Tensor) -> tuple[KVAllocator, RequestId, Tensor]:
    blocks = (key.shape[1] + 15) // 16
    pool = torch.full((22, 2, blocks + 1, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    request = RequestId("attention")
    allocator.create(KVReservation(request, blocks))
    allocator.append(request, key, value)
    return allocator, request, pool


def _reference(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    # Independent full-history FP64 oracle, permitted only in tests.
    heads = []
    for head in range(32):
        scores = query[0, head, 0].double() @ key[:, head // 8].double().T / 8
        heads.append(scores.softmax(dim=-1) @ value[:, head // 8].double())
    return torch.stack(heads).reshape(1, 32, 1, 64).half()


@pytest.mark.parametrize("tokens,layer", [(1, 0), (15, 21), (16, 7), (17, 0), (49, 21), (2048, 11)])
def test_paged_attention_matches_reference_at_cache_boundaries(tokens: int, layer: int) -> None:
    key = torch.randn(22, tokens, 4, 64).half()
    value = torch.randn_like(key)
    # A strided query must retain head ordering when grouped.
    query = torch.randn(1, 32, 1, 128).half()[..., ::2].requires_grad_()
    allocator, request, _ = _cache(key, value)
    before = allocator.capacity()

    actual = paged_attention(query, allocator, request, layer_index=layer)

    expected = _reference(query.detach(), key[layer], value[layer])
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert actual.shape == query.shape and actual.dtype == torch.float16
    assert actual.device == query.device and not actual.requires_grad
    assert torch.isfinite(actual).all()
    assert allocator.capacity() == before


def test_paged_attention_maps_eight_query_heads_to_each_kv_head_without_mutation() -> None:
    key = torch.zeros(22, 33, 4, 64, dtype=torch.float16)
    value = torch.zeros_like(key)
    for head in range(4):
        value[:, :, head] = (100 * head + torch.arange(33))[:, None]
    allocator, request, pool = _cache(key, value)
    before = pool.clone()
    table = allocator.block_table(request)
    counts = allocator.capacity()
    query = torch.zeros(1, 32, 1, 64, dtype=torch.float16)

    actual = paged_attention(query, allocator, request, layer_index=21)

    for head in range(32):
        torch.testing.assert_close(
            actual[0, head, 0], torch.full((64,), 100 * (head // 8) + 16).half(), atol=0, rtol=0
        )
    torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
    assert allocator.block_table(request) == table
    assert allocator.capacity() == counts
    allocator.release_all(request)
    assert allocator.capacity().owned_blocks == allocator.capacity().reserved_blocks == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("query_scale", [1.0, 1000.0])
def test_paged_attention_rescales_softmax_across_large_block_scores(
    reverse: bool, query_scale: float
) -> None:
    key = torch.full((22, 49, 4, 64), -1000.0, dtype=torch.float16)
    key[:, 16:32] = 0
    key[:, 32:] = 1000
    value = torch.randn_like(key)
    if reverse:
        key, value = key.flip(1), value.flip(1)
    allocator, request, _ = _cache(key, value)
    query = torch.full((1, 32, 1, 64), query_scale, dtype=torch.float16)

    actual = paged_attention(query, allocator, request, layer_index=0)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, _reference(query, key[0], value[0]), atol=2e-3, rtol=2e-3)


def test_paged_attention_keeps_fp32_math_inside_autocast() -> None:
    key = torch.randn(22, 49, 4, 64).half()
    value = torch.randn_like(key)
    query = torch.randn(1, 32, 1, 64).half()
    allocator, request, _ = _cache(key, value)
    expected = paged_attention(query, allocator, request, layer_index=0)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = paged_attention(query, allocator, request, layer_index=0)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("state", ["unknown", "empty", "released"])
def test_paged_attention_rejects_requests_without_cached_tokens(state: str) -> None:
    allocator = KVAllocator(torch.empty(22, 2, 1, 16, 4, 64, dtype=torch.float16))
    request = RequestId("missing")
    if state != "unknown":
        allocator.create(KVReservation(request, 1))
    if state == "released":
        allocator.release_all(request)
    before = allocator.capacity()

    with pytest.raises(ValueError if state == "empty" else KeyError):
        paged_attention(torch.zeros(1, 32, 1, 64).half(), allocator, request, layer_index=0)

    assert allocator.capacity() == before


@pytest.mark.parametrize(
    "query",
    [
        torch.zeros(2, 32, 1, 64).half(),
        torch.zeros(1, 31, 1, 64).half(),
        torch.zeros(1, 32, 2, 64).half(),
        torch.zeros(1, 32, 1, 63).half(),
        torch.zeros(1, 32, 1, 64),
    ],
)
def test_paged_attention_rejects_invalid_queries(query: Tensor) -> None:
    key = torch.zeros(22, 1, 4, 64).half()
    allocator, request, _ = _cache(key, key)
    with pytest.raises(ValueError, match="query must be FP16"):
        paged_attention(query, allocator, request, layer_index=0)


@pytest.mark.parametrize("layer", [-1, 22, True, 1.0])
def test_paged_attention_rejects_invalid_layers(layer: int) -> None:
    key = torch.zeros(22, 1, 4, 64).half()
    allocator, request, _ = _cache(key, key)
    with pytest.raises(ValueError, match="layer_index"):
        paged_attention(torch.zeros(1, 32, 1, 64).half(), allocator, request, layer_index=layer)


def test_paged_attention_rejects_query_on_a_different_device() -> None:
    key = torch.zeros(22, 1, 4, 64).half()
    allocator, request, _ = _cache(key, key)
    query = torch.empty(1, 32, 1, 64, dtype=torch.float16, device="meta")
    with pytest.raises(ValueError, match="same device"):
        paged_attention(query, allocator, request, layer_index=0)
