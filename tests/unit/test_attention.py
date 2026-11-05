import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from pagedcore.attention import _paged_value_sum, _paired_value_sum, paged_attention
from pagedcore.dense import dense_attention
from pagedcore.engine_types import KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parents[1] / "fixtures/t4_value_reduction.json").read_text())[
        "cases"
    ],
    ids=lambda case: f"layer-{case['layer']}",
)
def test_paired_reduction_matches_captured_t4_rounding(case, forbid_dense_kv_history) -> None:
    key = torch.zeros(22, 17, 4, 64, dtype=torch.float16)
    value = torch.zeros_like(key)
    layer, head, dimension = case["layer"], case["head"], case["dimension"]
    value[layer, :, head // 8, dimension] = torch.tensor(case["values"]).half()
    probabilities = torch.zeros(32, 17, dtype=torch.float16)
    probabilities[head] = torch.tensor(case["probabilities"]).half()
    allocator, request, pool = _cache(key, value)
    counts = allocator.capacity()
    with forbid_dense_kv_history(pool):
        actual = _paired_value_sum(probabilities.reshape(4, 8, 17), allocator, request, layer)
    assert actual[0, head, 0, dimension].item() == case["hf"]
    assert allocator.capacity() == counts


def test_guard_stops_tracking_kv_after_batched_weighted_value_reduction(
    forbid_dense_kv_history,
) -> None:
    pool = torch.ones(22, 2, 1, 16, 4, 64, dtype=torch.float16)
    probabilities = torch.full((4, 8, 16), 1 / 16)
    accumulator = torch.zeros(4, 8, 64)
    projection = torch.ones(5632, 64)

    with forbid_dense_kv_history(pool):
        page = pool[0, 1, 0].transpose(0, 1).float()
        attention = torch.baddbmm(accumulator, probabilities, page)
        output = torch.nn.functional.linear(attention[0, 0], projection)

    # The reduced attention output is an activation, not retained KV history.
    torch.testing.assert_close(output, torch.full_like(output, 64), atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [17, 31, 32, 33, 49, 2048])
def test_candidate_value_sum_keeps_page_storage_and_accounting(
    tokens, forbid_dense_kv_history
) -> None:
    generator = torch.Generator().manual_seed(20261102)
    key = torch.zeros(22, tokens, 4, 64, dtype=torch.float16)
    value = torch.randn(key.shape, generator=generator).half()
    probabilities = torch.randn(4, 8, tokens, generator=generator).softmax(-1).half()
    allocator, request, pool = _cache(key, value)
    before = pool.clone()
    counts = allocator.capacity()
    table = allocator.block_table(request)
    expected = probabilities.double() @ value[0].transpose(0, 1).double()

    with (
        forbid_dense_kv_history(pool),
        torch.inference_mode(),
        torch.autocast("cpu", enabled=False),
    ):
        actual = _paged_value_sum(probabilities, allocator, request, 0)

    torch.testing.assert_close(actual, expected.reshape_as(actual).half(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
    assert allocator.block_table(request) == table
    assert allocator.capacity() == counts
    allocator.release_all(request)
    assert allocator.capacity().owned_blocks == 0


def test_candidate_value_sum_rounds_only_after_cancelling_across_pages() -> None:
    key = torch.zeros(22, 17, 4, 64, dtype=torch.float16)
    value = torch.zeros_like(key)
    value[:, 0], value[:, 1], value[:, 16] = 2048, 1, -4096
    probabilities = torch.zeros(4, 8, 17, dtype=torch.float16)
    probabilities[..., 0], probabilities[..., 1], probabilities[..., 16] = 0.5, 0.25, 0.25
    allocator, request, _ = _cache(key, value)

    actual = _paged_value_sum(probabilities, allocator, request, 0)

    # Half-rounded page products would lose the quarter before cancellation.
    torch.testing.assert_close(actual, torch.full_like(actual, 0.25), atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 49, 2048])
def test_paged_attention_preserves_eager_score_and_probability_rounding(tokens: int) -> None:
    generator = torch.Generator().manual_seed(20261101)
    key = torch.randn(22, tokens, 4, 64, generator=generator).half() * 3
    value = torch.randn(key.shape, generator=generator).half() * 3
    query = torch.randn(1, 32, 1, 64, generator=generator).half() * 3
    allocator, request, _ = _cache(key, value)
    expected = dense_attention(
        query,
        key[0].transpose(0, 1)[None],
        value[0].transpose(0, 1)[None],
        torch.ones(1, 1, 1, tokens, dtype=torch.bool),
    )

    actual = paged_attention(query, allocator, request, layer_index=0, eager_rounding=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_actual = paged_attention(
            query, allocator, request, layer_index=0, eager_rounding=True
        )
    torch.testing.assert_close(autocast_actual, actual, atol=0, rtol=0)


def test_eager_probabilities_are_rounded_after_normalizing_all_pages() -> None:
    key = torch.zeros(22, 49, 4, 64, dtype=torch.float16)
    value = torch.full_like(key, 512)
    query = torch.zeros(1, 32, 1, 64, dtype=torch.float16)
    allocator, request, _ = _cache(key, value)
    expected = dense_attention(
        query,
        key[0].transpose(0, 1)[None],
        value[0].transpose(0, 1)[None],
        torch.ones(1, 1, 1, 49, dtype=torch.bool),
    )

    actual = paged_attention(query, allocator, request, layer_index=0, eager_rounding=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Retaining FP32 normalized probabilities would instead return exactly 512.
    assert not torch.equal(actual, value[0, :1].reshape(1, 4, 1, 64).repeat_interleave(8, 1))


def test_short_eager_attention_preserves_per_head_product_shapes() -> None:
    generator = torch.Generator().manual_seed(147)
    key = torch.randn(3, 4, 64, generator=generator).half()
    value = torch.randn(3, 4, 64, generator=generator).half()
    query = torch.randn(1, 32, 1, 64, generator=generator).half()
    allocator, request, _ = _cache(key[None].repeat(22, 1, 1, 1), value[None].repeat(22, 1, 1, 1))
    expected = dense_attention(
        query,
        key.transpose(0, 1)[None],
        value.transpose(0, 1)[None],
        torch.ones(1, 1, 1, 3, dtype=torch.bool),
    )

    actual = paged_attention(query, allocator, request, layer_index=0, eager_rounding=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
