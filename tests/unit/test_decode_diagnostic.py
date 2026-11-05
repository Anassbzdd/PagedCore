import json
from pathlib import Path

import pytest
import torch
from scripts import diagnose_paged_decode as diagnostic
from scripts.diagnose_paged_decode import AttentionScores, measurement

import pagedcore.attention as attention
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit

VALUE_CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/t4_value_reduction.json").read_text()
)["cases"]


@pytest.mark.parametrize("tokens", [16, 17, 18])
def test_page_value_replay_matches_boundary_scalar(
    tokens, forbid_dense_kv_history, monkeypatch
) -> None:
    case = VALUE_CASES[0] if tokens <= 17 else VALUE_CASES[-1]
    layer, head, channel = (case[name] for name in ("layer", "head", "dimension"))
    p = torch.zeros((4, 8, tokens), dtype=torch.float16)
    p[head // 8, head % 8] = torch.tensor(case["probabilities"][:tokens], dtype=torch.float16)
    key = torch.zeros((22, tokens, 4, 64), dtype=torch.float16)
    value = torch.zeros_like(key)
    value[layer, :, head // 8, channel] = torch.tensor(case["values"][:tokens], dtype=torch.float16)
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    cache = KVAllocator(pool)
    target, unrelated = RequestId("target"), RequestId("unrelated")
    cache.create(KVReservation(target, 2))
    cache.create(KVReservation(unrelated, 1))
    cache.append(target, key[:, :16], value[:, :16])
    sentinel = torch.full_like(key[:, :16], 100)
    cache.append(unrelated, -sentinel, sentinel)
    if tokens > 16:
        cache.append(target, key[:, 16:], value[:, 16:])
    table = (0,) if tokens == 16 else (0, 2)
    assert cache.block_table(target) == table
    assert cache.block_table(unrelated) == (1,)
    before, counts = pool.clone(), cache.capacity()
    expected = case["hf"]
    if tokens == 16:
        native_p = p.reshape(1, 32, 1, tokens)
        native_v = value[layer].transpose(0, 1).repeat_interleave(8, dim=0)
        expected = (native_p @ native_v)[0, head, 0, channel].item()

    monkeypatch.setattr(attention, "_multi_page_value_sum", diagnostic.fp64_page_value_sum)
    with forbid_dense_kv_history(pool):
        actual = attention._eager_value_sum(p, cache, target, layer)
        if tokens > 16:
            old = attention._striped_value_sum(
                p, cache, target, layer, lanes=4 if tokens == 17 else 2
            )
            assert old[0, head, 0, channel].item() != expected

    assert actual[0, head, 0, channel].item() == expected
    torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
    assert cache.capacity() == counts
    assert cache.cached_length(target) == tokens
    assert cache.block_table(target) == table
    cache.release_all(target)
    cache.release_all(unrelated)
    assert cache.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)


@pytest.mark.parametrize("difference", ["none", "probabilities", "values"])
def test_native_value_control_uses_owned_probabilities_and_checks_all_value_pages(
    difference, forbid_dense_kv_history, monkeypatch
) -> None:
    pool = torch.zeros((22, 2, 2, 16, 4, 64), dtype=torch.float16)
    cache = KVAllocator(pool)
    request = RequestId("target")
    cache.create(KVReservation(request, 2))
    key = torch.zeros((22, 17, 4, 64), dtype=torch.float16)
    value = key.clone()
    value[0, -1] = 1
    cache.append(request, key[:, :16], value[:, :16])
    query = torch.zeros((1, 32, 1, 64), dtype=torch.float16)
    hf_k, hf_v = (x[0].transpose(0, 1).unsqueeze(0).contiguous() for x in (key, value))
    hf_p = torch.full((1, 32, 1, 17), 1 / 17, dtype=torch.float16)
    if difference == "values":
        hf_v[..., -1, :] = 2
    elif difference == "probabilities":
        hf_p.fill_(0)
        hf_p[..., 0] = 1
    hf_output = (hf_p @ hf_v.repeat_interleave(8, dim=1)).transpose(1, 2)

    def decode(tokens, weights, allocator, requests):
        with allocator.append_decode(requests):
            for layer in range(22):
                allocator.write_decode_layer(request, layer, key[layer, -1], value[layer, -1])
            return diagnostic.decoder_module.paged_attention(
                query, allocator, request, layer_index=0, eager_rounding=True
            )

    monkeypatch.setattr(diagnostic, "decode_paged", decode)
    output, rows = diagnostic._trace_decode(
        torch.zeros((1, 1), dtype=torch.long),
        None,
        cache,
        (request,),
        [{0: (query, hf_k, hf_v, hf_output, hf_p.float(), hf_p)}],
        forbid_dense_kv_history(pool),
        "native_same_v",
    )

    result = rows[0]["layers"][0]
    assert result["p_equal"] == (difference != "probabilities")
    assert result["v_equal"] == (difference != "values")
    assert result["reducer"] == ("fallback" if difference == "values" else "native_same_v")
    assert result["out_equal"] == (difference == "none")
    assert torch.all(output == torch.tensor(1 / 17, dtype=torch.float16))


def test_observer_captures_probabilities_without_changing_softmax() -> None:
    scores = torch.randn(1, 32, 1, 17).half()
    expected = scores.softmax(-1, dtype=torch.float32)
    with AttentionScores() as observed:
        actual = scores.softmax(-1, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(observed.probabilities, expected, atol=0, rtol=0)
    actual.zero_()
    torch.testing.assert_close(observed.probabilities, expected, atol=0, rtol=0)


def test_observer_forces_reference_probabilities_without_replacing_measurement() -> None:
    scores = torch.zeros(4, 8, 17)
    reference = torch.zeros(1, 32, 1, 17)
    reference[..., 16] = 1
    with AttentionScores(reference) as observed:
        actual = scores.softmax(-1)
    torch.testing.assert_close(actual, reference.reshape_as(actual), atol=0, rtol=0)
    torch.testing.assert_close(observed.probabilities, scores.softmax(-1), atol=0, rtol=0)
    actual.zero_()
    assert reference[..., 16].eq(1).all()


def test_measurement_reports_first_coordinate_and_values() -> None:
    expected = torch.zeros(1, 2, 3)
    actual = expected.clone()
    actual[0, 1, 2] = 0.5
    result = measurement(actual, expected)
    assert result["first_unequal_index"] == (0, 1, 2)
    assert result["actual_at_first"] == 0.5
    assert result["expected_at_first"] == 0
    assert measurement(expected, expected)["first_unequal_index"] is None
