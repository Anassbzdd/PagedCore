import math
import struct
from dataclasses import replace

import pytest
import torch

from pagedcore import decoder
from pagedcore.dense import final_logits, swiglu
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.weights import LayerWeights, TinyLlamaWeights

pytestmark = pytest.mark.unit

WIDTH = 2048
V_ENTRIES = [(0, 0, 0.125), (64, 1, -0.125), (128, 65, 0.25), (192, 0, -0.0625), (192, 1, 0.125)]
O_ENTRIES = [(0, 0, 0.125), (17, 9 * 64, 0.25), (65, 25 * 64, -0.125)]
GATE_ENTRIES = [(0, 0, 0.0625), (0, 17, 0.03125), (1, 65, -0.0625), (2, 1, 0.0625)]
UP_ENTRIES = [(0, 1, -0.03125), (1, 17, 0.0625), (2, 65, 0.03125), (2, 0, 0.015625)]
DOWN_ENTRIES = [(1, 0, 0.5), (17, 1, -0.25), (65, 2, 0.25), (0, 1, 0.125)]
HEAD_ENTRIES = [
    (0, 65, 1.5),
    (1, 0, -1.0),
    (2, 17, 0.25),
    (3, 1, 2.0),
    (4, 65, 1.0),
    (4, 0, -1.0),
    (5, 17, -0.5),
    (6, 0, 1.0),
    (6, 1, 0.125),
]


def binary16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def scalar_projection(row, entries, output_size):
    result = [0.0] * output_size
    for output in {item[0] for item in entries}:
        result[output] = binary16(
            math.fsum(row[column] * scale for target, column, scale in entries if target == output)
        )
    return result


def scalar_norm(row, scales):
    denominator = math.sqrt(math.fsum(x * x for x in row) / len(row) + 1e-5)
    return [
        binary16(binary16(x / denominator) * scale) for x, scale in zip(row, scales, strict=True)
    ]


def scalar_mlp(row):
    gate = scalar_projection(row, GATE_ENTRIES, 3)
    up = scalar_projection(row, UP_ENTRIES, 3)
    values = [binary16(binary16(g / (1 + math.exp(-g))) * u) for g, u in zip(gate, up, strict=True)]
    return scalar_projection(values, DOWN_ENTRIES, WIDTH)


def matrix(rows, columns, entries):
    result = torch.zeros((rows, columns), dtype=torch.float16)
    for output, source, scale in entries:
        result[output, source] = scale
    return result


@pytest.fixture(scope="module")
def sparse_model():
    zeros = torch.zeros((WIDTH, WIDTH), dtype=torch.float16)
    kv_zeros = torch.zeros((256, WIDTH), dtype=torch.float16)
    norm = torch.ones(WIDTH, dtype=torch.float16)
    blank = LayerWeights(
        norm,
        zeros,
        kv_zeros,
        kv_zeros,
        zeros,
        norm,
        torch.zeros((3, WIDTH), dtype=torch.float16),
        torch.zeros((3, WIDTH), dtype=torch.float16),
        torch.zeros((WIDTH, 3), dtype=torch.float16),
    )
    active = replace(
        blank,
        v_proj=matrix(256, WIDTH, V_ENTRIES),
        o_proj=matrix(WIDTH, WIDTH, O_ENTRIES),
        gate_proj=matrix(3, WIDTH, GATE_ENTRIES),
        up_proj=matrix(3, WIDTH, UP_ENTRIES),
        down_proj=matrix(WIDTH, 3, DOWN_ENTRIES),
    )
    embedding = torch.zeros((7, WIDTH), dtype=torch.float16)
    embedding[:, [0, 1, 65]] = torch.tensor(
        [
            [0, 0, 0],
            [0.5, -0.25, 0.125],
            [-0.25, 0.5, -0.5],
            [1, 0.25, -0.125],
            [-0.5, -0.25, 0.5],
            [0.25, 1, -0.5],
            [-1, 0.5, 0.25],
        ],
        dtype=torch.float16,
    )
    final_norm = norm.clone()
    final_norm[[0, 1, 17, 65]] = torch.tensor([0.5, 1.5, 0.75, 1.25], dtype=torch.float16)
    weights = TinyLlamaWeights(embedding, (blank,) * 22, final_norm, matrix(7, WIDTH, HEAD_ENTRIES))
    return weights, active


@pytest.mark.parametrize("scale", [0.0, 8.0, 32.0, 64.0])
def test_swiglu_against_scalar_gate_up_and_down_definition(sparse_model, scale):
    weights, active = sparse_model
    hidden = weights.embedding[[1, 2, 4]][None] * scale
    expected = torch.tensor([[scalar_mlp(row) for row in hidden[0].tolist()]], dtype=torch.float16)
    before = hidden.clone()
    actual = swiglu(hidden, active)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.equal(hidden, before)


@pytest.mark.parametrize("token_ids", [[0, 1, 2], [6, 3, 6]])
def test_final_normalization_and_untied_head_against_scalar_definition(sparse_model, token_ids):
    weights, _ = sparse_model
    hidden = weights.embedding[token_ids][None]
    expected = torch.tensor(
        [
            [
                scalar_projection(scalar_norm(row, weights.final_norm.tolist()), HEAD_ENTRIES, 7)
                for row in hidden[0].tolist()
            ]
        ],
        dtype=torch.float16,
    )
    assert not torch.equal(weights.lm_head, weights.embedding)
    torch.testing.assert_close(final_logits(hidden, weights), expected, atol=5e-3, rtol=5e-3)


def scalar_decoder(token_ids, weights):
    rows = [weights.embedding[token].tolist() for token in token_ids]
    values = [scalar_projection(scalar_norm(row, [1.0] * WIDTH), V_ENTRIES, 256) for row in rows]
    logits = []
    for position, row in enumerate(rows):
        probability = binary16(1 / (position + 1))
        averaged = [
            binary16(math.fsum(probability * v[channel] for v in values[: position + 1]))
            for channel in range(256)
        ]
        # Query head h reads KV head h//8; O selects three different groups.
        projected = [0.0] * WIDTH
        for output, input_column, scale in O_ENTRIES:
            head, channel = divmod(input_column, 64)
            projected[output] = binary16(averaged[(head // 8) * 64 + channel] * scale)
        after_attention = [binary16(a + b) for a, b in zip(row, projected, strict=True)]
        mlp = scalar_mlp(scalar_norm(after_attention, [1.0] * WIDTH))
        hidden = [binary16(a + b) for a, b in zip(after_attention, mlp, strict=True)]
        logits.append(
            scalar_projection(scalar_norm(hidden, weights.final_norm.tolist()), HEAD_ENTRIES, 7)
        )
    return torch.tensor([logits], dtype=torch.float16), torch.tensor(values, dtype=torch.float16)


@pytest.mark.parametrize("active_index", [0, 21])
@pytest.mark.parametrize("prompt_ids", [[1], [2, 1, 3]])
def test_dense_and_paged_block_composition_against_scalar_decoder(
    sparse_model, forbid_dense_kv_history, active_index, prompt_ids
):
    weights, active = sparse_model
    layers = list(weights.layers)
    layers[active_index] = active
    weights = replace(weights, layers=tuple(layers))
    prompt = torch.tensor([prompt_ids])
    expected, expected_values = scalar_decoder(prompt_ids, weights)
    dense_logits, dense_cache = decoder.prefill(prompt, weights)
    torch.testing.assert_close(dense_logits, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(
        dense_cache.layers[active_index][1][0].transpose(0, 1),
        expected_values.reshape(-1, 4, 64),
        atol=2e-3,
        rtol=2e-3,
    )
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator, request = KVAllocator(pool), RequestId("scalar-math")
    allocator.create(KVReservation(request, 1))
    try:
        paged_logits = decoder.prefill_paged(prompt, weights, allocator, request)
        torch.testing.assert_close(paged_logits, expected, atol=5e-3, rtol=5e-3)
        next_ids = [*prompt_ids, 4]
        expected_next, expected_values = scalar_decoder(next_ids, weights)
        # The cached steps attend to two or four positions, whose uniform FP32
        # probabilities are exactly representable; prefill retains its FP16 policy.
        dense_next, _ = decoder.decode(
            torch.tensor([[4]]), dense_cache, weights, fp32_attention=True
        )
        with forbid_dense_kv_history(pool):
            paged_next = decoder.decode_paged(torch.tensor([[4]]), weights, allocator, (request,))
        for actual in (dense_next, paged_next):
            torch.testing.assert_close(actual, expected_next[:, -1:], atol=5e-3, rtol=5e-3)
        key, value = allocator.read_block(request, 0)
        assert torch.count_nonzero(key) == 0
        torch.testing.assert_close(
            value[active_index], expected_values.reshape(-1, 4, 64), atol=2e-3, rtol=2e-3
        )
        assert allocator.cached_length(request) == len(next_ids)
    finally:
        allocator.release_all(request)
        assert allocator.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)
