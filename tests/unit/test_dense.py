from types import SimpleNamespace

import pytest
import torch
from torch import Tensor
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from pagedcore.dense import (
    apply_rope,
    dense_attention,
    final_logits,
    output_projection,
    project_qkv,
    repeat_kv,
    rms_norm,
    swiglu,
)
from pagedcore.weights import LayerWeights, TinyLlamaWeights

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def components() -> SimpleNamespace:
    # Synthetic weights use the pinned head geometry, with a smaller MLP/vocabulary.
    config = LlamaConfig(
        hidden_size=2048,
        intermediate_size=32,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    attention = LlamaAttention(config, layer_idx=0).half().eval()
    mlp = LlamaMLP(config).half().eval()
    norm = LlamaRMSNorm(2048, eps=1e-5).half().eval()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 2048))
    weights = LayerWeights(
        input_norm=norm.weight.detach(),
        q_proj=attention.q_proj.weight.detach(),
        k_proj=attention.k_proj.weight.detach(),
        v_proj=attention.v_proj.weight.detach(),
        o_proj=attention.o_proj.weight.detach(),
        post_attention_norm=norm.weight.detach(),
        gate_proj=mlp.gate_proj.weight.detach(),
        up_proj=mlp.up_proj.weight.detach(),
        down_proj=mlp.down_proj.weight.detach(),
    )
    return SimpleNamespace(
        attention=attention,
        mlp=mlp,
        norm=norm,
        rope=LlamaRotaryEmbedding(config),
        weights=weights,
    )


@pytest.mark.parametrize("scale", [0.0, 1e-4, 1.0, 1000.0])
def test_rms_norm_matches_hf_with_fp32_accumulation(
    components: SimpleNamespace, scale: float
) -> None:
    hidden = (torch.randn(2, 3, 2048) * scale).half()
    before = hidden.clone()
    with torch.inference_mode():
        result = rms_norm(hidden, components.weights.input_norm)
        expected = components.norm(hidden)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()
    assert torch.equal(hidden, before)


def test_qkv_projection_layout_and_weight_orientation(components: SimpleNamespace) -> None:
    hidden = torch.zeros(2, 3, 2048, dtype=torch.float16)
    hidden[0, :, 7] = 1
    hidden[1, :, 91] = -2
    query, key, value = project_qkv(hidden, components.weights)
    for result, weight, heads in (
        (query, components.weights.q_proj, 32),
        (key, components.weights.k_proj, 4),
        (value, components.weights.v_proj, 4),
    ):
        assert result.shape == (2, heads, 3, 64)
        assert result.dtype == torch.float16
        for batch, column, scale in ((0, 7, 1), (1, 91, -2)):
            expected = (weight[:, column] * scale).reshape(heads, 1, 64).expand(-1, 3, -1)
            torch.testing.assert_close(result[batch], expected, atol=0, rtol=0)


@pytest.mark.parametrize("positions", [[[0, 15, 16, 17]], [[0, 1, 2], [511, 1023, 2047]]])
def test_rope_matches_hf_at_absolute_positions(
    components: SimpleNamespace, positions: list[list[int]]
) -> None:
    ids = torch.tensor(positions)
    query = torch.randn(2, 32, ids.shape[1], 64).half()
    key = torch.randn(2, 4, ids.shape[1], 64).half()
    original_query, original_key = query.clone(), key.clone()
    cosine, sine = components.rope(query, ids)
    expected = apply_rotary_pos_emb(query, key, cosine, sine)
    actual = apply_rope(query, key, ids)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
        assert result.dtype == torch.float16
    assert torch.equal(query, original_query)
    assert torch.equal(key, original_key)


def test_rope_rotates_half_pairs_and_preserves_absolute_decode_position() -> None:
    query = torch.zeros(1, 32, 2, 64, dtype=torch.float16)
    query[..., 0] = 1
    key = query[:, :4].clone()
    rotated, _ = apply_rope(query, key, torch.tensor([[0, 17]]))
    torch.testing.assert_close(rotated[:, :, 0], query[:, :, 0], atol=0, rtol=0)
    expected = torch.zeros(64, dtype=torch.float16)
    expected[0] = torch.tensor(17.0).cos().half()
    expected[32] = torch.tensor(17.0).sin().half()
    torch.testing.assert_close(rotated[0, 0, 1], expected, atol=0, rtol=0)
    decode, _ = apply_rope(query[:, :, 1:], key[:, :, 1:], torch.tensor([[17]]))
    torch.testing.assert_close(decode, rotated[:, :, 1:], atol=0, rtol=0)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0, 1]),
        torch.tensor([[0.0, 1.0]]),
        torch.zeros(3, 2).long(),
        torch.zeros(1, 1).long(),
    ],
)
def test_rope_rejects_positions_that_would_broadcast_incorrectly(positions: Tensor) -> None:
    with pytest.raises(ValueError, match="position_ids"):
        apply_rope(torch.zeros(2, 32, 2, 64), torch.zeros(2, 4, 2, 64), positions)


def test_grouped_query_mapping_repeats_consecutive_kv_heads() -> None:
    kv = torch.arange(4, dtype=torch.float16).view(1, 4, 1, 1).expand(2, 4, 3, 64)
    expanded = repeat_kv(kv)
    assert expanded.shape == (2, 32, 3, 64)
    for head in range(32):
        assert torch.equal(expanded[:, head], kv[:, head // 8])


@pytest.mark.parametrize("decode", [False, True])
def test_dense_attention_causality_and_gqa_with_uniform_scores(decode: bool) -> None:
    keys = 4
    queries = 1 if decode else keys
    query = torch.zeros(2, 32, queries, 64, dtype=torch.float16)
    key = torch.zeros(2, 4, keys, 64, dtype=torch.float16)
    value = torch.zeros_like(key)
    for batch in range(2):
        for head in range(4):
            value[batch, head] = (100 * batch + 10 * head + torch.arange(keys))[:, None]
    positions = torch.tensor([3]) if decode else torch.arange(keys)
    mask = (torch.arange(keys)[None, :] <= positions[:, None])[None, None]
    actual = dense_attention(query, key, value, mask)
    for batch in range(2):
        for head in range(32):
            for token, position in enumerate(positions):
                expected = value[batch, head // 8, : position + 1].float().mean(dim=0).half()
                torch.testing.assert_close(actual[batch, head, token], expected, atol=0, rtol=0)


@pytest.mark.parametrize("decode", [False, True])
def test_dense_attention_and_output_projection_match_hf(
    components: SimpleNamespace, decode: bool
) -> None:
    tokens = 1 if decode else 5
    positions = torch.tensor([[17]]) if decode else torch.arange(tokens)[None]
    hidden = torch.randn(2, tokens, 2048, dtype=torch.float16)
    mask = torch.ones(1, 1, tokens, tokens, dtype=torch.bool).tril()
    additive_mask = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill(
        ~mask, torch.finfo(hidden.dtype).min
    )
    with torch.inference_mode():
        query, key, value = project_qkv(hidden, components.weights)
        query, key = apply_rope(query, key, positions)
        actual = output_projection(dense_attention(query, key, value, mask), components.weights)
        expected, _ = components.attention(
            hidden,
            position_embeddings=components.rope(hidden, positions),
            attention_mask=additive_mask,
        )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert actual.shape == hidden.shape
    assert actual.dtype == torch.float16


def test_dense_attention_rejects_nonboolean_mask() -> None:
    with pytest.raises(ValueError, match="causal_mask"):
        dense_attention(
            torch.zeros(1, 32, 1, 64),
            torch.zeros(1, 4, 1, 64),
            torch.zeros(1, 4, 1, 64),
            torch.zeros(1, 1, 1, 1),
        )


def test_dense_attention_softmax_is_stable_for_large_scores() -> None:
    query = torch.full((1, 32, 1, 64), 10.0, dtype=torch.float16)
    key = torch.ones(1, 4, 2, 64, dtype=torch.float16)
    key[:, :, 1] = 0
    value = torch.ones_like(key)
    value[:, :, 1] = -100
    actual = dense_attention(query, key, value, torch.ones(1, 1, 1, 2, dtype=torch.bool))
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)


def test_output_projection_merges_heads_in_order(components: SimpleNamespace) -> None:
    attention = torch.zeros(2, 32, 3, 64, dtype=torch.float16)
    attention[:, 7, :, 11] = 1
    result = output_projection(attention, components.weights)
    expected = components.weights.o_proj[:, 7 * 64 + 11].expand(2, 3, -1)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_swiglu_matches_hf(components: SimpleNamespace) -> None:
    hidden = torch.randn(2, 3, 2048, dtype=torch.float16)
    with torch.inference_mode():
        actual = swiglu(hidden, components.weights)
        expected = components.mlp(hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.float16


def test_final_norm_and_lm_head_use_the_untied_head(components: SimpleNamespace) -> None:
    embedding = torch.zeros(11, 2048, dtype=torch.float16)
    head = torch.randn_like(embedding) * 0.01
    weights = TinyLlamaWeights(embedding, (), components.norm.weight.detach(), head)
    hidden = torch.randn(2, 3, 2048, dtype=torch.float16)
    with torch.inference_mode():
        expected_hidden = components.norm(hidden)
        expected = torch.stack(
            [(expected_hidden.float() * row.float()).sum(dim=-1) for row in head], dim=-1
        ).half()
        actual = final_logits(hidden, weights)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    assert actual.shape == (2, 3, 11)
    assert actual.dtype == torch.float16
    assert torch.equal(embedding, torch.zeros_like(embedding))
