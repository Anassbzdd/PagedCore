from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import pagedcore.decoder as decoder_module
from pagedcore.decoder import ContiguousCache, GreedyResult, decode, generate, prefill
from pagedcore.dense import apply_rope, project_qkv, rms_norm
from pagedcore.engine_types import FinishReason
from pagedcore.weights import LayerWeights, TinyLlamaWeights

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def decoder() -> SimpleNamespace:
    # Keep the pinned head geometry; reduced depth/MLP/vocabulary are synthetic.
    config = LlamaConfig(
        vocab_size=31,
        hidden_size=2048,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(config).half().eval()
    # .half() also rounds buffers; the reference requires FP32 RoPE frequencies.
    model.model.rotary_emb = LlamaRotaryEmbedding(config)
    layers = tuple(
        LayerWeights(
            input_norm=layer.input_layernorm.weight.detach(),
            q_proj=layer.self_attn.q_proj.weight.detach(),
            k_proj=layer.self_attn.k_proj.weight.detach(),
            v_proj=layer.self_attn.v_proj.weight.detach(),
            o_proj=layer.self_attn.o_proj.weight.detach(),
            post_attention_norm=layer.post_attention_layernorm.weight.detach(),
            gate_proj=layer.mlp.gate_proj.weight.detach(),
            up_proj=layer.mlp.up_proj.weight.detach(),
            down_proj=layer.mlp.down_proj.weight.detach(),
        )
        for layer in model.model.layers
    )
    weights = TinyLlamaWeights(
        model.model.embed_tokens.weight.detach(),
        layers,
        model.model.norm.weight.detach(),
        model.lm_head.weight.detach(),
    )
    return SimpleNamespace(model=model, weights=weights)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 49])
def test_prefill_matches_synthetic_hf_and_caches_prompt(
    decoder: SimpleNamespace, tokens: int
) -> None:
    ids = torch.arange(tokens)[None] % 31
    logits, cache = prefill(ids, decoder.weights)
    with torch.inference_mode():
        expected = decoder.model(ids, use_cache=False).logits
    torch.testing.assert_close(logits, expected, atol=5e-3, rtol=5e-3)
    assert logits.shape == (1, tokens, 31)
    assert logits.dtype == torch.float16
    assert not logits.requires_grad
    assert cache.length == tokens
    assert len(cache.layers) == 2
    for key, value in cache.layers:
        assert key.shape == value.shape == (1, 4, tokens, 64)
        assert key.dtype == value.dtype == torch.float16
        assert not key.requires_grad and not value.requires_grad
    # Verify K is rotated before storage and V is not rotated.
    hidden = decoder.weights.embedding[ids]
    _, key, value = project_qkv(
        rms_norm(hidden, decoder.weights.layers[0].input_norm), decoder.weights.layers[0]
    )
    query = torch.zeros(1, 32, tokens, 64, dtype=torch.float16)
    _, rotated = apply_rope(query, key, torch.arange(tokens)[None])
    torch.testing.assert_close(cache.layers[0][0], rotated, atol=0, rtol=0)
    torch.testing.assert_close(cache.layers[0][1], value, atol=0, rtol=0)


@pytest.mark.parametrize(
    "tokens,maximum,ignore_eos",
    [(1, 1, False), (15, 8, False), (16, 8, True), (17, 8, True), (32, 8, True), (49, 8, True)],
)
def test_greedy_generation_matches_synthetic_hf(
    decoder: SimpleNamespace,
    assert_decoder_greedy_parity: Callable[..., GreedyResult],
    tokens: int,
    maximum: int,
    ignore_eos: bool,
) -> None:
    ids = torch.arange(tokens)[None] % 31
    actual = assert_decoder_greedy_parity(decoder.model, decoder.weights, ids, maximum, ignore_eos)
    if ignore_eos:
        assert len(actual.emitted_ids) == maximum
        assert actual.finish_reason is FinishReason.LENGTH
    assert torch.equal(ids, torch.arange(tokens)[None] % 31)
    # A fresh invocation cannot inherit the previous request's cache or counters.
    assert generate(ids, decoder.weights, max_new_tokens=maximum, ignore_eos=ignore_eos) == actual


@pytest.mark.parametrize("maximum", [0, 257, True, 1.0, "1"])
def test_generation_rejects_invalid_output_limit(decoder: SimpleNamespace, maximum: Any) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(torch.tensor([[1]]), decoder.weights, max_new_tokens=maximum)


@pytest.mark.parametrize("ignore_eos", [0, None, "false"])
def test_generation_rejects_nonboolean_eos_policy(
    decoder: SimpleNamespace, ignore_eos: Any
) -> None:
    with pytest.raises(ValueError, match="ignore_eos"):
        generate(torch.tensor([[1]]), decoder.weights, ignore_eos=ignore_eos)


def test_generation_rejects_context_overflow_before_prefill(
    decoder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_prefill(*args: Any) -> None:
        pytest.fail("context overflow must fail before allocating a prompt cache")

    monkeypatch.setattr(decoder_module, "prefill", unexpected_prefill)
    with pytest.raises(ValueError, match="2048"):
        generate(torch.ones(1, 2048, dtype=torch.int64), decoder.weights, max_new_tokens=1)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 49])
def test_all_layers_match_synthetic_hf_prefill_and_cached_decode(
    decoder: SimpleNamespace,
    assert_decoder_layer_parity: Callable[[Any, TinyLlamaWeights, Tensor], None],
    tokens: int,
) -> None:
    assert_decoder_layer_parity(decoder.model, decoder.weights, torch.arange(tokens)[None] % 31)


def test_layer_comparison_identifies_owned_attention_mismatch(
    decoder: SimpleNamespace,
    assert_decoder_layer_parity: Callable[[Any, TinyLlamaWeights, Tensor], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = decoder_module.output_projection

    def corrupted_attention(value: Tensor, layer: LayerWeights) -> Tensor:
        result = original(value, layer)
        return result + 0.125 if layer is decoder.weights.layers[1] else result

    with monkeypatch.context() as patch:
        patch.setattr(decoder_module, "output_projection", corrupted_attention)
        with pytest.raises(AssertionError, match=r"prefill\.layer\.1\.attention"):
            assert_decoder_layer_parity(decoder.model, decoder.weights, torch.tensor([[1, 3]]))


@pytest.mark.parametrize("prompt_tokens", [1, 15, 16, 17, 49])
def test_repeated_decode_matches_full_prefix_and_preserves_cache(
    decoder: SimpleNamespace, prompt_tokens: int
) -> None:
    ids = torch.arange(prompt_tokens)[None] % 31
    logits, cache = prefill(ids, decoder.weights)
    for step in range(3):
        # Forward orchestration consumes a supplied token; stopping belongs to the caller.
        token = torch.tensor([[2 if step == 0 else 7 + step]])
        ids = torch.cat((ids, token), dim=1)
        previous = cache
        snapshots = tuple((key.clone(), value.clone()) for key, value in cache.layers)
        logits, cache = decode(token, previous, decoder.weights)
        full_logits, full_cache = prefill(ids, decoder.weights)
        torch.testing.assert_close(logits, full_logits[:, -1:], atol=5e-3, rtol=5e-3)
        assert logits.shape == (1, 1, 31)
        assert cache.length == prompt_tokens + step + 1
        assert previous.length == cache.length - 1
        for before, snapshot, after, full in zip(
            previous.layers, snapshots, cache.layers, full_cache.layers, strict=True
        ):
            for old, saved, new, expected in zip(before, snapshot, after, full, strict=True):
                assert torch.equal(old, saved)
                assert torch.equal(new[:, :, :-1], old)
                torch.testing.assert_close(new, expected, atol=2e-3, rtol=2e-3)


def test_prefill_is_causal_and_requests_have_independent_caches(decoder: SimpleNamespace) -> None:
    first = torch.tensor([[1, 3, 5, 7]])
    second = torch.tensor([[1, 3, 9, 11]])
    logits_a, cache_a = prefill(first, decoder.weights)
    logits_b, cache_b = prefill(second, decoder.weights)
    torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2], atol=0, rtol=0)
    for layer_a, layer_b in zip(cache_a.layers, cache_b.layers, strict=True):
        for a, b in zip(layer_a, layer_b, strict=True):
            assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    next_a, _ = decode(torch.tensor([[13]]), cache_a, decoder.weights)
    decode(torch.tensor([[17]]), cache_b, decoder.weights)
    repeated_a, _ = decode(torch.tensor([[13]]), cache_a, decoder.weights)
    torch.testing.assert_close(next_a, repeated_a, atol=0, rtol=0)


def test_decode_at_last_context_position_and_overflow_is_atomic(decoder: SimpleNamespace) -> None:
    _, initial = prefill(torch.tensor([[1]]), decoder.weights)
    cache = ContiguousCache(
        tuple(
            (key.repeat(1, 1, 2047, 1), value.repeat(1, 1, 2047, 1))
            for key, value in initial.layers
        )
    )
    logits, full = decode(torch.tensor([[3]]), cache, decoder.weights)
    assert torch.isfinite(logits).all()
    assert full.length == 2048
    assert cache.length == 2047
    with pytest.raises(ValueError, match="2048"):
        decode(torch.tensor([[5]]), full, decoder.weights)
    assert full.length == 2048


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([1]),
        torch.tensor([[1.0]]),
        torch.ones(2, 1).long(),
        torch.empty(1, 0).long(),
        torch.ones(1, 2049).long(),
    ],
)
def test_prefill_rejects_invalid_input_shape_dtype_or_context(
    decoder: SimpleNamespace, ids: Tensor
) -> None:
    with pytest.raises(ValueError):
        prefill(ids, decoder.weights)


@pytest.mark.parametrize(
    "ids",
    [torch.tensor([1]), torch.tensor([[1, 2]]), torch.tensor([[1.0]]), torch.ones(2, 1).long()],
)
def test_decode_requires_one_int64_token(decoder: SimpleNamespace, ids: Tensor) -> None:
    _, cache = prefill(torch.tensor([[1]]), decoder.weights)
    with pytest.raises(ValueError):
        decode(ids, cache, decoder.weights)
    assert cache.length == 1


def test_decode_rejects_cache_with_wrong_layer_count(decoder: SimpleNamespace) -> None:
    _, cache = prefill(torch.tensor([[1]]), decoder.weights)
    with pytest.raises(ValueError, match="layer count"):
        decode(torch.tensor([[3]]), replace(cache, layers=cache.layers[:1]), decoder.weights)


def test_failed_decode_preserves_all_layers(
    decoder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cache = prefill(torch.tensor([[1, 3]]), decoder.weights)
    snapshots = tuple((key.clone(), value.clone()) for key, value in cache.layers)
    attention = decoder_module.dense_attention
    calls = 0

    def fail_second_layer(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected attention failure")
        return attention(query, key, value, mask)

    with monkeypatch.context() as patch:
        patch.setattr(decoder_module, "dense_attention", fail_second_layer)
        with pytest.raises(RuntimeError, match="injected attention failure"):
            decode(torch.tensor([[5]]), cache, decoder.weights)
    assert calls == 2
    assert cache.length == 2
    for layer, snapshot in zip(cache.layers, snapshots, strict=True):
        for tensor, saved in zip(layer, snapshot, strict=True):
            assert torch.equal(tensor, saved)
    logits, updated = decode(torch.tensor([[5]]), cache, decoder.weights)
    expected, _ = prefill(torch.tensor([[1, 3, 5]]), decoder.weights)
    torch.testing.assert_close(logits, expected[:, -1:], atol=5e-3, rtol=5e-3)
    assert updated.length == 3


def test_forward_rejects_device_mismatch(decoder: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="same device"):
        prefill(torch.ones(1, 1, dtype=torch.int64, device="meta"), decoder.weights)
    key = torch.empty(1, 4, 1, 64, dtype=torch.float16, device="meta")
    cache = ContiguousCache(((key, key),) * 2)
    with pytest.raises(ValueError, match="same device"):
        decode(torch.tensor([[1]]), cache, decoder.weights)


@pytest.mark.parametrize("fault", ["empty", "rank", "length", "heads", "dtype", "device"])
def test_cache_rejects_inconsistent_storage(fault: str) -> None:
    key = torch.zeros(1, 4, 2, 64, dtype=torch.float16)
    bad = {
        "empty": (),
        "rank": ((key[0], key[0]),),
        "length": ((key, key[:, :, :1]),),
        "heads": ((key[:, :2], key[:, :2]),),
        "dtype": ((key, key.float()),),
        "device": ((key, key.to("meta")),),
    }[fault]
    with pytest.raises(ValueError, match="cache"):
        ContiguousCache(bad)
