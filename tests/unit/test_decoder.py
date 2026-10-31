import json
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
from pagedcore.decoder import (
    ContiguousCache,
    GreedyResult,
    decode,
    decode_paged,
    generate,
    prefill,
    prefill_paged,
)
from pagedcore.dense import apply_rope, project_qkv, rms_norm
from pagedcore.engine_types import CapacityCounts, FinishReason, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
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


@pytest.fixture(scope="module")
def paged_weights(decoder: SimpleNamespace) -> TinyLlamaWeights:
    # Repeat synthetic layers to exercise the real pool's 22-layer layout.
    return replace(decoder.weights, layers=decoder.weights.layers * 11)


@pytest.mark.parametrize("lengths", [(15,), (1, 15, 16, 17)])
def test_paged_decode_batches_distinct_positions_and_appends_each_layer(
    paged_weights: TinyLlamaWeights, lengths: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = torch.full((22, 2, 16, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    requests = tuple(RequestId(str(row)) for row in range(len(lengths)))
    caches = {}
    for request, length in zip(requests, lengths, strict=True):
        allocator.create(KVReservation(request, (length + 2 + 15) // 16))
        ids = (torch.arange(length)[None] + length) % 31
        _, caches[request] = prefill(ids, paged_weights)
        prefill_paged(ids, paged_weights, allocator, request)
    pointer = pool.untyped_storage().data_ptr()
    counts = allocator.capacity()
    project = decoder_module.project_qkv
    batch_sizes = []

    def record_projection(hidden: Tensor, layer: LayerWeights) -> tuple[Tensor, Tensor, Tensor]:
        batch_sizes.append(hidden.shape[0])
        return project(hidden, layer)

    for step in range(2):
        # Request order can change between worker iterations.
        order = requests if step == 0 else requests[::-1]
        tokens = torch.tensor([[2 + step + int(request)] for request in order])
        expected = [
            decode(tokens[row : row + 1], caches[request], paged_weights)
            for row, request in enumerate(order)
        ]
        before = pool.clone()
        with monkeypatch.context() as patch:
            patch.setattr(decoder_module, "project_qkv", record_projection)
            logits = decode_paged(tokens, paged_weights, allocator, order)
        assert logits.shape == (len(order), 1, 31)
        assert logits.dtype == torch.float16 and not logits.requires_grad
        assert torch.isfinite(logits).all()
        for row, request in enumerate(order):
            expected_logits, cache = expected[row]
            old = caches[request]
            torch.testing.assert_close(logits[row : row + 1], expected_logits, atol=5e-3, rtol=5e-3)
            assert allocator.cached_length(request) == old.length + 1
            for logical in range(len(allocator.block_table(request))):
                key, value = allocator.read_block(request, logical)
                start = logical * 16
                end = start + key.shape[1]
                for layer, (expected_key, expected_value) in enumerate(cache.layers):
                    physical = allocator.block_table(request)[logical]
                    for kind, actual, target in (
                        (0, key[layer], expected_key),
                        (1, value[layer], expected_value),
                    ):
                        torch.testing.assert_close(
                            actual, target[0, :, start:end].transpose(0, 1), atol=5e-3, rtol=5e-3
                        )
                        valid_old = max(0, min(end, old.length) - start)
                        torch.testing.assert_close(
                            actual[:valid_old],
                            before[layer, kind, physical, :valid_old],
                            atol=0,
                            rtol=0,
                        )
            caches[request] = cache
        current = allocator.capacity()
        assert current.live_cached_token_slots == sum(lengths) + len(order) * (step + 1)
        assert current.reserved_blocks == counts.reserved_blocks
        assert current.active_slots == len(order)
        assert pool.untyped_storage().data_ptr() == pointer
    assert batch_sizes == [len(requests)] * 44
    for request in requests:
        allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(16, 16, 0, 0, 0, 0)


@pytest.mark.parametrize("fault", ["attention", "logits", "copy"])
def test_paged_decode_failure_preserves_entire_batch_and_allows_retry(
    paged_weights: TinyLlamaWeights, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    pool = torch.full((22, 2, 6, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    requests = (RequestId("first"), RequestId("second"))
    caches = []
    for request, length in zip(requests, (15, 16), strict=True):
        allocator.create(KVReservation(request, 2))
        ids = torch.arange(length)[None] % 31
        prefill_paged(ids, paged_weights, allocator, request)
        caches.append(prefill(ids, paged_weights)[1])
    before = pool.clone()
    counts = allocator.capacity()
    tables = [allocator.block_table(request) for request in requests]
    calls = 0
    original = torch.Tensor.copy_ if fault == "copy" else decoder_module.paged_attention

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == (4 if fault == "copy" else 3) or fault == "logits":
            raise RuntimeError("injected decode failure")
        return original(*args, **kwargs)

    tokens = torch.tensor([[3], [7]])
    with monkeypatch.context() as patch:
        if fault == "copy":
            patch.setattr(torch.Tensor, "copy_", fail)
        else:
            patch.setattr(
                decoder_module, "final_logits" if fault == "logits" else "paged_attention", fail
            )
        with pytest.raises(RuntimeError, match="injected decode failure"):
            decode_paged(tokens, paged_weights, allocator, requests)
    assert allocator.capacity() == counts
    for request, table, length in zip(requests, tables, (15, 16), strict=True):
        assert allocator.block_table(request) == table
        assert allocator.cached_length(request) == length
        torch.testing.assert_close(
            pool[:, :, table[0], :length], before[:, :, table[0], :length], atol=0, rtol=0
        )
    logits = decode_paged(tokens, paged_weights, allocator, requests)
    for row, cache in enumerate(caches):
        expected, _ = decode(tokens[row : row + 1], cache, paged_weights)
        torch.testing.assert_close(logits[row : row + 1], expected, atol=5e-3, rtol=5e-3)
    for request in requests:
        allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(6, 6, 0, 0, 0, 0)


def test_paged_decode_at_last_context_position(paged_weights: TinyLlamaWeights) -> None:
    allocator = KVAllocator(torch.zeros(22, 2, 128, 16, 4, 64, dtype=torch.float16))
    request = RequestId("last")
    allocator.create(KVReservation(request, 128))
    history = torch.zeros(22, 2047, 4, 64, dtype=torch.float16)
    allocator.append(request, history, history)
    logits = decode_paged(torch.tensor([[3]]), paged_weights, allocator, (request,))
    assert logits.shape == (1, 1, 31) and torch.isfinite(logits).all()
    assert allocator.cached_length(request) == 2048
    counts = allocator.capacity()
    with pytest.raises(ValueError, match="2047"):
        decode_paged(torch.tensor([[5]]), paged_weights, allocator, (request,))
    assert allocator.capacity() == counts


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "dtype",
        "device",
        "layers",
        "duplicate",
        "empty",
        "unknown",
        "released",
        "unfilled",
        "reservation",
        "context",
    ],
)
def test_paged_decode_rejects_invalid_batch_without_cache_changes(
    paged_weights: TinyLlamaWeights, fault: str
) -> None:
    pool = torch.zeros(22, 2, 129 if fault == "context" else 4, 16, 4, 64, dtype=torch.float16)
    allocator = KVAllocator(pool)
    first, second = RequestId("first"), RequestId("second")
    length = 2048 if fault == "context" else 16
    credits = 128 if fault == "context" else 1 if fault == "reservation" else 2
    allocator.create(KVReservation(first, credits))
    allocator.append(
        first, torch.zeros(22, length, 4, 64).half(), torch.zeros(22, length, 4, 64).half()
    )
    allocator.create(KVReservation(second, 1))
    if fault != "unfilled":
        allocator.append(second, torch.zeros(22, 1, 4, 64).half(), torch.zeros(22, 1, 4, 64).half())
    if fault == "released":
        allocator.release_all(second)
    tokens = torch.tensor([[1], [3]])
    requests = (first, second)
    weights = paged_weights
    if fault == "shape":
        tokens = tokens.T
    elif fault == "dtype":
        tokens = tokens.float()
    elif fault == "device":
        tokens = tokens.to("meta")
    elif fault == "layers":
        weights = replace(weights, layers=weights.layers[:2])
    elif fault == "duplicate":
        requests = (first, first)
    elif fault == "empty":
        requests, tokens = (), tokens[:0]
    elif fault == "unknown":
        requests = (first, RequestId("unknown"))
    before = pool.clone()
    counts = allocator.capacity()
    error = {"unknown": KeyError, "released": KeyError, "reservation": RuntimeError}.get(
        fault, ValueError
    )
    with pytest.raises(error):
        decode_paged(tokens, weights, allocator, requests)
    assert allocator.capacity() == counts
    torch.testing.assert_close(pool, before, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 49])
def test_paged_prefill_writes_absolute_positions_to_reused_fragmented_pages(
    paged_weights: TinyLlamaWeights, tokens: int
) -> None:
    pool = torch.full((22, 2, 6, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    for name in ("released_first", "retained", "released_last"):
        request = RequestId(name)
        allocator.create(KVReservation(request, 1))
        key = torch.full((22, 16, 4, 64), -100.0, dtype=torch.float16)
        allocator.append(request, key, -key)
    allocator.release_all(RequestId("released_first"))
    allocator.release_all(RequestId("released_last"))
    request = RequestId("prompt")
    blocks = (tokens + 15) // 16
    allocator.create(KVReservation(request, blocks))
    before = pool.clone()
    pointer = pool.untyped_storage().data_ptr()
    ids = torch.arange(tokens)[None] % 31
    expected_logits, expected_cache = prefill(ids, paged_weights)

    logits = prefill_paged(ids, paged_weights, allocator, request)

    torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
    assert logits.dtype == torch.float16 and not logits.requires_grad
    assert pool.untyped_storage().data_ptr() == pointer
    assert allocator.block_table(request) == (2, 0, 3, 4)[:blocks]
    assert allocator.cached_length(request) == tokens
    assert allocator.capacity() == CapacityCounts(
        6, 5 - blocks, blocks + 1, blocks + 1, 2, tokens + 16
    )
    for logical, physical in enumerate(allocator.block_table(request)):
        start = logical * 16
        valid = min(16, tokens - start)
        for layer, (key, value) in enumerate(expected_cache.layers):
            torch.testing.assert_close(
                pool[layer, 0, physical, :valid],
                key[0, :, start : start + valid].transpose(0, 1),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                pool[layer, 1, physical, :valid],
                value[0, :, start : start + valid].transpose(0, 1),
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(
            pool[:, :, physical, valid:], before[:, :, physical, valid:], equal_nan=True
        )
    for physical in set(range(6)) - set(allocator.block_table(request)):
        torch.testing.assert_close(pool[:, :, physical], before[:, :, physical], equal_nan=True)
    allocator.release_all(request)
    allocator.release_all(RequestId("retained"))
    assert allocator.capacity() == CapacityCounts(6, 6, 0, 0, 0, 0)


@pytest.mark.parametrize("state", ["unknown", "populated", "released"])
def test_paged_prefill_requires_an_admitted_empty_cache_before_running_model(
    paged_weights: TinyLlamaWeights, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    pool = torch.zeros((22, 2, 1, 16, 4, 64), dtype=torch.float16)
    allocator = KVAllocator(pool)
    request = RequestId("prompt")
    if state != "unknown":
        allocator.create(KVReservation(request, 1))
        key = torch.ones((22, 1, 4, 64), dtype=torch.float16)
        allocator.append(request, key, -key)
        if state == "released":
            allocator.release_all(request)
    before = allocator.capacity()
    snapshot = pool.clone()

    def unexpected_prefill(*args: Any) -> None:
        pytest.fail("invalid request cache must fail before running the model")

    monkeypatch.setattr(decoder_module, "prefill", unexpected_prefill)
    error = ValueError if state == "populated" else KeyError
    with pytest.raises(error):
        prefill_paged(torch.tensor([[1]]), paged_weights, allocator, request)
    assert allocator.capacity() == before
    assert torch.equal(pool, snapshot)


@pytest.mark.parametrize("fault", ["input", "layers", "reservation", "forward", "copy"])
def test_paged_prefill_failure_preserves_admission_and_allows_retry(
    paged_weights: TinyLlamaWeights, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    pool = torch.full((22, 2, 2, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    request = RequestId("prompt")
    allocator.create(KVReservation(request, 1 if fault == "reservation" else 2))
    before = allocator.capacity()
    ids = torch.arange(17)[None] % 31
    weights = paged_weights
    copy = torch.Tensor.copy_
    calls = 0

    def failing_copy(destination: Tensor, source: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return copy(destination, source)

    def failing_attention(*args: Any) -> Tensor:
        raise RuntimeError("injected attention failure")

    with monkeypatch.context() as patch:
        if fault == "input":
            ids = torch.empty((1, 0), dtype=torch.int64)
        elif fault == "layers":
            weights = replace(weights, layers=weights.layers[:2])
        elif fault == "forward":
            patch.setattr(decoder_module, "dense_attention", failing_attention)
        elif fault == "copy":
            patch.setattr(torch.Tensor, "copy_", failing_copy)
        with pytest.raises(ValueError if fault in ("input", "layers") else RuntimeError):
            prefill_paged(ids, weights, allocator, request)
    assert allocator.capacity() == before
    assert allocator.cached_length(request) == 0
    assert allocator.block_table(request) == ()
    if fault != "copy":
        assert torch.isnan(pool).all()
    # Even a failed V copy may leave unpublished K slots; retry must overwrite them.
    ids = torch.tensor([[7, 9, 11]])
    logits = prefill_paged(ids, paged_weights, allocator, request)
    expected, cache = prefill(ids, paged_weights)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    key, value = allocator.read_block(request, 0)
    for layer, (expected_key, expected_value) in enumerate(cache.layers):
        torch.testing.assert_close(key[layer], expected_key[0].transpose(0, 1), atol=0, rtol=0)
        torch.testing.assert_close(value[layer], expected_value[0].transpose(0, 1), atol=0, rtol=0)
    assert allocator.capacity() == replace(
        before, free_blocks=1, owned_blocks=1, live_cached_token_slots=3
    )
    allocator.release_all(request)
    assert allocator.capacity() == CapacityCounts(2, 2, 0, 0, 0, 0)


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
    assert_decoder_layer_parity: Callable[
        [Any, TinyLlamaWeights, Tensor], dict[str, dict[str, float]]
    ],
    record_testsuite_property: Callable[[str, object], None],
    tokens: int,
) -> None:
    errors = assert_decoder_layer_parity(
        decoder.model, decoder.weights, torch.arange(tokens)[None] % 31
    )
    assert len(errors) == 4 * (5 * len(decoder.weights.layers) + 3)
    for step in ("prefill", "decode.1", "decode.2", "decode.3"):
        assert errors[f"{step}.embedding"] == {
            "max_absolute_error": 0.0,
            "max_tolerance_ratio": 0.0,
        }
        assert f"{step}.logits" in errors
    assert all(0 <= error["max_tolerance_ratio"] <= 1 for error in errors.values())
    record_testsuite_property(
        f"synthetic_decoder_errors_{tokens}", json.dumps(errors, allow_nan=False)
    )


@pytest.mark.parametrize("reference_value", [0.0, -2.0])
def test_layer_measurements_use_fp32_errors_and_combined_tolerance(
    decoder: SimpleNamespace,
    assert_decoder_layer_parity: Callable[
        [Any, TinyLlamaWeights, Tensor], dict[str, dict[str, float]]
    ],
    monkeypatch: pytest.MonkeyPatch,
    reference_value: float,
) -> None:
    original = decoder_module.final_logits
    actual_value = torch.tensor(reference_value + 0.003, dtype=torch.float16).item()

    def logits(hidden: Tensor, weights: TinyLlamaWeights) -> Tensor:
        # Still run final normalization so the comparison observes every layer.
        result = original(hidden, weights)
        return torch.full_like(result, actual_value)

    def reference_logits(module: Any, args: Any, output: Tensor) -> Tensor:
        return torch.full_like(output, reference_value)

    handle = decoder.model.lm_head.register_forward_hook(reference_logits)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(decoder_module, "final_logits", logits)
            errors = assert_decoder_layer_parity(
                decoder.model, decoder.weights, torch.tensor([[1, 3]])
            )
    finally:
        handle.remove()
    absolute = abs(actual_value - reference_value)
    for step in ("prefill", "decode.1", "decode.2", "decode.3"):
        assert errors[f"{step}.logits"]["max_absolute_error"] == absolute
        assert errors[f"{step}.logits"]["max_tolerance_ratio"] == pytest.approx(
            absolute / (5e-3 + 5e-3 * abs(reference_value))
        )


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
