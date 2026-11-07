import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM

import pagedcore.attention as attention_module
import pagedcore.decoder as decoder_module
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.decoder import GreedyResult, decode_paged, prefill_paged
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import TinyLlamaWeights, load_weights

pytestmark = pytest.mark.gpu

FIXTURES = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "reference_prompts.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def pinned_decoders() -> tuple[Any, TinyLlamaWeights]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert "t4" in torch.cuda.get_device_name(0).casefold(), "requires an NVIDIA T4"
    # Enforce the reference math policy before loading or running either decoder.
    with reference_math():
        weights = load_weights(device="cuda:0")
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )
            .to("cuda:0")
            .eval()
        )
    assert len(weights.layers) == len(model.model.layers) == 22
    return model, weights


def test_pinned_decoder_layer_parity_on_t4(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    assert_decoder_layer_parity: Callable[
        [Any, TinyLlamaWeights, Tensor], dict[str, dict[str, float]]
    ],
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    model, weights = pinned_decoders
    cases = {case["name"]: case["input_ids"] for case in FIXTURES["prompts"]}
    prompts = {"bos_only": [1]} | {
        name: cases[name]
        for name in ("boundary_15", "boundary_16", "boundary_17", "maximum_output")
    }
    measurements = {}
    with reference_math() as math_settings:
        for name, ids in prompts.items():
            measurements[name] = {
                "prompt_tokens": len(ids),
                "tensors": assert_decoder_layer_parity(
                    model, weights, torch.tensor([ids], device="cuda:0")
                ),
            }
    record_testsuite_property(
        "decoder_tolerance_envelope",
        json.dumps(
            {
                "scope": "PagedCore vs Hugging Face; prefill and three reference-token decode steps",
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "seed": 0,
                "dtype": "float16",
                "attention_backend": model.config._attn_implementation,
                "model_config": model.config.to_dict(),
                "math_settings": math_settings,
                "tolerances": {
                    "intermediates": {"atol": 2e-3, "rtol": 2e-3},
                    "logits": {"atol": 5e-3, "rtol": 5e-3},
                },
                "cases": measurements,
            },
            allow_nan=False,
        ),
    )


def test_pinned_paged_prefill_on_t4(pinned_decoders: tuple[Any, TinyLlamaWeights]) -> None:
    model, weights = pinned_decoders
    pool = torch.full((22, 2, 6, 16, 4, 64), torch.nan, dtype=torch.float16, device="cuda:0")
    allocator = KVAllocator(pool)
    pointer = pool.untyped_storage().data_ptr()
    cases = {case["name"]: case["input_ids"] for case in FIXTURES["prompts"]}
    prompts = [[1]] + [
        cases[name] for name in ("boundary_15", "boundary_16", "boundary_17", "maximum_output")
    ]
    with reference_math(), torch.inference_mode():
        for ids in prompts:
            for name in ("first", "retained", "last"):
                request = RequestId(name)
                allocator.create(KVReservation(request, 1))
                key = torch.full((22, 16, 4, 64), -100.0, dtype=torch.float16, device=pool.device)
                allocator.append(request, key, -key)
            allocator.release_all(RequestId("first"))
            allocator.release_all(RequestId("last"))
            request = RequestId("prompt")
            blocks = (len(ids) + 15) // 16
            allocator.create(KVReservation(request, blocks))
            before = pool.clone()
            inputs = torch.tensor([ids], device=pool.device)
            expected = model(inputs, use_cache=True)
            logits = prefill_paged(inputs, weights, allocator, request)
            torch.testing.assert_close(logits, expected.logits, atol=5e-3, rtol=5e-3)
            table = allocator.block_table(request)
            assert len(table) == blocks
            assert allocator.cached_length(request) == len(ids)
            # Release order shuffles pages; inspect actual IDs rather than assuming contiguity.
            if blocks > 1:
                assert table != tuple(range(table[0], table[0] + blocks))
            for logical, physical in enumerate(table):
                start = logical * 16
                valid = min(16, len(ids) - start)
                for layer in range(22):
                    key, value = expected.past_key_values[layer]
                    for kind, reference in enumerate((key, value)):
                        torch.testing.assert_close(
                            pool[layer, kind, physical, :valid],
                            reference[0, :, start : start + valid].transpose(0, 1),
                            atol=2e-3,
                            rtol=2e-3,
                        )
                torch.testing.assert_close(
                    pool[:, :, physical, valid:], before[:, :, physical, valid:], equal_nan=True
                )
            for physical in set(range(6)) - set(table):
                torch.testing.assert_close(
                    pool[:, :, physical], before[:, :, physical], equal_nan=True
                )
            assert allocator.capacity() == CapacityCounts(
                6, 5 - blocks, blocks + 1, blocks + 1, 2, len(ids) + 16
            )
            allocator.release_all(request)
            allocator.release_all(RequestId("retained"))
            assert allocator.capacity() == CapacityCounts(6, 6, 0, 0, 0, 0)
            assert pool.untyped_storage().data_ptr() == pointer
        torch.cuda.synchronize(pool.device)


def _assert_pinned_paged_decode(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    lengths: tuple[int, ...],
    forbid_dense_kv_history: Callable[[Tensor], TorchDispatchMode],
    *,
    fragmented: bool = False,
    steps: int = 3,
) -> None:
    model, weights = pinned_decoders
    pool = torch.full(
        (22, 2, 16, 16, 4, 64), torch.nan, dtype=torch.float16, device=weights.embedding.device
    )
    allocator = KVAllocator(pool)
    pointer = pool.untyped_storage().data_ptr()
    requests = tuple(RequestId(str(row)) for row in range(len(lengths)))
    public_ids = next(
        case["input_ids"] for case in FIXTURES["prompts"] if case["name"] == "maximum_output"
    )
    references = {}
    blocker = RequestId("unrelated")
    blocker_table = ()
    boundary_request = requests[lengths.index(15)] if fragmented else None
    with reference_math(), torch.inference_mode():
        for request, length in zip(requests, lengths, strict=True):
            allocator.create(KVReservation(request, (length + steps + 15) // 16))
            inputs = torch.tensor([public_ids[:length]], device=pool.device)
            references[request] = model(inputs, use_cache=True)
            logits = prefill_paged(inputs, weights, allocator, request)
            torch.testing.assert_close(logits, references[request].logits, atol=5e-3, rtol=5e-3)
            if request == boundary_request:
                allocator.create(KVReservation(blocker, 1))
                sentinel = torch.full((22, 16, 4, 64), 100, dtype=pool.dtype, device=pool.device)
                allocator.append(blocker, -sentinel, sentinel)
                blocker_table = allocator.block_table(blocker)
                assert blocker_table == (allocator.block_table(request)[0] + 1,)
        reserved = allocator.capacity().reserved_blocks
        for step in range(steps):
            # Row order is independent of request identity and absolute position.
            order = requests if step % 2 == 0 else requests[::-1]
            tokens = torch.cat([references[request].logits[:, -1:].argmax(-1) for request in order])
            before = pool.clone()
            old_lengths = {request: allocator.cached_length(request) for request in order}
            old_tables = {request: allocator.block_table(request) for request in order}
            expected = {
                request: model(
                    tokens[row : row + 1],
                    past_key_values=references[request].past_key_values,
                    use_cache=True,
                )
                for row, request in enumerate(order)
            }
            if fragmented and step == 1:
                counts = allocator.capacity()
                # Final logits run after every layer has written its staged K/V.
                with (
                    patch.object(
                        decoder_module, "final_logits", side_effect=RuntimeError("logit failure")
                    ),
                    forbid_dense_kv_history(pool),
                    pytest.raises(RuntimeError, match="logit failure"),
                ):
                    decode_paged(tokens, weights, allocator, order)
                assert allocator.capacity() == counts
                assert allocator.block_table(blocker) == blocker_table
                for request in order:
                    assert allocator.cached_length(request) == old_lengths[request]
                    assert allocator.block_table(request) == old_tables[request]
                    for logical, physical in enumerate(old_tables[request]):
                        valid = min(16, old_lengths[request] - logical * 16)
                        torch.testing.assert_close(
                            pool[:, :, physical, :valid],
                            before[:, :, physical, :valid],
                            atol=0,
                            rtol=0,
                        )
            with forbid_dense_kv_history(pool):
                logits = decode_paged(tokens, weights, allocator, order)
            assert logits.shape == (len(order), 1, model.config.vocab_size)
            assert logits.dtype == torch.float16 and torch.isfinite(logits).all()
            for row, request in enumerate(order):
                reference = expected[request]
                torch.testing.assert_close(
                    logits[row : row + 1], reference.logits, atol=5e-3, rtol=5e-3
                )
                assert torch.equal(logits[row : row + 1].argmax(-1), reference.logits.argmax(-1))
                assert allocator.cached_length(request) == old_lengths[request] + 1
                table = allocator.block_table(request)
                assert table[: len(old_tables[request])] == old_tables[request]
                if fragmented:
                    if request == boundary_request and step >= 1:
                        assert len(table) == 2 and table[1] > table[0] + 1
                    logical, slot = divmod(old_lengths[request], 16)
                    physical = table[logical]
                    before[:, :, physical, slot] = pool[:, :, physical, slot]
                for logical, physical in enumerate(table):
                    key, value = allocator.read_block(request, logical)
                    start = logical * 16
                    end = start + key.shape[1]
                    for layer, pair in enumerate(reference.past_key_values):
                        for kind, actual, target in (
                            (0, key[layer], pair[0]),
                            (1, value[layer], pair[1]),
                        ):
                            torch.testing.assert_close(
                                actual,
                                target[0, :, start:end].transpose(0, 1),
                                atol=2e-3,
                                rtol=2e-3,
                            )
                            old_slots = max(0, min(end, old_lengths[request]) - start)
                            torch.testing.assert_close(
                                actual[:old_slots],
                                before[layer, kind, physical, :old_slots],
                                atol=0,
                                rtol=0,
                            )
                references[request] = reference
            counts = allocator.capacity()
            assert counts.live_cached_token_slots == (
                sum(lengths) + len(order) * (step + 1) + 16 * len(blocker_table)
            )
            assert counts.reserved_blocks == reserved
            assert counts.active_slots == len(order) + len(blocker_table)
            assert pool.untyped_storage().data_ptr() == pointer
            if fragmented:
                assert allocator.block_table(blocker) == blocker_table
                torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
        torch.cuda.synchronize(pool.device)
        for request in requests:
            allocator.release_all(request)
        if fragmented:
            allocator.release_all(blocker)
        assert allocator.capacity() == CapacityCounts(16, 16, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("lengths", "fragmented"),
    [((15,), False), ((1, 15, 16, 17, 49), False), ((1, 15, 16, 17, 49), True)],
    ids=["single", "mixed", "fragmented"],
)
def test_pinned_paged_decode_on_t4(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    lengths: tuple[int, ...],
    fragmented: bool,
    forbid_dense_kv_history: Callable[[Tensor], TorchDispatchMode],
) -> None:
    _assert_pinned_paged_decode(
        pinned_decoders, lengths, forbid_dense_kv_history, fragmented=fragmented
    )


@pytest.mark.parametrize("lengths", [(15,), (1, 15, 16, 17, 49)], ids=["single", "mixed"])
def test_pinned_candidate_paged_decode_on_t4(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    lengths: tuple[int, ...],
    forbid_dense_kv_history: Callable[[Tensor], TorchDispatchMode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_module, "_multi_page_value_sum", attention_module._paged_value_sum
    )
    _assert_pinned_paged_decode(pinned_decoders, lengths, forbid_dense_kv_history)


@pytest.mark.usefixtures("paired_at_17")
@pytest.mark.parametrize(
    ("lengths", "fragmented", "steps"),
    [
        ((15,), True, 2),
        ((15,), False, 3),
        ((1, 15, 16, 17, 49), False, 3),
        ((1, 15, 16, 17, 49), True, 3),
    ],
    ids=["boundary", "single", "mixed", "fragmented"],
)
def test_paired_at_17_pinned_decode_on_t4(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    lengths: tuple[int, ...],
    fragmented: bool,
    steps: int,
    forbid_dense_kv_history: Callable[[Tensor], TorchDispatchMode],
) -> None:
    _assert_pinned_paged_decode(
        pinned_decoders, lengths, forbid_dense_kv_history, fragmented=fragmented, steps=steps
    )


@pytest.mark.parametrize("case", FIXTURES["prompts"], ids=lambda case: case["name"])
def test_pinned_decoder_greedy_parity_on_t4(
    pinned_decoders: tuple[Any, TinyLlamaWeights],
    assert_decoder_greedy_parity: Callable[..., GreedyResult],
    case: dict[str, Any],
) -> None:
    model, weights = pinned_decoders
    assert_decoder_greedy_parity(
        model,
        weights,
        torch.tensor([case["input_ids"]], device="cuda:0"),
        case["max_new_tokens"],
        case["ignore_eos"],
    )
