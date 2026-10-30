import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.decoder import GreedyResult, prefill_paged
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
