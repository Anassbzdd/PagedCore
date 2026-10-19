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
from pagedcore.decoder import GreedyResult
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
