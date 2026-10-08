import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from pagedcore.config import MODEL_ID, MODEL_REVISION, PagedCoreConfig
from pagedcore.engine_types import FinishReason

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "reference_prompts.json"
REFERENCE_DATA: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
PROMPTS: list[dict[str, Any]] = REFERENCE_DATA["prompts"]


@pytest.mark.parametrize("fixture", PROMPTS, ids=[item["name"] for item in PROMPTS])
def test_public_prompt_contract(fixture: dict[str, Any]) -> None:
    config = PagedCoreConfig()
    settings = REFERENCE_DATA["tokenizer"]
    ids = fixture["input_ids"]
    count = fixture["prompt_tokens"]
    maximum = fixture["max_new_tokens"]

    assert isinstance(fixture["prompt"], str) and fixture["prompt"]
    fixture["prompt"].encode("utf-8")
    assert type(count) is int and count == len(ids)
    assert all(type(token_id) is int and 0 <= token_id < settings["vocab_size"] for token_id in ids)
    assert ids[0] == settings["bos_token_id"]
    assert type(maximum) is int and config.min_new_tokens <= maximum <= config.max_new_tokens
    assert type(fixture["ignore_eos"]) is bool
    assert count + maximum <= config.context_limit
    assert fixture["prompt_blocks"] == (count + config.block_tokens - 1) // config.block_tokens
    maximum_cached_tokens = count + maximum - 1
    assert (
        fixture["reserved_blocks"]
        == (maximum_cached_tokens + config.block_tokens - 1) // config.block_tokens
    )


def test_reference_tokenizer_contract() -> None:
    settings = REFERENCE_DATA["tokenizer"]
    assert settings["model_id"] == MODEL_ID
    assert settings["revision"] == MODEL_REVISION
    assert settings["tokenizer_class"] == "LlamaTokenizerFast"
    assert settings["add_special_tokens"] is True
    assert settings["truncation"] is False
    assert settings["add_bos_token"] is True
    assert settings["add_eos_token"] is False
    assert settings["bos_token_id"] == 1
    assert settings["eos_token_id"] == 2
    assert settings["vocab_size"] == 32000
    hashes = settings["source_sha256"]
    assert set(hashes) == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in hashes.values())


def test_public_prompts_cover_required_lengths() -> None:
    by_name = {item["name"]: item for item in PROMPTS}
    assert len(by_name) == len(PROMPTS)
    assert {15, 16, 17, 32, 49} <= {item["prompt_tokens"] for item in PROMPTS}
    maximum = by_name["maximum_output"]
    assert maximum["max_new_tokens"] == PagedCoreConfig().max_new_tokens
    assert maximum["ignore_eos"] is True
    assert by_name["eos_enabled"]["ignore_eos"] is False
    whitespace = by_name["boundary_17"]["prompt"]
    assert whitespace.startswith("  ") and whitespace.endswith("\n")

    mixed = [by_name[name] for name in REFERENCE_DATA["mixed_lengths"]]
    assert len(mixed) >= 2
    assert len({item["prompt_tokens"] for item in mixed}) == len(mixed)
    assert len({item["max_new_tokens"] for item in mixed}) == len(mixed)


@pytest.mark.integration
def test_public_prompt_ids_match_cached_pinned_tokenizer() -> None:
    """Recheck real tokenizer IDs offline; an absent cache is not a verification pass."""
    hub = importlib.import_module("huggingface_hub")
    transformers = importlib.import_module("transformers")
    settings = REFERENCE_DATA["tokenizer"]
    for name, expected_hash in settings["source_sha256"].items():
        cached = hub.try_to_load_from_cache(MODEL_ID, name, revision=MODEL_REVISION)
        if not isinstance(cached, str):
            pytest.skip(
                "Pinned tokenizer files are not cached; see the reference fixture procedure."
            )
        assert hashlib.sha256(Path(cached).read_bytes()).hexdigest() == expected_hash

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    assert type(tokenizer).__name__ == settings["tokenizer_class"]
    for name in ("add_bos_token", "add_eos_token", "bos_token_id", "eos_token_id", "vocab_size"):
        assert getattr(tokenizer, name) == settings[name]

    for fixture in PROMPTS:
        encoded = tokenizer(fixture["prompt"], add_special_tokens=True, truncation=False)
        assert encoded.input_ids == fixture["input_ids"], fixture["name"]
        without_specials = tokenizer(
            fixture["prompt"], add_special_tokens=False, truncation=False
        ).input_ids
        assert encoded.input_ids == [settings["bos_token_id"], *without_specials]


class ControlledLogits:
    """Force unique greedy winners without depending on a checkpoint's response."""

    def __init__(self, token_ids: tuple[int, ...], prompt_tokens: int) -> None:
        self.token_ids = token_ids
        self.prompt_tokens = prompt_tokens
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = input_ids.shape[-1] - self.prompt_tokens
        self.calls += 1
        forced = torch.full_like(scores, -torch.inf)
        forced[:, self.token_ids[step]] = 0
        return forced


@pytest.fixture(scope="module")
def synthetic_model() -> Any:
    transformers = importlib.import_module("transformers")
    config = transformers.LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return transformers.LlamaForCausalLM(config).eval()


@pytest.mark.parametrize(
    "script,maximum,ignore_eos,expected,reason,calls",
    [
        ((2, 3), 3, False, (), FinishReason.EOS, 1),
        ((2, 3), 1, False, (), FinishReason.EOS, 1),
        ((3, 4, 2, 5), 4, False, (3, 4), FinishReason.EOS, 3),
        ((3, 4, 5, 2), 3, False, (3, 4, 5), FinishReason.LENGTH, 3),
        ((3, 4, 2, 5), 3, False, (3, 4), FinishReason.EOS, 3),
        ((2, 3, 2, 4), 3, True, (2, 3, 2), FinishReason.LENGTH, 3),
        ((3,) * 256 + (2,), 256, True, (3,) * 256, FinishReason.LENGTH, 256),
    ],
    ids=[
        "immediate_eos",
        "immediate_eos_at_limit",
        "delayed_eos",
        "length",
        "eos_at_limit",
        "ignored_eos",
        "maximum_output",
    ],
)
def test_synthetic_stopping_reference(
    synthetic_model: Any,
    script: tuple[int, ...],
    maximum: int,
    ignore_eos: bool,
    expected: tuple[int, ...],
    reason: FinishReason,
    calls: int,
) -> None:
    """HF's controlled CPU reference is test data, not owned-decoder or checkpoint parity."""
    transformers = importlib.import_module("transformers")
    prompt = torch.tensor([[1]], dtype=torch.long)
    logits = ControlledLogits(script, prompt_tokens=prompt.shape[-1])
    with torch.inference_mode():
        result = synthetic_model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=maximum,
            eos_token_id=None if ignore_eos else 2,
            pad_token_id=0,
            logits_processor=transformers.LogitsProcessorList([logits]),
        )
    selected = cast(list[int], result[0, prompt.shape[-1] :].tolist())
    # HF includes a stopping EOS in its return; the public emitted-token contract excludes it.
    stopped_on_eos = not ignore_eos and selected[-1] == 2
    emitted = selected[:-1] if stopped_on_eos else selected
    actual_reason = FinishReason.EOS if stopped_on_eos else FinishReason.LENGTH

    assert tuple(emitted) == expected
    assert actual_reason is reason
    assert logits.calls == calls
