from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import GenerationConfig

import pagedcore.decoder as decoder_module
import pagedcore.dense as dense_module
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import FinishReason
from pagedcore.oracle import reference_math
from pagedcore.weights import LayerWeights, TinyLlamaWeights

TEST_SEED = 0


@pytest.fixture(autouse=True)
def deterministic_test_state() -> None:
    """Start every test with the same Python and PyTorch RNG state."""
    seed_everything(TEST_SEED)


@pytest.fixture
def assert_decoder_greedy_parity() -> Callable[..., decoder_module.GreedyResult]:
    def compare(
        model: Any,
        weights: TinyLlamaWeights,
        input_ids: Tensor,
        max_new_tokens: int,
        ignore_eos: bool,
    ) -> decoder_module.GreedyResult:
        assert model.config._attn_implementation == "eager"
        assert not model.training
        generation = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=None if ignore_eos else 2,
            bos_token_id=1,
            pad_token_id=2,
            use_cache=True,
        )
        with reference_math(), torch.inference_mode():
            reference = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                generation_config=generation,
                use_model_defaults=False,
                eos_token_id=None if ignore_eos else 2,
                logits_to_keep=0,
            )
            actual = decoder_module.generate(
                input_ids, weights, max_new_tokens=max_new_tokens, ignore_eos=ignore_eos
            )
        expected = tuple(reference[0, input_ids.shape[1] :].tolist())
        stopped = not ignore_eos and expected[-1] == 2
        assert actual.greedy_ids == expected, "selected greedy IDs differ"
        assert actual.emitted_ids == (expected[:-1] if stopped else expected), "emitted IDs differ"
        assert actual.finish_reason is (FinishReason.EOS if stopped else FinishReason.LENGTH)
        return actual

    return compare


@pytest.fixture
def assert_decoder_layer_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any, TinyLlamaWeights, Tensor], None]:
    def compare(model: Any, weights: TinyLlamaWeights, input_ids: Tensor) -> None:
        assert model.config._attn_implementation == "eager"
        assert not model.training
        assert len(model.model.layers) == len(weights.layers)
        modules = {"embedding": model.model.embed_tokens}
        for index, layer in enumerate(model.model.layers):
            for name, module in (
                ("input_norm", layer.input_layernorm),
                ("attention", layer.self_attn),
                ("post_attention_norm", layer.post_attention_layernorm),
                ("mlp", layer.mlp),
                ("hidden", layer),
            ):
                modules[f"layer.{index}.{name}"] = module
        modules.update(final_norm=model.model.norm, logits=model.lm_head)

        expected: dict[str, Tensor] = {}
        actual: dict[str, Tensor] = {}
        norm_calls = attention_calls = mlp_calls = 0
        original_norm = decoder_module.rms_norm
        original_attention = decoder_module.output_projection
        original_mlp = decoder_module.swiglu

        def snapshot(value: Tensor) -> Tensor:
            return value.detach().to("cpu", copy=True)

        def hook(name: str) -> Callable[..., None]:
            def save(module: Any, args: Any, output: Any) -> None:
                expected[name] = snapshot(output[0] if isinstance(output, tuple) else output)

            return save

        def norm(hidden: Tensor, weight: Tensor) -> Tensor:
            nonlocal norm_calls
            index, post_attention = divmod(norm_calls, 2)
            if not post_attention:
                if index == 0:
                    actual["embedding"] = snapshot(hidden)
                else:
                    # The next norm receives the previous layer's residual output.
                    actual[f"layer.{index - 1}.hidden"] = snapshot(hidden)
            result = original_norm(hidden, weight)
            if index == len(weights.layers):
                name = "final_norm"
            else:
                name = f"layer.{index}.{'post_attention_norm' if post_attention else 'input_norm'}"
            actual[name] = snapshot(result)
            norm_calls += 1
            return result

        def attention(value: Tensor, layer: LayerWeights) -> Tensor:
            nonlocal attention_calls
            result = original_attention(value, layer)
            # HF self_attn includes the output projection and returns [batch, tokens, hidden].
            actual[f"layer.{attention_calls}.attention"] = snapshot(result)
            attention_calls += 1
            return result

        def mlp(hidden: Tensor, layer: LayerWeights) -> Tensor:
            nonlocal mlp_calls
            result = original_mlp(hidden, layer)
            actual[f"layer.{mlp_calls}.mlp"] = snapshot(result)
            mlp_calls += 1
            return result

        with ExitStack() as cleanup:
            for name, module in modules.items():
                cleanup.callback(module.register_forward_hook(hook(name)).remove)
            patch = cleanup.enter_context(monkeypatch.context())
            patch.setattr(decoder_module, "rms_norm", norm)
            patch.setattr(dense_module, "rms_norm", norm)
            patch.setattr(decoder_module, "output_projection", attention)
            patch.setattr(decoder_module, "swiglu", mlp)
            cleanup.enter_context(reference_math())
            cleanup.enter_context(torch.inference_mode())
            cache = None
            reference_cache = None
            for step in range(4):
                prefix = "prefill" if step == 0 else f"decode.{step}"
                expected.clear()
                actual.clear()
                norm_calls = attention_calls = mlp_calls = 0
                reference = model(input_ids, past_key_values=reference_cache, use_cache=True)
                if cache is None:
                    logits, cache = decoder_module.prefill(input_ids, weights)
                else:
                    logits, cache = decoder_module.decode(input_ids, cache, weights)
                actual["logits"] = snapshot(logits)
                assert actual.keys() == expected.keys() == modules.keys(), prefix
                assert norm_calls == 2 * len(weights.layers) + 1, prefix
                assert attention_calls == mlp_calls == len(weights.layers), prefix
                for name, target in expected.items():
                    value = actual[name]
                    label = f"{prefix}.{name}"
                    assert value.dtype == target.dtype == torch.float16, label
                    assert torch.isfinite(value).all() and torch.isfinite(target).all(), label
                    tolerance = 5e-3 if name == "logits" else 2e-3
                    torch.testing.assert_close(
                        value,
                        target,
                        atol=tolerance,
                        rtol=tolerance,
                        msg=lambda message, label=label: f"{label}: {message}",
                    )
                reference_cache = reference.past_key_values
                # Replay reference tokens to isolate forward parity from generation/stopping.
                input_ids = reference.logits[:, -1:].argmax(dim=-1)

    return compare
