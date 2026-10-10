import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pagedcore import cli, oracle
from pagedcore.engine_types import FinishReason
from pagedcore.oracle import (
    ReferenceTrace,
    capture_reference,
    compare_traces,
    load_reference,
    reference_math,
    save_reference,
)
from pagedcore.verification import TargetVerificationError


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, ids: tuple[int, ...] = (1, 3, 4)) -> None:
        self.ids = ids
        self.prompt = ""

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs == {"add_special_tokens": True, "truncation": False, "return_tensors": "pt"}
        self.prompt = prompt
        inputs = torch.tensor([self.ids])
        return {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}

    def decode(self, ids: tuple[int, ...], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return " ".join(str(item) for item in ids if item not in (1, 2))


@pytest.fixture
def model() -> Any:
    transformers = import_module("transformers")
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        attn_implementation="eager",
    )
    return transformers.LlamaForCausalLM(config)


@pytest.mark.parametrize(
    "script,maximum,ignore_eos,emitted,reason",
    [
        ((2,), 1, False, (), FinishReason.EOS),
        ((3, 2), 3, False, (3,), FinishReason.EOS),
        ((3, 4, 2), 3, False, (3, 4), FinishReason.EOS),
        ((3, 4, 5), 3, False, (3, 4, 5), FinishReason.LENGTH),
        ((2, 3, 2), 3, True, (2, 3, 2), FinishReason.LENGTH),
        ((3,) * 256, 256, True, (3,) * 256, FinishReason.LENGTH),
    ],
    ids=["immediate_eos", "delayed_eos", "eos_at_limit", "length", "ignored_eos", "maximum"],
)
def test_capture_stopping_and_step_snapshots(
    model: Any,
    script: tuple[int, ...],
    maximum: int,
    ignore_eos: bool,
    emitted: tuple[int, ...],
    reason: FinishReason,
) -> None:
    calls = 0
    forwards = []

    def force(module: Any, args: Any, output: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        forced = torch.full_like(output, -10)
        forced[..., script[calls]] = 10
        calls += 1
        return forced

    def observe(module: Any, args: Any, kwargs: Any) -> None:
        assert not module.training
        assert torch.is_inference_mode_enabled()
        forwards.append(kwargs["input_ids"].clone())

    forcing = model.lm_head.register_forward_hook(force)
    observing = model.register_forward_pre_hook(observe, with_kwargs=True)
    tokenizer = Tokenizer()
    try:
        result = capture_reference(
            model, tokenizer, "  public prompt\n", maximum, ignore_eos, expected_input_ids=(1, 3, 4)
        )
    finally:
        forcing.remove()
        observing.remove()
    assert result.greedy_ids == script
    assert result.emitted_ids == emitted
    assert result.finish_reason is reason
    assert result.decoded_text == " ".join(str(item) for item in emitted if item not in (1, 2))
    assert tokenizer.prompt == "  public prompt\n"
    assert calls == len(script)
    assert forwards[0].tolist() == [[1, 3, 4]]
    assert [inputs.item() for inputs in forwards[1:]] == list(script[:-1])
    assert result.tensors["prefill.logits"].shape == (1, 3, 16)
    assert result.tensors["prefill.embedding"].shape == (1, 3, 16)
    assert result.tensors["prefill.layer.0.attention"].shape == (1, 3, 16)
    assert result.tensors["prefill.layer.1.mlp"].shape == (1, 3, 16)
    assert result.tensors["prefill.final_norm"].shape == (1, 3, 16)
    for step, token in enumerate(script):
        prefix = "prefill" if step == 0 else f"decode.{step}"
        assert result.tensors[f"{prefix}.logits"][0, -1].argmax().item() == token
    assert all(
        value.device.type == "cpu" and not value.requires_grad for value in result.tensors.values()
    )
    assert model.training
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def test_capture_matches_independent_full_history_and_neutral_hf_generation(model: Any) -> None:
    transformers = import_module("transformers")
    model.eval()
    # Deliberately hostile checkpoint defaults must not affect the oracle's raw greedy policy.
    model.generation_config.repetition_penalty = 100.0
    model.generation_config.forced_bos_token_id = 9
    model.generation_config.do_sample = True
    model.generation_config.transformers_version = "4.52.4"
    result = capture_reference(model, Tokenizer(), "public", 3, True)
    with reference_math(), torch.inference_mode():
        independent = model.generate(
            input_ids=torch.tensor([[1, 3, 4]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            generation_config=transformers.GenerationConfig(max_new_tokens=3, pad_token_id=2),
            use_model_defaults=False,
            eos_token_id=None,
        )
        assert result.greedy_ids == tuple(independent[0, 3:].tolist())
        history = [1, 3, 4]
        for step, token in enumerate(result.greedy_ids):
            full = model(input_ids=torch.tensor([history]), use_cache=False).logits
            prefix = "prefill" if step == 0 else f"decode.{step}"
            torch.testing.assert_close(result.tensors[f"{prefix}.logits"][:, -1], full[:, -1])
            assert token == full[0, -1].argmax().item()
            history.append(token)
    before = result.tensors["prefill.embedding"].clone()
    capture_reference(model, Tokenizer(), "public", 1, True)
    model.model.embed_tokens.weight.data.zero_()
    assert torch.equal(result.tensors["prefill.embedding"], before)
    assert not model.training


def test_failed_forward_removes_hooks_and_restores_model_and_math(model: Any) -> None:
    def fail(module: Any, args: Any) -> None:
        raise RuntimeError("injected forward failure")

    handle = model.model.layers[0].register_forward_pre_hook(fail)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        with pytest.raises(RuntimeError, match="injected"):
            capture_reference(model, Tokenizer(), "public", 2, True)
    finally:
        handle.remove()
    assert model.training
    assert torch.backends.cuda.matmul.allow_tf32 == previous
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def test_math_context_freezes_and_restores_on_failure() -> None:
    owners = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
        (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "deterministic"),
    )
    previous = [getattr(owner, name) for owner, name in owners]
    precision = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError), reference_math() as settings:
        assert settings["float32_matmul_precision"] == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        assert torch.backends.cudnn.deterministic
        raise RuntimeError("injected")
    assert [getattr(owner, name) for owner, name in owners] == previous
    assert torch.get_float32_matmul_precision() == precision


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_math_context_enforces_determinism_and_restores_caller(
    enabled: bool, warn_only: bool, fail: bool
) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        try:
            with reference_math() as settings:
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                assert settings["deterministic_algorithms"] is True
                with reference_math():
                    assert torch.are_deterministic_algorithms_enabled()
                assert torch.are_deterministic_algorithms_enabled()
                if fail:
                    raise RuntimeError("injected capture failure")
        except RuntimeError as error:
            assert fail and str(error) == "injected capture failure"
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


@pytest.mark.parametrize("workspace", [None, "", ":4096:2"])
def test_cuda_reference_rejects_missing_or_unsupported_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace: str | None, capsys: Any
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla T4")
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    previous = torch.are_deterministic_algorithms_enabled()
    with (
        pytest.raises(TargetVerificationError, match="CUBLAS_WORKSPACE_CONFIG"),
        reference_math(),
    ):
        pytest.fail("invalid CUDA configuration must fail before capture")
    assert torch.are_deterministic_algorithms_enabled() == previous
    fixtures = Path(__file__).parents[1] / "fixtures" / "reference_prompts.json"
    with pytest.raises(TargetVerificationError, match="CUBLAS_WORKSPACE_CONFIG"):
        oracle.run_oracle(fixtures, "boundary_15", Path("unused.pt"))
    assert cli.main(["oracle"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "CUBLAS_WORKSPACE_CONFIG" in json.loads(output.err)["error"]


def test_math_context_rejects_nondeterministic_operation() -> None:
    with reference_math(), pytest.raises(RuntimeError, match="deterministic"):
        torch.zeros(1).put_(torch.tensor([0, 0]), torch.tensor([1.0, 2.0]), accumulate=False)


@pytest.mark.parametrize("workspace", [":4096:8", ":16:8"])
def test_cuda_reference_records_supported_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with reference_math() as settings:
        assert settings["cublas_workspace_config"] == workspace
        assert settings["deterministic_algorithms"] is True


@pytest.mark.parametrize("maximum,ignore", [(0, True), (257, True), (True, True), (1, 1)])
def test_invalid_generation_contract(model: Any, maximum: int, ignore: bool) -> None:
    with pytest.raises(ValueError):
        capture_reference(model, Tokenizer(), "public", maximum, ignore)


def test_input_ids_context_and_backend_are_checked(model: Any) -> None:
    with pytest.raises(ValueError, match="prompt IDs"):
        capture_reference(model, Tokenizer(), "public", 1, True, expected_input_ids=(1, 5))
    with pytest.raises(ValueError, match="context"):
        capture_reference(model, Tokenizer((1,) * 2048), "public", 1, True)
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        capture_reference(model, Tokenizer(), "public", 1, True)


def trace() -> ReferenceTrace:
    return ReferenceTrace(
        (1, 3),
        (4,),
        (4,),
        FinishReason.LENGTH,
        "4",
        {"prefill.embedding": torch.ones(1, 2, 3), "prefill.logits": torch.ones(1, 2, 16)},
    )


def test_comparison_reports_errors_with_fixed_tensor_and_logit_tolerances() -> None:
    expected = trace()
    actual = replace(
        expected,
        tensors={
            "prefill.embedding": expected.tensors["prefill.embedding"] + 0.003,
            "prefill.logits": expected.tensors["prefill.logits"] + 0.008,
        },
    )
    discrepancies = compare_traces(actual, expected)
    assert discrepancies["prefill.embedding"] == pytest.approx(0.003, abs=1e-7)
    assert discrepancies["prefill.logits"] == pytest.approx(0.008, abs=1e-7)
    actual.tensors["prefill.embedding"] += 0.002
    with pytest.raises(AssertionError, match=r"prefill\.embedding"):
        compare_traces(actual, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_ids", (1, 4)),
        ("greedy_ids", (5,)),
        ("emitted_ids", (5,)),
        ("finish_reason", FinishReason.EOS),
        ("decoded_text", "different"),
    ],
)
def test_comparison_requires_exact_ids_and_outcome(field: str, value: Any) -> None:
    expected = trace()
    with pytest.raises(AssertionError, match=field):
        compare_traces(replace(expected, **{field: value}), expected)


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "nan", "inf"])
def test_comparison_rejects_invalid_tensor_evidence(change: str) -> None:
    expected = trace()
    tensors = dict(expected.tensors)
    if change == "missing":
        tensors.pop("prefill.logits")
    else:
        tensors["prefill.logits"] = {
            "shape": torch.ones(1, 1, 16),
            "dtype": torch.ones(1, 2, 16, dtype=torch.float16),
            "nan": torch.full((1, 2, 16), float("nan")),
            "inf": torch.full((1, 2, 16), float("inf")),
        }[change]
    with pytest.raises(AssertionError):
        compare_traces(replace(expected, tensors=tensors), expected)


def test_artifact_round_trip_uses_plain_tensor_records(tmp_path: Path) -> None:
    path = tmp_path / "reference.pt"
    save_reference(path, trace(), {"scope": "synthetic", "settings": {"eager": True}})
    loaded, metadata = load_reference(path)
    assert metadata == {"scope": "synthetic", "settings": {"eager": True}}
    assert all(error == 0 for error in compare_traces(loaded, trace()).values())
    torch.save({"format_version": 99}, path)
    with pytest.raises(ValueError, match="unsupported"):
        load_reference(path)


def test_failed_artifact_write_preserves_previous_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "reference.pt"
    save_reference(path, trace(), {})
    previous = path.read_bytes()

    def fail(payload: Any, destination: Path) -> None:
        destination.write_bytes(b"partial write")
        raise OSError("injected failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected"):
        save_reference(path, trace(), {})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_ids", (True,)),
        ("greedy_ids", ()),
        ("emitted_ids", None),
        ("tensors", {}),
        ("metadata", None),
        ("finish_reason", "invalid"),
    ],
)
def test_malformed_reference_records_fail(tmp_path: Path, field: str, value: Any) -> None:
    path = tmp_path / "reference.pt"
    save_reference(path, trace(), {})
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_reference(path)


def test_runner_pins_loading_and_retains_comparison_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Mock target/loading plumbing only; this test cannot establish T4 evidence.
    public = Path(__file__).parents[1] / "fixtures" / "reference_prompts.json"
    fixtures = json.loads(public.read_text())
    fixtures["tokenizer"]["source_sha256"] = {}
    fixtures["tokenizer"]["tokenizer_class"] = "SimpleNamespace"
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(json.dumps(fixtures))
    snapshot = tmp_path / fixtures["tokenizer"]["revision"]
    tokenizer = SimpleNamespace(
        **{
            name: fixtures["tokenizer"][name]
            for name in (
                "add_bos_token",
                "add_eos_token",
                "bos_token_id",
                "eos_token_id",
                "vocab_size",
            )
        }
    )
    calls = []

    class LoadedModel:
        config = SimpleNamespace(
            _attn_implementation="eager",
            to_dict=lambda: {
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "num_key_value_heads": 4,
            },
        )

        def to(self, device: str) -> "LoadedModel":
            assert device == "cuda:0"
            return self

        def eval(self) -> "LoadedModel":
            return self

    def load(*args: Any, **kwargs: Any) -> LoadedModel:
        calls.append((args, kwargs))
        return LoadedModel()

    transformers = import_module("transformers")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla T4")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="Tesla T4", uuid="synthetic", major=7, minor=5, total_memory=1024
        ),
    )
    monkeypatch.setattr(oracle, "_download_pinned_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(
        oracle, "_validate_checkpoint", lambda *args, **kwargs: {"revision": snapshot.name}
    )
    monkeypatch.setattr(oracle, "seed_everything", lambda seed: None)
    monkeypatch.setattr(oracle, "capture_reference", lambda *args, **kwargs: trace())
    monkeypatch.setattr(
        cli, "collect_environment_diagnostics", lambda *args: {"scope": "synthetic"}
    )
    monkeypatch.setattr(
        cli, "collect_validation_provenance", lambda: {"source_sha256": {"synthetic": "hash"}}
    )
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    oracle.run_oracle(fixture_path, "boundary_15", first, command_arguments=["oracle"])
    result = oracle.run_oracle(fixture_path, "boundary_15", second, first)
    _, metadata = load_reference(second)
    assert calls[0] == (
        (str(snapshot),),
        {"local_files_only": True, "torch_dtype": torch.float16, "attn_implementation": "eager"},
    )
    assert metadata["math_settings"]["matmul_allow_tf32"] is False
    assert metadata["math_settings"]["deterministic_algorithms"] is True
    assert metadata["math_settings"]["cublas_workspace_config"] == ":4096:8"
    assert metadata["model_config"]["rope_theta"] == 10000
    assert metadata["decoding"]["chat_template"] is False
    assert metadata["comparison"]["maximum_absolute_errors"] == result["maximum_absolute_errors"]
    _, old_metadata = load_reference(first)
    old_metadata["math_settings"]["deterministic_algorithms"] = False
    save_reference(first, trace(), old_metadata)
    with pytest.raises(ValueError, match="policy mismatch: math_settings"):
        oracle.run_oracle(fixture_path, "boundary_15", tmp_path / "legacy.pt", first)
    metadata["decoding"]["ignore_eos"] = False
    save_reference(second, trace(), metadata)
    with pytest.raises(ValueError, match="policy mismatch: decoding"):
        oracle.run_oracle(fixture_path, "boundary_15", tmp_path / "third.pt", second)


def test_runner_requires_t4_before_checkpoint_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = Path(__file__).parents[1] / "fixtures" / "reference_prompts.json"
    with pytest.raises(TargetVerificationError, match="NVIDIA T4"):
        oracle.run_oracle(path, "boundary_15", Path("unused.pt"))


def test_oracle_command_sanitizes_paths_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: Any,
) -> None:
    calls = []

    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"status": "passed", "scope": "synthetic"}

    monkeypatch.setattr(oracle, "run_oracle", run)
    assert (
        cli.main(
            [
                "oracle",
                "--case=boundary_16",
                "--output",
                str(tmp_path / "secret.pt"),
                "--compare=private.pt",
                "--fixtures=private.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert calls[0][1]["command_arguments"] == [
        "oracle",
        "--case=boundary_16",
        "--output",
        "<path>",
        "--compare=<path>",
        "--fixtures=<path>",
    ]

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError("private-path-sentinel")

    monkeypatch.setattr(oracle, "run_oracle", fail)
    assert cli.main(["oracle"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "status": "failed",
        "error": "oracle capture or comparison failed",
    }
    assert "private-path" not in output.err
