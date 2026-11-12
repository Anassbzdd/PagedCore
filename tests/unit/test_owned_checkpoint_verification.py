import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts import verify_owned_checkpoint

pytestmark = pytest.mark.unit


def test_owned_verifier_rejects_cpu_before_loading_checkpoint(monkeypatch):
    monkeypatch.setattr(verify_owned_checkpoint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        verify_owned_checkpoint,
        "load_weights",
        lambda **kwargs: pytest.fail("checkpoint loaded before device validation"),
    )
    with pytest.raises(ValueError, match="one CUDA-visible T4"):
        verify_owned_checkpoint.verify()


@pytest.mark.parametrize("fault", [None, "file_hash", "tokenizer_setting", "nonfinite", "cache"])
def test_owned_verifier_records_owned_smoke_and_rejects_invalid_evidence(
    tmp_path, monkeypatch, fault
):
    module = verify_owned_checkpoint
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("pinned tokenizer fixture")
    settings = {
        "model_id": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "tokenizer_class": "Tokenizer",
        "source_sha256": {
            "tokenizer.json": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
        },
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "vocab_size": 32000,
    }
    fixtures = {
        "tokenizer": settings,
        "prompts": [{"name": "public", "input_ids": [1, 3], "prompt": "public"}],
    }
    if fault == "file_hash":
        tokenizer_file.write_text("corrupted")
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            json.dumps(fixtures)
            if path.name == "reference_prompts.json"
            else original_read(path, *a, **kw)
        ),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda index: "Tesla T4")
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module, "hf_hub_download", lambda *a, **kw: str(tokenizer_file))

    class Tokenizer:
        def __init__(self):
            self.__dict__.update(settings)
            if fault == "tokenizer_setting":
                self.add_bos_token = False

        def __call__(self, text, **kwargs):
            if "return_tensors" in kwargs:
                return {"input_ids": SimpleNamespace(to=lambda device: torch.tensor([[1, 3]]))}
            return SimpleNamespace(input_ids=[1, 3])

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(module, "reference_math", lambda: nullcontext({"strict": True}))
    owned_weights = object()
    monkeypatch.setattr(module, "load_weights", lambda **kw: owned_weights)

    def owned_prefill(ids, weights):
        assert weights is owned_weights and torch.is_inference_mode_enabled()
        logits = torch.zeros(1, 2, 32000).half()
        if fault == "nonfinite":
            logits[..., 0] = float("nan")
        return logits, SimpleNamespace(length=1 if fault == "cache" else 2)

    monkeypatch.setattr(module, "prefill", owned_prefill)
    monkeypatch.setattr(module, "collect_environment_diagnostics", lambda: {"synthetic": True})
    monkeypatch.setattr(module, "collect_validation_provenance", lambda: {"source": "synthetic"})
    if fault:
        with pytest.raises(ValueError):
            module.verify()
    else:
        result = module.verify()
        assert result["manifest_version"] == 3 and result["status"] == "passed"
        assert result["math_settings"] == {"strict": True}
        assert result["tokenizer"] == settings
