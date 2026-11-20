from types import SimpleNamespace

import pytest
import torch
from scripts.check_checkpoint_trajectory import component_audit

from pagedcore import decoder

pytestmark = pytest.mark.unit


def test_component_audit_preserves_native_outputs_and_saves_raw_math(tmp_path):
    hidden = torch.tensor([[[1.0, -2.0, 0.5, 0.0]]], dtype=torch.float16)
    scale = torch.ones(4, dtype=torch.float16)
    layer = SimpleNamespace(
        gate_proj=torch.ones(2, 4).half() / 4,
        up_proj=torch.arange(8).reshape(2, 4).half() / 8,
        down_proj=torch.ones(4, 2).half() / 4,
    )
    model = SimpleNamespace(final_norm=scale, lm_head=torch.eye(4).half())
    expected = [
        decoder.rms_norm(hidden, scale),
        decoder.swiglu(hidden, layer),
        decoder.final_logits(hidden, model),
    ]
    records = []
    with component_audit(records, tmp_path, "cpu-smoke"):
        actual = [
            decoder.rms_norm(hidden, scale),
            decoder.swiglu(hidden, layer),
            decoder.final_logits(hidden, model),
        ]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert len(records) == 3 and len(list(tmp_path.glob("component-*.pt"))) == 3
    assert all(row["max_absolute_error"] >= 0 for row in records)


def test_component_audit_restores_functions_after_an_error(tmp_path):
    original = decoder.swiglu
    with pytest.raises(RuntimeError), component_audit([], tmp_path, "failure"):
        raise RuntimeError("injected")
    assert decoder.swiglu is original
