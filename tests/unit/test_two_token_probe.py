import pytest
import torch
from scripts import probe_two_token_decode as probe

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("changed", [None, "decode_input", "paged", "owned_reference"])
def test_batch_comparison_distinguishes_input_reference_and_output_drift(tmp_path, changed):
    raw = {
        "prompt_ids": torch.tensor([1]),
        "decode_input": torch.tensor([[7]]),
        "prefill_logits": torch.tensor([[[0.0, 1.0]]], dtype=torch.float16),
        "paged": torch.tensor([[[0.0, 1.25]]], dtype=torch.float16),
        "owned_reference": torch.tensor([[[0.0, 1.0]]], dtype=torch.float16),
    }
    for name in ("single", "mixed"):
        directory = tmp_path / name
        directory.mkdir()
        tensors = {key: value.clone() for key, value in raw.items()}
        if name == "mixed" and changed is not None:
            tensors[changed].view(-1)[-1] += 1
        torch.save(tensors, directory / "target.pt")
    controls = probe.compare_cases(tmp_path)
    if changed is None:
        assert all(controls.values())
    else:
        key = "paged_logits_exact" if changed == "paged" else changed + "_exact"
        assert not controls[key]
        if changed in ("paged", "owned_reference"):
            assert not controls["signed_errors_exact"]
