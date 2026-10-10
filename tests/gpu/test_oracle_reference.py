from pathlib import Path

import pytest
import torch

from pagedcore.oracle import load_reference, run_oracle

pytestmark = pytest.mark.gpu


def test_public_reference_capture_and_repeatability_on_t4(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    fixtures = Path(__file__).parents[1] / "fixtures" / "reference_prompts.json"
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    run_oracle(fixtures, "boundary_15", first)
    result = run_oracle(fixtures, "boundary_15", second, first)
    trace, metadata = load_reference(second)
    assert result["status"] == "passed"
    assert trace.tensors["prefill.logits"].shape == (1, 15, 32000)
    assert all(value.dtype == torch.float16 for value in trace.tensors.values())
    assert metadata["attention_backend"] == "eager"
    assert metadata["math_settings"]["deterministic_algorithms"] is True
    assert metadata["math_settings"]["deterministic_warn_only"] is False
    assert metadata["math_settings"]["cublas_workspace_config"] in (":4096:8", ":16:8")
    assert metadata["model_config"]["num_key_value_heads"] == 4
    assert metadata["provenance"]["source_sha256"]
    assert "t4" in metadata["target"]["name"].casefold()
