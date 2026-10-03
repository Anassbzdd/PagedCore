import pytest

from pagedcore.verification import verify_target

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu


def test_pinned_checkpoint_loads_and_runs_one_reference_forward_pass() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    manifest = verify_target()

    assert manifest["status"] == "passed"
    assert manifest["target"]["name"].casefold().find("t4") >= 0
    assert manifest["checkpoint"]["revision"] == (
        "af8e934848d8dd00074cc2cd8a40a9b05c3b011e"
    )
    assert manifest["checkpoint"]["dtype"] == "torch.float16"
    assert manifest["checkpoint"]["config"]["max_position_embeddings"] == 2048
    assert manifest["checkpoint"]["tokenizer"]["revision"] == (
        "af8e934848d8dd00074cc2cd8a40a9b05c3b011e"
    )
    assert manifest["checkpoint"]["tokenizer"]["resolved_revision"] == (
        "af8e934848d8dd00074cc2cd8a40a9b05c3b011e"
    )
    assert manifest["forward_pass"]["status"] == "passed"
