import random

import pytest

from pagedcore.determinism import inference_mode, seed_everything

torch = pytest.importorskip("torch")


def test_seed_everything_repeats_python_and_torch_streams() -> None:
    seed_everything(17)
    first_python = random.random()
    first_torch = torch.rand(4)

    seed_everything(17)
    second_python = random.random()
    second_torch = torch.rand(4)

    assert second_python == first_python
    torch.testing.assert_close(second_torch, first_torch)


@pytest.mark.parametrize("seed", [-1, True, "17"])
def test_seed_everything_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_everything(seed)


def test_inference_mode_disables_gradients_and_restores_state() -> None:
    assert torch.is_grad_enabled()

    with inference_mode():
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        result = torch.ones(1) + 1
        assert result.is_inference()

    assert torch.is_grad_enabled()
