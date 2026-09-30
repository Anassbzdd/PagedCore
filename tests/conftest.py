import pytest

from pagedcore.determinism import seed_everything

TEST_SEED = 0

@pytest.fixture(autouse=True)
def deterministic_test_state() -> None:
    """Start every test with the same Python and PyTorch RNG state."""
    seed_everything(TEST_SEED)
