import pytest
import torch
from scripts.diagnose_paged_decode import AttentionScores, measurement

pytestmark = pytest.mark.unit


def test_observer_captures_probabilities_without_changing_softmax() -> None:
    scores = torch.randn(1, 32, 1, 17).half()
    expected = scores.softmax(-1, dtype=torch.float32)
    with AttentionScores() as observed:
        actual = scores.softmax(-1, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(observed.probabilities, expected, atol=0, rtol=0)
    actual.zero_()
    torch.testing.assert_close(observed.probabilities, expected, atol=0, rtol=0)


def test_observer_forces_reference_probabilities_without_replacing_measurement() -> None:
    scores = torch.zeros(4, 8, 17)
    reference = torch.zeros(1, 32, 1, 17)
    reference[..., 16] = 1
    with AttentionScores(reference) as observed:
        actual = scores.softmax(-1)
    torch.testing.assert_close(actual, reference.reshape_as(actual), atol=0, rtol=0)
    torch.testing.assert_close(observed.probabilities, scores.softmax(-1), atol=0, rtol=0)
    actual.zero_()
    assert reference[..., 16].eq(1).all()


def test_measurement_reports_first_coordinate_and_values() -> None:
    expected = torch.zeros(1, 2, 3)
    actual = expected.clone()
    actual[0, 1, 2] = 0.5
    result = measurement(actual, expected)
    assert result["first_unequal_index"] == (0, 1, 2)
    assert result["actual_at_first"] == 0.5
    assert result["expected_at_first"] == 0
    assert measurement(expected, expected)["first_unequal_index"] is None
