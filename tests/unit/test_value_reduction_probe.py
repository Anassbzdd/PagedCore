import pytest
import torch
from scripts.probe_value_reduction import probe_addition_order, scalar_reductions

pytestmark = pytest.mark.unit


def test_addition_order_probe_keeps_layout_and_distinguishes_cancellation() -> None:
    probabilities = torch.empty((1, 1, 1, 34), dtype=torch.float16)[..., ::2]
    values = torch.empty((1, 1, 17, 8), dtype=torch.float16)[..., ::2]
    probabilities.fill_(1)
    values.fill_(2)

    report = probe_addition_order(probabilities, values, head=0, channel=2)

    assert report["probability_stride"] == list(probabilities.stride())
    assert report["value_stride"] == list(values.stride())
    assert len(set(report["trial_positions"])) == 17 * 16 * 15
    outputs = dict(zip(report["trial_positions"], report["observed_outputs"], strict=True))
    assert set(outputs.values()) == {0.0, 2**-19}
    assert report["mismatches_by_model"]["fp64"] == sum(value == 0 for value in outputs.values())
    assert probabilities.eq(1).all()
    assert values.eq(2).all()


def test_scalar_probe_exposes_reduction_order_without_rounding_products_to_half() -> None:
    probabilities = torch.full((17,), 1 / 17, dtype=torch.float16)
    values = torch.zeros(17, dtype=torch.float16)
    values[0], values[1], values[16] = 65504, 0.001, -65504

    result = scalar_reductions(probabilities, values)

    assert result["sequential_rounded"] == 0
    assert result["striped_rounded"] == result["fp64_rounded"] > 0
    assert result["probabilities"] == probabilities.tolist()
    assert result["values"] == values.tolist()


@pytest.mark.parametrize("length", [1, 2, 3, 4, 15, 16, 17, 18, 49])
def test_scalar_probe_preserves_small_exact_sums(length: int) -> None:
    result = scalar_reductions(torch.ones(length).half(), torch.arange(length).half())
    expected = length * (length - 1) / 2
    assert result["fp64_sum"] == result["sequential_fp32_sum"] == expected
    assert result["striped_fp32_sum"] == expected


@pytest.mark.parametrize("case", ["empty", "shape", "dtype", "nan"])
def test_scalar_probe_rejects_invalid_operands(case: str) -> None:
    probabilities, values = torch.ones(17).half(), torch.ones(17).half()
    if case == "empty":
        probabilities, values = probabilities[:0], values[:0]
    elif case == "shape":
        values = values[None]
    elif case == "dtype":
        values = values.float()
    else:
        values[0] = torch.nan
    with pytest.raises(ValueError):
        scalar_reductions(probabilities, values)
