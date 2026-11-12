from copy import deepcopy

import pytest
import torch

from pagedcore.accuracy import ACCURACY_POLICY, compare_logits, valid_accuracy_report
from pagedcore.dense import dense_attention_fp32

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("length", [1, 17, 18, 33, 129])
def test_contiguous_fp32_attention_matches_independent_fp64(length):
    q = torch.randn(1, 32, 1, 64).half()
    k, v = (torch.randn(1, 4, length, 64).half() for _ in range(2))
    mask = torch.ones(1, 1, 1, length, dtype=torch.bool)
    expected = []
    for head in range(32):
        scores = q[0, head, 0].double() @ k[0, head // 8].double().T / 8
        expected.append(scores.softmax(-1) @ v[0, head // 8].double())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = dense_attention_fp32(q, k, v, mask)
    torch.testing.assert_close(
        actual, torch.stack(expected).reshape_as(actual).half(), atol=2e-3, rtol=2e-3
    )
    assert actual.dtype == torch.float16


def test_fp32_attention_preserves_causality_and_avoids_half_score_overflow():
    q = torch.full((1, 32, 1, 64), 100.0).half()
    k = torch.full((1, 4, 2, 64), 100.0).half()
    v = torch.ones_like(k)
    v[..., 1, :] = 100
    mask = torch.tensor([[[[True, False]]]])
    actual = dense_attention_fp32(q, k, v, mask)
    assert actual.eq(1).all()


def report():
    metrics = compare_logits(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]]))
    return {
        "policy": ACCURACY_POLICY,
        "lengths": [15],
        "steps": 3,
        "comparisons": [
            {
                "step": step,
                "request": 0,
                "attended_tokens": 15 + step,
                "owned_reference": metrics.copy(),
                "max_kv_tolerance_ratio": 0.0,
            }
            for step in range(1, 4)
        ],
    }


def test_hf_is_unnecessary_while_owned_accuracy_remains_a_hard_gate():
    data = report()
    assert valid_accuracy_report(data, (15,))
    mismatch = compare_logits(torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    assert valid_accuracy_report(data, (15,))
    data["comparisons"][0]["owned_reference"] = mismatch
    assert not valid_accuracy_report(data, (15,))


@pytest.mark.parametrize(
    "fault",
    [
        "policy",
        "lengths",
        "missing_row",
        "duplicate",
        "infinite_reference",
        "own_greedy",
        "kv",
        "step",
    ],
)
def test_accuracy_report_rejects_incomplete_or_invalid_evidence(fault):
    data = report()
    first = data["comparisons"][0]
    if fault == "policy":
        data["policy"] = "old-hf-only"
    elif fault == "lengths":
        data["lengths"] = [17]
    elif fault == "missing_row":
        data["comparisons"].pop()
    elif fault == "duplicate":
        data["comparisons"][-1] = deepcopy(first)
    elif fault == "infinite_reference":
        first["owned_reference"]["max_absolute_error"] = float("inf")
    elif fault == "own_greedy":
        first["owned_reference"]["greedy_matches"] = False
    elif fault == "kv":
        first["max_kv_tolerance_ratio"] = 1.1
    else:
        first["step"] = True
    assert not valid_accuracy_report(data, (15,))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_logit_measurements_reject_nonfinite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        compare_logits(torch.tensor([[bad, 0.0]]), torch.zeros(1, 2))
