"""Finite logit comparisons without retaining request content or token identities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

ACCURACY_POLICY = "fp32-paged-v2-independent"


def compare_logits(actual: Tensor, reference: Tensor) -> dict[str, float | bool]:
    import torch

    if actual.shape != reference.shape or actual.ndim < 1 or actual.shape[-1] < 2:
        raise ValueError("logit comparisons require matching shapes and at least two entries")
    if actual.dtype != reference.dtype or not actual.is_floating_point():
        raise ValueError("logit comparisons require matching floating-point dtypes")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("logit comparisons require finite values")
    values, target = actual.double(), reference.double()
    error = values - target
    rms = error.square().mean().sqrt()
    scale = target.square().mean().sqrt().clamp_min(1.0)
    top = target.topk(2, dim=-1).values
    return {
        "max_absolute_error": float(error.abs().max().item()),
        "rms_error": float(rms.item()),
        "normalized_rms_error": float((rms / scale).item()),
        "max_tolerance_ratio": float((error.abs() / (5e-3 * (1 + target.abs()))).max().item()),
        "greedy_matches": bool(torch.equal(values.argmax(-1), target.argmax(-1))),
        "reference_top1_margin_min": float((top[..., 0] - top[..., 1]).min().item()),
    }


def valid_accuracy_report(report: object, lengths: tuple[int, ...], steps: int = 3) -> bool:
    """Reject missing or failing owned-reference comparisons independently of pytest status."""
    if not isinstance(report, dict) or report.get("policy") != ACCURACY_POLICY:
        return False
    if report.get("lengths") != list(lengths) or report.get("steps") != steps:
        return False
    rows = report.get("comparisons")
    if not isinstance(rows, list) or len(rows) != len(lengths) * steps:
        return False
    observed = set()
    metrics = (
        "max_absolute_error",
        "rms_error",
        "normalized_rms_error",
        "max_tolerance_ratio",
        "reference_top1_margin_min",
    )
    for row in rows:
        if not isinstance(row, dict):
            return False
        step, request = row.get("step"), row.get("request")
        if type(step) is not int or type(request) is not int:
            return False
        if not 1 <= step <= steps or not 0 <= request < len(lengths):
            return False
        if row.get("attended_tokens") != lengths[request] + step:
            return False
        observed.add((step, request))
        comparison = row.get("owned_reference")
        if not isinstance(comparison, dict) or type(comparison.get("greedy_matches")) is not bool:
            return False
        for key in metrics:
            number = comparison.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not math.isfinite(number)
                or number < 0
            ):
                return False
        if comparison["max_tolerance_ratio"] > 1 or not comparison["greedy_matches"]:
            return False
        kv_ratio = row.get("max_kv_tolerance_ratio")
        if (
            isinstance(kv_ratio, bool)
            or not isinstance(kv_ratio, (float, int))
            or not math.isfinite(kv_ratio)
            or not 0 <= kv_ratio <= 1
        ):
            return False
    return len(observed) == len(rows)
