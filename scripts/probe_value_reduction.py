"""Capture the small weighted-value reductions behind the paged-decode mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama as llama

from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.oracle import reference_math


def scalar_reductions(probabilities: Tensor, values: Tensor) -> dict[str, Any]:
    if probabilities.ndim != 1 or probabilities.shape != values.shape or not values.numel():
        raise ValueError("probabilities and values must be equal-length nonempty vectors")
    if probabilities.dtype != torch.float16 or values.dtype != torch.float16:
        raise ValueError("the replay requires exact FP16 operands")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(values).all():
        raise ValueError("the replay requires finite operands")
    probabilities, values = (tensor.detach().cpu() for tensor in (probabilities, values))
    products = probabilities.float() * values.float()
    serial = torch.zeros(())
    lanes = torch.zeros(4)
    tail = products.numel() - products.numel() % 4
    for position, product in enumerate(products):
        serial.add_(product)
        lanes[position % 4 if position < tail else 0].add_(product)
    striped = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]
    high_precision = products.double().sum()
    return {
        "probabilities": probabilities.tolist(),
        "values": values.tolist(),
        "fp64_sum": high_precision.item(),
        "fp64_rounded": high_precision.half().item(),
        "sequential_fp32_sum": serial.item(),
        "sequential_rounded": serial.half().item(),
        "striped_fp32_sum": striped.item(),
        "striped_rounded": striped.half().item(),
    }


def _profile_matmul(probabilities: Tensor, values: Tensor) -> tuple[Tensor, list[str]]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        output = torch.matmul(probabilities, values)
        torch.cuda.synchronize(probabilities.device)
    kernels = sorted(
        {event.name for event in profiler.events() if "CUDA" in str(event.device_type)}
    )
    return output, kernels


def probe_addition_order(
    probabilities: Tensor, values: Tensor, *, head: int, channel: int
) -> dict[str, Any]:
    positions = list(permutations(range(17), 3))
    probability = 2**-5
    terms = (32768.0, -32768.0, 2**-14)
    vectors = torch.zeros((len(positions), 17), dtype=torch.float16)
    indices = torch.tensor(positions)
    vectors.scatter_(1, indices, torch.tensor(terms, dtype=vectors.dtype).expand_as(indices))
    products = vectors.float() * probability
    probe_p, probe_v = (
        torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        for x in (probabilities, values)
    )
    probe_p.fill_(probability)
    observed = []
    for vector in vectors.to(values.device):
        # A small term disappears if added to a large term before cancellation.
        probe_v.copy_(vector[None, None, :, None])
        observed.append((probe_p @ probe_v)[0, head, 0, channel].clone())
    actual = torch.stack(observed).cpu()

    mismatches = {"fp64": int(actual.ne(terms[2] * probability).sum().item())}
    for name, lanes in (("sequential_fp32", 1), ("two_lane_fp32", 2), ("four_lane_fp32", 4)):
        partials = torch.zeros((lanes, len(positions)))
        tail = 17 - 17 % lanes
        for slot in range(17):
            partials[slot % lanes if slot < tail else 0].add_(products[:, slot])
        output = partials[0]
        for partial in partials[1:]:
            output = output + partial
        mismatches[name] = int(output.half().ne(actual).sum().item())

    kernels = _profile_matmul(probe_p, probe_v)[1] if probabilities.is_cuda else []
    return {
        "probability_shape": list(probe_p.shape),
        "probability_stride": list(probe_p.stride()),
        "value_shape": list(probe_v.shape),
        "value_stride": list(probe_v.stride()),
        "head": head,
        "channel": channel,
        "probability": probability,
        "values_in_trial_order": terms,
        "cuda_kernels": kernels,
        "mismatches_by_model": mismatches,
        "trial_positions": positions,
        "observed_outputs": actual.tolist(),
    }


@torch.inference_mode()
def run(device: str, *, profile: bool = False, probe_order: bool = False) -> dict[str, Any]:
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/reference_prompts.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))["prompts"]
    ids = next(case["input_ids"] for case in cases if case["name"] == "maximum_output")[:15]
    original = llama.eager_attention_forward
    records = []

    def observe(
        module: Any, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kw: Any
    ) -> Any:
        result = original(module, query, key, value, *args, **kw)
        if query.shape[-2] != 1 or key.shape[-2] != 17:
            return result
        probabilities = result[1].reshape(32, 17)
        packed_values = llama.repeat_kv(value, 8)
        values = packed_values.reshape(32, 17, 64)
        target = result[0].reshape(32, 64)
        high_precision = torch.matmul(probabilities.double()[:, None], values.double())[:, 0]
        serial = torch.zeros((32, 64), device=query.device)
        for slot in range(17):
            serial.add_(probabilities[:, slot, None].float() * values[:, slot].float())
        record: dict[str, Any] = {
            "layer": module.layer_idx,
            "fp64_unequal": int((high_precision.half() != target).sum().item()),
            "sequential_unequal": int((serial.half() != target).sum().item()),
        }
        selected = {1: (2, 41), 4: (20, 19)}
        if module.layer_idx in selected:
            head, dimension = selected[module.layer_idx]
            record.update(scalar_reductions(probabilities[head], values[head, :, dimension]))
            record.update(head=head, dimension=dimension, hf=target[head, dimension].item())
            sample_order = probe_order and module.layer_idx == 1
            if (profile or sample_order) and query.is_cuda:
                replay, kernels = _profile_matmul(result[1], packed_values)
                torch.testing.assert_close(replay.transpose(1, 2), result[0], atol=0, rtol=0)
                record["cuda_kernels"] = kernels
            if sample_order:
                order = probe_addition_order(result[1], packed_values, head=head, channel=dimension)
                order["shared_gemvx_kernels"] = sorted(
                    name
                    for name in set(record.get("cuda_kernels", ())) & set(order["cuda_kernels"])
                    if "gemvx" in name
                )
                record["addition_order"] = order
        records.append(record)
        return result

    with reference_math() as policy:
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        with patch.object(llama, "eager_attention_forward", observe):
            reference = model(torch.tensor([ids], device=device), use_cache=True)
            for _ in range(2):
                reference = model(
                    reference.logits[:, -1:].argmax(-1),
                    past_key_values=reference.past_key_values,
                    use_cache=True,
                )
    if len(records) != 22:
        raise RuntimeError("expected one 17-token replay for every decoder layer")
    return {
        "scope": "17-token weighted-value diagnostic; not release evidence",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "device": device,
        "environment": collect_environment_diagnostics(),
        "provenance": collect_validation_provenance(),
        "math_policy": policy,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--probe-order", action="store_true", help="sweep three-term cancellation at layer 1"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/local/value-reduction-probe.json")
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = run(arguments.device, profile=arguments.profile, probe_order=arguments.probe_order)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if arguments.probe_order:
        order = next(
            record["addition_order"] for record in report["records"] if "addition_order" in record
        )
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "trials": len(order["trial_positions"]),
                    "shared_gemvx_kernels": order["shared_gemvx_kernels"],
                    "mismatches_by_model": order["mismatches_by_model"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))
