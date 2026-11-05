"""Capture the small weighted-value reductions behind the paged-decode mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
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


@torch.inference_mode()
def run(device: str, *, profile: bool = False) -> dict[str, Any]:
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
        values = llama.repeat_kv(value, 8).reshape(32, 17, 64)
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
            if profile and query.is_cuda:
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with torch.profiler.profile(activities=activities) as profiler:
                    replay = torch.matmul(result[1], llama.repeat_kv(value, 8))
                    torch.cuda.synchronize(query.device)
                torch.testing.assert_close(replay.transpose(1, 2), result[0], atol=0, rtol=0)
                record["cuda_kernels"] = sorted(
                    {event.name for event in profiler.events() if "CUDA" in str(event.device_type)}
                )
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
        "--output", type=Path, default=Path("results/local/value-reduction-probe.json")
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = run(arguments.device, profile=arguments.profile)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
