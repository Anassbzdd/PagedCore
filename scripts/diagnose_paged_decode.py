"""Replay pinned eager inputs to separate attention rounding from batch arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama as llama

from pagedcore.attention import paged_attention
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.decoder import decode_paged, prefill_paged
from pagedcore.dense import apply_rope, rms_norm
from pagedcore.engine_types import KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights


class AttentionScores(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.value: Tensor | None = None

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
    ) -> Any:
        if func in (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default):
            self.value = args[0].detach().clone()
        return func(*args, **(kwargs or {}))


def measurement(actual: Tensor, expected: Tensor, tolerance: float = 2e-3) -> dict[str, Any]:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.float() - expected.float()).abs()
    return {
        "unequal": int((actual != expected).sum().item()),
        "max_absolute_error": float(error.max().item()),
        "max_tolerance_ratio": float(
            (error / (tolerance * (1 + expected.float().abs()))).max().item()
        ),
    }


def component_controls(
    model: Any, tokens: Tensor, references: list[Any]
) -> tuple[list[Any], list[dict[str, tuple[Tensor, Tensor]]], list[dict[int, tuple[Tensor, ...]]]]:
    """Observe the unchanged oracle; each control later replays its exact operands."""
    rows: list[dict[str, tuple[Tensor, Tensor]]] = [{} for _ in references]
    attention: list[dict[int, tuple[Tensor, ...]]] = [{} for _ in references]
    row = 0

    def hook(name: str) -> Any:
        def save(module: Any, args: Any, output: Tensor) -> None:
            rows[row][name] = (args[0].detach().clone(), output.detach().clone())

        return save

    original = llama.eager_attention_forward
    original_rope = llama.apply_rotary_pos_emb
    rope_layer = 0

    def rope(query: Tensor, key: Tensor, *args: Any, **kw: Any) -> Any:
        nonlocal rope_layer
        result = original_rope(query, key, *args, **kw)
        for kind, source, target in (("query", query, result[0]), ("key", key, result[1])):
            rows[row][f"model.layers.{rope_layer}.rope_{kind}"] = (
                source.detach().clone(),
                target.detach().clone(),
            )
        rope_layer += 1
        return result

    def eager(module: Any, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kw: Any) -> Any:
        result = original(module, query, key, value, *args, **kw)
        attention[row][module.layer_idx] = tuple(
            tensor.detach().clone() for tensor in (query, key, value, result[0])
        )
        return result

    expected = []
    with ExitStack() as cleanup:
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.SiLU, llama.LlamaRMSNorm)):
                cleanup.callback(module.register_forward_hook(hook(name)).remove)
        cleanup.enter_context(patch.object(llama, "eager_attention_forward", eager))
        cleanup.enter_context(patch.object(llama, "apply_rotary_pos_emb", rope))
        for row, reference in enumerate(references):
            rope_layer = 0
            expected.append(
                model(
                    tokens[row : row + 1], past_key_values=reference.past_key_values, use_cache=True
                )
            )
    return expected, rows, attention


def run(device: str, steps: int = 3) -> dict[str, Any]:
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/reference_prompts.json"
    ids = next(
        case["input_ids"]
        for case in json.loads(fixture.read_text(encoding="utf-8"))["prompts"]
        if case["name"] == "maximum_output"
    )
    records: list[dict[str, Any]] = []
    with reference_math() as policy, torch.inference_mode():
        weights = load_weights(device=device)
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
        modules = dict(model.named_modules())
        for lengths in ((15,), (1, 15, 16, 17, 49)):
            requests = tuple(RequestId(str(row)) for row in range(len(lengths)))
            pool = torch.full((22, 2, 16, 16, 4, 64), torch.nan, dtype=torch.float16, device=device)
            allocator = KVAllocator(pool)
            references = []
            for request, length in zip(requests, lengths, strict=True):
                allocator.create(KVReservation(request, (length + 18) // 16))
                inputs = torch.tensor([ids[:length]], device=device)
                references.append(model(inputs, use_cache=True))
                prefill_paged(inputs, weights, allocator, request)
            for step in range(steps):
                order = (
                    tuple(range(len(requests)))
                    if step % 2 == 0
                    else tuple(reversed(range(len(requests))))
                )
                ordered = [references[row] for row in order]
                tokens = torch.cat([reference.logits[:, -1:].argmax(-1) for reference in ordered])
                expected, rows, attention = component_controls(model, tokens, ordered)

                def record(
                    mode: str,
                    operation: str,
                    actual: Tensor,
                    target: Tensor,
                    lengths: tuple[int, ...] = lengths,
                    step: int = step,
                    **extra: Any,
                ) -> None:
                    records.append(
                        {
                            "lengths": lengths,
                            "step": step + 1,
                            "mode": mode,
                            "operation": operation,
                            **extra,
                            **measurement(actual, target, 5e-3 if operation == "logits" else 2e-3),
                        }
                    )

                # Replaying each HF operation's input excludes propagated upstream drift.
                for name in rows[0]:
                    if name.endswith("rope_key"):
                        continue
                    if name.endswith("rope_query"):
                        key_name = name.replace("rope_query", "rope_key")
                        positions = torch.tensor(
                            [
                                [reference.past_key_values.get_seq_length() - 1]
                                for reference in expected
                            ],
                            device=device,
                        )
                        queries = torch.cat([values[name][0] for values in rows])
                        keys = torch.cat([values[key_name][0] for values in rows])
                        rotated = apply_rope(queries, keys, positions)
                        isolated = [
                            apply_rope(
                                values[name][0], values[key_name][0], positions[row : row + 1]
                            )
                            for row, values in enumerate(rows)
                        ]
                        for index, label in enumerate((name, key_name)):
                            target = torch.cat([values[label][1] for values in rows])
                            record("batched_components", label, rotated[index], target)
                            record(
                                "isolated_components",
                                label,
                                torch.cat([value[index] for value in isolated]),
                                target,
                            )
                        continue
                    inputs = torch.cat([values[name][0] for values in rows])
                    target = torch.cat([values[name][1] for values in rows])
                    module = modules[name]

                    def operation(value: Tensor, module: Any = module) -> Tensor:
                        if isinstance(module, torch.nn.Linear):
                            return F.linear(value, module.weight)
                        if isinstance(module, llama.LlamaRMSNorm):
                            return rms_norm(value, module.weight)
                        return F.silu(value)

                    record("batched_components", name, operation(inputs), target)
                    record(
                        "isolated_components",
                        name,
                        torch.cat([operation(value[name][0]) for value in rows]),
                        target,
                    )

                # Attention uses the oracle's exact Q and complete current cache,
                # copied into pages only outside the serving forward.
                control_pool = torch.full_like(pool, torch.nan)
                control = KVAllocator(control_pool)
                for row, reference in enumerate(expected):
                    length = reference.past_key_values.get_seq_length()
                    request = requests[order[row]]
                    control.create(KVReservation(request, (length + 15) // 16))
                    control.append(
                        request,
                        torch.stack(
                            [key[0].transpose(0, 1) for key, _ in reference.past_key_values]
                        ),
                        torch.stack(
                            [value[0].transpose(0, 1) for _, value in reference.past_key_values]
                        ),
                    )
                    for layer, (query, key, _value, target) in attention[row].items():
                        # Full score tensors are diagnostic outputs, never gathered K/V.
                        product = torch.matmul(query, llama.repeat_kv(key, 8).transpose(-1, -2))
                        legacy_scores = torch.empty_like(product, dtype=torch.float32)
                        start = 0
                        for page, _ in control.iter_layer_blocks(request, layer_index=layer):
                            stop = start + page.shape[0]
                            partial = torch.matmul(
                                query.reshape(4, 8, 64).float(), page.permute(1, 2, 0).float()
                            ).reshape(1, 32, 1, -1)
                            legacy_scores[..., start:stop] = partial / 8
                            start = stop
                        record(
                            "online_attention",
                            f"layer.{layer}.scaled_scores",
                            legacy_scores,
                            product / 8,
                            row=row,
                        )
                        for mode, eager_rounding in (
                            ("online_attention", False),
                            ("eager_attention", True),
                        ):
                            with AttentionScores() as scores:
                                actual = paged_attention(
                                    query,
                                    control,
                                    request,
                                    layer_index=layer,
                                    eager_rounding=eager_rounding,
                                )
                            if eager_rounding:
                                assert scores.value is not None
                                record(
                                    mode,
                                    f"layer.{layer}.scaled_scores",
                                    scores.value.reshape_as(product),
                                    product / 8,
                                    row=row,
                                )
                            record(
                                mode,
                                f"layer.{layer}.attention",
                                actual.transpose(1, 2),
                                target,
                                row=row,
                            )
                    control.release_all(request)

                actual = decode_paged(
                    tokens, weights, allocator, tuple(requests[row] for row in order)
                )
                for row, reference in enumerate(expected):
                    record(
                        "integrated",
                        "logits",
                        actual[row : row + 1],
                        reference.logits,
                        row=row,
                        greedy_matches=torch.equal(
                            actual[row : row + 1].argmax(-1), reference.logits.argmax(-1)
                        ),
                    )
                    request = requests[order[row]]
                    for logical in range(len(allocator.block_table(request))):
                        key, value = allocator.read_block(request, logical)
                        start = logical * 16
                        for layer, pair in enumerate(reference.past_key_values):
                            for kind, actual_kv, target_kv in (
                                ("key", key[layer], pair[0]),
                                ("value", value[layer], pair[1]),
                            ):
                                record(
                                    "integrated",
                                    f"layer.{layer}.{kind}",
                                    actual_kv,
                                    target_kv[0, :, start : start + actual_kv.shape[0]].transpose(
                                        0, 1
                                    ),
                                    row=row,
                                )
                    references[order[row]] = reference
                print(f"compared batch={len(lengths)} step={step + 1}", flush=True)
            for request in requests:
                allocator.release_all(request)
    summaries = {}
    for mode in dict.fromkeys(record["mode"] for record in records):
        selected = [record for record in records if record["mode"] == mode]
        summaries[mode] = {
            "first_unequal": next((record for record in selected if record["unequal"]), None),
            "first_failure": next(
                (record for record in selected if record["max_tolerance_ratio"] > 1), None
            ),
            "max_tolerance_ratio": max(record["max_tolerance_ratio"] for record in selected),
            "all_greedy_matches": all(record.get("greedy_matches", True) for record in selected),
        }
    return {
        "scope": "Diagnostic replay; not T4 release evidence",
        "device": device,
        "steps": steps,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "model_revision": MODEL_REVISION,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": collect_validation_provenance(),
        "environment": collect_environment_diagnostics(),
        "math_policy": policy,
        "summaries": summaries,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/local/paged-decode-arithmetic.json")
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = run(arguments.device, arguments.steps)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2))
    if (
        report["summaries"]["integrated"]["max_tolerance_ratio"] > 1
        or not report["summaries"]["integrated"]["all_greedy_matches"]
    ):
        raise SystemExit("Integrated parity still fails; retain this diagnostic report.")
