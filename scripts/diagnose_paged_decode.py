"""Replay pinned eager inputs to separate attention rounding from batch arithmetic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import runpy
import subprocess
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama as llama

import pagedcore.attention as attention_module
import pagedcore.decoder as decoder_module
from pagedcore.attention import paged_attention
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.decoder import decode_paged, prefill_paged
from pagedcore.dense import apply_rope, rms_norm
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import TinyLlamaWeights, load_weights

ROOT = Path(__file__).resolve().parents[1]
VALUE_REDUCTIONS = ("striped", "paged", "paired", "fp64", "native_same_v")


def legacy_paged_attention(*args: Any, **kwargs: Any) -> Tensor:
    kwargs["eager_rounding"] = True
    return paged_attention(*args, **kwargs)


class AttentionScores(TorchDispatchMode):
    def __init__(self, forced_probabilities: Tensor | None = None) -> None:
        super().__init__()
        self.value: Tensor | None = None
        self.probabilities: Tensor | None = None
        self.forced_probabilities = forced_probabilities

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
    ) -> Any:
        if func in (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default):
            self.value = args[0].detach().clone()
            result = func(*args, **(kwargs or {}))
            self.probabilities = result.detach().clone()
            if self.forced_probabilities is not None:
                forced = self.forced_probabilities.reshape_as(result)
                assert forced.dtype == result.dtype and forced.device == result.device
                return forced.clone()
            return result
        return func(*args, **(kwargs or {}))


def measurement(actual: Tensor, expected: Tensor, tolerance: float = 2e-3) -> dict[str, Any]:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.float() - expected.float()).abs()
    indices = (actual != expected).nonzero()
    first = tuple(indices[0].tolist()) if indices.shape[0] else None
    return {
        "shape": list(actual.shape),
        "actual_stride": list(actual.stride()),
        "expected_stride": list(expected.stride()),
        "first_unequal_index": first,
        "actual_at_first": float(actual[first].item()) if first is not None else None,
        "expected_at_first": float(expected[first].item()) if first is not None else None,
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
        with AttentionScores() as observed:
            result = original(module, query, key, value, *args, **kw)
        assert observed.probabilities is not None
        attention[row][module.layer_idx] = tuple(
            tensor.detach().clone()
            for tensor in (query, key, value, result[0], observed.probabilities, result[1])
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


def fp64_page_value_sum(
    probabilities: Tensor, allocator: KVAllocator, request: RequestId, layer_index: int
) -> Tensor:
    total = torch.zeros((4, 8, 64), dtype=torch.float64, device=probabilities.device)
    start = 0
    for _, value in allocator.iter_layer_blocks(request, layer_index=layer_index):
        stop = start + value.shape[0]
        total = torch.baddbmm(
            total, probabilities[..., start:stop].double(), value.transpose(0, 1).double()
        )
        start = stop
    return total.reshape(1, 32, 1, 64).half()


def _layout(tensor: Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }


def _trace_decode(
    tokens: Tensor,
    weights: TinyLlamaWeights,
    allocator: KVAllocator,
    requests: tuple[RequestId, ...],
    captures: list[dict[int, tuple[Tensor, ...]]],
    guard: TorchDispatchMode,
    mode: str,
) -> tuple[Tensor, list[dict[str, Any]]]:
    reference = dict(zip(requests, captures, strict=True))
    rows: dict[RequestId, dict[str, Any]] = {
        request: {
            "request": request,
            "attended_tokens": allocator.cached_length(request) + 1,
            "layers": [],
        }
        for request in requests
    }
    expanded = {}
    if mode == "native_same_v":
        # Only independent HF storage is expanded, outside the guarded decoder.
        expanded = {
            (request, layer): llama.repeat_kv(capture[2], 8)
            for request, layers in reference.items()
            if rows[request]["attended_tokens"] > 16
            for layer, capture in layers.items()
        }

    base_attention = decoder_module.paged_attention
    base_value = attention_module._eager_value_sum

    def observe(
        query: Tensor,
        cache: KVAllocator,
        request: RequestId,
        *,
        layer_index: int,
        eager_rounding: bool,
    ) -> Tensor:
        hf_query, hf_key, hf_value, hf_output, _, hf_p = reference[request][layer_index]
        checks: dict[str, Any] = {"layer": layer_index, "q_equal": torch.equal(query, hf_query)}

        def reduce(p: Tensor, alloc: KVAllocator, req: RequestId, layer: int) -> Tensor:
            actual_p = p.reshape_as(hf_p).contiguous()
            start = 0
            k_equal = v_equal = True
            page_layouts = []
            for key, value in alloc.iter_layer_blocks(req, layer_index=layer):
                stop = start + value.shape[0]
                k_equal &= torch.equal(key, hf_key[0, :, start:stop].transpose(0, 1))
                v_equal &= torch.equal(value, hf_value[0, :, start:stop].transpose(0, 1))
                if layer == 0:
                    page_layouts.append(_layout(value))
                start = stop
            assert start == p.shape[-1] == hf_p.shape[-1]
            p_equal = torch.equal(actual_p, hf_p)
            checks.update(k_equal=k_equal, v_equal=v_equal, p_equal=p_equal)
            native_v = expanded.get((req, layer))
            if layer == 0:
                checks["layouts"] = {
                    "owned_q": _layout(query),
                    "owned_p": _layout(p),
                    "pages": page_layouts,
                    "hf_q": _layout(hf_query),
                    "hf_k": _layout(hf_key),
                    "hf_v": _layout(hf_value),
                    "hf_p": _layout(hf_p),
                }
                if native_v is not None:
                    checks["layouts"]["hf_expanded_v"] = _layout(native_v)
            if native_v is not None and v_equal:
                assert actual_p.stride() == hf_p.stride()
                output = actual_p @ native_v
                if p_equal:
                    assert torch.equal(output.transpose(1, 2), hf_output), (
                        "Native value control failed to reproduce the reference on equal operands"
                    )
                checks["reducer"] = "native_same_v"
                return output

            checks["reducer"] = (
                "fallback" if native_v is not None else ("one_page" if start <= 16 else mode)
            )
            return base_value(p, alloc, req, layer)

        with patch.object(attention_module, "_eager_value_sum", reduce):
            output = base_attention(
                query, cache, request, layer_index=layer_index, eager_rounding=eager_rounding
            )
        checks["out_equal"] = torch.equal(output.transpose(1, 2), hf_output)
        rows[request]["layers"].append(checks)
        return output

    with patch.object(decoder_module, "paged_attention", observe), guard:
        logits = decode_paged(tokens, weights, allocator, requests)
    return logits, list(rows.values())


def run(
    device: str, steps: int = 3, value_reduction: str = "striped", *, trace_value: bool = False
) -> dict[str, Any]:
    if value_reduction not in VALUE_REDUCTIONS:
        raise ValueError(f"value_reduction must be one of {VALUE_REDUCTIONS}")
    trace_value = trace_value or value_reduction in ("fp64", "native_same_v")
    guard = (
        runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
        if trace_value
        else None
    )
    fixture = ROOT / "tests/fixtures/reference_prompts.json"
    ids = next(
        case["input_ids"]
        for case in json.loads(fixture.read_text(encoding="utf-8"))["prompts"]
        if case["name"] == "maximum_output"
    )
    records: list[dict[str, Any]] = []
    value_trace: list[dict[str, Any]] = []
    with ExitStack() as cleanup, reference_math() as policy, torch.inference_mode():
        cleanup.enter_context(
            patch.object(decoder_module, "paged_attention", legacy_paged_attention)
        )
        reducer = {
            "paged": attention_module._paged_value_sum,
            "paired": attention_module._paired_value_sum,
            "fp64": fp64_page_value_sum,
        }.get(value_reduction)
        if reducer is not None:
            cleanup.enter_context(patch.object(attention_module, "_multi_page_value_sum", reducer))
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
                    for layer, (
                        query,
                        key,
                        value,
                        target,
                        hf_probabilities,
                        hf_rounded,
                    ) in attention[row].items():
                        start = 0
                        for page_key, page_value in control.iter_layer_blocks(
                            request, layer_index=layer
                        ):
                            stop = start + page_key.shape[0]
                            for kind, page_tensor, reference_tensor in (
                                ("key", page_key, key),
                                ("value", page_value, value),
                            ):
                                record(
                                    "control_storage",
                                    f"layer.{layer}.{kind}",
                                    page_tensor,
                                    reference_tensor[0, :, start:stop].transpose(0, 1),
                                    row=row,
                                )
                            start = stop
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
                            "fp32_paged_attention",
                            f"layer.{layer}.scaled_scores",
                            legacy_scores,
                            product / 8,
                            row=row,
                        )
                        for mode, eager_rounding in (
                            ("fp32_paged_attention", False),
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
                                assert scores.probabilities is not None
                                record(
                                    mode,
                                    f"layer.{layer}.scaled_scores",
                                    scores.value.reshape_as(product),
                                    product / 8,
                                    row=row,
                                )
                                record(
                                    mode,
                                    f"layer.{layer}.probabilities_fp32",
                                    scores.probabilities.reshape_as(hf_probabilities),
                                    hf_probabilities,
                                    row=row,
                                )
                                record(
                                    mode,
                                    f"layer.{layer}.probabilities_fp16",
                                    scores.probabilities.half().reshape_as(hf_rounded),
                                    hf_rounded,
                                    row=row,
                                )
                            record(
                                mode,
                                f"layer.{layer}.attention",
                                actual.transpose(1, 2),
                                target,
                                row=row,
                            )
                        with AttentionScores(hf_probabilities):
                            fixed = paged_attention(
                                query, control, request, layer_index=layer, eager_rounding=True
                            )
                        record(
                            "fixed_probabilities",
                            f"layer.{layer}.attention",
                            fixed.transpose(1, 2),
                            target,
                            row=row,
                        )
                    control.release_all(request)

                request_order = tuple(requests[row] for row in order)
                if guard is None:
                    actual = decode_paged(tokens, weights, allocator, request_order)
                else:
                    actual, layer_checks = _trace_decode(
                        tokens,
                        weights,
                        allocator,
                        request_order,
                        attention,
                        guard(pool),
                        value_reduction,
                    )
                    value_trace.append({"lengths": lengths, "step": step + 1, "rows": layer_checks})
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
        "value_reduction": value_reduction,
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
        "integrated_value_trace": value_trace,
    }


def _integrated_passed(report: dict[str, Any]) -> bool:
    summary = report["summaries"]["integrated"]
    return summary["max_tolerance_ratio"] <= 1 and summary["all_greedy_matches"]


def _probe_value_reduction(device: str, steps: int, output: Path) -> None:
    runtime: dict[str, object] = {"device": device, "seed": 0, "steps": steps}
    if device == "cuda:0":
        runtime.update(
            cublas_wheel=version("nvidia-cublas-cu12"),
            selected_uuid=str(torch.cuda.get_device_properties(0).uuid),
            compute_capability=torch.cuda.get_device_capability(0),
            nvidia_smi=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("runtime.json", json.dumps(runtime, indent=2))
        for mode in ("striped", "fp64", "native_same_v"):
            seed_everything(0)
            report = run(device, steps, mode, trace_value=True)
            archive.writestr(f"{mode}.json", json.dumps(report, allow_nan=False))
            print("PARITY", mode, json.dumps(report["summaries"]["integrated"]), flush=True)
            passed = _integrated_passed(report)
            if mode == "striped" and passed:
                print("Baseline passed. Check runtime provenance before interpreting the probe.")
                break
            if mode == "fp64" and passed:
                break
            del report
            gc.collect()
            if device == "cuda:0":
                torch.cuda.empty_cache()
    print("REPORT", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--value-reduction", choices=VALUE_REDUCTIONS, default="striped")
    mode.add_argument(
        "--probe-value-reduction", action="store_true", help="Run the controlled A/B probe"
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    if arguments.probe_value_reduction:
        _probe_value_reduction(
            arguments.device,
            arguments.steps,
            arguments.output or Path("results/local/pagedcore-parity-probe.zip"),
        )
        raise SystemExit(0)
    output = arguments.output or Path("results/local/paged-decode-arithmetic.json")
    report = run(arguments.device, arguments.steps, arguments.value_reduction)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2))
    if not _integrated_passed(report):
        raise SystemExit("Integrated parity still fails; retain this diagnostic report.")
