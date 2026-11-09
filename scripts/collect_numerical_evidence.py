"""Collect real eager operands and guarded decoder controls without changing serving math."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import runpy
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from scripts.diagnose_paged_decode import AttentionScores, measurement
from scripts.probe_value_reduction import _profile_matmul
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama as llama

import pagedcore.attention as attention
import pagedcore.decoder as decoder
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (
    4,
    15,
    16,
    17,
    18,
    19,
    24,
    30,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
)
ARMS = ("baseline", "paired17", "family_hypothesis")
Guard = runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def layout(value: Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
    }


def guarded_measurement(actual: Tensor, expected: Tensor) -> dict[str, Any]:
    """Find one differing coordinate without an oversized nonzero index buffer."""
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    mask = actual != expected
    count = int(mask.sum().item())
    first = None
    if count:
        flat = int(mask.flatten().to(torch.uint8).argmax().item())
        indices = []
        for size in reversed(actual.shape):
            indices.append(flat % size)
            flat //= size
        first = tuple(reversed(indices))
    error = (actual.float() - expected.float()).abs()
    return {
        "shape": list(actual.shape),
        "actual_stride": list(actual.stride()),
        "expected_stride": list(expected.stride()),
        "first_unequal_index": first,
        "actual_at_first": float(actual[first].item()) if first is not None else None,
        "expected_at_first": float(expected[first].item()) if first is not None else None,
        "unequal": count,
        "max_absolute_error": float(error.max().item()),
        "max_tolerance_ratio": float((error / (2e-3 * (1 + expected.float().abs()))).max().item()),
    }


def family(names: list[str]) -> str:
    variants = {
        match.group(1)
        for name in names
        if (match := re.search(r"false, (\d+), false, cublasGemv", name))
    }
    if len(variants) == 1:
        return "gemvx_" + variants.pop()
    if any("gemv" in name.lower() and "nsp" in name.lower() for name in names):
        return "gemv_nsp"
    return "unknown"


def profile_product(left: Tensor, right: Tensor) -> tuple[Tensor, list[str]]:
    return _profile_matmul(left, right) if left.is_cuda else (left @ right, [])


def predictions(probabilities: Tensor, values: Tensor) -> dict[str, Tensor]:
    """Replay all head/channel outputs in FP32; these are reference-only CPU arrays."""
    p = probabilities.detach().cpu().numpy().reshape(32, -1)
    v = values.detach().cpu().numpy()[0]
    length = p.shape[-1]
    partials = {
        name: np.zeros((lanes, 32, 64), dtype=np.float32)
        for name, lanes in (("alternating", 2), ("halves", 2), ("striped4", 4))
    }
    precise = np.zeros((32, 64), dtype=np.float64)
    for slot in range(length):
        product = p[:, slot, None].astype(np.float32) * v[np.arange(32) // 8, slot].astype(
            np.float32
        )
        precise += product
        partials["alternating"][slot % 2] += product
        partials["halves"][int(slot >= (length + 1) // 2)] += product
        partials["striped4"][slot % 4 if slot < length - length % 4 else 0] += product
    outputs = {"fp64": torch.from_numpy(precise.astype(np.float16)).reshape(1, 32, 1, 64)}
    for name, lanes in partials.items():
        total = lanes[0].copy()
        for lane in lanes[1:]:
            total += lane
        outputs[name] = torch.from_numpy(total.astype(np.float16)).reshape(1, 32, 1, 64)
    return outputs


def halves_page_sum(p: Tensor, allocator: KVAllocator, request: RequestId, layer: int) -> Tensor:
    halves = [torch.zeros((4, 8, 64), dtype=torch.float32, device=p.device) for _ in range(2)]
    middle = (p.shape[-1] + 1) // 2
    position = 0
    for _, page in allocator.iter_layer_blocks(request, layer_index=layer):
        values = page.float()
        for slot in range(page.shape[0]):
            # A rank-one product preserves the scalar arithmetic and marks the
            # transition from cached V to attention output for the storage guard.
            halves[int(position >= middle)].add_(
                torch.matmul(p[..., position, None].float(), values[slot, :, None, :])
            )
            position += 1
    assert position == p.shape[-1]
    return (halves[0] + halves[1]).reshape(1, 32, 1, 64).half()


def choose_reducer(arm: str, length: int, kernel_family: str) -> tuple[str, Any]:
    if arm == "paired17" and length == 17:
        return "alternating", attention._paired_value_sum
    if arm == "family_hypothesis":
        if kernel_family == "gemvx_5":
            return "alternating", attention._paired_value_sum
        if kernel_family == "gemvx_6" and length % 2 == 0:
            return "halves", halves_page_sum
    return "baseline", attention._eager_value_sum


@contextmanager
def capture_eager(directory: Path):
    """Record original layouts before copies; every replay must reproduce eager exactly."""
    captures: dict[int, dict[str, Tensor]] = {}
    records: list[dict[str, Any]] = []
    original = llama.eager_attention_forward
    directory.mkdir(parents=True, exist_ok=True)

    def observe(module: Any, q: Tensor, k: Tensor, v: Tensor, *args: Any, **kwargs: Any):
        eager_kernels = []
        if q.is_cuda:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with (
                torch.profiler.profile(activities=activities) as profiler,
                AttentionScores() as observed,
            ):
                result = original(module, q, k, v, *args, **kwargs)
                torch.cuda.synchronize(q.device)
            eager_kernels = sorted(
                {event.name for event in profiler.events() if "CUDA" in str(event.device_type)}
            )
        else:
            with AttentionScores() as observed:
                result = original(module, q, k, v, *args, **kwargs)
        assert observed.value is not None and observed.probabilities is not None
        p = result[1]
        expanded_k, expanded_v = llama.repeat_kv(k, 8), llama.repeat_kv(v, 8)
        raw_dot, score_kernels = profile_product(q, expanded_k.transpose(-1, -2))
        value_out, value_kernels = profile_product(p, expanded_v)
        torch.testing.assert_close(value_out.transpose(1, 2), result[0], atol=0, rtol=0)
        # No mask is needed for a single last-position query and a valid-length cache.
        scaled = raw_dot * module.scaling
        torch.testing.assert_close(scaled.float(), observed.value.float(), atol=0, rtol=0)
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "scores": observed.value,
            "probabilities_fp32": observed.probabilities,
            "probabilities": p,
            "output": value_out,
            "raw_dot": raw_dot,
        }
        record: dict[str, Any] = {
            "layer": module.layer_idx,
            "attended_tokens": k.shape[-2],
            "layouts": {name: layout(tensor) for name, tensor in tensors.items()},
            "matmul_layouts": {
                "score_left": layout(q),
                "score_right": layout(expanded_k.transpose(-1, -2)),
                "value_left": layout(p),
                "value_right": layout(expanded_v),
            },
            "score_kernels": score_kernels,
            "value_kernels": value_kernels,
            "actual_eager_kernels": eager_kernels,
            "shared_value_kernels": sorted(set(value_kernels) & set(eager_kernels)),
            "kernel_capture_complete": bool(eager_kernels and score_kernels and value_kernels)
            if q.is_cuda
            else None,
            "value_family": family(sorted(set(value_kernels) & set(eager_kernels))),
            "native_replay_exact": True,
        }
        captures[module.layer_idx] = tensors
        outputs = predictions(p, v)
        record["models"] = {
            name: measurement(out, value_out.cpu()) for name, out in outputs.items()
        }
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in tensors.items()} | outputs,
            directory / f"layer-{module.layer_idx:02d}.pt",
        )
        records.append(record)
        write_json(directory / "reference.json", records)
        return result

    with patch.object(llama, "eager_attention_forward", observe):
        yield captures, records


def public_input(length: int, device: str) -> Tensor:
    fixture = json.loads((ROOT / "tests/fixtures/reference_prompts.json").read_text())
    ids = next(case["input_ids"] for case in fixture["prompts"] if case["name"] == "maximum_output")
    # Longer synthetic contexts repeat fixed public token IDs; no text is logged.
    repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    return torch.tensor([repeated], device=device)


def replay_score_layouts(captured: dict[str, Tensor], directory: Path) -> dict[str, Any]:
    """Change packing and output width separately on independent reference storage."""
    q, k = captured["q"], llama.repeat_kv(captured["k"], 8)
    length = k.shape[-2]
    outputs = {
        name: torch.empty_like(captured["raw_dot"])
        for name in ("page_view", "page_packed", "full_width_view", "full_width_packed")
    }
    records = []
    for start in range(0, length, 16):
        stop = min(start + 16, length)
        page = k[..., start:stop, :]
        wide = torch.zeros_like(k)
        wide[..., start:stop, :] = page
        for name, right in (
            ("page_view", page.transpose(-1, -2)),
            ("page_packed", page.transpose(-1, -2).contiguous()),
            ("full_width_view", wide.transpose(-1, -2)),
            ("full_width_packed", wide.transpose(-1, -2).contiguous()),
        ):
            out, kernels = profile_product(q, right)
            selected = out if name.startswith("page") else out[..., start:stop]
            outputs[name][..., start:stop] = selected
            records.append(
                {
                    "start": start,
                    "stop": stop,
                    "variant": name,
                    "left": layout(q),
                    "right": layout(right),
                    "kernels": kernels,
                    "raw_dot": measurement(selected, captured["raw_dot"][..., start:stop]),
                }
            )
    comparisons = {}
    for name, out in outputs.items():
        scaled = out * (64**-0.5)
        p32 = scaled.softmax(-1, dtype=torch.float32)
        comparisons[name] = {
            "raw_dot": measurement(out, captured["raw_dot"]),
            "scaled_scores": measurement(scaled.float(), captured["scores"].float()),
            "probabilities": measurement(p32.half(), captured["probabilities"]),
        }
    torch.save(
        {name: value.cpu() for name, value in outputs.items()}, directory / "score-layouts.pt"
    )
    return {
        "scope": "Reference-only K; full-width controls are forbidden in serving",
        "pages": records,
        "comparisons": comparisons,
    }


def score_controls(model: Any, length: int, directory: Path) -> dict[str, Any]:
    device = str(model.device)
    reference = model(public_input(length - 1, device), use_cache=True)
    with capture_eager(directory) as (captures, records):
        reference = model(
            reference.logits[:, -1:].argmax(-1),
            past_key_values=reference.past_key_values,
            use_cache=True,
        )
    assert [record["layer"] for record in records] == list(range(22))
    blocks = (length + 15) // 16
    pool = torch.full((22, 2, blocks * 2, 16, 4, 64), torch.nan, device=device, dtype=torch.float16)
    allocator = KVAllocator(pool)
    request, blocker = RequestId("control"), RequestId("blocker")
    allocator.create(KVReservation(request, blocks))
    allocator.create(KVReservation(blocker, blocks))
    # Alternate real and poisoned pages. Preparation is outside all guarded forwards.
    for start in range(0, length, 16):
        stop = min(start + 16, length)
        allocator.append(
            request,
            torch.stack(
                [k[0, :, start:stop].transpose(0, 1) for k, _ in reference.past_key_values]
            ),
            torch.stack(
                [v[0, :, start:stop].transpose(0, 1) for _, v in reference.past_key_values]
            ),
        )
        sentinel = torch.full((22, 16, 4, 64), 100, device=device, dtype=torch.float16)
        allocator.append(blocker, -sentinel, sentinel)
    before = allocator.capacity()
    layout_replayed = False
    for record in records:
        layer = record["layer"]
        captured = captures[layer]
        # Isolate the score path from the known value error and avoid a long scalar GPU reduction.
        with (
            patch.object(attention, "_eager_value_sum", return_value=captured["output"]),
            Guard(pool),
            AttentionScores() as observed,
        ):
            attention.paged_attention(
                captured["q"], allocator, request, layer_index=layer, eager_rounding=True
            )
        record["paged_scores"] = measurement(
            observed.value.reshape_as(captured["scores"]), captured["scores"]
        )
        record["paged_probabilities_fp32"] = measurement(
            observed.probabilities.reshape_as(captured["probabilities_fp32"]),
            captured["probabilities_fp32"],
        )
        record["paged_probabilities_fp16"] = measurement(
            observed.probabilities.half().reshape_as(captured["probabilities"]),
            captured["probabilities"],
        )
        if record["paged_scores"]["unequal"] and not layout_replayed:
            record["score_layout_controls"] = replay_score_layouts(captured, directory)
            layout_replayed = True
        # Retain the actual paged scores/probabilities for offline score-layout experiments.
        torch.save(
            {"scores": observed.value.cpu(), "probabilities_fp32": observed.probabilities.cpu()},
            directory / f"paged-scores-{layer:02d}.pt",
        )
        assert allocator.capacity() == before
        write_json(directory / "reference.json", records)
    tables = {str(req): allocator.block_table(req) for req in (request, blocker)}
    assert tables[str(request)] == tuple(range(0, blocks * 2, 2))
    allocator.release_all(request)
    allocator.release_all(blocker)
    assert allocator.capacity() == CapacityCounts(blocks * 2, blocks * 2, 0, 0, 0, 0)
    return {
        "length": length,
        "layers": len(records),
        "tables": tables,
        "cleanup": asdict(allocator.capacity()),
        "guard": "passed",
        "kernel_capture_complete": all(record["kernel_capture_complete"] for record in records),
    }


def first_difference(layers: list[dict[str, Any]]) -> dict[str, Any] | None:
    order = ("q", "k", "v", "scores", "probabilities_fp32", "probabilities", "weighted_value")
    for layer in layers:
        for operation in order:
            check = layer[operation]
            if check["unequal"]:
                return {"layer": layer["layer"], "operation": operation, "measurement": check}
    return None


def trace_decode(
    tokens: Tensor,
    weights: Any,
    allocator: KVAllocator,
    requests: tuple[RequestId, ...],
    captures: dict[RequestId, Any],
    records: dict[RequestId, Any],
    arm: str,
    directory: Path,
    guard: Any,
) -> tuple[Tensor, dict[RequestId, Any]]:
    base = decoder.paged_attention
    results: dict[RequestId, Any] = {req: [] for req in requests}
    owned: dict[str, Tensor] = {}

    def observe(
        q: Tensor, cache: KVAllocator, request: RequestId, *, layer_index: int, eager_rounding: bool
    ):
        hf = captures[request][layer_index]
        kernel_family = records[request][layer_index]["value_family"]
        checks: dict[str, Any] = {"layer": layer_index, "q": guarded_measurement(q, hf["q"])}
        for kind in ("k", "v"):
            pages = []
            start = 0
            for pair in cache.iter_layer_blocks(request, layer_index=layer_index):
                page = pair[kind == "v"]
                stop = start + page.shape[0]
                pages.append(guarded_measurement(page, hf[kind][0, :, start:stop].transpose(0, 1)))
                start = stop
            assert start == hf[kind].shape[-2]
            checks[kind] = {"unequal": sum(item["unequal"] for item in pages), "pages": pages}
        name, reducer = choose_reducer(arm, hf["probabilities"].shape[-1], kernel_family)
        checks.update(value_family=kernel_family, reducer=name)

        def reduce(p: Tensor, alloc: KVAllocator, req: RequestId, layer: int):
            checks["probabilities"] = guarded_measurement(
                p.reshape_as(hf["probabilities"]), hf["probabilities"]
            )
            out = reducer(p, alloc, req, layer)
            checks["weighted_value"] = guarded_measurement(out, hf["output"])
            return out

        with patch.object(attention, "_eager_value_sum", reduce), AttentionScores() as scores:
            out = base(q, cache, request, layer_index=layer_index, eager_rounding=True)
        checks["scores"] = guarded_measurement(scores.value.reshape_as(hf["scores"]), hf["scores"])
        checks["probabilities_fp32"] = guarded_measurement(
            scores.probabilities.reshape_as(hf["probabilities_fp32"]), hf["probabilities_fp32"]
        )
        results[request].append(checks)
        # Outputs may span context; no owned K/V history is copied or retained here.
        for name, value in {
            "q": q,
            "scores": scores.value,
            "p32": scores.probabilities,
            "output": out,
        }.items():
            owned[f"{request}/layer-{layer_index:02d}/{name}"] = value.detach().cpu()
        return out

    try:
        with patch.object(decoder, "paged_attention", observe), guard:
            output = decoder.decode_paged(tokens, weights, allocator, requests)
    finally:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(owned, directory / "owned-attention.pt")
        write_json(directory / "owned-trace.json", results)
    return output, results


def integrated(model: Any, weights: Any, arm: str, case: str, directory: Path) -> dict[str, Any]:
    lengths = (15,) if case == "single" else (1, 15, 16, 17, 49)
    fragmented = case == "fragmented"
    directory.mkdir(parents=True, exist_ok=True)
    pool = torch.full((22, 2, 16, 16, 4, 64), torch.nan, dtype=torch.float16, device=model.device)
    allocator = KVAllocator(pool)
    requests = tuple(RequestId(str(row)) for row in range(len(lengths)))
    references, prefill_checks = {}, []
    blocker = RequestId("blocker")
    live = []
    rows: list[dict[str, Any]] = []
    pointer = pool.untyped_storage().data_ptr()
    try:
        for request, length in zip(requests, lengths, strict=True):
            allocator.create(KVReservation(request, (length + 18) // 16))
            live.append(request)
            inputs = public_input(length, str(model.device))
            references[request] = model(inputs, use_cache=True)
            logits = decoder.prefill_paged(inputs, weights, allocator, request)
            prefill_checks.append(measurement(logits, references[request].logits, 5e-3))
            prefill_checks[-1]["greedy_matches"] = torch.equal(
                logits[:, -1:].argmax(-1), references[request].logits[:, -1:].argmax(-1)
            )
            if fragmented and length == 15:
                allocator.create(KVReservation(blocker, 1))
                live.append(blocker)
                sentinel = torch.full(
                    (22, 16, 4, 64), 100, device=model.device, dtype=torch.float16
                )
                allocator.append(blocker, -sentinel, sentinel)
        reservation = allocator.capacity().reserved_blocks
        for step in range(3):
            order = requests if step % 2 == 0 else requests[::-1]
            step_dir = directory / f"step-{step + 1}"
            tokens = torch.cat([references[req].logits[:, -1:].argmax(-1) for req in order])
            captures, native_records, expected = {}, {}, {}
            for row, req in enumerate(order):
                with capture_eager(step_dir / f"request-{req}") as (captured, recorded):
                    expected[req] = model(
                        tokens[row : row + 1],
                        past_key_values=references[req].past_key_values,
                        use_cache=True,
                    )
                captures[req], native_records[req] = captured, recorded
                assert [record["layer"] for record in recorded] == list(range(22))
            before = pool.clone()
            old_lengths = {req: allocator.cached_length(req) for req in live}
            tables = {req: allocator.block_table(req) for req in live}
            rollback = "not_injected"
            if fragmented and step == 1:
                counts = allocator.capacity()
                try:
                    with (
                        patch.object(
                            decoder,
                            "final_logits",
                            side_effect=RuntimeError("injected logit failure"),
                        ),
                        Guard(pool),
                    ):
                        decoder.decode_paged(tokens, weights, allocator, order)
                except RuntimeError as error:
                    assert str(error) == "injected logit failure"
                else:
                    raise AssertionError("injected failure did not execute")
                assert allocator.capacity() == counts
                assert all(
                    allocator.cached_length(req) == old_lengths[req]
                    and allocator.block_table(req) == tables[req]
                    for req in live
                )
                for req in live:
                    for logical, physical in enumerate(tables[req]):
                        valid = min(16, old_lengths[req] - logical * 16)
                        torch.testing.assert_close(
                            pool[:, :, physical, :valid],
                            before[:, :, physical, :valid],
                            atol=0,
                            rtol=0,
                        )
                rollback = "passed"
            actual, traced = trace_decode(
                tokens,
                weights,
                allocator,
                order,
                captures,
                native_records,
                arm,
                step_dir,
                Guard(pool),
            )
            assert all(len(traced[req]) == 22 for req in order)
            torch.save(
                {
                    "actual": actual.detach().cpu(),
                    "reference": torch.cat([expected[req].logits for req in order]).cpu(),
                },
                step_dir / "logits.pt",
            )
            for row, req in enumerate(order):
                target = expected[req]
                kv_checks = []
                for logical, physical in enumerate(allocator.block_table(req)):
                    k, v = allocator.read_block(req, logical)
                    start = logical * 16
                    old_slots = max(0, min(k.shape[1], old_lengths[req] - start))
                    for layer, pair in enumerate(target.past_key_values):
                        for kind, value in enumerate((k[layer], v[layer])):
                            kv_checks.append(
                                measurement(
                                    value,
                                    pair[kind][0, :, start : start + value.shape[0]].transpose(
                                        0, 1
                                    ),
                                )
                            )
                            torch.testing.assert_close(
                                value[:old_slots],
                                before[layer, kind, physical, :old_slots],
                                atol=0,
                                rtol=0,
                            )
                assert allocator.cached_length(req) == old_lengths[req] + 1
                assert allocator.block_table(req)[: len(tables[req])] == tables[req]
                logit_check = measurement(actual[row : row + 1], target.logits, 5e-3)
                greedy = torch.equal(actual[row : row + 1].argmax(-1), target.logits.argmax(-1))
                item = {
                    "step": step + 1,
                    "request": req,
                    "row": row,
                    "attended_tokens": allocator.cached_length(req),
                    "arm": arm,
                    "layers": traced[req],
                    "first_difference": first_difference(traced[req]),
                    "logits": logit_check,
                    "greedy_matches": greedy,
                    "kv": kv_checks,
                    "block_table": allocator.block_table(req),
                    "passed": logit_check["max_tolerance_ratio"] <= 1
                    and greedy
                    and all(x["max_tolerance_ratio"] <= 1 for x in kv_checks),
                    "rollback": rollback,
                }
                rows.append(item)
                references[req] = target
            if fragmented:
                assert allocator.block_table(blocker) == tables[blocker]
                for req in order:
                    logical, slot = divmod(old_lengths[req], 16)
                    physical = allocator.block_table(req)[logical]
                    before[:, :, physical, slot] = pool[:, :, physical, slot]
                torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
                boundary = requests[lengths.index(15)]
                if step >= 1:
                    table = allocator.block_table(boundary)
                    assert table[1] > table[0] + 1
            counts = allocator.capacity()
            assert counts.reserved_blocks == reservation and counts.active_slots == len(live)
            assert counts.live_cached_token_slots == sum(lengths) + len(order) * (step + 1) + (
                16 if fragmented else 0
            )
            assert pool.untyped_storage().data_ptr() == pointer
            write_json(directory / "integrated.json", {"prefill": prefill_checks, "rows": rows})
    finally:
        # Synchronous diagnostic failures release the same admitted requests once.
        if pool.is_cuda:
            torch.cuda.synchronize(pool.device)
        for req in live:
            allocator.release_all(req)
        assert allocator.capacity() == CapacityCounts(16, 16, 0, 0, 0, 0)
        write_json(directory / "cleanup.json", asdict(allocator.capacity()))
    return {
        "arm": arm,
        "case": case,
        "rows": len(rows),
        "failed_rows": sum(not r["passed"] for r in rows),
        "prefill_passed": all(
            r["max_tolerance_ratio"] <= 1 and r["greedy_matches"] for r in prefill_checks
        ),
        "guard": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    lengths = sorted(set(map(int, args.lengths.split(","))))
    arms = args.arms.split(",")
    if (
        not lengths
        or any(not 2 <= length <= 2048 for length in lengths)
        or any(arm not in ARMS for arm in arms)
    ):
        parser.error("lengths must be 2..2048 and arms must be known diagnostic controls")
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "Diagnostic evidence; not release acceptance",
        "planned_lengths": lengths,
        "planned_arms": arms,
        "planned_cases": ["single", "mixed", "fragmented"],
        "expected_layers": 22,
        "lengths": [],
        "integrated": [],
        "errors": [],
    }
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert "T4" in torch.cuda.get_device_name(0)
        assert torch.__version__ == "2.7.1+cu126"
        seed_everything(0)
        torch.set_num_threads(1)
        with reference_math() as policy, torch.inference_mode():
            properties = torch.cuda.get_device_properties(0)
            write_json(
                destination / "environment.json",
                {
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "environment": collect_environment_diagnostics(),
                    "provenance": collect_validation_provenance(),
                    "math_policy": policy,
                    "arguments": sys.argv,
                    "selected_device": {
                        "index": 0,
                        "name": properties.name,
                        "uuid": str(properties.uuid),
                        "capability": [properties.major, properties.minor],
                        "total_memory": properties.total_memory,
                    },
                    "model_revision": MODEL_REVISION,
                    "attention_backend": "eager",
                    "seed": 0,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "module_paths": {
                        "attention": attention.__file__,
                        "decoder": decoder.__file__,
                        "torch": torch.__file__,
                        "reference_llama": llama.__file__,
                    },
                },
            )
            model = (
                AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                )
                .to("cuda:0")
                .eval()
            )
            write_json(destination / "model-config.json", model.config.to_dict())
            for length in lengths:
                try:
                    result = score_controls(model, length, destination / f"length-{length}")
                    summary["lengths"].append(result)
                    if not result["kernel_capture_complete"]:
                        summary["errors"].append(f"kernel-capture-length-{length}")
                except Exception as error:
                    summary["errors"].append(f"length-{length}")
                    (destination / f"error-length-{length}.txt").write_text(traceback.format_exc())
                    if isinstance(error, RuntimeError):
                        # Unknown CUDA or ownership state must not be reused in this process.
                        raise
                write_json(destination / "summary.json", summary)
                print(f"length={length} saved; errors={len(summary['errors'])}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            weights = load_weights(device="cuda:0")
            for arm in arms:
                for case in ("single", "mixed", "fragmented"):
                    try:
                        result = integrated(model, weights, arm, case, destination / arm / case)
                        summary["integrated"].append(result)
                    except Exception as error:
                        summary["errors"].append(f"{arm}-{case}")
                        (destination / f"error-{arm}-{case}.txt").write_text(traceback.format_exc())
                        if isinstance(error, RuntimeError):
                            raise
                    write_json(destination / "summary.json", summary)
                    print(f"arm={arm} case={case} saved", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    except BaseException:
        (destination / "fatal.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json(destination / "summary.json", summary)
    if summary["errors"]:
        raise SystemExit("Some experiments failed; partial evidence was retained")


if __name__ == "__main__":
    main()
