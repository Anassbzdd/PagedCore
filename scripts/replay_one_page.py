"""Replay the first differing one-page attention layers on shared operands."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.trace_two_token_decode import (
    AttentionMath,
    attention_stages,
    difference,
    restore_operand,
)

from pagedcore.attention import paged_attention
from pagedcore.dense import dense_attention_fp32
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator


def fp64_attention(query, key, value):
    # Independent head-by-head reductions; no repeated KV heads or page traversal.
    query, key, value = (x.cpu().double() for x in (query, key, value))
    dots, probabilities, outputs = [], [], []
    for head in range(32):
        kv_head = head // 8
        scores = (query[0, head, 0] * key[0, kv_head]).sum(-1) * 0.125
        weights = scores.softmax(-1)
        dots.append(scores / 0.125)
        probabilities.append(weights)
        outputs.append((weights[:, None] * value[0, kv_head]).sum(0))
    return {
        "query_key_dot": torch.stack(dots).reshape(1, 32, 1, -1),
        "probabilities": torch.stack(probabilities).reshape(1, 32, 1, -1),
        "precast_attention": torch.stack(outputs).reshape(1, 32, 1, 64),
    }


def error_against_fp64(actual, expected):
    delta = actual.double() - expected.double()
    return {
        "max_absolute_error": float(delta.abs().max()),
        "rms_error": float(delta.square().mean().sqrt()),
        "fp16_mismatches": int((round_to_fp16(actual) != round_to_fp16(expected)).sum()),
        "attention_tolerance_ratio": float(
            (delta.abs() / (2e-3 * (1 + expected.double().abs()))).max()
        ),
    }


def round_to_fp16(value):
    if value.dtype != torch.float64:
        return value.cpu().half()
    # Tensor double->half can round through FP32 and turn a near-midpoint value
    # into an exact tie. Pack binary64 directly to binary16 for this diagnostic.
    scalars = value.cpu().reshape(-1).tolist()
    packed = struct.pack("<" + "e" * len(scalars), *scalars)
    return torch.frombuffer(bytearray(packed), dtype=torch.float16).reshape(value.shape).clone()


def event_stages(events, *, paged):
    # Reuse stage extraction, not the production arithmetic.
    return attention_stages(SimpleNamespace(math=[events]), 0, paged=paged)


@torch.inference_mode()
def replay_layer(path, device):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    layer = int(path.stem.split("-")[-1])
    length = raw["paged"]["k_read"]["value"].shape[0]
    report, tensors = {"layer": layer, "attended_tokens": length, "common_inputs": {}}, {}
    for source_name in ("reference", "paged"):
        source = raw[source_name]
        query = restore_operand(source["q_rotated"], device)
        if source_name == "reference":
            key, value = (restore_operand(source[n], device) for n in ("k_read", "v_read"))
        else:
            key, value = (
                source[n]["value"].permute(1, 0, 2)[None].contiguous().to(device)
                for n in ("k_read", "v_read")
            )
        pool = torch.full((22, 2, 32, 16, 4, 64), torch.nan, dtype=torch.float16, device=device)
        allocator, request = KVAllocator(pool), RequestId("replay")
        allocator.create(KVReservation(request, 2))
        inputs_k = torch.zeros((22, length, 4, 64), dtype=torch.float16, device=device)
        inputs_v = torch.zeros_like(inputs_k)
        inputs_k[layer].copy_(key[0].transpose(0, 1))
        inputs_v[layer].copy_(value[0].transpose(0, 1))
        allocator.append(request, inputs_k, inputs_v)
        try:
            reference_math, paged_math = AttentionMath(), AttentionMath()
            mask = torch.ones((1, 1, 1, length), dtype=torch.bool, device=device)
            with reference_math:
                reference = dense_attention_fp32(query, key, value, mask)
            with paged_math:
                paged = paged_attention(query, allocator, request, layer_index=layer)
            stages = {
                "reference": event_stages(reference_math.events, paged=False),
                "paged": event_stages(paged_math.events, paged=True),
            }
            fp64 = fp64_attention(query, key, value)
            own = reference if source_name == "reference" else paged
            recorded_math = event_stages(
                raw[source_name + "_attention_math"], paged=source_name == "paged"
            )
            native = {
                stage: torch.equal(stages[source_name][stage]["value"], saved["value"])
                for stage, saved in recorded_math.items()
            }
            native["attention_output"] = torch.equal(own.cpu(), source["attention_output"]["value"])
            read_key, read_value = next(allocator.iter_layer_blocks(request, layer_index=layer))
            result = {
                "native_replays_exact": native,
                "common_input_controls": {
                    "q_shape": list(query.shape),
                    "gqa_mapping": "query head h reads KV head h // 8",
                    "absolute_position": length - 1,
                    "pool_k_exact": torch.equal(read_key, key[0].transpose(0, 1)),
                    "pool_v_exact": torch.equal(read_value, value[0].transpose(0, 1)),
                },
                "stages": {},
                "attention_output_pair": difference(paged.cpu(), reference.cpu()),
            }
            for stage in ("query_key_dot", "probabilities", "precast_attention"):
                expected = fp64[stage]
                a, b = (
                    stages[name][stage]["value"].reshape(expected.shape)
                    for name in ("paged", "reference")
                )
                result["stages"][stage] = {
                    "pair": difference(a, b),
                    "paged_vs_fp64": error_against_fp64(a, expected),
                    "reference_vs_fp64": error_against_fp64(b, expected),
                }
                tensors[f"{source_name}_{stage}_paged"] = a
                tensors[f"{source_name}_{stage}_reference"] = b
                tensors[f"{source_name}_{stage}_fp64"] = expected

            # Isolate P@V: both native layouts now receive exactly the same P and V.
            products = {
                name: [e for e in events if "operands" in e][-1]
                for name, events in (
                    ("reference", reference_math.events),
                    ("paged", paged_math.events),
                )
            }
            common_probabilities = stages["reference"]["probabilities"]["value"]
            pv_outputs = {}
            pv_values = {}
            for name, product in products.items():
                operands = [restore_operand(saved, device) for saved in product["operands"]]
                operands[0].copy_(common_probabilities.to(device).reshape(operands[0].shape))
                pv_values[name] = operands[1].cpu()
                pv_outputs[name] = torch.matmul(*operands).cpu().reshape(1, 32, 1, 64)
            values_identical = torch.equal(
                pv_values["reference"].reshape(4, 8, length, 64),
                pv_values["paged"][:, None].expand(4, 8, length, 64),
            )
            assert values_identical
            common_p = common_probabilities.double().reshape(32, length)
            v64 = value.cpu().double()
            independent_pv = torch.stack(
                [(common_p[h, :, None] * v64[0, h // 8]).sum(0) for h in range(32)]
            ).reshape(1, 32, 1, 64)
            result["identical_probability_value_replay"] = {
                "value_operands_exact": values_identical,
                "pair": difference(pv_outputs["paged"], pv_outputs["reference"]),
                **{
                    name + "_vs_fp64": error_against_fp64(v, independent_pv)
                    for name, v in pv_outputs.items()
                },
            }
            tensors.update({source_name + "_common_pv_" + k: v for k, v in pv_outputs.items()})
            tensors[source_name + "_common_pv_fp64"] = independent_pv
            report["common_inputs"][source_name] = result
        finally:
            allocator.release_all(request)
            assert allocator.capacity() == CapacityCounts(32, 32, 0, 0, 0, 0)
    return report, tensors


def replay_first_layers(directory: Path, device):
    trace = json.loads((directory / "trace.json").read_text())
    layers = sorted(
        {
            row["layer"]
            for name in ("first_fp32_difference", "first_fp16_difference")
            if (row := trace.get(name)) is not None
        }
    )
    report = {
        "scope": "First differing attention layers only; no repair or acceptance decision",
        "fp64_device": "CPU, explicit per-head FP64 reductions",
        "layers": [],
    }
    for layer in layers:
        result, tensors = replay_layer(directory / "raw" / f"layer-{layer:02}.pt", device)
        report["layers"].append(result)
        torch.save(tensors, directory / f"common-input-layer-{layer:02}.pt")
    report["all_native_replays_exact"] = all(
        all(arm["native_replays_exact"].values())
        for layer in report["layers"]
        for arm in layer["common_inputs"].values()
    )
    (directory / "common-input-replay.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
