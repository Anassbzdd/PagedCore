"""Trace actual two-token decode operations without replacing their arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.probe_two_token_decode import ROOT, run_case
from torch.utils._python_dispatch import TorchDispatchMode

from pagedcore import decoder
from pagedcore.accuracy import compare_logits
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights

STAGES = (
    "hidden_in",
    "input_norm",
    "q_projected",
    "k_projected",
    "v_projected",
    "q_rotated",
    "k_rotated",
    "v_unrotated",
    "k_read",
    "v_read",
    "query_key_dot",
    "scaled_scores",
    "precast_attention",
    "attention_output",
    "output_projection",
    "attention_residual",
    "post_attention_norm",
    "mlp",
    "hidden_out",
)


def snapshot(value):
    return {
        "layout": {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "contiguous": value.is_contiguous(),
            "storage_nbytes": value.untyped_storage().nbytes(),
        },
        # Layout above describes the original runtime tensor, before this diagnostic copy.
        "value": value.detach().to("cpu", copy=True),
    }


class AttentionMath(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        kind = name.split(".")[1]
        if isinstance(result, torch.Tensor) and kind in {
            "matmul",
            "bmm",
            "mm",
            "mul",
            "softmax",
            "_softmax",
            "exp",
            "div",
            "amax",
            "sum",
            "maximum",
            "sub",
            "add",
            "_to_copy",
            "to",
        }:
            event = {"operation": name, "output": snapshot(result)}
            if kind in ("matmul", "bmm", "mm"):
                event["operands"] = [snapshot(value) for value in args[:2]]
            if name == "aten.mul.Tensor" and isinstance(args[1], (float, int)):
                event["scalar"] = args[1]
            self.events.append(event)
        return result


class ForwardTrace:
    def __init__(self):
        self.layers = [{} for _ in range(22)]
        self.math = [[] for _ in range(22)]
        self.norm_calls = 0
        self.layer = -1

    def put(self, name, value, *, layer=None):
        index = self.layer if layer is None else layer
        assert name not in self.layers[index], f"duplicate trace point {index}/{name}"
        self.layers[index][name] = snapshot(value)


@contextmanager
def capture_forward(trace, allocator=None):
    originals = {
        name: getattr(decoder, name)
        for name in (
            "rms_norm",
            "project_qkv",
            "apply_rope",
            "dense_attention_fp32",
            "paged_attention",
            "output_projection",
            "swiglu",
            "final_logits",
        )
    }

    def norm(hidden, weight):
        index, post = divmod(trace.norm_calls, 2)
        trace.norm_calls += 1
        trace.layer = index
        if not post:
            if index:
                trace.put("hidden_out", hidden, layer=index - 1)
            trace.put("hidden_in", hidden)
        else:
            trace.put("attention_residual", hidden)
        result = originals["rms_norm"](hidden, weight)
        trace.put("post_attention_norm" if post else "input_norm", result)
        return result

    def project(hidden, weights):
        result = originals["project_qkv"](hidden, weights)
        for name, value in zip(("q_projected", "k_projected", "v_projected"), result, strict=True):
            trace.put(name, value)
        return result

    def rotate(query, key, positions):
        result = originals["apply_rope"](query, key, positions)
        trace.put("q_rotated", result[0])
        trace.put("k_rotated", result[1])
        # RoPE never rotates V; retain the unchanged operand explicitly.
        trace.layers[trace.layer]["v_unrotated"] = trace.layers[trace.layer]["v_projected"]
        return result

    def attend(*args, **kwargs):
        dense = allocator is None
        if dense:
            trace.put("k_read", args[1])
            trace.put("v_read", args[2])
        mode = AttentionMath()
        try:
            with mode:
                result = originals["dense_attention_fp32" if dense else "paged_attention"](
                    *args, **kwargs
                )
        finally:
            trace.math[trace.layer] = mode.events
        trace.put("attention_output", result)
        return result

    def projection(value, weights):
        result = originals["output_projection"](value, weights)
        trace.put("output_projection", result)
        return result

    def mlp(value, weights):
        result = originals["swiglu"](value, weights)
        trace.put("mlp", result)
        return result

    def final(hidden, weights):
        trace.put("hidden_out", hidden)
        return originals["final_logits"](hidden, weights)

    with ExitStack() as stack:
        for name, function in (
            ("rms_norm", norm),
            ("project_qkv", project),
            ("apply_rope", rotate),
            ("dense_attention_fp32" if allocator is None else "paged_attention", attend),
            ("output_projection", projection),
            ("swiglu", mlp),
            ("final_logits", final),
        ):
            stack.enter_context(patch.object(decoder, name, function))
        if allocator is not None:
            original_read = allocator.iter_layer_blocks

            def read(request, *, layer_index):
                assert layer_index == trace.layer
                for logical, pair in enumerate(original_read(request, layer_index=layer_index)):
                    assert logical == 0 and pair[0].shape == (2, 4, 64)
                    for name, value in zip(("k_read", "v_read"), pair, strict=True):
                        trace.put(name, value)
                        trace.layers[layer_index][name]["pool_read"] = {
                            "logical_block": logical,
                            "physical_block": int(allocator.block_table(request)[logical]),
                            "shares_pool_storage": value.untyped_storage().data_ptr()
                            == allocator._pool.untyped_storage().data_ptr(),
                            "history_positions": [0],
                            "appended_position": 1,
                        }
                    yield pair

            stack.enter_context(patch.object(allocator, "iter_layer_blocks", read))
        yield


def attention_stages(trace, layer, *, paged):
    events = trace.math[layer]
    products = [e for e in events if e["operation"].split(".")[1] in ("matmul", "bmm", "mm")]
    scales = [e for e in events if e.get("scalar") == 0.125]
    divisions = [e for e in events if e["operation"] == "aten.div.Tensor"]
    result = {}
    if products:
        result["query_key_dot"] = products[0]["output"]
    if scales:
        result["scaled_scores"] = scales[0]["output"]
    if paged and divisions:
        result["precast_attention"] = divisions[-1]["output"]
    elif not paged and len(products) == 2:
        result["precast_attention"] = products[-1]["output"]
    return result


def difference(actual, expected, *, tolerance=2e-3):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.double() - expected.double()).abs()
    unequal = actual != expected
    coordinate = tuple(unequal.nonzero()[0].tolist()) if unequal.any() else None
    return {
        "exact": torch.equal(actual, expected),
        "unequal_elements": int(unequal.sum()),
        "max_absolute_error": float(error.max()),
        "max_tolerance_ratio": float((error / (tolerance * (1 + expected.double().abs()))).max()),
        "first_unequal_index": coordinate,
        "actual_at_first": float(actual[coordinate]) if coordinate is not None else None,
        "reference_at_first": float(expected[coordinate]) if coordinate is not None else None,
    }


def compare_traces(reference, paged):
    comparisons, missing = [], []
    for layer in range(22):
        expected = reference.layers[layer] | attention_stages(reference, layer, paged=False)
        actual = paged.layers[layer] | attention_stages(paged, layer, paged=True)
        for stage in STAGES:
            if stage not in actual or stage not in expected:
                missing.append({"layer": layer, "operation": stage})
                continue
            a, e = actual[stage]["value"], expected[stage]["value"]
            if stage in ("k_read", "v_read"):
                a = a.permute(1, 0, 2).unsqueeze(0)
            elif stage in ("query_key_dot", "scaled_scores"):
                a, e = a.reshape(4, 8, 2), e.reshape(4, 8, 2)
            elif stage == "precast_attention":
                a, e = a.reshape(1, 32, 1, 64), e.reshape(1, 32, 1, 64)
            row = {
                "layer": layer,
                "operation": stage,
                **difference(a, e),
                "comparison_shape": list(a.shape),
                "paged_layout": actual[stage]["layout"],
                "reference_layout": expected[stage]["layout"],
                "layout_equal": actual[stage]["layout"] == expected[stage]["layout"],
            }
            if stage in ("k_read", "v_read"):
                row["pool_read"] = actual[stage]["pool_read"]
                row["history"] = difference(a[:, :, :1], e[:, :, :1])
                row["appended"] = difference(a[:, :, 1:], e[:, :, 1:])
                written = actual["k_rotated" if stage == "k_read" else "v_projected"]["value"]
                row["appended_matches_local_projection"] = torch.equal(a[:, :, 1:], written)
            comparisons.append(row)
    kv = [row for row in comparisons if row["operation"] in ("k_read", "v_read")]
    return {
        "layer_indexing": "zero-based, 0 through 21",
        "complete": not missing,
        "missing": missing,
        "comparisons": comparisons,
        "first_difference": next((row for row in comparisons if not row["exact"]), None),
        "kv_measurement": {
            "status": "complete" if len(kv) == 44 else "partial" if kv else "not_measured",
            "measured_pairs": len(kv),
            "expected_pairs": 44,
            "max_kv_tolerance_ratio": max((r["max_tolerance_ratio"] for r in kv), default=None),
        },
    }


def restore_operand(saved, device):
    value, layout = saved["value"], saved["layout"]
    storage = torch.empty(
        layout["storage_nbytes"] // value.element_size(), dtype=value.dtype, device=device
    )
    result = storage.as_strided(layout["shape"], layout["stride"], layout["storage_offset"])
    result.copy_(value)
    return result


@torch.inference_mode()
def investigate_first_product(reference, paged, first, device, directory):
    """Replay only the first differing QK product, outside the guarded serving forward."""
    if first is None or first["operation"] != "query_key_dot":
        return {"status": "not_applicable", "reason": "First difference is not a QK product"}
    layer = first["layer"]
    products = {}
    for name, trace in (("reference", reference), ("paged", paged)):
        event = next(
            e for e in trace.math[layer] if e["operation"].split(".")[1] in ("matmul", "bmm", "mm")
        )
        operands = [restore_operand(value, device) for value in event["operands"]]
        native = torch.matmul(*operands)
        packed = torch.matmul(*(value.contiguous() for value in operands))
        products[name] = {"event": event, "operands": operands, "native": native, "packed": packed}
    r, p = products["reference"], products["paged"]
    rq, rk = r["operands"]
    pq, pk = p["operands"]
    controls = {
        "query_values_exact": torch.equal(rq.reshape(4, 8, 64), pq),
        "key_values_exact": torch.equal(rk.reshape(4, 8, 64, 2), pk[:, None].expand(4, 8, 64, 2)),
    }
    # This small dense calculation is an isolated diagnostic, never a serving fallback.
    dense_geometry = torch.matmul(pq.reshape(1, 32, 1, 64), pk.repeat_interleave(8, dim=0)[None])
    outputs = {"dense_geometry_from_paged": dense_geometry.detach().cpu()}
    result = {"status": "measured", "layer": layer, "operand_controls": controls, "replays": {}}
    for name, product in products.items():
        target = product["event"]["output"]["value"].reshape(4, 8, 2)
        other = products["paged" if name == "reference" else "reference"]["event"]["output"][
            "value"
        ]
        result["replays"][name] = {}
        for variant in ("native", "packed"):
            actual = product[variant].detach().cpu().reshape(4, 8, 2)
            result["replays"][name][variant] = {
                "against_recorded_own_product": difference(actual, target),
                "against_recorded_other_product": difference(actual, other.reshape(4, 8, 2)),
            }
            outputs[f"{name}_{variant}"] = actual
    result["dense_geometry_from_paged"] = {
        name: difference(
            dense_geometry.cpu().reshape(4, 8, 2),
            product["event"]["output"]["value"].reshape(4, 8, 2),
        )
        for name, product in products.items()
    }
    result["native_replays_exact"] = all(
        r["native"]["against_recorded_own_product"]["exact"] for r in result["replays"].values()
    )
    torch.save(outputs, directory / "first-product-replays.pt")
    return result


@torch.inference_mode()
def trace_case(weights, guard, directory):
    directory.mkdir(parents=True, exist_ok=True)
    baseline_report = run_case(weights, (1,), guard, directory / "baseline")
    assert "error" not in baseline_report, baseline_report.get("error")
    baseline = torch.load(directory / "baseline/target.pt", weights_only=True)
    pool = torch.full(
        (22, 2, 32, 16, 4, 64), torch.nan, dtype=torch.float16, device=weights.embedding.device
    )
    allocator = KVAllocator(pool)
    request = RequestId("0")
    reference, paged = ForwardTrace(), ForwardTrace()
    report = {"scope": "Standalone two-token layer trace; no acceptance or repair claim"}
    try:
        with reference_math() as settings:
            report["math_settings"] = settings
            allocator.create(KVReservation(request, 1))
            prompt = baseline["prompt_ids"][None].to(pool.device)
            prefill, cache = decoder.prefill(prompt, weights)
            decoder.prefill_paged(prompt, weights, allocator, request)
            tokens = prefill[:, -1:].argmax(-1)
            assert torch.equal(tokens.cpu(), baseline["decode_input"])
            pointer = pool.untyped_storage().data_ptr()
            with capture_forward(reference):
                reference_logits, _ = decoder.decode(tokens, cache, weights, fp32_attention=True)
            with guard(pool), capture_forward(paged, allocator):
                paged_logits = decoder.decode_paged(tokens, weights, allocator, (request,))
            assert pool.untyped_storage().data_ptr() == pointer
            assert allocator.cached_length(request) == 2
            report["instrumentation_controls"] = {
                "reference_logits_exact": torch.equal(
                    reference_logits.cpu(), baseline["owned_reference"]
                ),
                "paged_logits_exact": torch.equal(paged_logits.cpu(), baseline["paged"]),
            }
            report["logits"] = compare_logits(paged_logits.cpu(), reference_logits.cpu())
            assert all(report["instrumentation_controls"].values()), (
                "instrumentation changed logits"
            )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if pool.is_cuda:
            torch.cuda.synchronize(pool.device)
        allocator.release_all(request)
        report["cleanup"] = asdict(allocator.capacity())
        assert allocator.capacity() == CapacityCounts(32, 32, 0, 0, 0, 0)
        raw = directory / "raw"
        raw.mkdir(exist_ok=True)
        for layer in range(22):
            torch.save(
                {
                    "reference": reference.layers[layer],
                    "paged": paged.layers[layer],
                    "reference_attention_math": reference.math[layer],
                    "paged_attention_math": paged.math[layer],
                },
                raw / f"layer-{layer:02}.pt",
            )
        report.update(compare_traces(reference, paged))
        report["first_difference_valid"] = (
            report["complete"]
            and "error" not in report
            and all(report.get("instrumentation_controls", {}).values())
        )
        if report["first_difference_valid"]:
            try:
                with reference_math():
                    report["first_operation_investigation"] = investigate_first_product(
                        reference, paged, report["first_difference"], pool.device, directory
                    )
            except Exception:
                report["error"] = traceback.format_exc()
        (directory / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        if args.device != "cpu":
            assert torch.cuda.is_available() and "t4" in torch.cuda.get_device_name(0).lower()
            assert torch.__version__ == "2.7.1+cu126"
        seed_everything(0)
        report["environment"] = collect_environment_diagnostics()
        report["source"] = collect_validation_provenance()
        report["diagnostic_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/trace_two_token_decode.py",
                "scripts/probe_two_token_decode.py",
                "tests/fixtures/reference_prompts.json",
                "docs/NUMERICAL_POLICY.md",
            )
        }
        with reference_math():
            weights = load_weights(device=args.device)
        guard = runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
        report["trace"] = trace_case(weights, guard, args.output_dir)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        trace = report.get("trace", {})
        print(
            json.dumps(
                {
                    key: trace.get(key)
                    for key in (
                        "complete",
                        "first_difference",
                        "kv_measurement",
                        "instrumentation_controls",
                        "error",
                    )
                },
                indent=2,
            )
        )
        print("Report:", args.output_dir / "summary.json")
    return int(
        "error" in report
        or "error" in report.get("trace", {})
        or not report.get("trace", {}).get("complete")
    )


if __name__ == "__main__":
    raise SystemExit(main())
