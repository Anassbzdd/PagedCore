"""Measure downstream drift after replacing one captured FP16 attention scalar."""

from __future__ import annotations

import argparse
import json
import runpy
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.trace_two_token_decode import (
    ROOT,
    STAGES,
    ForwardTrace,
    attention_stages,
    capture_forward,
    compare_traces,
)

from pagedcore import decoder
from pagedcore.accuracy import compare_logits
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights


@contextmanager
def replace_attention_scalar(layer, coordinate, expected, replacement):
    original = decoder.paged_attention
    observation = {"calls": 0}

    def attend(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs["layer_index"] != layer:
            return output
        observation["calls"] += 1
        assert observation["calls"] == 1, "target attention was called more than once"
        before = output.detach().cpu().clone()
        assert torch.equal(before, expected), "target output differs from the retained trace"
        # Both arms clone and write one scalar; the sham writes its original value.
        changed = output.clone()
        changed[coordinate] = replacement
        after = changed.detach().cpu()
        mask = torch.zeros_like(before, dtype=torch.bool)
        mask[coordinate] = True
        assert torch.equal(before[~mask], after[~mask]), "intervention changed another element"
        assert changed.shape == output.shape and changed.stride() == output.stride()
        observation.update(
            coordinate=list(coordinate),
            value_before=float(before[coordinate]),
            value_after=float(after[coordinate]),
            changed_elements=int((before != after).sum()),
            other_elements_exact=True,
        )
        return changed

    with patch.object(decoder, "paged_attention", attend):
        yield observation
    assert observation["calls"] == 1, "target attention was not called"


@torch.inference_mode()
def run_arm(weights, guard, baseline, target, directory, *, intervene):
    directory.mkdir(parents=True, exist_ok=True)
    pool = torch.full(
        (22, 2, 32, 16, 4, 64), torch.nan, dtype=torch.float16, device=weights.embedding.device
    )
    allocator, request = KVAllocator(pool), RequestId("0")
    reference, paged = ForwardTrace(), ForwardTrace()
    report = {"arm": "single_scalar" if intervene else "sham", "controls": {}}
    try:
        with reference_math() as settings:
            report["math_settings"] = settings
            prompt = baseline["prompt_ids"][None].to(pool.device)
            prompt_tokens = prompt.shape[1]
            allocator.create(KVReservation(request, (prompt_tokens + 3 + 15) // 16))
            prefill, cache = decoder.prefill(prompt, weights)
            paged_prefill = decoder.prefill_paged(prompt, weights, allocator, request)
            assert torch.equal(prefill.cpu(), baseline["prefill_logits"])
            assert torch.equal(prefill, paged_prefill)
            tokens = prefill[:, -1:].argmax(-1)
            assert torch.equal(tokens.cpu(), baseline["decode_input"])
            history = {
                (layer, name): value.cpu().clone()
                for layer in range(22)
                for pair in allocator.iter_layer_blocks(request, layer_index=layer)
                for name, value in zip(("k_read", "v_read"), pair, strict=True)
            }
            pointer = pool.untyped_storage().data_ptr()
            capacity = allocator.capacity()
            table = allocator.block_table(request)
            with capture_forward(reference):
                reference_logits, _ = decoder.decode(tokens, cache, weights, fp32_attention=True)
            replacement = target["reference_value"] if intervene else target["paged_value"]
            # Install the intervention first so capture_forward observes its returned value.
            with (
                replace_attention_scalar(
                    target["layer"], target["coordinate"], target["paged_output"], replacement
                ) as observation,
                guard(pool),
                capture_forward(paged, allocator),
            ):
                logits = decoder.decode_paged(tokens, weights, allocator, (request,))
            report["intervention"] = observation
            report["controls"] = {
                "reference_logits_exact": torch.equal(
                    reference_logits.cpu(), baseline["owned_reference"]
                ),
                "prefill_and_token_exact": True,
                "pool_pointer_unchanged": pool.untyped_storage().data_ptr() == pointer,
                "cached_length_correct": allocator.cached_length(request) == prompt_tokens + 1,
                "block_table_unchanged": allocator.block_table(request) == table,
                "owned_blocks_unchanged": allocator.capacity().owned_blocks
                == capacity.owned_blocks,
                "reserved_blocks_unchanged": allocator.capacity().reserved_blocks
                == capacity.reserved_blocks,
                "active_slots_unchanged": allocator.capacity().active_slots
                == capacity.active_slots,
                "old_history_unchanged": all(
                    torch.equal(paged.layers[layer][name]["value"][:-1], value)
                    for (layer, name), value in history.items()
                ),
                "absolute_positions_correct": all(
                    row["absolute_positions"] == [[prompt_tokens]] for row in paged.layers
                ),
                "changed_only_selected_scalar": observation["changed_elements"] == int(intervene),
            }
            assert all(report["controls"].values()), "arm controls failed"
            report["logits"] = compare_logits(logits.cpu(), reference_logits.cpu())
            ratio = (logits.double() - reference_logits.double()).abs() / (
                5e-3 * (1 + reference_logits.double().abs())
            )
            report["failing_logit_elements"] = int((ratio > 1).sum())
            output = paged.layers[target["layer"]]["attention_output"]["value"]
            report["target_output_equals_reference"] = torch.equal(
                output, target["reference_output"]
            )
            report["target_hidden_out_equals_reference"] = torch.equal(
                paged.layers[target["layer"]]["hidden_out"]["value"],
                reference.layers[target["layer"]]["hidden_out"]["value"],
            )
            torch.save(
                {"paged": logits.cpu(), "owned_reference": reference_logits.cpu()},
                directory / "logits.pt",
            )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if pool.is_cuda:
            torch.cuda.synchronize(pool.device)
        allocator.release_all(request)
        report["cleanup"] = asdict(allocator.capacity())
        assert allocator.capacity() == CapacityCounts(32, 32, 0, 0, 0, 0)
        (directory / "raw").mkdir(exist_ok=True)
        for layer in range(22):
            torch.save(
                {
                    "reference": reference.layers[layer],
                    "paged": paged.layers[layer],
                    "reference_attention_math": reference.math[layer],
                    "paged_attention_math": paged.math[layer],
                },
                directory / "raw" / f"layer-{layer:02}.pt",
            )
        report["trace"] = compare_traces(reference, paged)
        kv = [r for r in report["trace"]["comparisons"] if r["operation"] in ("k_read", "v_read")]
        report["appended_writes_match_local_projections"] = len(kv) == 44 and all(
            r["appended_matches_local_projection"] for r in kv
        )
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, reference, paged


def load_target(trace_dir):
    report = json.loads((trace_dir / "trace.json").read_text())
    assert report["first_difference_valid"] and report["complete"]
    first = report["first_fp16_difference"]
    assert first["operation"] == "attention_output"
    layer, coordinate = first["layer"], tuple(first["first_unequal_index"])
    raw = torch.load(trace_dir / f"layer-{layer:02}.pt", map_location="cpu", weights_only=True)
    paged, reference = (raw[name]["attention_output"]["value"] for name in ("paged", "reference"))
    assert paged.dtype == reference.dtype == torch.float16
    assert paged[coordinate] != reference[coordinate]
    return report, {
        "layer": layer,
        "coordinate": coordinate,
        "paged_output": paged,
        "reference_output": reference,
        "paged_value": float(paged[coordinate]),
        "reference_value": float(reference[coordinate]),
    }


def prefix_controls(sham, changed, target):
    checks = []
    for layer in range(target["layer"] + 1):
        a = sham.layers[layer] | attention_stages(sham, layer, paged=True)
        b = changed.layers[layer] | attention_stages(changed, layer, paged=True)
        for stage in STAGES:
            if layer == target["layer"] and stage == "attention_output":
                return {"measured_stages": len(checks), "all_exact": all(checks)}
            checks.append(torch.equal(a[stage]["value"], b[stage]["value"]))
    raise AssertionError("target output was not reached")


def run_control(weights, guard, baseline_dir, trace_dir, directory):
    directory.mkdir(parents=True, exist_ok=True)
    baseline = torch.load(baseline_dir / "target.pt", map_location="cpu", weights_only=True)
    archived, target = load_target(trace_dir)
    report = {
        "scope": "Single-scalar causal control; never a serving repair or accuracy pass",
        "arms": {},
    }
    traces = {}
    for name, enabled in (("sham", False), ("single_scalar", True)):
        arm, reference, paged = run_arm(
            weights, guard, baseline, target, directory / name, intervene=enabled
        )
        report["arms"][name] = arm
        traces[name] = (reference, paged)
    if all("error" not in arm and arm["trace"]["complete"] for arm in report["arms"].values()):
        sham = torch.load(directory / "sham/logits.pt", weights_only=True)
        comparisons = report["arms"]["sham"]["trace"]["comparisons"]
        prefix = prefix_controls(traces["sham"][1], traces["single_scalar"][1], target)
        report["controls"] = {
            "sham_logits_equal_baseline": torch.equal(sham["paged"], baseline["paged"]),
            "sham_comparisons_equal_retained_trace": json.loads(json.dumps(comparisons))
            == archived["comparisons"],
            "prefix_before_target_exact": prefix["all_exact"],
            "both_reference_logits_equal_baseline": all(
                a["controls"]["reference_logits_exact"] for a in report["arms"].values()
            ),
            "both_append_local_projections_exact": all(
                a["appended_writes_match_local_projections"] for a in report["arms"].values()
            ),
        }
        report["prefix_stages_measured"] = prefix["measured_stages"]
        report["target"] = {k: v for k, v in target.items() if not isinstance(v, torch.Tensor)}
        report["remaining_first_fp16_difference"] = report["arms"]["single_scalar"]["trace"][
            "first_fp16_difference"
        ]
    report["valid_control"] = (
        bool(report.get("controls"))
        and all(report["controls"].values())
        and all(
            "error" not in a and a["trace"]["complete"] and all(a["controls"].values())
            for a in report["arms"].values()
        )
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        if args.device != "cpu":
            assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
            assert torch.__version__ == "2.7.1+cu126"
        archived, _ = load_target(args.trace_dir)
        assert (
            archived["prompt_tokens"] == 15
            and archived["first_fp16_difference"]["unequal_elements"] == 1
        )
        seed_everything(0)
        report["environment"] = collect_environment_diagnostics()
        report["source"] = collect_validation_provenance()
        guard = runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
        with reference_math():
            weights = load_weights(device=args.device)
            report["experiment"] = run_control(
                weights, guard, args.baseline_dir, args.trace_dir, args.output_dir
            )
    except Exception:
        report["error"] = traceback.format_exc()
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "experiment"}, indent=2))
    print("Report:", args.output_dir / "summary.json")
    return int("error" in report or not report.get("experiment", {}).get("valid_control"))


if __name__ == "__main__":
    raise SystemExit(main())
