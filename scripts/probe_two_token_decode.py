"""Compare one decode step alone and in the original mixed batch; diagnostic only."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from pagedcore import decoder
from pagedcore.accuracy import ACCURACY_POLICY, compare_logits
from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math

ROOT = Path(__file__).resolve().parents[1]
MIXED_LENGTHS = (1, 15, 16, 17, 49)


@torch.inference_mode()
def run_case(weights, lengths, guard, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "tests/fixtures/reference_prompts.json"
    public_ids = next(
        case["input_ids"]
        for case in json.loads(fixture.read_text(encoding="utf-8"))["prompts"]
        if case["name"] == "maximum_output"
    )
    pool = torch.full(
        (22, 2, 32, 16, 4, 64),
        torch.nan,
        dtype=torch.float16,
        device=weights.embedding.device,
    )
    allocator = KVAllocator(pool)
    requests = tuple(RequestId(str(row)) for row in range(len(lengths)))
    live, caches, prefill_logits = [], {}, {}
    result = {"lengths": list(lengths), "decode_steps": 1, "attended_tokens": 2}
    try:
        with reference_math() as settings:
            result["math_settings"] = settings
            for request, length in zip(requests, lengths, strict=True):
                # Preserve the failed three-step case's reservations and physical layout.
                allocator.create(KVReservation(request, (length + 3 + 15) // 16))
                live.append(request)
                inputs = torch.tensor([public_ids[:length]], device=pool.device)
                logits, caches[request] = decoder.prefill(inputs, weights)
                prefill_logits[request] = logits
                paged_prefill = decoder.prefill_paged(inputs, weights, allocator, request)
                torch.testing.assert_close(paged_prefill, logits, atol=5e-3, rtol=5e-3)
            tokens = torch.cat([prefill_logits[req][:, -1:].argmax(-1) for req in requests])
            reference, reference_cache = decoder.decode(
                tokens[:1], caches[requests[0]], weights, fp32_attention=True
            )
            pointer = pool.untyped_storage().data_ptr()
            before = allocator.capacity()
            with guard(pool):
                actual = decoder.decode_paged(tokens, weights, allocator, requests)
            assert pool.untyped_storage().data_ptr() == pointer
            assert allocator.cached_length(requests[0]) == 2
            assert allocator.capacity().reserved_blocks == before.reserved_blocks
            assert allocator.capacity().active_slots == before.active_slots
            assert allocator.capacity().live_cached_token_slots == sum(lengths) + len(lengths)
            actual = actual[:1].cpu()
            reference = reference.cpu()
            metrics = compare_logits(actual, reference)
            error = (actual.double() - reference.double()).abs()
            ratio = error / (5e-3 * (1 + reference.double().abs()))
            index = int(ratio.flatten().argmax().item())
            result.update(
                owned_reference=metrics,
                discrepancy_reproduced=metrics["max_tolerance_ratio"] > 1,
                worst_logit={
                    "index": index,
                    "paged": actual.flatten()[index].item(),
                    "reference": reference.flatten()[index].item(),
                },
                block_tables=[list(allocator.block_table(req)) for req in requests],
            )
            kv_ratio = 0.0
            for logical, _ in enumerate(allocator.block_table(requests[0])):
                pair = allocator.read_block(requests[0], logical)
                start = logical * 16
                for layer, targets in enumerate(reference_cache.layers):
                    for page, target in zip(pair, targets, strict=True):
                        expected = target[0, :, start : start + page.shape[1]].transpose(0, 1)
                        assert torch.isfinite(page[layer]).all() and torch.isfinite(expected).all()
                        kv_ratio = max(
                            kv_ratio,
                            float(
                                (
                                    (page[layer].double() - expected.double()).abs()
                                    / (2e-3 * (1 + expected.double().abs()))
                                )
                                .max()
                                .item()
                            ),
                        )
            result["max_kv_tolerance_ratio"] = kv_ratio
            raw = {
                "prompt_ids": torch.tensor(public_ids[:1]),
                "decode_input": tokens[:1].cpu(),
                "prefill_logits": prefill_logits[requests[0]].cpu(),
                "paged": actual,
                "owned_reference": reference,
            }
            torch.save(raw, directory / "target.pt")
            result["raw_sha256"] = hashlib.sha256(
                (directory / "target.pt").read_bytes()
            ).hexdigest()
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if pool.is_cuda:
            torch.cuda.synchronize(pool.device)
        for request in live:
            allocator.release_all(request)
        result["cleanup"] = asdict(allocator.capacity())
        assert allocator.capacity() == CapacityCounts(32, 32, 0, 0, 0, 0)
        (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_cases(directory):
    single, mixed = (
        torch.load(directory / name / "target.pt", weights_only=True)
        for name in ("single", "mixed")
    )
    controls = {
        key + "_exact": torch.equal(single[key], mixed[key])
        for key in ("prompt_ids", "decode_input", "prefill_logits", "owned_reference")
    }
    controls["paged_logits_exact"] = torch.equal(single["paged"], mixed["paged"])
    controls["signed_errors_exact"] = torch.equal(
        single["paged"].double() - single["owned_reference"].double(),
        mixed["paged"].double() - mixed["owned_reference"].double(),
    )
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Two-token diagnostic; does not replace acceptance or close a gate",
        "policy": ACCURACY_POLICY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "cases": {},
    }
    try:
        if args.device != "cpu":
            assert torch.cuda.is_available(), (
                "CUDA is required; use --device cpu only for smoke checks"
            )
            assert torch.__version__ == "2.7.1+cu126", "requires the failed run's Torch build"
            assert "t4" in torch.cuda.get_device_name(0).casefold(), "requires an NVIDIA T4"
        report["environment"] = collect_environment_diagnostics()
        report["source"] = collect_validation_provenance()
        report["diagnostic_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/probe_two_token_decode.py",
                "tests/fixtures/reference_prompts.json",
                "docs/NUMERICAL_POLICY.md",
            )
        }
        from pagedcore.weights import load_weights

        seed_everything(0)
        with reference_math():
            weights = load_weights(device=args.device)
        guard = runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
        for name, lengths in (("single", (1,)), ("mixed", MIXED_LENGTHS)):
            seed_everything(0)
            report["cases"][name] = run_case(weights, lengths, guard, args.output_dir / name)
        if not any("error" in case for case in report["cases"].values()):
            controls = compare_cases(args.output_dir)
            controls["math_settings_exact"] = (
                report["cases"]["single"]["math_settings"]
                == report["cases"]["mixed"]["math_settings"]
            )
            report["controls"] = controls
            controlled = all(
                value
                for key, value in controls.items()
                if key not in ("paged_logits_exact", "signed_errors_exact")
            )
            report["controlled_inputs_match"] = controlled
            report["both_exceed_logit_tolerance"] = all(
                case["discrepancy_reproduced"] for case in report["cases"].values()
            )
            report["batch_membership_changes_target_logits"] = (
                not controls["paged_logits_exact"] if controlled else None
            )
            report["same_discrepancy"] = (
                controlled
                and controls["signed_errors_exact"]
                and report["both_exceed_logit_tolerance"]
            )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        destination = args.output_dir / "summary.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("environment", "source")
                },
                indent=2,
            )
        )
        print("Diagnostic report:", destination)
    return int("error" in report or any("error" in case for case in report["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
