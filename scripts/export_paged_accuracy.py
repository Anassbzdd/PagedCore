"""Validate and export paged-decoder accuracy evidence without accepting partial runs."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pagedcore.accuracy import ACCURACY_POLICY, valid_accuracy_report

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "single": (15,),
    "mixed": (1, 15, 16, 17, 49),
    "fragmented": (1, 15, 16, 17, 49),
    "held_out": (37, 73, 111, 145),
}


def required_gpu_cases() -> set[str]:
    names = {
        "test_pinned_checkpoint_loads_and_runs_one_reference_forward_pass",
        "test_seed_everything_repeats_cuda_random_streams",
        "test_public_reference_capture_and_repeatability_on_t4",
        "test_pinned_decoder_layer_parity_on_t4",
        "test_pinned_paged_prefill_on_t4",
        "test_paged_attention_boundaries_on_t4",
        "test_paged_attention_large_scores_on_t4",
        "test_kv_pool_storage_on_t4",
        "test_kv_allocator_storage_and_stream_ownership_on_t4",
    }
    names.update(f"test_pinned_paged_decode_accuracy_on_t4[{case}]" for case in CASES)
    fixtures = json.loads((ROOT / "tests/fixtures/reference_prompts.json").read_text())
    names.update(
        f"test_pinned_decoder_greedy_parity_on_t4[{case['name']}]" for case in fixtures["prompts"]
    )
    return names


def successful_cases(path: Path) -> list[ET.Element]:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases or any(
        case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
    ):
        raise ValueError(f"Required tests must execute and pass: {path.name}")
    return cases


def export(directory: Path) -> dict[str, Any]:
    commands = json.loads((directory / "commands.json").read_text())
    for name in (
        "target",
        "checkpoint-verify",
        "lint",
        "format",
        "types",
        "cpu-tests",
        "gpu-regressions",
    ):
        matches = [command for command in commands if command.get("name") == name]
        if len(matches) != 1 or matches[0].get("returncode") != 0 or matches[0].get("interrupted"):
            raise ValueError(f"Required command did not complete successfully: {name}")
    cpu = successful_cases(directory / "cpu-tests.xml")
    gpu = successful_cases(directory / "gpu-tests.xml")
    if not required_gpu_cases() <= {case.get("name") for case in gpu}:
        raise ValueError(
            "Required T4 architecture, allocator, attention and decode cases are missing"
        )
    reports = {}
    for name, lengths in CASES.items():
        tests = [
            case
            for case in gpu
            if case.get("name") == f"test_pinned_paged_decode_accuracy_on_t4[{name}]"
        ]
        if len(tests) != 1:
            raise ValueError(f"Expected one accuracy test: {name}")
        properties = tests[0].findall("./properties/property[@name='paged_decode_accuracy']")
        if len(properties) != 1:
            raise ValueError(f"Versioned accuracy and HF evidence missing: {name}")
        report = json.loads(properties[0].attrib["value"])
        if not valid_accuracy_report(report, lengths):
            raise ValueError(f"Incomplete or invalid accuracy/HF evidence: {name}")
        reports[name] = report
    target = json.loads((directory / "target.json").read_text())
    if (
        target.get("torch") != "2.7.1+cu126"
        or "T4" not in target.get("name", "")
        or target.get("visible_devices") != 1
    ):
        raise ValueError("Evidence must identify the pinned Torch build and one visible T4")
    comparisons = [row for report in reports.values() for row in report["comparisons"]]
    return {
        "status": "passed",
        "policy": ACCURACY_POLICY,
        "scope": "Paged-decoder accuracy; not full release, workspace or service acceptance",
        "cpu_passed": len(cpu),
        "gpu_passed": len(gpu),
        "target": target,
        "cases": reports,
        "hf_fidelity_summary": {
            "comparisons": len(comparisons),
            "greedy_matches": sum(row["hf_fidelity"]["greedy_matches"] for row in comparisons),
            "max_absolute_error": max(
                row["hf_fidelity"]["max_absolute_error"] for row in comparisons
            ),
            "max_normalized_rms_error": max(
                row["hf_fidelity"]["normalized_rms_error"] for row in comparisons
            ),
            "max_previous_hf_tolerance_ratio": max(
                row["hf_fidelity"]["max_tolerance_ratio"] for row in comparisons
            ),
            "scope": "Common owned-reference token histories, not free-running HF generation",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        summary = export(args.directory)
    except (ValueError, KeyError, OSError, ET.ParseError) as error:
        (args.directory / "accuracy-summary.json").write_text(
            json.dumps({"status": "rejected", "reason": str(error)}, indent=2) + "\n"
        )
        raise SystemExit(f"Accuracy evidence rejected: {error}") from error
    (args.directory / "accuracy-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({key: summary[key] for key in ("status", "policy", "cpu_passed", "gpu_passed")})
    )
