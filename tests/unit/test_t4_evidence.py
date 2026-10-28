import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]


@pytest.fixture
def gpu_report() -> ET.Element:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    required = [
        "test_pinned_checkpoint_loads_and_runs_one_reference_forward_pass",
        "test_seed_everything_repeats_cuda_random_streams",
        "test_public_reference_capture_and_repeatability_on_t4",
        "test_pinned_decoder_layer_parity_on_t4",
        "test_kv_pool_storage_on_t4",
        "test_kv_allocator_storage_and_stream_ownership_on_t4",
    ]
    fixtures = json.loads((ROOT / "tests/fixtures/reference_prompts.json").read_text())
    required.extend(
        f"test_pinned_decoder_greedy_parity_on_t4[{case['name']}]" for case in fixtures["prompts"]
    )
    for name in required:
        ET.SubElement(suite, "testcase", name=name)
    return root


def _run_collector(directory: Path) -> subprocess.CompletedProcess[str]:
    # Execute the collector's exporter itself without requiring Bash or a GPU.
    script = (ROOT / "scripts/validate_t4.sh").read_text().rsplit("<<'PY'\n", 1)[1]
    return subprocess.run(
        [sys.executable, "-", str(directory)],
        input=script.removesuffix("PY\n"),
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )


@pytest.mark.parametrize(
    "missing_tests",
    [
        ("test_kv_pool_storage_on_t4",),
        ("test_kv_allocator_storage_and_stream_ownership_on_t4",),
        ("test_kv_pool_storage_on_t4", "test_kv_allocator_storage_and_stream_ownership_on_t4"),
    ],
    ids=["missing_pool", "missing_allocator", "missing_both"],
)
def test_collector_rejects_missing_kv_tests(
    tmp_path: Path, gpu_report: ET.Element, missing_tests: tuple[str, ...]
) -> None:
    suite = gpu_report[0]
    for case in list(suite):
        if case.get("name") in missing_tests:
            suite.remove(case)
    ET.ElementTree(gpu_report).write(tmp_path / "gpu-tests.xml")

    result = _run_collector(tmp_path)

    assert result.returncode != 0
    assert "GPU evidence rejected: required tests must execute and pass." in result.stderr
    assert not tmp_path.with_suffix(".zip").exists()


@pytest.mark.parametrize(
    "test_name",
    ["test_kv_pool_storage_on_t4", "test_kv_allocator_storage_and_stream_ownership_on_t4"],
)
@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_collector_rejects_unsuccessful_kv_tests(
    tmp_path: Path, gpu_report: ET.Element, test_name: str, outcome: str
) -> None:
    case = next(case for case in gpu_report[0] if case.get("name") == test_name)
    ET.SubElement(case, outcome)
    ET.ElementTree(gpu_report).write(tmp_path / "gpu-tests.xml")

    result = _run_collector(tmp_path)

    assert result.returncode != 0
    assert "GPU evidence rejected: required tests must execute and pass." in result.stderr
    assert not tmp_path.with_suffix(".zip").exists()


@pytest.mark.parametrize("include_measurements", [False, True])
@pytest.mark.parametrize("workspace_state", ["missing", "partial", "missing_samples", "complete"])
@pytest.mark.parametrize(
    "pool_state",
    ["complete", "missing", "partial", "unreserved", "mismatch", "false_certification"],
)
def test_collector_requires_and_retains_owned_decoder_measurements(
    tmp_path: Path,
    gpu_report: ET.Element,
    include_measurements: bool,
    workspace_state: str,
    pool_state: str,
) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    root = gpu_report
    suite = root[0]
    measurements = {
        "scope": "synthetic exporter test",
        "cases": {
            "first": {
                "tensors": {
                    "prefill.final_norm": {"max_absolute_error": 0.01, "max_tolerance_ratio": 0.2},
                    "prefill.logits": {"max_absolute_error": 0.02, "max_tolerance_ratio": 0.1},
                }
            },
            "second": {
                "tensors": {
                    "decode.3.final_norm": {
                        "max_absolute_error": 0.001,
                        "max_tolerance_ratio": 0.9,
                    },
                    "decode.3.logits": {"max_absolute_error": 0.03, "max_tolerance_ratio": 0.8},
                }
            },
        },
    }
    if include_measurements:
        properties = ET.SubElement(suite, "properties")
        ET.SubElement(
            properties,
            "property",
            name="decoder_tolerance_envelope",
            value=json.dumps(measurements),
        )
    ET.ElementTree(root).write(directory / "gpu-tests.xml")
    manifest = {
        "environment": {
            "python": "synthetic",
            "settings": {"kv_pool_mib": 1, "workspace_margin_mib": 1, "max_active_sequences": 4},
        },
        "provenance": {"git_revision": "test"},
    }
    target = {"name": "synthetic device", "uuid": "test-device"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "target.json").write_text(json.dumps(target))
    workspace_runs = [
        {
            "case": name,
            "prompt_tokens": prompt,
            "output_tokens": output,
            "repetition": repetition,
            "status": "passed",
            "samples": [
                {"phase": phase, "allocator_stats": {"num_ooms": 0}}
                for phase in ("before_forward", "after_forward", "after_cleanup")
            ],
        }
        for name, prompt, output in (
            ("maximum_prefill", 2048, 0),
            ("maximum_prompt", 2047, 1),
            ("maximum_output", 1792, 256),
        )
        for repetition in range(3)
    ]
    workspace = {"status": "passed", "runs": workspace_runs}
    pool_workspace = deepcopy(workspace)
    pool_workspace.update(
        mode="pool_pressure",
        model_baseline={
            "allocated_bytes": 100,
            "device_free_bytes": 16 * 2**20,
            "allocator_stats": {"num_alloc_retries": 0, "num_ooms": 0},
        },
        pool_baseline={
            "allocated_bytes": 100 + 2**20,
            "device_free_bytes": 15 * 2**20,
            "allocator_stats": {"num_alloc_retries": 0, "num_ooms": 0},
        },
        pool_pressure={
            "kind": "unused_fp16_reservation",
            "requested_bytes": 2**20,
            "tensor_bytes": 2**20,
            "workspace_margin_bytes": 2**20,
            "configured_max_active_sequences": 4,
            "measured_active_sequences": 1,
            "capacity_validated": False,
            "max_dense_increment_bytes": 800,
            "dense_increment_exceeds_margin": False,
        },
    )
    for run in pool_workspace["runs"]:
        run["peak_allocated_above_baseline_bytes"] = 800
        for sample in run["samples"]:
            sample["allocated_bytes"] = 100 + 2**20
    if pool_state == "partial":
        pool_workspace["runs"].pop()
    elif pool_state == "unreserved":
        pool_workspace["runs"][0]["samples"][0]["allocated_bytes"] = 100
    elif pool_state == "mismatch":
        pool_workspace["pool_pressure"]["requested_bytes"] = 2 * 2**20
    elif pool_state == "false_certification":
        pool_workspace["pool_pressure"]["capacity_validated"] = True
    if pool_state != "missing":
        (directory / "pool-dense-workspace.json").write_text(json.dumps(pool_workspace))
        (directory / "pool-dense-workspace-allocator.txt").write_text("synthetic pool diagnostics")
    if workspace_state == "partial":
        workspace_runs.pop()
    elif workspace_state == "missing_samples":
        workspace_runs[0]["samples"] = []
    if workspace_state != "missing":
        (directory / "dense-workspace.json").write_text(json.dumps(workspace))
        (directory / "dense-workspace-allocator.txt").write_text("synthetic diagnostics")
    result = _run_collector(directory)
    archive = directory.with_suffix(".zip")
    if not include_measurements:
        assert result.returncode != 0
        assert "tolerance measurements are required" in result.stderr
        assert not archive.exists()
        return
    if workspace_state != "complete":
        assert result.returncode != 0
        assert "dense workspace" in result.stderr
        assert not archive.exists()
        return
    if pool_state != "complete":
        assert result.returncode != 0
        assert "pool" in result.stderr
        assert not archive.exists()
        return
    assert result.returncode == 0, result.stderr
    report = json.loads((directory / "tolerance-envelope.json").read_text())
    assert report["cases"] == measurements["cases"]
    assert report["environment"] == manifest["environment"]
    assert report["provenance"] == manifest["provenance"]
    assert report["target"] == target
    maxima = report["maxima"]
    assert maxima["intermediates"]["max_absolute_error"]["case"] == "first"
    assert maxima["intermediates"]["max_tolerance_ratio"]["case"] == "second"
    assert maxima["logits"]["max_absolute_error"]["max_absolute_error"] == 0.03
    assert maxima["logits"]["max_tolerance_ratio"]["max_tolerance_ratio"] == 0.8
    with zipfile.ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence/tolerance-envelope.json")) == report
        assert json.loads(bundle.read("evidence/dense-workspace.json")) == workspace
        assert bundle.read("evidence/dense-workspace-allocator.txt") == b"synthetic diagnostics"
        assert json.loads(bundle.read("evidence/pool-dense-workspace.json")) == pool_workspace
        assert (
            bundle.read("evidence/pool-dense-workspace-allocator.txt")
            == b"synthetic pool diagnostics"
        )
