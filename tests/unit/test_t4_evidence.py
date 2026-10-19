import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("include_measurements", [False, True])
def test_collector_requires_and_retains_owned_decoder_measurements(
    tmp_path: Path, include_measurements: bool
) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    required = [
        "test_pinned_checkpoint_loads_and_runs_one_reference_forward_pass",
        "test_seed_everything_repeats_cuda_random_streams",
        "test_public_reference_capture_and_repeatability_on_t4",
        "test_pinned_decoder_layer_parity_on_t4",
    ]
    fixtures = json.loads((ROOT / "tests/fixtures/reference_prompts.json").read_text())
    required.extend(
        f"test_pinned_decoder_greedy_parity_on_t4[{case['name']}]" for case in fixtures["prompts"]
    )
    for name in required:
        ET.SubElement(suite, "testcase", name=name)
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
    manifest = {"environment": {"python": "synthetic"}, "provenance": {"git_revision": "test"}}
    target = {"name": "synthetic device", "uuid": "test-device"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "target.json").write_text(json.dumps(target))
    # Execute the collector's exporter itself without requiring Bash or a GPU.
    script = (ROOT / "scripts/validate_t4.sh").read_text().rsplit("<<'PY'\n", 1)[1]
    script = script.removesuffix("PY\n")
    result = subprocess.run(
        [sys.executable, "-", str(directory)],
        input=script,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    archive = directory.with_suffix(".zip")
    if not include_measurements:
        assert result.returncode != 0
        assert "tolerance measurements are required" in result.stderr
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
