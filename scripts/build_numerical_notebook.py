"""Bundle the exact working source into a standalone Kaggle evidence notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = r"""
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from IPython.display import FileLink, display

# SOURCE_CONSTANTS

working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
run_root = Path(tempfile.mkdtemp(prefix="pagedcore-evidence-", dir=working))
source = run_root / "source"
evidence = run_root / "evidence"
evidence.mkdir()
payload = base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest() == SNAPSHOT_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(source)
identity = json.loads((source / "source-identity.json").read_text())
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
commands = []

def run(name, command, *, required=False):
    record = {"name": name, "argv": list(map(str, command)), "cwd": str(source)}
    commands.append(record)
    print(name, flush=True)
    with (evidence / (name + ".log")).open("w") as log:
        process = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(name + " is running; progress is saved in " + str(log.name), flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            record["interrupted"] = True
            raise
    record["returncode"] = process.returncode
    (evidence / "commands.json").write_text(json.dumps(commands, indent=2))
    print(name, "exit", process.returncode, flush=True)
    if required and process.returncode:
        raise RuntimeError(name + " failed; inspect its retained log")

def verify_source():
    checks = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in identity["sha256"]}
    assert checks == identity["sha256"], "source changed after notebook generation"
    return checks

try:
    verify_source()
    run("setup-uv", [sys.executable, "-m", "pip", "install", "uv==0.11.27"], required=True)
    run("setup-frozen", [sys.executable, "-m", "uv", "sync", "--frozen", "--extra", "cuda",
                         "--extra", "dev", "--python", "3.11"], required=True)
    python = source / ".venv/bin/python"
    run("resolved-dependencies", [sys.executable, "-m", "uv", "pip", "freeze", "--python", str(python)])
    run("gpu-inventory", ["nvidia-smi", "-q"])
    with zipfile.ZipFile(source / "results/validation/2026-11-07/value-kernel-sweep-t4.zip") as prior:
        sweep = prior.read("sweep_source.py")
        assert hashlib.sha256(sweep).hexdigest() == json.loads(prior.read("metadata.json"))["source_sha256"]
    (evidence / "synthetic-source.py").write_bytes(sweep)
    run("synthetic-controls", [str(python), str(evidence / "synthetic-source.py"),
         "--lengths", "17,18,24,95,96,2048", "--samples", "512",
         "--output", str(evidence / "synthetic-controls.zip")])
    run("diagnostics", [str(python), "-m", "scripts.collect_numerical_evidence",
                        "--output-dir", str(evidence / "diagnostics")])
    run("existing-component-diagnostic", [str(python), "-m", "scripts.diagnose_paged_decode",
         "--device", "cuda:0", "--steps", "3", "--value-reduction", "striped",
         "--output", str(evidence / "existing-components.json")])
    run("gpu-regressions", [str(python), "-m", "pytest", "-v", "-ra", "tests/gpu",
         "-k", "test_pinned_paged_decode_on_t4 or paired_at_17 or eager_value_sum_at_17_tokens",
         "--junitxml=" + str(evidence / "gpu-regressions.xml")])
    (evidence / "source-after.json").write_text(json.dumps(verify_source(), indent=2))
except BaseException:
    (evidence / "notebook-failure.txt").write_text(traceback.format_exc())
    print("Run incomplete; retained the failure and partial evidence.", flush=True)
finally:
    (evidence / "commands.json").write_text(json.dumps(commands, indent=2))
    report = evidence / "gpu-regressions.xml"
    if report.exists():
        try:
            tests = list(ET.parse(report).getroot().iter("testcase"))
            counts = {kind: sum(test.find(kind) is not None for test in tests)
                      for kind in ("failure", "error", "skipped")}
            counts.update(selected=len(tests), executed=len(tests) - counts["skipped"],
                          passed=len(tests) - sum(counts.values()))
            (evidence / "gpu-test-counts.json").write_text(json.dumps({
                **counts, "expected_selected": 9, "cases": [test.get("name") for test in tests],
                "all_required_executed": len(tests) == 9 and counts["skipped"] == 0 and counts["error"] == 0
            }, indent=2))
        except Exception:
            (evidence / "gpu-test-counts-error.txt").write_text(traceback.format_exc())
    (evidence / "notebook-run.json").write_text(json.dumps({
        "snapshot_sha256": SNAPSHOT_SHA256, "base_revision": identity["base_revision"],
        "scope": "Diagnostic collection, not decoder release acceptance",
        "environment_overrides": {key: environment[key] for key in
            ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    }, indent=2))
    files = {path.relative_to(evidence).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in evidence.rglob("*") if path.is_file()}
    (evidence / "checksums.json").write_text(json.dumps(files, indent=2))
    archive_path = working / (run_root.name + ".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source-snapshot.zip", payload)
        for path in evidence.rglob("*"):
            if path.is_file():
                archive.write(path, "evidence/" + path.relative_to(evidence).as_posix())
    print("DOWNLOAD", archive_path, "bytes", archive_path.stat().st_size, flush=True)
    display(FileLink(str(archive_path)))
"""


def build(output: Path) -> None:
    paths = []
    for directory in ("src", "scripts", "tests", "docs"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".json", ".md", ".sh")
            and "__pycache__" not in path.parts
        )
    paths.extend(
        ROOT / name
        for name in (
            "AGENTS.md",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "README.md",
            "LICENSE",
        )
    )
    paths.extend(
        ROOT / name
        for name in (
            "results/validation/2026-11-07/value-kernel-sweep-t4.zip",
            "results/validation/2026-11-07/paired-at-17-console.txt",
        )
    )
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(set(paths))}
    identity = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "source_identity": "Complete bundled working files plus per-file SHA-256; not a clean-commit claim",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("source-identity.json", json.dumps(identity, indent=2))
    payload = stream.getvalue()
    constants = (
        f"PAYLOAD = {base64.b64encode(payload).decode()!r}\n"
        f"SNAPSHOT_SHA256 = {hashlib.sha256(payload).hexdigest()!r}\n"
    )
    code = textwrap.dedent(BOOTSTRAP).lstrip().replace("# SOURCE_CONSTANTS", constants)
    intro = """# PagedCore numerical evidence on T4

Enable **Internet** and **GPU T4** in Kaggle, then **Run All**. The subprocess uses
one T4 even if Kaggle assigns two. No GitHub push, manual patch, or login is needed.
Use a fresh session with enough working-disk space for raw tensors and their archive.
This notebook contains the exact source snapshot; do not replace it with latest main.

1. **Real operands:** At 34 representative attended lengths from 4 to 2,048,
   capture all 22 layers, native score/value kernel names, Q/K/V, raw dots,
   scaled scores, FP32/FP16 probabilities, output and original shapes/strides.
   Native replays must be exact. This tests whether the synthetic kernel families
   and inferred orders apply to every real head/channel, rather than six samples.
   The earlier synthetic sweep is also rerun at six representative lengths using
   its retained, hash-verified source to detect environment or kernel changes.
2. **Separate score control:** Copy independent HF K/V to alternating physical
   pages, then compare paged scores and both probability dtypes with fixed value
   output. At the first score mismatch for each length, vary K packing and logical
   output width separately. This tests which factor explains the score error.
   Full-width K controls use independent reference storage outside serving.
3. **Integrated controls:** Baseline, two sums only at 17, and the measured-family
   hypothesis (alternating for variant 5, contiguous halves for even variant 6,
   baseline elsewhere). Run single, mixed and poisoned-fragmented cases for three
   steps, reverse rows, inject rollback/retry, check all K/V and frozen logit
   tolerances, and require exact greedy tokens. Every row records its first
   attention difference in operation order. Continue after numerical mismatches.
   Inputs follow reference tokens so each failed step remains comparable.
4. **Existing GPU regressions:** Retain their JUnit report, including failures.

All source, hashes, setup commands/logs, selected GPU identity, physical inventory,
dependencies, math policy, raw tensors and partial failures go into **one ZIP** in
`/kaggle/working`. Download it from the final link or Kaggle Output and return it.
Failures are expected evidence; a completed collection is not a parity pass.
No runtime reducer or score repair is selected by this notebook. The full T4
collector, end-to-end generation, workspace and CUDA reuse gates remain separate.
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "id": "instructions",
                "cell_type": "markdown",
                "metadata": {},
                "source": intro.splitlines(True),
            },
            {
                "id": "collect",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(True),
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(output)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(output)], check=True)
    print(
        json.dumps(
            {
                "notebook": str(output),
                "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                "files": len(contents),
                "bytes": output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "notebooks/p53_numerical_evidence.ipynb"
    )
    build(parser.parse_args().output)
