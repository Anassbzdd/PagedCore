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

ACCURACY_RUN = r"""
    target_code = "import json,sys,torch; from pathlib import Path; from pagedcore.cli import collect_environment_diagnostics; assert torch.cuda.is_available() and torch.cuda.device_count()==1; assert 'T4' in torch.cuda.get_device_name(0); assert torch.__version__=='2.7.1+cu126'; p=torch.cuda.get_device_properties(0); Path(sys.argv[1]).write_text(json.dumps(dict(torch=torch.__version__,name=p.name,uuid=str(p.uuid),capability=[p.major,p.minor],visible_devices=torch.cuda.device_count(),total_memory=p.total_memory,environment=collect_environment_diagnostics()),indent=2))"
    run("target", [str(python), "-c", target_code, str(evidence / "target.json")], required=True)
    run("checkpoint-verify", [str(python), "-m", "scripts.verify_owned_checkpoint",
         "--output", str(evidence / "environment-manifest.json")], required=True)
    run("lint", [str(python), "-m", "ruff", "check", "."], required=True)
    run("format", [str(python), "-m", "ruff", "format", "--check", "."], required=True)
    run("types", [str(python), "-m", "mypy", "--strict", "src/pagedcore"], required=True)
    run("cpu-tests", [str(python), "-m", "pytest", "-m", "not gpu and not benchmark", "-ra",
         "--junitxml=" + str(evidence / "cpu-tests.xml")])
    environment["PAGEDCORE_ACCURACY_DIR"] = str(evidence / "raw-logits")
    run("gpu-regressions", [str(python), "-m", "pytest", "-m", "gpu", "-v", "-ra",
         "-k", "not legacy and not paired_at_17 and not test_pinned_candidate_paged_decode_on_t4",
         "--junitxml=" + str(evidence / "gpu-tests.xml")])
    run("accuracy-export", [str(python), "-m", "scripts.export_paged_accuracy", str(evidence)])
"""

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

output_override = os.environ.get("PAGEDCORE_EVIDENCE_WORKDIR")
working = (
    Path(output_override).expanduser().resolve()
    if output_override
    else Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
)
working.mkdir(parents=True, exist_ok=True)
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
    run("gpu-regressions", [str(python), "-m", "pytest", "--run-hf-diagnostics", "-v", "-ra", "tests/gpu",
         "-k", "test_legacy_hf_paged_decode_on_t4 or paired_at_17 or eager_value_sum_at_17_tokens",
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


def build(
    output: Path, *, accuracy: bool = False, two_token: bool = False, layer_trace: bool = False
) -> None:
    two_token = two_token or layer_trace
    if accuracy and two_token:
        raise ValueError("choose one notebook mode")
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
    if not (accuracy or two_token):
        paths.extend(
            ROOT / name
            for name in (
                "results/validation/2026-11-07/value-kernel-sweep-t4.zip",
                "results/validation/2026-11-07/paired-at-17-console.txt",
            )
        )
    paths = [path for path in paths if path.name != "AGENTS.md" or path.is_file()]
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
    if accuracy or two_token:
        code = code.replace(
            '"--python", "3.11"', '"--python", (source / ".python-version").read_text().strip()'
        )
        start = code.index('    with zipfile.ZipFile(source / "results/validation/')
        end = code.index('    (evidence / "source-after.json")', start)
        commands = ACCURACY_RUN
        if two_token:
            commands = commands[: commands.index('    run("lint"')]
            module = (
                "scripts.trace_two_token_decode"
                if layer_trace
                else "scripts.probe_two_token_decode"
            )
            folder = "layer-trace" if layer_trace else "two-token"
            commands += f'''    run("{folder}", [str(python), "-m", "{module}",
         "--output-dir", str(evidence / "{folder}")], required=True)
'''
        code = code[:start] + commands + code[end:]
        start = code.index('    report = evidence / "gpu-regressions.xml"')
        end = code.index('    (evidence / "notebook-run.json")', start)
        code = code[:start] + code[end:]
        code = code.replace("import xml.etree.ElementTree as ET\n", "")
        code = code.replace('prefix="pagedcore-evidence-"', 'prefix="pagedcore-accuracy-"')
        code = code.replace(
            "Diagnostic collection, not decoder release acceptance",
            "Paged-decoder accuracy validation, not full release acceptance",
        )
        if two_token:
            code = code.replace("pagedcore-accuracy-", "pagedcore-two-token-")
            code = code.replace(
                "Paged-decoder accuracy validation, not full release acceptance",
                "Two-token diagnostic, not decoder release acceptance",
            )
            if layer_trace:
                code = code.replace("pagedcore-two-token-", "pagedcore-layer-trace-")
    intro = """# PagedCore numerical evidence on T4

Enable **Internet** and **GPU T4** in Kaggle, then **Run All**. The subprocess uses
one T4 even if Kaggle assigns two. No GitHub push, manual patch, or login is needed.
Use a fresh session with enough working-disk space for raw tensors and their archive.
This notebook contains the exact source snapshot; do not replace it with latest main.
Set `PAGEDCORE_EVIDENCE_WORKDIR` to override the output folder; otherwise Kaggle
uses `/kaggle/working` and other environments use the current folder.

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
    if accuracy:
        intro = """# PagedCore FP32 paging accuracy

Enable **Internet** and **T4 GPU**, then **Run All**. No Input dataset, previous
notebook session, Git push or manual patch is needed. Exact source is embedded.
Set `PAGEDCORE_EVIDENCE_WORKDIR` to override the output folder; otherwise Kaggle
uses `/kaggle/working` and other environments use the current folder.

This runs CPU checks and the required T4 suite, including independent FP64
attention, owned checkpoint/prefill smoke, and paged decode versus
independent contiguous FP32 attention at the existing tolerances. Single, mixed,
fragmented and held-out layouts are required. The decoder still uses FP16 weights
and cache. No HF model forward or HF fidelity report is required. Optional HF
comparisons are deselected by default. No cuBLAS kernel profiling is needed.

Download the final ZIP even when tests fail. `accuracy-summary.json` says passed
only when CPU/GPU tests and all versioned reports are complete and valid. Setup
logs, source hashes, target identity, raw logit tensors and failures are retained.
This collector covers paged accuracy; end-to-end generation, maximum-active
workspace, CUDA reuse and service gates remain separate. Read NUMERICAL_POLICY.md
in the source snapshot for the November 11 independent validation contract. Model quality, full-distribution
KL, and expanded component coverage remain planned and are not certified here.
"""
    if two_token:
        intro = """# PagedCore two-token reproduction on T4

Enable **Internet** and **T4 GPU**, then **Run All**. Exact working source is
embedded; no Git push or Input dataset is required. Setup uses the frozen lock.
Set `PAGEDCORE_EVIDENCE_WORKDIR` to override `/kaggle/working`.

This runs one cached decode step after the `maximum_output` fixture's first token:
first alone, then as row 0 of the original 1/15/16/17/49-token mixed batch.
The original 32-block pool, three-step reservation sizes, FP16 weights/cache,
FP32 attention policy, deterministic settings and no-full-history guard remain.
The target attends to exactly two tokens in both arms. The owned contiguous
prefill selects the decode input; cached reference decode uses FP32 attention.
The probe verifies identical input tokens, prefill logits and reference logits.

The ZIP retains raw target logits, both unchanged tolerance comparisons,
measured K/V error, cleanup, math settings, source hashes and environment.
`evidence/two-token/summary.json` reports whether both failures reproduce and
their signed logit errors are identical. Check `controls` and both case reports;
an error means the comparison is incomplete. Numerical failures are retained
without aborting the other arm. Download the ZIP even when collection fails.
This diagnostic leaves serving arithmetic and the acceptance suite unchanged.
"""
    if layer_trace:
        intro = """# PagedCore two-token layer trace on T4

Enable **Internet** and **T4 GPU**, then **Run All**. Exact working source and
frozen setup are embedded. No Git push or Input dataset is required.
Set `PAGEDCORE_EVIDENCE_WORKDIR` to override `/kaggle/working`.

This traces the standalone one-token prompt plus one decode step through all
22 layers. It records hidden input, normalization, projected Q/K/V, rotated Q/K
(V stays unrotated), actual pool K/V reads including both token slots, attention,
output projection, both residuals, post-attention normalization and MLP output.
Each comparison reports exact value equality, maximum error, tolerance ratio,
first differing coordinate, shapes, original strides, offset, dtype and storage
size. Layers are numbered 0 through 21. Layout differences are reported separately
from value differences. Raw tensors and observed FP32 attention operation inputs
and outputs are saved for every layer, including differences within tolerance.

An uninstrumented baseline must match the traced final logits exactly. The first
exact mismatch is selected in layer/operation order; attention substeps compare
the actual QK product, scaled scores and FP32 output before its FP16 cast.
If the first difference is the QK product, it replays the recorded operands with
their original shapes/strides/offsets, compares packed operands, and tests dense
geometry using the same paged operand values. These controls run outside serving.
K/V history and the appended slot are measured separately. An incomplete K/V
trace is marked not_measured or partial with counts and a nullable maximum;
the initialized zero in an older failed acceptance report is not a measurement.

Download the final ZIP even on failure. Inspect `evidence/layer-trace/trace.json`
and `raw/layer-00.pt` through `layer-21.pt`. `first_difference` identifies the
earliest observed value difference; `complete` and `instrumentation_controls`
must pass before interpreting it. All arithmetic, tolerances, ownership checks
and GPU acceptance tests remain unchanged. This is diagnostic evidence only.
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
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--accuracy", action="store_true", help="build the revised accuracy collector"
    )
    modes.add_argument("--two-token", action="store_true", help="build the minimal decode probe")
    modes.add_argument(
        "--layer-trace", action="store_true", help="trace all two-token decoder layers"
    )
    args = parser.parse_args()
    output = args.output or ROOT / "notebooks" / (
        "two_token_layer_trace.ipynb"
        if args.layer_trace
        else "two_token_decode.ipynb"
        if args.two_token
        else "paged_decode_accuracy.ipynb"
        if args.accuracy
        else "p53_numerical_evidence.ipynb"
    )
    build(output, accuracy=args.accuracy, two_token=args.two_token, layer_trace=args.layer_trace)
