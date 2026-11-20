"""Embed the experimental native backend and frozen T4 acceptance suite."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.build_numerical_notebook import ROOT, build

# Pin the JIT source artifact, rather than guessing a wheel's CUDA/Torch ABI.
FLASHINFER_URL = (
    "https://files.pythonhosted.org/packages/b2/c4/"
    "9ec0f79e2480fc5c93307c4a1ac903e5cf33c551c0eaeb648196234b55af/"
    "flashinfer_python-0.2.5.tar.gz"
)
FLASHINFER_SHA256 = "990aa090ef781783e76b836696ece4efd23956f72b5696d622fc619a61162aef"

INSTALL = r'''
    environment["TORCH_CUDA_ARCH_LIST"] = "7.5"
    environment["MAX_JOBS"] = "2"
    # Record the actual SDK/compiler; JIT compatibility must be proved by execution.
    run("nvcc", ["nvcc", "--version"], required=True)
    run("compiler", ["c++", "--version"], required=True)
    run("setup-ninja", [sys.executable, "-m", "uv", "pip", "install", "--python", str(python),
        "--no-deps", "ninja==1.11.1.4"], required=True)
    download = """
import hashlib, sys, urllib.request
from pathlib import Path
data = urllib.request.urlopen(sys.argv[1], timeout=120).read()
assert hashlib.sha256(data).hexdigest() == sys.argv[2], 'FlashInfer source checksum mismatch'
Path(sys.argv[3]).write_bytes(data)
"""
    artifact = evidence / "flashinfer_python-0.2.5.tar.gz"
    run("download-flashinfer", [str(python), "-c", download, FLASHINFER_URL,
        FLASHINFER_SHA256, str(artifact)], required=True)
    run("setup-flashinfer", [sys.executable, "-m", "uv", "pip", "install", "--python", str(python),
        "--no-deps", "--no-build-isolation", str(artifact)], required=True)
    run("candidate-dependencies", [sys.executable, "-m", "uv", "pip", "freeze",
        "--python", str(python)], required=True)
    run("candidate-dependency-check", [sys.executable, "-m", "uv", "pip", "check",
        "--python", str(python)], required=True)
    installed = """
import json, sys, torch
from importlib.metadata import version
from pathlib import Path
assert torch.__version__ == '2.7.1+cu126', 'candidate install changed Torch'
assert version('flashinfer-python') == '0.2.5'
Path(sys.argv[1]).write_text(json.dumps(dict(torch=torch.__version__,
    flashinfer=version('flashinfer-python'), ninja=version('ninja'),
    cuda=torch.version.cuda, source_sha256=sys.argv[2]), indent=2))
"""
    run("candidate-identity", [str(python), "-c", installed,
        str(evidence / "candidate.json"), FLASHINFER_SHA256], required=True)
'''

CHECKS = """checks_ok = False
try:
    assert setup_ok, "Setup did not finish"
    run("lint", [str(python), "-m", "ruff", "check", "."], required=True)
    run("format", [str(python), "-m", "ruff", "format", "--check", "."], required=True)
    run("types", [str(python), "-m", "mypy", "--strict", "src/pagedcore"], required=True)
    run("cpu-tests", [str(python), "-m", "pytest", "-m", "not gpu and not benchmark", "-ra",
        "--junitxml=" + str(evidence / "cpu-tests.xml")], required=True)
    checks_ok = True
except BaseException:
    (evidence / "checks-failure.txt").write_text(traceback.format_exc())
    print("Checks incomplete; keep running to the export cell.")
"""

KERNEL = """kernel_ok = False
try:
    assert checks_ok, "CPU checks did not finish"
    run("native-kernel-tests", [str(python), "-m", "pytest", "tests/gpu/test_flashinfer_cuda.py",
        "--decode-backend=flashinfer", "-v", "-ra",
        "--junitxml=" + str(evidence / "native-kernel-tests.xml")], required=True)
    kernel_ok = True
except BaseException:
    (evidence / "kernel-failure.txt").write_text(traceback.format_exc())
    print("Native kernel checks incomplete; export the failure logs.")
"""

INTEGRATION = """try:
    assert kernel_ok, "Native kernel checks did not pass"
    for backend in ("pagedcore", "flashinfer"):
        run("focused-15-plus-1-" + backend, [str(python), "-m", "scripts.probe_two_token_decode",
            "--prompt-tokens", "15", "--single-only", "--attention-backend", backend,
            "--output-dir", str(evidence / ("focused-" + backend))], required=True)
    # A focused numerical failure is retained by the diagnostic; it must not hide the suite.
    environment["PAGEDCORE_ACCURACY_DIR"] = str(evidence / "raw-logits")
    run("gpu-regressions", [str(python), "-m", "pytest", "-m", "gpu",
        "--decode-backend=flashinfer", "-v", "-ra",
        "--junitxml=" + str(evidence / "gpu-tests.xml")])
    run("accuracy-export", [str(python), "-m", "scripts.export_paged_accuracy", str(evidence)])
    run("candidate-cached-trajectory", [str(python), "-m", "scripts.check_checkpoint_trajectory",
        "--attention-backend", "flashinfer", "--output-dir", str(evidence / "trajectory")], required=True)
    summary = json.loads((evidence / "accuracy-summary.json").read_text())
    print("Frozen paging accuracy:", summary["status"])
    print("This is an experimental backend result; P5.3 needs review of the exported evidence.")
except BaseException:
    (evidence / "integration-failure.txt").write_text(traceback.format_exc())
    print("Integration incomplete; export the failure and partial evidence.")
"""


def build_candidate(output: Path) -> None:
    # Use current-source accuracy mode. Do not bypass the old trace builder's
    # unchanged-runtime control or claim the new source is the frozen baseline.
    build(output, accuracy=True)
    notebook = json.loads(output.read_text())
    code = "".join(notebook["cells"][1]["source"])
    start = code.index('    run("lint"')
    export_start = code.index("finally:\n", start) + len("finally:\n")
    setup = code[:start].replace(
        "try:\n    verify_source()", "setup_ok = False\ntry:\n    verify_source()", 1
    )
    setup += (
        INSTALL
        + """
    setup_ok = True
except BaseException:
    (evidence / "setup-failure.txt").write_text(traceback.format_exc())
    print("Setup incomplete; run the final cell to download its logs.")
"""
    )
    setup = setup.replace("pagedcore-accuracy-", "pagedcore-flashinfer-")
    setup = setup.replace(
        "PAYLOAD =",
        f"FLASHINFER_URL = {FLASHINFER_URL!r}\nFLASHINFER_SHA256 = {FLASHINFER_SHA256!r}\nPAYLOAD =",
        1,
    )
    export = textwrap.dedent(code[export_start:])
    export = (
        """try:
    (evidence / "source-after.json").write_text(json.dumps(verify_source(), indent=2))
except BaseException:
    (evidence / "source-verification-failure.txt").write_text(traceback.format_exc())
"""
        + export
    )
    intro = """# Experimental native paged attention: FlashInfer

Fresh Kaggle session: enable Internet and T4 GPU, then Run All. Source is embedded;
no upload, Git push or manual source edits are needed. Only GPU 0 is visible.
Download the final ZIP even on failure.

This changes only the explicitly selected decode attention kernel. PagedCore owns
the decoder, RoPE, FP16 K/V pool, block mapping and staged commits. The existing
backend remains the default while the candidate awaits T4 acceptance.

The cells install the frozen Torch 2.7.1+cu126 environment plus hash-verified
FlashInfer 0.2.5 JIT source and pinned Ninja. NVCC and a C++ compiler are required;
compilation support on Kaggle is unproven until this runs. Installation must not
change Torch. Missing dependencies or compilation failures export logs; there is
no automatic fallback or passing skip.

First check native attention with independent CPU FP64, poisoned fragmented pages,
lengths 1/15/16/17/49/2048, large scores, staged rollback and worker/stream ownership.
Then compare the current baseline and candidate at exactly 15 prefill + 1 decode.
Run the frozen single/mixed/fragmented/held-out paging suite with the candidate,
then a fixed 32-step cached trajectory (including an instrumentation control).
Logit/KV thresholds and exact greedy checks are unchanged. Retain old failures.

A passing notebook covers this experiment only. Whole-model independent quality,
service behavior, throughput and maximum-active workspace need separate evidence.
"""
    cells = [
        {
            "id": "instructions",
            "cell_type": "markdown",
            "metadata": {},
            "source": intro.splitlines(True),
        }
    ]
    for name, source in (
        ("setup", setup),
        ("checks", CHECKS),
        ("kernel", KERNEL),
        ("integration", INTEGRATION),
        ("export", export),
    ):
        compile(source, name, "exec")
        cells.append(
            {
                "id": name,
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.splitlines(True),
            }
        )
    notebook["cells"] = cells
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(output)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(output)], check=True)
    print(output)


if __name__ == "__main__":
    build_candidate(ROOT / "notebooks/flashinfer_backend_validation.ipynb")
