"""Package the explicit native backend with unchanged T4 accuracy checks."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from scripts.build_numerical_notebook import ROOT, build

FLASHINFER_URL = (
    "https://files.pythonhosted.org/packages/b2/c4/"
    "9ec0f79e2480fc5c93307c4a1ac903e5cf33c551c0eaeb648196234b55af/"
    "flashinfer_python-0.2.5.tar.gz"
)
FLASHINFER_SHA256 = "990aa090ef781783e76b836696ece4efd23956f72b5696d622fc619a61162aef"

INSTALL = r'''
    environment["TORCH_CUDA_ARCH_LIST"] = "7.5"
    environment["MAX_JOBS"] = "2"
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

INTEGRATION = r'''
try:
    assert kernel_ok, "Native kernel checks did not pass"
    for backend in ("pagedcore", "flashinfer"):
        run("focused-15-plus-1-" + backend, [str(python), "-m", "scripts.probe_two_token_decode",
            "--prompt-tokens", "15", "--single-only", "--attention-backend", backend,
            "--output-dir", str(evidence / ("focused-" + backend))], required=True)
    control = """
import json, sys, torch
from pathlib import Path
root, output = map(Path, sys.argv[1:])
prior = torch.load(root / 'prior-failure/target.pt', map_location='cpu', weights_only=True)
current = torch.load(output / 'focused-pagedcore/single/target.pt', map_location='cpu', weights_only=True)
controls = {name + '_exact': torch.equal(prior[name], current[name]) for name in prior}
(output / 'prior-failure-controls.json').write_text(json.dumps(controls, indent=2))
assert all(controls.values()), 'Baseline changed; do not interpret the candidate as a controlled improvement'
"""
    run("prior-failure-control", [str(python), "-c", control, str(source), str(evidence)], required=True)
    for backend in ("pagedcore", "flashinfer"):
        result = json.loads((evidence / ("focused-" + backend) / "single/report.json").read_text())
        print(backend, "logit ratio:", result["owned_reference"]["max_tolerance_ratio"],
              "greedy agrees:", result["owned_reference"]["greedy_matches"])
    # Numerical failures remain recorded; collect the whole frozen suite regardless.
    environment["PAGEDCORE_ACCURACY_DIR"] = str(evidence / "raw-logits")
    run("gpu-regressions", [str(python), "-m", "pytest", "-m", "gpu",
        "--decode-backend=flashinfer", "-v", "-ra",
        "--junitxml=" + str(evidence / "gpu-tests.xml")])
    run("accuracy-export", [str(python), "-m", "scripts.export_paged_accuracy", str(evidence)])
    run("candidate-cached-trajectory", [str(python), "-m", "scripts.check_checkpoint_trajectory",
        "--attention-backend", "flashinfer", "--output-dir", str(evidence / "trajectory")], required=True)
    summary = json.loads((evidence / "accuracy-summary.json").read_text())
    if summary["status"] == "passed":
        assert all(c["attention_backend"] == "flashinfer-0.2.5" for c in summary["cases"].values())
    print("Frozen paging accuracy:", summary["status"])
    print("Return the exported ZIP for analysis. Default promotion and P5.3 closure require evidence review.")
except BaseException:
    (evidence / "integration-failure.txt").write_text(traceback.format_exc())
    print("Integration incomplete; export the failure and partial evidence.")
'''


def build_candidate(output: Path) -> None:
    # Different source is explicit. The old trace builder's runtime identity check
    # is preserved; this accuracy experiment instead compares retained raw logits.
    retained = ("prior-failure/target.pt", "prior-failure/metadata.json")
    if (ROOT / "restored-source-identity.json").exists():
        retained += ("restored-source-identity.json",)
    build(output, accuracy=True, retained_files=retained)
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
    export = """try:
    (evidence / "source-after.json").write_text(json.dumps(verify_source(), indent=2))
except BaseException:
    (evidence / "source-verification-failure.txt").write_text(traceback.format_exc())
""" + textwrap.dedent(code[export_start:])
    export = export.replace(
        "display(FileLink(str(archive_path)))",
        "display(FileLink(os.path.relpath(archive_path, Path.cwd())))",
    )
    export = export.replace(
        "Paged-decoder accuracy validation, not full release acceptance",
        "Experimental FlashInfer candidate; not default promotion or full release",
    )
    intro = """# Experimental native paged attention: FlashInfer

Fresh Kaggle session: enable Internet and T4 GPU, then Run All. Exact source is
embedded; no upload, Git push or manual patch is needed. Only GPU 0 is visible.
Download the final ZIP, even on failure, using the link or Kaggle's output files.

PagedCore still owns the decoder, RoPE, FP16 K/V pool and transactional writes.
Only explicitly selected decode attention uses FlashInfer. The existing backend
remains the default pending T4 acceptance. Thresholds are unchanged.

Setup installs the frozen Python 3.11.15 / Torch 2.7.1+cu126 environment plus
hash-verified FlashInfer 0.2.5 JIT source and pinned Ninja. An available NVCC and
C++ compiler are required; Kaggle compilation/runtime compatibility is unproven.
Failures export logs; no fallback or passing skip is used.

Run full CPU checks, then eight native tests using independent CPU FP64,
fragmented/poisoned pages, lengths 1/15/16/17/49/2048, large scores, staged rollback
and worker/stream ownership. Compare baseline and candidate at exactly 15 prefill
tokens + 1 decode token; baseline raw logits must match the retained failure.
Then run the frozen single/mixed/fragmented/held-out T4 suite with the candidate
and collect 32 cached teacher-forced steps with instrumentation controls.

This current source was recovered from the retained checkpoint evidence after
the Windows workspace disconnected. Source origin/hashes are included. The
old source remains the baseline; this notebook contains the explicit adapter changes.
No T4 passing result is claimed before execution and review. Model quality,
service behavior, throughput and maximum-active workspace remain separate gates.
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
