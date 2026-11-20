"""Package the checkpoint diagnostic using the frozen evidence bootstrap."""

import json
import subprocess
import sys

from scripts.build_numerical_notebook import ROOT, build

output = ROOT / "notebooks/checkpoint_trajectory_validation.ipynb"
build(output, layer_trace=True, prompt_tokens=15)
notebook = json.loads(output.read_text())
notebook["cells"][0]["source"] = [
    """# Real-checkpoint component and cached-trajectory diagnostic

Enable Internet and T4 GPU, then Run All in a fresh Kaggle session.
Exact source and the frozen environment are embedded; no dataset or Git push is needed.

The four cells set up/verify the pinned checkpoint, run CPU checks, run the T4
diagnostic and export one ZIP. Download it even on failure.

Use 15 fixture tokens then feed 32 fixed ground-truth continuation IDs. Both paths
see identical histories and attend to lengths 16 through 47. This exercises actual
KV reuse and page boundaries; it is not greedy generation or a quality dataset.
Record complete logit/KV errors at all steps without discarding numerical failures.

At steps 0, 15 and 31, inspect both paths' real-weight RMSNorm, SwiGLU and final-head
operations against separately written FP64 reductions with documented FP16 rounding
boundaries. Save inputs and native/reference outputs. These are diagnostic errors,
not new pass/fail limits or a full independent model oracle. Residual composition
has sparse CPU coverage; this does not independently validate its real-weight math.

Run once without instrumentation and once with it. Both paths' logits must be exact
between runs at every step, or stop interpretation and retain the failure.
Only GPU 0 is visible. The full T4 runtime and end-to-end behavior of this new
collector remain unvalidated until its returned ZIP is reviewed.

This does not change serving arithmetic, thresholds or the comparator; P5.3 stays
open. vLLM quality comparison and long-context coverage up to 2048 remain future work.
"""
]
notebook["cells"][1]["source"] = (
    "".join(notebook["cells"][1]["source"])
    .replace("pagedcore-layer-trace-", "pagedcore-checkpoint-")
    .splitlines(True)
)
notebook["cells"][2]["source"] = """checks_ok = False
try:
    assert setup_ok
    run("cpu-component-checks", [str(python), "-m", "pytest",
        "tests/unit/test_checkpoint_diagnostic.py", "tests/unit/test_independent_components.py",
        "tests/unit/test_independent_decoder_math.py", "-q",
        "--junitxml=" + str(evidence / "cpu-components.xml")], required=True)
    checks_ok = True
except BaseException:
    (evidence / "cpu-check-failure.txt").write_text(traceback.format_exc())
    print("CPU checks incomplete; run the export cell.")
""".splitlines(True)
notebook["cells"][3]["source"] = """try:
    assert checks_ok
    run("checkpoint-trajectory", [str(python), "-m", "scripts.check_checkpoint_trajectory",
        "--output-dir", str(evidence / "checkpoint-trajectory")], required=True)
    result = json.loads((evidence / "checkpoint-trajectory/audited/report.json").read_text())
    print("Recorded cached steps:", len(result["rows"]))
    print("Component checks:", len(result["components"]))
    print("Instrumentation exact:", all(r["instrumentation_logits_exact"] for r in result["rows"]))
    print("Final step:", result["rows"][-1])
except BaseException:
    (evidence / "trajectory-failure.txt").write_text(traceback.format_exc())
    print("Run incomplete; export its partial evidence using the final cell.")
""".splitlines(True)
notebook["cells"][4]["source"] = (
    "".join(notebook["cells"][4]["source"])
    .replace(
        "15-token prefill + one decode step; cause localization, not acceptance",
        "Real-checkpoint component and 32-step cached diagnostic; not accuracy or quality acceptance",
    )
    .splitlines(True)
)
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), cell["id"], "exec")
output.write_text(json.dumps(notebook, indent=1) + "\n")
subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(output)], check=True)
subprocess.run([sys.executable, "-m", "ruff", "format", str(output)], check=True)
print(output)
