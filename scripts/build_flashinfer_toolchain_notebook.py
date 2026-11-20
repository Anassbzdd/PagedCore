"""Replay the exact returned source snapshot with the CUDA 13 compiler recovery."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from scripts.flashinfer_toolchain import WARMUP, recovery_environment


def embedded_constants(code):
    values = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("PAYLOAD", "SNAPSHOT_SHA256"):
                    values[target.id] = ast.literal_eval(node.value)
    return values


def insert_at_run(code, command_name, insertion, *, after=False):
    calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == command_name
    ]
    assert len(calls) == 1, f"expected exactly one {command_name} command"
    lines = code.splitlines(True)
    position = calls[0].end_lineno if after else calls[0].lineno - 1
    assert position is not None
    return "".join(lines[:position]) + insertion + "".join(lines[position:])


def build_replay(original_notebook: Path, returned_archive: Path, output: Path):
    notebook = json.loads(original_notebook.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][1]["source"])
    constants = embedded_constants(setup)
    payload = base64.b64decode(constants["PAYLOAD"])
    assert hashlib.sha256(payload).hexdigest() == constants["SNAPSHOT_SHA256"]
    with ZipFile(returned_archive) as archive:
        assert payload == archive.read("source-snapshot.zip"), (
            "notebook does not match returned run"
        )
        checks = json.loads(archive.read("evidence/checksums.json"))
        assert all(
            hashlib.sha256(archive.read("evidence/" + name)).hexdigest() == digest
            for name, digest in checks.items()
        )
    with ZipFile(io.BytesIO(payload)) as source:
        identity = json.loads(source.read("source-identity.json"))
        assert all(
            hashlib.sha256(source.read(name)).hexdigest() == digest
            for name, digest in identity["sha256"].items()
        )
    helper = inspect.getsource(recovery_environment)
    origin = {
        "returned_archive": returned_archive.name,
        "returned_archive_sha256": hashlib.sha256(returned_archive.read_bytes()).hexdigest(),
        "original_notebook_sha256": hashlib.sha256(original_notebook.read_bytes()).hexdigest(),
        "source_snapshot_sha256": constants["SNAPSHOT_SHA256"],
        "source_snapshot_unchanged": True,
        "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(),
        "warmup_sha256": hashlib.sha256(WARMUP.encode()).hexdigest(),
    }
    # Preserve the complete payload byte-for-byte, including every runtime file.
    insertion = (
        "TOOLCHAIN_HELPER_SOURCE = "
        + repr(helper)
        + "\n"
        + "FLASHINFER_WARMUP = "
        + repr(WARMUP)
        + "\n"
        + "TOOLCHAIN_REPLAY_ORIGIN = "
        + repr(origin)
        + "\n"
        + helper
        + "\nPAYLOAD ="
    )
    setup = setup.replace("PAYLOAD =", insertion, 1)
    configure = """    recovery = recovery_environment(
        (evidence / "nvcc.log").read_text(), run_root / "nvcc13-fresh-workspace",
        environment.get("NVCC_APPEND_FLAGS", ""))
    environment.update(recovery["environment_overrides"])
    (evidence / "toolchain-recovery.py").write_text(TOOLCHAIN_HELPER_SOURCE)
    (evidence / "native-warmup.py").write_text(FLASHINFER_WARMUP)
    (evidence / "toolchain-recovery.json").write_text(json.dumps(
        {**TOOLCHAIN_REPLAY_ORIGIN, **recovery}, indent=2))
"""
    notebook["cells"][1]["source"] = insert_at_run(setup, "nvcc", configure, after=True).splitlines(
        True
    )
    kernel = next(cell for cell in notebook["cells"] if cell["id"] == "kernel")
    code = "".join(kernel["source"])
    warmup = """    run("native-compile-smoke", [str(python), str(evidence / "native-warmup.py"),
        str(evidence / "native-warmup.json")], required=True)
"""
    kernel["source"] = insert_at_run(code, "native-kernel-tests", warmup).splitlines(True)
    intro = """# FlashInfer CUDA 13 compiler recovery

Fresh Kaggle: enable Internet + T4, then Run All. Return the final ZIP even if it fails.

The returned run passed 808 CPU checks. Its first native build failed because
CUDA 13 made template launch stubs local to individual source files; the older
FlashInfer kernel expects to link them across files. Subsequent missing-library
errors follow from that failed build. No native attention result was collected.

This replay keeps the original source snapshot byte-for-byte, including all
decoder math, FlashInfer 0.2.5, tests, thresholds and retained failure control.
It detects the actual NVCC release (not Torch's CUDA runtime version) and adds
`--static-global-template-stub=false` on CUDA 13+. Each run uses a fresh isolated
FlashInfer build directory. No package source is patched and no old cache is deleted.

A one-key build/load smoke runs first, then the same eight native tests, 15+1
comparison, frozen paging suite and cached trajectory. The compiler fix remains
unvalidated on T4 until execution. P5.3 and whole-model acceptance stay open.
"""
    notebook["cells"][0]["source"] = intro.splitlines(True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(output)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(output)], check=True)
    print(json.dumps({"notebook": str(output), **origin}, indent=2))
    return origin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-notebook", type=Path, required=True)
    parser.add_argument("--returned-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_replay(args.original_notebook, args.returned_archive, args.output)
