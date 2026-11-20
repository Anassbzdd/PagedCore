import base64
import hashlib
import io
import json
import subprocess
from zipfile import ZipFile

import pytest
from scripts.build_flashinfer_notebook import build_candidate
from scripts.build_flashinfer_toolchain_notebook import (
    build_replay,
    embedded_constants,
    insert_at_run,
)
from scripts.flashinfer_toolchain import WARMUP, recovery_environment

pytestmark = pytest.mark.unit

NVCC_13 = """nvcc: NVIDIA (R) Cuda compiler driver
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
"""


def test_actual_kaggle_compiler_gets_legacy_linkage_and_isolated_workspace(tmp_path):
    result = recovery_environment(NVCC_13, tmp_path / "fresh-cache")
    assert result["nvcc_release"] == [13, 0]
    assert result["added_nvcc_flags"] == ["--static-global-template-stub=false"]
    assert result["environment_overrides"] == {
        "NVCC_APPEND_FLAGS": "--static-global-template-stub=false",
        "FLASHINFER_JIT_VERBOSE": "1",
        "FLASHINFER_WORKSPACE_BASE": str((tmp_path / "fresh-cache").resolve()),
    }
    assert not (tmp_path / "fresh-cache").exists()  # configuration never deletes or creates caches


@pytest.mark.parametrize("release", ["11.8", "12.6", "12.8"])
def test_older_compilers_keep_their_default_and_existing_flags(tmp_path, release):
    result = recovery_environment(NVCC_13.replace("release 13.0", "release " + release), tmp_path)
    assert result["added_nvcc_flags"] == []
    assert "NVCC_APPEND_FLAGS" not in result["environment_overrides"]
    with_existing = recovery_environment(
        NVCC_13.replace("release 13.0", "release " + release), tmp_path, "-DKEEP=1"
    )
    assert with_existing["environment_overrides"]["NVCC_APPEND_FLAGS"] == "-DKEEP=1"


@pytest.mark.parametrize(
    "flags", ["--static-global-template-stub=false", "-static-global-template-stub false"]
)
def test_recovery_is_idempotent_and_preserves_other_flags(tmp_path, flags):
    existing = "-DKEEP=1 " + flags
    result = recovery_environment(NVCC_13, tmp_path, existing)
    assert result["environment_overrides"]["NVCC_APPEND_FLAGS"] == existing
    assert result["added_nvcc_flags"] == []


@pytest.mark.parametrize(
    "flags",
    [
        "--static-global-template-stub=true",
        "-static-global-template-stub true",
        "--static-global-template-stub",
    ],
)
def test_conflicting_legacy_flags_are_not_silently_overridden(tmp_path, flags):
    with pytest.raises(ValueError, match="conflict"):
        recovery_environment(NVCC_13, tmp_path, flags)


def test_unidentified_toolkit_is_rejected_and_warmup_compiles_as_python(tmp_path):
    with pytest.raises(ValueError, match="actual NVCC"):
        recovery_environment("torch 2.7.1+cu126, driver supports CUDA 13", tmp_path)
    compile(WARMUP, "native-warmup", "exec")


@pytest.mark.parametrize("multiline", [False, True])
def test_command_insertion_preserves_formatted_and_compact_notebooks(multiline):
    call = (
        '    run(\n        "nvcc",\n        ["nvcc"],\n    )\n'
        if multiline
        else '    run("nvcc", ["nvcc"])\n'
    )
    code = "try:\n" + call + "except Exception:\n    pass\n"
    result = insert_at_run(code, "nvcc", "    recorded = True\n", after=True)
    compile(result, "insertion", "exec")
    assert result.index("recorded = True") > result.index('"nvcc"')


def test_replay_retains_every_payload_byte_and_records_compiler_change(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "fixture-identity")
    original = tmp_path / "original.ipynb"
    build_candidate(original)
    old = json.loads(original.read_text())
    constants = embedded_constants("".join(old["cells"][1]["source"]))
    payload = base64.b64decode(constants["PAYLOAD"])
    archive = tmp_path / "returned.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("source-snapshot.zip", payload)
        z.writestr("evidence/control.txt", b"retained")
        z.writestr(
            "evidence/checksums.json",
            json.dumps({"control.txt": hashlib.sha256(b"retained").hexdigest()}),
        )
    output = tmp_path / "replay.ipynb"
    origin = build_replay(original, archive, output)
    replay = json.loads(output.read_text())
    setup = "".join(replay["cells"][1]["source"])
    assert embedded_constants(setup) == constants
    assert origin["source_snapshot_unchanged"]
    assert "NVCC_APPEND_FLAGS" in setup and "toolchain-recovery.json" in setup
    kernel = "".join(next(c for c in replay["cells"] if c["id"] == "kernel")["source"])
    assert kernel.index('"native-compile-smoke"') < kernel.index('"native-kernel-tests"')
    with ZipFile(io.BytesIO(payload)) as z:
        identity = json.loads(z.read("source-identity.json"))
        assert all(
            hashlib.sha256(z.read(name)).hexdigest() == digest
            for name, digest in identity["sha256"].items()
        )
    for cell in replay["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
