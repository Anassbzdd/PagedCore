import hashlib
import io
import json
import subprocess
import sys
from types import ModuleType
from zipfile import ZipFile

import pytest
from scripts.build_flashinfer_notebook import build_candidate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mutate_source", [False, True])
def test_candidate_exports_setup_failure_and_verifies_embedded_source(
    tmp_path, monkeypatch, mutate_source
):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "test-source-identity")
    path = tmp_path / "candidate.ipynb"
    build_candidate(path)
    notebook = json.loads(path.read_text())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(codes) == 5
    assert "--no-deps" in codes[0] and "--no-build-isolation" in codes[0]
    assert "--decode-backend=flashinfer" in codes[3]
    assert '"--prompt-tokens", "15", "--single-only"' in codes[3]
    assert "prior-failure-control" in codes[3]

    class FailedInstall:
        returncode = 1

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 1

    display = ModuleType("IPython.display")
    display.FileLink, display.display = lambda path: path, lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(subprocess, "Popen", FailedInstall)
    destination = tmp_path / "output"
    monkeypatch.setenv("PAGEDCORE_EVIDENCE_WORKDIR", str(destination))
    namespace = {}
    for index, code in enumerate(codes):
        if index == len(codes) - 1 and mutate_source:
            (namespace["source"] / "src/pagedcore/flashinfer_backend.py").write_text("# mutation")
        exec(compile(code, f"cell-{index}", "exec"), namespace)
    archive_path = next(destination.glob("pagedcore-flashinfer-*.zip"))
    with ZipFile(archive_path) as archive:
        assert "evidence/setup-failure.txt" in archive.namelist()
        assert "evidence/integration-failure.txt" in archive.namelist()
        assert ("evidence/source-verification-failure.txt" in archive.namelist()) == mutate_source
        assert json.loads(archive.read("evidence/commands.json"))[0]["returncode"] == 1
        checks = json.loads(archive.read("evidence/checksums.json"))
        assert all(
            hashlib.sha256(archive.read("evidence/" + name)).hexdigest() == digest
            for name, digest in checks.items()
        )
        with ZipFile(io.BytesIO(archive.read("source-snapshot.zip"))) as source:
            identity = json.loads(source.read("source-identity.json"))
            assert "src/pagedcore/flashinfer_backend.py" in identity["sha256"]
            assert "prior-failure/target.pt" in identity["sha256"]
            assert all(
                hashlib.sha256(source.read(name)).hexdigest() == digest
                for name, digest in identity["sha256"].items()
            )
