import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pagedcore import cli
from pagedcore.config import PagedCoreConfig


def test_collect_environment_diagnostics_reports_safe_resolved_values(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_read_torch_environment",
        lambda: {
            "version": "2.7.1",
            "cuda_runtime": "12.6",
            "cuda_available": True,
            "gpus": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_read_nvidia_smi",
        lambda: ("550.54.15", [{"name": "NVIDIA T4", "memory_mib": 15360}]),
    )

    diagnostics = cli.collect_environment_diagnostics(
        PagedCoreConfig(kv_pool_mib=7168, bind_address="127.0.0.1")
    )

    assert diagnostics["python"]
    assert diagnostics["cuda"] == {"available": True, "runtime": "12.6"}
    assert diagnostics["driver"] == "550.54.15"
    assert diagnostics["gpu"] == [{"name": "NVIDIA T4", "memory_mib": 15360}]
    assert diagnostics["model"] == {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "af8e934848d8dd00074cc2cd8a40a9b05c3b011e",
    }
    assert diagnostics["settings"]["kv_pool_mib"] == 7168
    assert diagnostics["settings"]["bind_address"] == "127.0.0.1"


def test_env_command_prints_json_without_unrelated_environment_values(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_read_torch_environment",
        lambda: {
            "version": "2.7.1+cpu",
            "cuda_runtime": None,
            "cuda_available": False,
            "gpus": [],
        },
    )
    monkeypatch.setattr(cli, "_read_nvidia_smi", lambda: (None, []))
    monkeypatch.setenv("PAGEDCORE_PRIVATE_TOKEN", "diagnostics-secret-sentinel")
    monkeypatch.setenv("HF_TOKEN", "diagnostics-secret-sentinel")

    assert cli.main(["env"]) == 0

    output = capsys.readouterr().out
    diagnostics = json.loads(output)
    assert diagnostics["settings"]["log_level"] == "INFO"
    assert "diagnostics-secret-sentinel" not in output


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 2)],
)
def test_nvidia_smi_diagnostic_falls_back_when_command_fails(monkeypatch, failure) -> None:
    def fail_to_run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli.subprocess, "run", fail_to_run)

    assert cli._read_nvidia_smi() == (None, [])


def test_environment_diagnostics_fall_back_to_torch_gpu_list(monkeypatch) -> None:
    expected_gpus = [{"name": "Tesla T4", "memory_mib": 15360}]
    monkeypatch.setattr(
        cli,
        "_read_torch_environment",
        lambda: {
            "version": "2.7.1+cu126",
            "cuda_runtime": "12.6",
            "cuda_available": True,
            "gpus": expected_gpus,
        },
    )
    monkeypatch.setattr(cli, "_read_nvidia_smi", lambda: (None, []))

    assert cli.collect_environment_diagnostics()["gpu"] == expected_gpus


def test_torch_diagnostic_handles_an_unavailable_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert cli._read_torch_environment() == {
        "version": None,
        "cuda_runtime": None,
        "cuda_available": False,
        "gpus": [],
    }


def test_validation_provenance_contains_the_lock_file_hash() -> None:
    provenance = cli.collect_validation_provenance()
    expected_hash = hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()

    assert provenance["setup_command"] is None
    assert provenance["command_arguments"] is None
    assert provenance["uv_lock_sha256"] == expected_hash
    assert (
        provenance["source_sha256"]["src/pagedcore/cli.py"]
        == hashlib.sha256(Path("src/pagedcore/cli.py").read_bytes()).hexdigest()
    )


def test_validation_provenance_identifies_changed_source_without_git(monkeypatch, tmp_path) -> None:
    source_path = tmp_path / "src/pagedcore/cli.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("original", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(source_path))

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(cli.subprocess, "run", missing_git)
    before = cli.collect_validation_provenance()
    source_path.write_text("changed", encoding="utf-8")
    after = cli.collect_validation_provenance()

    assert before["git_revision"] is None
    assert before["working_tree_clean"] is None
    assert before["uv_lock_sha256"] == hashlib.sha256(b"lock").hexdigest()
    assert (
        before["source_sha256"]["src/pagedcore/cli.py"]
        != after["source_sha256"]["src/pagedcore/cli.py"]
    )
    assert after["source_sha256"][".python-version"] is None


def test_validation_provenance_does_not_guess_outside_a_checkout(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "cli.py"))

    provenance = cli.collect_validation_provenance()

    assert provenance["git_revision"] is None
    assert provenance["uv_lock_sha256"] is None
    assert provenance["source_sha256"] == {}


def test_verify_command_writes_manifest(monkeypatch, tmp_path, capsys) -> None:
    manifest = {
        "manifest_version": 1,
        "status": "passed",
        "forward_pass": {"status": "passed"},
    }
    monkeypatch.setattr(cli, "verify_target", lambda: manifest)
    manifest_path = tmp_path / "nested" / "manifest.json"

    assert cli.main(["verify", "--manifest-path", str(manifest_path)]) == 0

    assert manifest["provenance"]["command_arguments"] == [
        "verify",
        "--manifest-path",
        "<manifest-path>",
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert json.loads(capsys.readouterr().out) == manifest


def test_verify_command_reports_target_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "verify_target",
        lambda: (_ for _ in ()).throw(cli.TargetVerificationError("requires T4")),
    )

    assert cli.main(["verify"]) == 1

    assert json.loads(capsys.readouterr().err) == {
        "status": "failed",
        "error": "requires T4",
    }


def test_verify_command_records_only_arguments_that_were_supplied(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "verify_target", lambda: {"status": "passed"})

    assert cli.main(["verify"]) == 0

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["provenance"]["command_arguments"] == ["verify"]


@pytest.mark.parametrize("command", ["env", "verify"])
def test_commands_report_invalid_configuration(monkeypatch, capsys, command) -> None:
    monkeypatch.setenv("PAGEDCORE_KV_POOL_MIB", "invalid-private-value")

    assert cli.main([command]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "failed",
        "error": "PAGEDCORE_KV_POOL_MIB must be an integer",
    }
    assert "invalid-private-value" not in captured.err


@pytest.mark.parametrize("operation", ["mkdir", "write_text"])
@pytest.mark.parametrize("error_type", [PermissionError, ValueError])
def test_verify_command_reports_manifest_io_failure(
    monkeypatch, tmp_path, capsys, operation, error_type
) -> None:
    monkeypatch.setattr(cli, "verify_target", lambda: {"status": "passed"})

    def fail(*args, **kwargs):
        raise error_type("private-path-sentinel")

    monkeypatch.setattr(Path, operation, fail)
    assert cli.main(["verify", "--manifest-path", str(tmp_path / "manifest.json")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "failed",
        "error": "could not write verification manifest",
    }
    assert "private-path-sentinel" not in captured.err


def test_verify_command_captures_process_arguments_without_private_paths(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(cli, "verify_target", lambda: {"status": "passed"})
    manifest_path = tmp_path / "private-path-sentinel.json"
    monkeypatch.setattr(sys, "argv", ["pagedcore", "verify", f"--manifest-path={manifest_path}"])

    assert cli.main() == 0

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["provenance"]["command_arguments"] == [
        "verify",
        "--manifest-path=<manifest-path>",
    ]
    assert "private-path-sentinel" not in json.dumps(manifest)
