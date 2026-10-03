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

    assert provenance["setup_command"] == "uv sync --frozen --extra cuda --extra dev"
    assert provenance["command"] == "uv run pagedcore verify"
    assert provenance["uv_lock_sha256"] == expected_hash


def test_verify_command_writes_manifest(monkeypatch, tmp_path, capsys) -> None:
    manifest = {
        "manifest_version": 1,
        "status": "passed",
        "forward_pass": {"status": "passed"},
    }
    monkeypatch.setattr(cli, "verify_target", lambda: manifest)
    manifest_path = tmp_path / "nested" / "manifest.json"

    assert cli.main(["verify", "--manifest-path", str(manifest_path)]) == 0

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
