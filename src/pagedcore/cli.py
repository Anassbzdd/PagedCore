from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import fields
from importlib import metadata
from pathlib import Path

from pagedcore.config import ConfigurationError, PagedCoreConfig, load_config
from pagedcore.verification import TargetVerificationError, verify_target


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _read_nvidia_smi() -> tuple[str | None, list[dict[str, object]]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None, []

    driver: str | None = None
    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3 or not all(values):
            continue
        current_driver, name, memory_mib = values
        driver = driver or current_driver
        try:
            total_memory_mib: int | None = int(memory_mib)
        except ValueError:
            total_memory_mib = None
        gpus.append({"name": name, "memory_mib": total_memory_mib})
    return driver, gpus


def _read_torch_environment() -> dict[str, object]:
    try:
        import torch
    except (ImportError, OSError):
        return {
            "version": None,
            "cuda_runtime": None,
            "cuda_available": False,
            "gpus": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    gpus: list[dict[str, object]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(index)
            except (RuntimeError, OSError):
                continue
            gpus.append(
                {
                    "name": str(properties.name),
                    "memory_mib": properties.total_memory // (1024 * 1024),
                }
            )
    return {
        "version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpus": gpus,
    }


def _resolved_settings(config: PagedCoreConfig) -> dict[str, object]:
    return {
        field.name: (
            getattr(config, field.name).value
            if hasattr(getattr(config, field.name), "value")
            else getattr(config, field.name)
        )
        for field in fields(config)
    }


def collect_environment_diagnostics(
    config: PagedCoreConfig | None = None,
) -> dict[str, object]:
    resolved_config = load_config() if config is None else config
    torch_environment = _read_torch_environment()
    driver, nvidia_gpus = _read_nvidia_smi()
    gpus = nvidia_gpus or torch_environment["gpus"]
    return {
        "python": platform.python_version(),
        "package": {
            "name": "pagedcore",
            "version": _package_version("pagedcore"),
        },
        "dependencies": {
            "torch": torch_environment["version"],
            "transformers": _package_version("transformers"),
        },
        "cuda": {
            "available": torch_environment["cuda_available"],
            "runtime": torch_environment["cuda_runtime"],
        },
        "driver": driver,
        "gpu": gpus,
        "model": {
            "id": resolved_config.model_id,
            "revision": resolved_config.model_revision,
        },
        "settings": _resolved_settings(resolved_config),
    }


def collect_validation_provenance() -> dict[str, object]:
    project_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
        ),
        None,
    )
    lock_hash: str | None = None
    source_hashes: dict[str, str | None] = {}
    revision: str | None = None
    working_tree_clean: bool | None = None
    if project_root is not None:
        source_paths = [project_root / "pyproject.toml", project_root / ".python-version"]
        for directory in ("src/pagedcore", "tests"):
            source_paths.extend((project_root / directory).rglob("*.py"))
        for path in sorted(source_paths):
            relative_path = path.relative_to(project_root).as_posix()
            source_hashes[relative_path] = None
            with suppress(OSError):
                source_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        lock_path = project_root / "uv.lock"
        with suppress(OSError):
            lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        try:
            revision_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            revision = revision_result.stdout.strip()
            working_tree_clean = not status_result.stdout.strip()

    return {
        "setup_command": None,
        "command_arguments": None,
        "git_revision": revision,
        "working_tree_clean": working_tree_clean,
        "uv_lock_sha256": lock_hash,
        "source_sha256": source_hashes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagedcore")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "env",
        help="print safe Python, package, CUDA, GPU, model, and configuration diagnostics",
    )
    verify = commands.add_parser(
        "verify",
        help="load the pinned checkpoint and run one T4 reference forward pass",
    )
    verify.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/local/environment-manifest.json"),
        help="where to write the prompt-free verification manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    actual_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = _build_parser().parse_args(actual_arguments)
    try:
        if arguments.command == "env":
            print(json.dumps(collect_environment_diagnostics(), indent=2, sort_keys=True))
            return 0
        if arguments.command == "verify":
            manifest = verify_target()
            sanitized_arguments = [actual_arguments[0]]
            for argument in actual_arguments[1:]:
                if argument.startswith("-"):
                    option, separator, _ = argument.partition("=")
                    sanitized_arguments.append(option + ("=<manifest-path>" if separator else ""))
                else:
                    sanitized_arguments.append("<manifest-path>")
            manifest.setdefault("provenance", {})["command_arguments"] = sanitized_arguments
            try:
                arguments.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                arguments.manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except (OSError, ValueError) as error:
                raise TargetVerificationError("could not write verification manifest") from error
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
    except (ConfigurationError, TargetVerificationError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    raise AssertionError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
