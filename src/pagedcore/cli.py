from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import fields
from importlib import metadata

from pagedcore.config import PagedCoreConfig, load_config


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagedcore")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "env",
        help="print safe Python, package, CUDA, GPU, model, and configuration diagnostics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "env":
        print(json.dumps(collect_environment_diagnostics(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
