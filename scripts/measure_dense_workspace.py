"""Record single-sequence dense memory in a process without an HF model or KV pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns, time_ns
from typing import Any, TextIO

import torch

from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.decoder import generate, prefill
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import FinishReason
from pagedcore.oracle import reference_math
from pagedcore.weights import TinyLlamaWeights, load_weights

CASES = (
    ("maximum_prefill", 2048, 0),
    ("maximum_prompt", 2047, 1),
    ("maximum_output", 1792, 256),
)
REPETITIONS = 3


def _sample(phase: str) -> dict[str, Any]:
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    return {
        "phase": phase,
        "timestamp_ns": time_ns(),
        "monotonic_ns": monotonic_ns(),
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "allocator_stats": dict(torch.cuda.memory_stats(0)),
    }


def _diagnostics(stream: TextIO, label: str) -> None:
    stream.write(f"\n{label}\n{torch.cuda.memory_summary(0)}\n")
    stream.flush()


def _run_path(weights: TinyLlamaWeights, prompt_tokens: int, output_tokens: int) -> dict[str, Any]:
    # Fixed IDs reproduce tensor shapes without retaining any prompt or output text.
    input_ids = torch.arange(prompt_tokens, device="cuda:0")[None] % 31
    if output_tokens:
        result = generate(input_ids, weights, max_new_tokens=output_tokens, ignore_eos=True)
        if (
            result.finish_reason is not FinishReason.LENGTH
            or len(result.greedy_ids) != output_tokens
            or len(result.emitted_ids) != output_tokens
        ):
            raise RuntimeError("dense generation did not reach the requested output limit")
    else:
        logits, cache = prefill(input_ids, weights)
        if logits.shape != (1, prompt_tokens, 32000) or cache.length != prompt_tokens:
            raise RuntimeError("dense prefill did not reach the requested context length")
    # Keep prefill's returned logits/cache alive through this sample, as a caller would.
    return _sample("after_forward")


def measure(directory: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("dense workspace measurement requires exactly one CUDA-visible T4")
    properties = torch.cuda.get_device_properties(0)
    if "t4" not in properties.name.casefold():
        raise RuntimeError("dense workspace measurement requires an NVIDIA T4")
    if (
        torch.cuda.memory.get_allocator_backend() != "native"
        or os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
    ):
        raise RuntimeError("dense workspace peaks require the native CUDA caching allocator")

    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "dense-workspace.json"
    report: dict[str, Any] = {
        "scope": "Single-sequence owned dense decoder; no HF model or paged KV pool",
        "status": "running",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": collect_environment_diagnostics(),
        "provenance": collect_validation_provenance(),
        "target": {
            "device": "cuda:0",
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "memory_bytes": properties.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_visible_device_count": torch.cuda.device_count(),
        },
        "allocator": {
            "backend": torch.cuda.memory.get_allocator_backend(),
            "configuration": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "seed": 0,
        "input_pattern": "arange(prompt_tokens) % 31",
        "ignore_eos": True,
        "repetitions": REPETITIONS,
        "runs": [],
        "limitations": [
            "Device-free samples are synchronized boundaries, not device-wide transient peaks.",
            "Allocator peaks exclude allocations made outside PyTorch's CUDA allocator.",
            "Peak increments include input, KV cache and logits as well as temporary tensors.",
            "No pool, batched decode or service memory safety is established by this record.",
        ],
    }
    # Refuse stale evidence; subsequent writes preserve progress if a later path fails.
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def save() -> None:
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    with (directory / "dense-workspace-allocator.txt").open("x", encoding="utf-8") as diagnostics:
        try:
            seed_everything(0)
            with reference_math() as math_settings:
                report["math_settings"] = math_settings
                report["before_weights"] = _sample("before_weights")
                save()
                weights = load_weights(device="cuda:0")
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                report["model_baseline"] = _sample("model_baseline")
                _diagnostics(diagnostics, "model_baseline")
                save()
                for name, prompt_tokens, output_tokens in CASES:
                    # Only the first run of each path releases cached allocator segments.
                    torch.cuda.empty_cache()
                    for repetition in range(REPETITIONS):
                        run: dict[str, Any] = {
                            "case": name,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "decode_steps": max(0, output_tokens - 1),
                            "repetition": repetition,
                            "allocator_state": "empty_cache" if repetition == 0 else "reused",
                            "status": "running",
                            "samples": [_sample("before_forward")],
                        }
                        report["runs"].append(run)
                        save()
                        torch.cuda.reset_peak_memory_stats(0)
                        run["samples"].append(_run_path(weights, prompt_tokens, output_tokens))
                        run["samples"].append(_sample("after_cleanup"))
                        before, after, cleanup = run["samples"]
                        run["peak_allocated_above_baseline_bytes"] = (
                            after["peak_allocated_bytes"] - before["allocated_bytes"]
                        )
                        run["peak_reserved_above_baseline_bytes"] = (
                            after["peak_reserved_bytes"] - before["reserved_bytes"]
                        )
                        run["retained_allocated_above_baseline_bytes"] = (
                            cleanup["allocated_bytes"] - before["allocated_bytes"]
                        )
                        _diagnostics(diagnostics, f"{name} repetition={repetition}")
                        # First use can retain library workspace. Repeats must stabilize.
                        if (
                            repetition > 0
                            and cleanup["allocated_bytes"] != before["allocated_bytes"]
                        ):
                            raise RuntimeError(
                                "dense path retained live tensor allocations after cleanup"
                            )
                        for counter in ("num_alloc_retries", "num_ooms"):
                            if (
                                cleanup["allocator_stats"][counter]
                                != before["allocator_stats"][counter]
                            ):
                                raise RuntimeError(
                                    f"dense path increased allocator counter {counter}"
                                )
                        run["status"] = "passed"
                        save()
            report["status"] = "passed"
            save()
        except Exception as error:
            report["status"] = "failed"
            if report["runs"] and report["runs"][-1]["status"] == "running":
                report["runs"][-1]["status"] = "failed"
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            save()
            # A fatal CUDA error can also prevent diagnostics; retain the earlier raw samples.
            with suppress(RuntimeError):
                report["failure_sample"] = _sample("failure")
                _diagnostics(diagnostics, "failure")
                save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    measure(parser.parse_args().directory)
