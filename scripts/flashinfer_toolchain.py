"""Compiler-only recovery for the retained FlashInfer 0.2.5 CUDA 13 link failure."""

from __future__ import annotations


def recovery_environment(nvcc_output, workspace_base, existing_append_flags=""):
    """Use the actual NVCC release, independently of Torch's CUDA runtime version.

    This function is embedded unchanged in the replay notebook, so imports stay
    local. NVCC_APPEND_FLAGS is NVIDIA's compiler hook, not a FlashInfer patch.
    """
    import re
    import shlex
    from pathlib import Path

    match = re.search(r"Cuda compilation tools,\s+release (\d+)\.(\d+)", nvcc_output)
    if match is None:
        raise ValueError("cannot identify the actual NVCC release")
    release = tuple(map(int, match.groups()))
    flags = existing_append_flags.strip()
    selected = []
    if release >= (13, 0):
        tokens = shlex.split(flags)
        settings = []
        for index, token in enumerate(tokens):
            name, _, value = token.partition("=")
            if name in ("-static-global-template-stub", "--static-global-template-stub"):
                settings.append(
                    value if "=" in token else tokens[index + 1] if index + 1 < len(tokens) else ""
                )
        if any(setting != "false" for setting in settings):
            raise ValueError("existing NVCC flags conflict with the legacy template-stub recovery")
        if not settings:
            selected = ["--static-global-template-stub=false"]
            flags = (flags + " " + selected[0]).strip()
    overrides = {
        "FLASHINFER_WORKSPACE_BASE": str(Path(workspace_base).resolve()),
        "FLASHINFER_JIT_VERBOSE": "1",
    }
    if flags:
        overrides["NVCC_APPEND_FLAGS"] = flags
    return {
        "nvcc_release": list(release),
        "added_nvcc_flags": selected,
        "environment_overrides": overrides,
        "scope": "Compiler linkage only; same pinned FlashInfer and decoder arithmetic",
    }


WARMUP = """import json, sys, traceback
from pathlib import Path
import torch
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.flashinfer_backend import FlashInferAttention
from pagedcore.kv_cache import KVAllocator

report = {"scope": "One-key native build/load smoke; not model accuracy acceptance"}
allocator = None
request = RequestId("compile-smoke")
try:
    backend = FlashInferAttention(device="cuda:0")
    pool = torch.full((22, 2, 1, 16, 4, 64), torch.nan, dtype=torch.float16, device="cuda:0")
    allocator = KVAllocator(pool)
    allocator.create(KVReservation(request, 1))
    key = torch.zeros((22, 1, 4, 64), dtype=torch.float16, device=pool.device)
    value = torch.ones_like(key)
    allocator.append(request, key, value)
    query = torch.zeros((1, 32, 1, 64), dtype=torch.float16, device=pool.device)
    output = backend(query, allocator, request, layer_index=0)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)
    report.update(native_library_loaded=True, one_key_output_exact=True)
except BaseException:
    report["error"] = traceback.format_exc()
    raise
finally:
    if allocator is not None:
        allocator.release_all(request)
        assert allocator.capacity() == CapacityCounts(1, 1, 0, 0, 0, 0)
        report["cleanup_complete"] = True
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
"""
