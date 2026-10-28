#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Commit the reviewed changes before collecting T4 evidence.\n' >&2
    exit 1
fi

evidence_dir="$(pwd)/results/local/t4-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$evidence_dir")"
mkdir "$evidence_dir"
exec 3>&1 4>&2
exec > >(tee "$evidence_dir/commands.log") 2>&1
log_pid=$!
set -x

date -u
git rev-parse HEAD
git status --porcelain
sha256sum uv.lock scripts/validate_t4.sh scripts/measure_dense_workspace.py
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi > "$evidence_dir/physical-gpus.txt"
uv --version
if [[ -d .venv ]]; then
    printf 'Setup: synchronize an existing environment.\n'
else
    printf 'Setup: create a fresh environment.\n'
fi
uv sync --frozen --extra cuda --extra dev
uv pip freeze --python .venv/bin/python > "$evidence_dir/dependencies.txt"
uv run --no-sync python - "$evidence_dir" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Validation requires exactly one CUDA-visible T4.")
properties = torch.cuda.get_device_properties(0)
if "t4" not in properties.name.casefold():
    raise SystemExit("Validation requires an NVIDIA T4.")
target = {
    "platform": platform.platform(),
    "device": "cuda:0",
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "cuda_visible_device_count": torch.cuda.device_count(),
    "name": properties.name,
    "uuid": str(properties.uuid),
    "compute_capability": f"{properties.major}.{properties.minor}",
    "memory_bytes": properties.total_memory,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
}
(Path(sys.argv[1]) / "target.json").write_text(json.dumps(target, indent=2) + "\n")
PY
uv run --no-sync pagedcore verify --manifest-path "$evidence_dir/manifest.json"
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m ruff check .
uv run --no-sync python -m mypy --strict src/pagedcore
uv run --no-sync python -m pytest -m "not gpu and not benchmark" -ra \
    --junitxml="$evidence_dir/cpu-tests.xml"
uv run --no-sync python scripts/measure_dense_workspace.py "$evidence_dir"
uv run --no-sync python scripts/measure_dense_workspace.py "$evidence_dir" --with-pool
uv run --no-sync python -m pytest -m gpu -v -ra \
    --junitxml="$evidence_dir/gpu-tests.xml"
uv run --no-sync pagedcore oracle --case boundary_15 \
    --output "$evidence_dir/reference.pt" > "$evidence_dir/capture.json"
uv run --no-sync pagedcore oracle --case boundary_15 \
    --compare "$evidence_dir/reference.pt" \
    --output "$evidence_dir/reference-repeat.pt" > "$evidence_dir/comparison.json"
git status --porcelain
test -z "$(git status --porcelain)"
date -u
set +x
# Finish tee before packaging so the archive retains the complete command log.
exec 1>&3 2>&4 3>&- 4>&-
wait "$log_pid"

uv run --no-sync python - "$evidence_dir" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

directory = Path(sys.argv[1])
report = ET.parse(directory / "gpu-tests.xml")
cases = report.findall(".//testcase")
required = {
    "test_pinned_checkpoint_loads_and_runs_one_reference_forward_pass",
    "test_seed_everything_repeats_cuda_random_streams",
    "test_public_reference_capture_and_repeatability_on_t4",
    "test_pinned_decoder_layer_parity_on_t4",
    "test_kv_pool_storage_on_t4",
    "test_kv_allocator_storage_and_stream_ownership_on_t4",
}
fixtures = json.loads(Path("tests/fixtures/reference_prompts.json").read_text())
required.update(
    f"test_pinned_decoder_greedy_parity_on_t4[{case['name']}]"
    for case in fixtures["prompts"]
)
if not required <= {case.attrib["name"] for case in cases} or any(
    case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
):
    raise SystemExit("GPU evidence rejected: required tests must execute and pass.")
properties = report.findall(".//testsuite/properties/property[@name='decoder_tolerance_envelope']")
if len(properties) != 1:
    raise SystemExit("GPU evidence rejected: owned-decoder tolerance measurements are required.")
envelope = json.loads(properties[0].attrib["value"])
manifest = json.loads((directory / "manifest.json").read_text())
envelope.update(
    environment=manifest["environment"],
    provenance=manifest["provenance"],
    target=json.loads((directory / "target.json").read_text()),
)
maxima = {}
for group in ("intermediates", "logits"):
    entries = [
        {"case": case, "tensor": tensor, **errors}
        for case, measurement in envelope["cases"].items()
        for tensor, errors in measurement["tensors"].items()
        if tensor.endswith(".logits") == (group == "logits")
    ]
    maxima[group] = {
        metric: max(entries, key=lambda entry: entry[metric])
        for metric in ("max_absolute_error", "max_tolerance_ratio")
    }
envelope["maxima"] = maxima
(directory / "tolerance-envelope.json").write_text(
    json.dumps(envelope, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
expected_runs = {
    (name, prompt, output, repetition)
    for name, prompt, output in (
        ("maximum_prefill", 2048, 0),
        ("maximum_prompt", 2047, 1),
        ("maximum_output", 1792, 256),
    )
    for repetition in range(3)
}
for stem in ("dense-workspace", "pool-dense-workspace"):
    workspace_path = directory / f"{stem}.json"
    diagnostics_path = directory / f"{stem}-allocator.txt"
    if (
        not workspace_path.is_file()
        or not diagnostics_path.is_file()
        or not diagnostics_path.stat().st_size
    ):
        raise SystemExit(f"GPU evidence rejected: {stem} dense workspace data/diagnostics required.")
    workspace = json.loads(workspace_path.read_text())
    runs = workspace.get("runs", [])
    if (
        workspace.get("status") != "passed"
        or len(runs) != len(expected_runs)
        or {
            (run["case"], run["prompt_tokens"], run["output_tokens"], run["repetition"])
            for run in runs
        } != expected_runs
        or any(
            run.get("status") != "passed"
            or [sample["phase"] for sample in run["samples"]]
            != ["before_forward", "after_forward", "after_cleanup"]
            or any(not sample.get("allocator_stats") for sample in run["samples"])
            for run in runs
        )
    ):
        raise SystemExit(f"GPU evidence rejected: all {stem} dense workspace paths/samples must pass.")
    if stem != "pool-dense-workspace":
        continue
    pool = workspace.get("pool_pressure", {})
    settings = manifest["environment"]["settings"]
    pool_bytes = settings["kv_pool_mib"] * 2**20
    margin_bytes = settings["workspace_margin_mib"] * 2**20
    model = workspace["model_baseline"]
    baseline = workspace["pool_baseline"]
    increment = max(run["peak_allocated_above_baseline_bytes"] for run in runs)
    if (
        workspace.get("mode") != "pool_pressure"
        or pool.get("kind") != "unused_fp16_reservation"
        or pool.get("requested_bytes") != pool_bytes
        or pool.get("tensor_bytes") != pool_bytes
        or pool.get("workspace_margin_bytes") != margin_bytes
        or pool.get("configured_max_active_sequences") != settings["max_active_sequences"]
        or pool.get("measured_active_sequences") != 1
        or pool.get("capacity_validated") is not False
        or pool.get("max_dense_increment_bytes") != increment
        or pool.get("dense_increment_exceeds_margin") is not (increment > margin_bytes)
        or model["device_free_bytes"] < pool_bytes + margin_bytes
        or baseline["device_free_bytes"] < margin_bytes
        or baseline["allocated_bytes"] < model["allocated_bytes"] + pool_bytes
        or any(
            sample["allocated_bytes"] < model["allocated_bytes"] + pool_bytes
            for run in runs for sample in run["samples"]
        )
        or any(
            baseline["allocator_stats"][counter] != model["allocator_stats"][counter]
            for counter in ("num_alloc_retries", "num_ooms")
        )
    ):
        raise SystemExit("GPU evidence rejected: pool pressure reservation/settings do not match.")
archive = directory.with_suffix(".zip")
with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(directory.iterdir()):
        bundle.write(path, f"{directory.name}/{path.name}")
print(f"Saved T4 evidence: {archive}")
PY
