import json
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from scripts import measure_dense_workspace as workspace

from pagedcore.config import PagedCoreConfig
from pagedcore.decoder import GreedyResult
from pagedcore.engine_types import FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        allocated=100, reserved=128, peak=100, retries=0, ooms=0, events=[], total=16 * 2**20
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="synthetic T4", uuid="test", major=7, minor=5, total_memory=4096
        ),
    )
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.delenv("PYTORCH_NO_CUDA_MEMORY_CACHING", raising=False)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: state.events.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (state.total - state.reserved, state.total)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: state.allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: state.reserved)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: state.peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: state.reserved)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device: {"num_alloc_retries": state.retries, "num_ooms": state.ooms},
    )
    monkeypatch.setattr(
        torch.cuda, "memory_summary", lambda device: "synthetic allocator diagnostics"
    )

    def empty_cache() -> None:
        state.events.append("empty_cache")
        state.reserved = state.allocated + 28

    def reset(device: int) -> None:
        state.events.append("reset")
        state.peak = state.allocated

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(workspace, "seed_everything", lambda seed: None)
    monkeypatch.setattr(workspace, "reference_math", lambda: nullcontext({"synthetic": True}))
    monkeypatch.setattr(workspace, "load_weights", lambda **kwargs: object())
    monkeypatch.setattr(workspace, "collect_environment_diagnostics", lambda: {"synthetic": True})
    monkeypatch.setattr(
        workspace, "collect_validation_provenance", lambda: {"git_revision": "test"}
    )

    def path(weights: object, prompt: int, output: int) -> dict[str, Any]:
        state.events.append((prompt, output))
        baseline = state.allocated
        state.allocated, state.reserved, state.peak = baseline + 50, baseline + 924, baseline + 800
        sample = workspace._sample("after_forward")
        state.allocated = baseline
        return sample

    state.path = path
    monkeypatch.setattr(workspace, "_run_path", path)
    return state


def test_recorder_retains_transient_peaks_boundaries_and_warm_reuse(
    recorder: SimpleNamespace, tmp_path: Path
) -> None:
    workspace.measure(tmp_path)
    report = json.loads((tmp_path / "dense-workspace.json").read_text())
    assert report["status"] == "passed"
    assert report["environment"] == {"synthetic": True}
    assert report["provenance"]["git_revision"] == "test"
    assert report["target"]["uuid"] == "test"
    assert len(report["recorder_sha256"]) == 64
    assert report["math_settings"] == {"synthetic": True}
    assert recorder.events.count((2048, 0)) == 3
    assert recorder.events.count((2047, 1)) == 3
    assert recorder.events.count((1792, 256)) == 3
    assert recorder.events.count("reset") == 9
    assert recorder.events.count("empty_cache") == 4
    for run in report["runs"]:
        assert run["status"] == "passed"
        assert run["decode_steps"] == (255 if run["output_tokens"] == 256 else 0)
        before, after, cleanup = run["samples"]
        assert [sample["phase"] for sample in run["samples"]] == [
            "before_forward",
            "after_forward",
            "after_cleanup",
        ]
        assert before["reserved_bytes"] == (128 if run["repetition"] == 0 else 1024)
        assert after["peak_allocated_bytes"] == 900 > after["allocated_bytes"]
        assert run["peak_allocated_above_baseline_bytes"] == 800
        assert cleanup["allocated_bytes"] == before["allocated_bytes"]
        assert cleanup["reserved_bytes"] > cleanup["allocated_bytes"]
        assert after["device_free_bytes"] == recorder.total - 1024
        assert before["monotonic_ns"] <= after["monotonic_ns"] <= cleanup["monotonic_ns"]
    diagnostics = (tmp_path / "dense-workspace-allocator.txt").read_text()
    assert "maximum_prefill repetition=0" in diagnostics
    assert "maximum_output repetition=2" in diagnostics
    assert "synthetic allocator diagnostics" in diagnostics
    assert "greedy_ids" not in json.dumps(report)


@pytest.fixture
def pool_recorder(recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        workspace,
        "load_config",
        lambda: PagedCoreConfig(kv_pool_mib=1, workspace_margin_mib=1, max_active_sequences=4),
    )
    empty = torch.empty

    def allocate(count: int, *, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert dtype is torch.float16 and device == "cuda:0"
        tensor = empty(count, dtype=dtype)
        size = tensor.numel() * tensor.element_size()
        recorder.events.append(("pool", size))
        recorder.allocated += size
        recorder.reserved += size
        recorder.pool_reference = weakref.ref(tensor)

        def release() -> None:
            recorder.allocated -= size

        weakref.finalize(tensor, release)
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    return recorder


def test_pool_pressure_holds_one_allocation_and_preserves_dense_baseline(
    pool_recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace.measure(tmp_path)
    baseline = (tmp_path / "dense-workspace.json").read_bytes()

    def path(weights: object, prompt: int, output: int) -> dict[str, Any]:
        assert pool_recorder.pool_reference() is not None
        sample = pool_recorder.path(weights, prompt, output)
        # An increment above the margin is an observation, not isolated scratch demand.
        sample["peak_allocated_bytes"] += 2**20
        return sample

    monkeypatch.setattr(workspace, "_run_path", path)
    workspace.measure(tmp_path, with_pool=True)
    report = json.loads((tmp_path / "pool-dense-workspace.json").read_text())
    assert (tmp_path / "dense-workspace.json").read_bytes() == baseline
    assert report["status"] == "passed"
    assert report["mode"] == "pool_pressure"
    assert report["pool_pressure"] == {
        "kind": "unused_fp16_reservation",
        "requested_bytes": 2**20,
        "tensor_bytes": 2**20,
        "workspace_margin_bytes": 2**20,
        "configured_max_active_sequences": 4,
        "measured_active_sequences": 1,
        "capacity_validated": False,
        "max_dense_increment_bytes": 2**20 + 800,
        "dense_increment_exceeds_margin": True,
    }
    assert pool_recorder.events.count(("pool", 2**20)) == 1
    assert report["pool_baseline"]["allocated_bytes"] == 100 + 2**20
    assert all(run["samples"][0]["allocated_bytes"] == 100 + 2**20 for run in report["runs"])
    assert pool_recorder.pool_reference() is None
    assert pool_recorder.allocated == 100
    assert "pool_baseline" in (tmp_path / "pool-dense-workspace-allocator.txt").read_text()


@pytest.mark.parametrize("failure", ["headroom", "oom", "post_allocation_headroom", "retry"])
def test_pool_pressure_retains_failure_and_never_starts_forwards(
    pool_recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    allocate = torch.empty

    def fail_allocation(*args: Any, **kwargs: Any) -> torch.Tensor:
        if failure == "oom":
            raise torch.OutOfMemoryError("synthetic pool exhaustion")
        tensor = allocate(*args, **kwargs)
        if failure == "retry":
            pool_recorder.retries += 1
        else:
            pool_recorder.reserved = pool_recorder.total - 2**20 + 1
        return tensor

    if failure == "headroom":
        pool_recorder.total = 2 * 2**20
    else:
        monkeypatch.setattr(torch, "empty", fail_allocation)
    with pytest.raises(RuntimeError):
        workspace.measure(tmp_path, with_pool=True)
    report = json.loads((tmp_path / "pool-dense-workspace.json").read_text())
    assert report["status"] == "failed"
    assert report["runs"] == []
    assert report["pool_pressure"]["capacity_validated"] is False
    assert report["failure"]["type"] == ("OutOfMemoryError" if failure == "oom" else "RuntimeError")
    assert report["failure_sample"]["phase"] == "failure"
    assert not (tmp_path / "dense-workspace.json").exists()


@pytest.mark.parametrize("failure", ["oom", "retained_tensor", "retry"])
def test_failed_path_retains_raw_progress_and_allocator_diagnostics(
    recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    def fail(weights: object, prompt: int, output: int) -> dict[str, Any]:
        if failure == "oom":
            recorder.ooms += 1
            raise torch.OutOfMemoryError("synthetic exhaustion")
        sample = recorder.path(weights, prompt, output)
        if failure == "retained_tensor":
            recorder.allocated += 1
        else:
            recorder.retries += 1
        return sample

    monkeypatch.setattr(workspace, "_run_path", fail)
    with pytest.raises(RuntimeError):
        workspace.measure(tmp_path)
    report = json.loads((tmp_path / "dense-workspace.json").read_text())
    assert report["status"] == "failed"
    assert report["runs"][-1]["status"] == "failed"
    assert report["runs"][0]["samples"][0]["phase"] == "before_forward"
    assert report["failure"]["type"] == ("OutOfMemoryError" if failure == "oom" else "RuntimeError")
    assert report["failure_sample"]["allocator_stats"]["num_ooms"] == (failure == "oom")
    assert "failure" in (tmp_path / "dense-workspace-allocator.txt").read_text()


def test_recorder_retains_first_use_workspace_and_requires_stable_repeats(
    recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def first_use(weights: object, prompt: int, output: int) -> dict[str, Any]:
        sample = recorder.path(weights, prompt, output)
        if recorder.events.count((prompt, output)) == 1:
            recorder.allocated += 64
        return sample

    monkeypatch.setattr(workspace, "_run_path", first_use)
    workspace.measure(tmp_path)
    report = json.loads((tmp_path / "dense-workspace.json").read_text())
    assert report["status"] == "passed"
    assert [run["retained_allocated_above_baseline_bytes"] for run in report["runs"]] == [
        64,
        0,
        0,
        64,
        0,
        0,
        64,
        0,
        0,
    ]


@pytest.mark.parametrize("target", ["cpu", "multiple", "other_gpu", "async_allocator"])
def test_recorder_rejects_unsupported_measurement_environment(
    recorder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    if target == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif target == "multiple":
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    elif target == "other_gpu":
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="A100")
        )
    else:
        monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync")
    with pytest.raises(RuntimeError):
        workspace.measure(tmp_path)
    assert not (tmp_path / "dense-workspace.json").exists()


@pytest.mark.parametrize("prompt,output", [(2048, 0), (2047, 1), (1792, 256)])
def test_measured_path_uses_owned_forwards_and_forces_full_output(
    monkeypatch: pytest.MonkeyPatch, prompt: int, output: int
) -> None:
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda count, **kwargs: arange(count))
    monkeypatch.setattr(workspace, "_sample", lambda phase: {"phase": phase})
    calls = []

    def prefill(ids: torch.Tensor, weights: object) -> tuple[torch.Tensor, SimpleNamespace]:
        calls.append((ids.shape[1], 0))
        return torch.empty((1, prompt, 32000), device="meta"), SimpleNamespace(length=prompt)

    def generate(ids: torch.Tensor, weights: object, **kwargs: Any) -> GreedyResult:
        assert kwargs == {"max_new_tokens": output, "ignore_eos": True}
        calls.append((ids.shape[1], output))
        # EOS IDs must not shorten the measurement.
        return GreedyResult((2,) * output, (2,) * output, FinishReason.LENGTH)

    monkeypatch.setattr(workspace, "prefill", prefill)
    monkeypatch.setattr(workspace, "generate", generate)
    assert workspace._run_path(object(), prompt, output) == {"phase": "after_forward"}
    assert calls == [(prompt, output)]
