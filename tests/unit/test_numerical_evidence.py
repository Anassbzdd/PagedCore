import base64
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from scripts import build_numerical_notebook as notebook
from scripts import collect_numerical_evidence as evidence
from transformers.models.llama import modeling_llama as llama

import pagedcore.decoder as decoder
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit


def fragmented_cache(tokens):
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    cache = KVAllocator(pool)
    request, other = RequestId("target"), RequestId("other")
    cache.create(KVReservation(request, 2))
    cache.create(KVReservation(other, 1))
    keys = torch.zeros((22, tokens, 4, 64), dtype=torch.float16)
    values = keys.clone()
    cache.append(request, keys[:, :16], values[:, :16])
    cache.append(other, torch.ones_like(keys[:, :16]), torch.ones_like(keys[:, :16]))
    if tokens > 16:
        cache.append(request, keys[:, 16:], values[:, 16:])
    return pool, cache, request, other


def test_contiguous_halves_replay_matches_captured_t4_scalar_on_fragmented_pages(
    forbid_dense_kv_history,
):
    cases = json.loads((evidence.ROOT / "tests/fixtures/t4_value_reduction.json").read_text())[
        "cases"
    ]
    case = next(case for case in cases if len(case["probabilities"]) == 18)
    pool, cache, request, other = fragmented_cache(18)
    layer, head, channel = (case[key] for key in ("layer", "head", "dimension"))
    p = torch.zeros((4, 8, 18), dtype=torch.float16)
    p[head // 8, head % 8] = torch.tensor(case["probabilities"])
    for position, value in enumerate(case["values"]):
        logical, slot = divmod(position, 16)
        pool[layer, 1, cache.block_table(request)[logical], slot, head // 8, channel] = value
    before, counts = pool.clone(), cache.capacity()
    with forbid_dense_kv_history(pool):
        actual = evidence.halves_page_sum(p, cache, request, layer)
        rejected = evidence.attention._paired_value_sum(p, cache, request, layer)
    assert actual[0, head, 0, channel].item() == case["hf"]
    assert rejected[0, head, 0, channel].item() != case["hf"]
    torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
    assert cache.capacity() == counts and cache.block_table(request) == (0, 2)
    cache.release_all(request)
    cache.release_all(other)
    assert cache.capacity() == CapacityCounts(4, 4, 0, 0, 0, 0)


def test_capture_preserves_original_layout_metadata_and_raw_reference_operands(tmp_path):
    module = SimpleNamespace(layer_idx=0, num_key_value_groups=8, training=False, scaling=0.125)
    q = torch.randn(1, 32, 1, 64).half()
    k, v = (torch.randn(1, 18, 4, 64).half().transpose(1, 2) for _ in range(2))
    with torch.inference_mode(), evidence.capture_eager(tmp_path) as (captures, records):
        output, _ = llama.eager_attention_forward(module, q, k, v, None, scaling=0.125)
    saved = torch.load(tmp_path / "layer-00.pt", weights_only=True)
    torch.testing.assert_close(saved["v"], v, atol=0, rtol=0)
    torch.testing.assert_close(saved["output"].transpose(1, 2), output, atol=0, rtol=0)
    assert records[0]["layouts"]["v"] == evidence.layout(v)
    assert records[0]["native_replay_exact"] and records[0]["value_family"] == "unknown"
    assert set(records[0]["models"]) == {"alternating", "halves", "striped4", "fp64"}
    replay = evidence.replay_score_layouts(captures[0], tmp_path)
    assert set(replay["comparisons"]) == {
        "page_view",
        "page_packed",
        "full_width_view",
        "full_width_packed",
    }
    assert replay["comparisons"]["full_width_view"]["raw_dot"]["unequal"] == 0


def test_half_sum_can_feed_mlp_without_marking_hidden_states_as_kv(forbid_dense_kv_history):
    pool, cache, request, other = fragmented_cache(18)
    probabilities = torch.full((4, 8, 18), 1 / 18, dtype=torch.float16)
    projection = torch.zeros((5632, 2048), dtype=torch.float16)
    with forbid_dense_kv_history(pool):
        output = evidence.halves_page_sum(probabilities, cache, request, 0)
        hidden = torch.nn.functional.linear(output.reshape(1, 1, 2048), projection)
    assert hidden.shape == (1, 1, 5632) and hidden.count_nonzero() == 0
    cache.release_all(request)
    cache.release_all(other)


def test_large_page_difference_does_not_break_storage_guard(forbid_dense_kv_history):
    pool, cache, request, other = fragmented_cache(18)
    page = next(cache.iter_layer_blocks(request, layer_index=0))[0]
    with forbid_dense_kv_history(pool):
        result = evidence.guarded_measurement(page, torch.ones_like(page))
    assert result["unequal"] == page.numel()
    assert result["first_unequal_index"] == (0, 0, 0)
    cache.release_all(request)
    cache.release_all(other)


@pytest.mark.parametrize("fail", [False, True])
def test_guarded_trace_records_the_actual_row_and_retains_partial_failure(
    tmp_path,
    monkeypatch,
    forbid_dense_kv_history,
    fail,
):
    pool, cache, request, other = fragmented_cache(17)
    q = torch.zeros((1, 32, 1, 64), dtype=torch.float16)
    k = v = torch.zeros((1, 4, 18, 64), dtype=torch.float16)
    p32 = torch.full((1, 32, 1, 18), 1 / 18)
    captured = {
        "q": q,
        "k": k,
        "v": v,
        "scores": torch.zeros_like(p32),
        "probabilities_fp32": p32,
        "probabilities": p32.half(),
        "output": q,
    }

    def decode(tokens, weights, allocator, requests):
        with allocator.append_decode(requests):
            for layer in range(22):
                allocator.write_decode_layer(
                    request, layer, torch.zeros(4, 64).half(), torch.zeros(4, 64).half()
                )
            output = decoder.paged_attention(
                q, allocator, request, layer_index=0, eager_rounding=True
            )
            if fail:
                raise RuntimeError("failure after attention")
            return output

    monkeypatch.setattr(decoder, "decode_paged", decode)
    args = (
        torch.zeros((1, 1), dtype=torch.long),
        None,
        cache,
        (request,),
        {request: {0: captured}},
        {request: [{"value_family": "gemvx_6"}]},
        "family_hypothesis",
        tmp_path,
        forbid_dense_kv_history(pool),
    )
    if fail:
        with pytest.raises(RuntimeError, match="failure after attention"):
            evidence.trace_decode(*args)
        assert cache.cached_length(request) == 17
    else:
        output, rows = evidence.trace_decode(*args)
        assert torch.equal(output, q) and evidence.first_difference(rows[request]) is None
    trace = json.loads((tmp_path / "owned-trace.json").read_text())
    assert trace[request][0]["reducer"] == "halves"
    assert (tmp_path / "owned-attention.pt").is_file()
    trace[request][0]["scores"]["unequal"] = 1
    assert evidence.first_difference(trace[request])["operation"] == "scores"
    trace[request][0]["q"]["unequal"] = 1
    assert evidence.first_difference(trace[request])["operation"] == "q"
    cache.release_all(request)
    cache.release_all(other)


@pytest.mark.parametrize(
    ("length", "family", "expected"),
    [
        (17, "gemvx_5", "alternating"),
        (18, "gemvx_6", "halves"),
        (19, "gemvx_6", "baseline"),
        (95, "gemvx_7", "baseline"),
        (96, "gemv_nsp", "baseline"),
        (2048, "gemvx_8", "baseline"),
    ],
)
def test_hypothesis_dispatch_requires_a_measured_supported_family(length, family, expected):
    assert evidence.choose_reducer("family_hypothesis", length, family)[0] == expected
    assert evidence.choose_reducer("baseline", length, family)[0] == "baseline"


@pytest.mark.parametrize(
    "accuracy,two_token,layer_trace",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize("kaggle_available", [False, True])
def test_notebook_source_identity_and_archive_survive_setup_failure(
    tmp_path, monkeypatch, accuracy, two_token, layer_trace, kaggle_available
):
    source_root = tmp_path / "input"
    source_root.mkdir()
    for name in (
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "results/validation/2026-11-07/value-kernel-sweep-t4.zip",
        "results/validation/2026-11-07/paired-at-17-console.txt",
    ):
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("source fixture\n")
    monkeypatch.setattr(notebook, "ROOT", source_root)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kw: "base-commit" if "rev-parse" in command else " M source\n",
    )
    path = tmp_path / "evidence.ipynb"
    notebook.build(path, accuracy=accuracy, two_token=two_token, layer_trace=layer_trace)
    cells = json.loads(path.read_text())["cells"]
    code = "".join(cells[-1]["source"])
    if accuracy:
        assert "scripts.export_paged_accuracy" in code
        assert "synthetic-controls" not in code
    if two_token:
        assert "scripts.probe_two_token_decode" in code
        assert "gpu-regressions" not in code
        assert "scripts.export_paged_accuracy" not in code
        assert "synthetic-controls" not in code
    if layer_trace:
        assert "scripts.trace_two_token_decode" in code
        assert "scripts.probe_two_token_decode" not in code
        assert "gpu-regressions" not in code
        assert "scripts.export_paged_accuracy" not in code
        assert "synthetic-controls" not in code
    compile(code, str(path), "exec")

    # Only the setup subprocess is simulated; all extraction, hashes and archive writes execute.
    class FailedSetup:
        returncode = 1

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 1

    display_module = ModuleType("IPython.display")
    display_module.FileLink = lambda path: path
    display_module.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.setattr(subprocess, "Popen", FailedSetup)
    output = tmp_path / "output"
    monkeypatch.setenv("PAGEDCORE_EVIDENCE_WORKDIR", str(output))
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: kaggle_available if path == Path("/kaggle/working") else original_exists(path),
    )
    original_mkdtemp = tempfile.mkdtemp

    def isolated_mkdtemp(*args, **kwargs):
        assert Path(kwargs["dir"]) == output
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkdtemp", isolated_mkdtemp)
    monkeypatch.chdir(tmp_path)
    exec(code, {})
    prefix = (
        "pagedcore-layer-trace"
        if layer_trace
        else "pagedcore-two-token"
        if two_token
        else ("pagedcore-accuracy" if accuracy else "pagedcore-evidence")
    )
    archive_path = next(output.glob(f"{prefix}-*.zip"))
    with ZipFile(archive_path) as archive:
        assert "evidence/notebook-failure.txt" in archive.namelist()
        assert json.loads(archive.read("evidence/commands.json"))[0]["returncode"] == 1
        checks = json.loads(archive.read("evidence/checksums.json"))
        for name, digest in checks.items():
            assert hashlib.sha256(archive.read("evidence/" + name)).hexdigest() == digest
        payload = archive.read("source-snapshot.zip")
    with ZipFile(io.BytesIO(payload)) as source:
        identity = json.loads(source.read("source-identity.json"))
        for name, digest in identity["sha256"].items():
            assert hashlib.sha256(source.read(name)).hexdigest() == digest
    assert base64.b64encode(payload).decode() in code
