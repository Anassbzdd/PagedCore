import json
import runpy
import shutil
from unittest.mock import patch

import pytest
import torch
from scripts import probe_two_token_decode as probe
from scripts import trace_two_token_decode as trace
from scripts.intervene_decode_scalar import replace_attention_scalar, run_control
from scripts.replay_one_page import error_against_fp64, fp64_attention, replay_layer, round_to_fp16

from pagedcore import decoder
from pagedcore.weights import LayerWeights, TinyLlamaWeights

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", params=[1, 15])
def recorded_trace(tmp_path_factory, request):
    def matrix(rows, columns):
        return (torch.randn(rows, columns) * 0.01).half()

    layer = LayerWeights(
        torch.ones(2048).half(),
        matrix(2048, 2048),
        matrix(256, 2048),
        matrix(256, 2048),
        matrix(2048, 2048),
        torch.ones(2048).half(),
        matrix(8, 2048),
        matrix(8, 2048),
        matrix(2048, 8),
    )
    weights = TinyLlamaWeights(
        matrix(31, 2048), (layer,) * 22, torch.ones(2048).half(), matrix(31, 2048)
    )
    guard = runpy.run_path(str(trace.ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
    directory = tmp_path_factory.mktemp("actual-layer-trace")
    fixture = directory / "fixture/tests/fixtures/reference_prompts.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(json.dumps({"prompts": [{"name": "maximum_output", "input_ids": [1] * 15}]}))
    with patch.object(probe, "ROOT", directory / "fixture"):
        report = trace.trace_case(weights, guard, directory, prompt_tokens=request.param)
    return directory, report, weights


def load_records(directory):
    reference, paged = trace.ForwardTrace(), trace.ForwardTrace()
    for index in range(22):
        saved = torch.load(directory / "raw" / f"layer-{index:02}.pt", weights_only=True)
        reference.layers[index], paged.layers[index] = saved["reference"], saved["paged"]
        reference.math[index], paged.math[index] = (
            saved["reference_attention_math"],
            saved["paged_attention_math"],
        )
    return reference, paged


def test_trace_observes_live_staged_reads_and_preserves_actual_forward(recorded_trace):
    directory, report, _ = recorded_trace
    assert "error" not in report, report.get("error")
    assert report["complete"] and all(report["instrumentation_controls"].values())
    length = report["attended_tokens"]
    baseline = torch.load(directory / "baseline/target.pt", weights_only=True)
    assert len(baseline["prompt_ids"]) == length - 1
    assert report["baseline"]["attended_tokens"] == length
    assert report["old_history_unchanged"] and report["absolute_positions_correct"]
    assert len(report["comparisons"]) == 22 * len(trace.STAGES)
    assert report["kv_measurement"]["status"] == "complete"
    assert report["kv_measurement"]["measured_pairs"] == 44
    assert report["cleanup"]["owned_blocks"] == report["cleanup"]["reserved_blocks"] == 0
    reads = [r for r in report["comparisons"] if r["operation"] in ("k_read", "v_read")]
    assert len(reads) == 44
    for row in reads:
        assert row["paged_layout"]["shape"] == [length, 4, 64]
        assert row["reference_layout"]["shape"] == [1, 4, length, 64]
        assert row["pool_read"]["shares_pool_storage"]
        assert row["pool_read"]["appended_position"] == length - 1
        assert row["pool_read"]["history_positions"] == list(range(length - 1))
        assert row["appended_matches_local_projection"]
        assert "history" in row and "appended" in row
    saved = torch.load(directory / "raw/layer-00.pt", weights_only=True)
    assert any(e.get("operands") for e in saved["paged_attention_math"])
    if report["first_difference"]["operation"] == "query_key_dot":
        investigation = report["first_operation_investigation"]
        assert investigation["native_replays_exact"]
        assert all(investigation["operand_controls"].values())


def test_first_exact_difference_is_reported_inside_tolerance(recorded_trace):
    directory, _, _ = recorded_trace
    reference, paged = load_records(directory)
    value = paged.layers[0]["input_norm"]["value"]
    first = value.view(-1)[0]
    first.copy_(torch.nextafter(first, torch.full_like(first, float("inf"))))
    result = trace.compare_traces(reference, paged)
    first = result["first_difference"]
    assert first["layer"] == 0 and first["operation"] == "input_norm"
    assert 0 < first["max_tolerance_ratio"] < 1


def test_incomplete_kv_measurements_are_explicit(recorded_trace):
    empty = trace.compare_traces(trace.ForwardTrace(), trace.ForwardTrace())
    assert not empty["complete"]
    assert empty["kv_measurement"]["status"] == "not_measured"
    assert empty["kv_measurement"]["max_kv_tolerance_ratio"] is None
    directory, _, _ = recorded_trace
    reference, paged = load_records(directory)
    del paged.layers[21]["v_read"]
    partial = trace.compare_traces(reference, paged)
    assert not partial["complete"]
    assert partial["kv_measurement"]["status"] == "partial"
    assert partial["kv_measurement"]["measured_pairs"] == 43


def test_instrumentation_is_restored_after_exception():
    original = decoder.rms_norm
    with (
        pytest.raises(RuntimeError, match="injected trace failure"),
        trace.capture_forward(trace.ForwardTrace()),
    ):
        raise RuntimeError("injected trace failure")
    assert decoder.rms_norm is original


def test_operand_replay_restores_stride_offset_and_backing_storage():
    backing = torch.arange(256, dtype=torch.float32)
    value = backing.as_strided((2, 3), (8, 2), 7)
    actual = trace.restore_operand(trace.snapshot(value), "cpu")
    assert torch.equal(actual, value)
    assert actual.stride() == value.stride()
    assert actual.storage_offset() == value.storage_offset()
    assert actual.untyped_storage().nbytes() == value.untyped_storage().nbytes()


def test_shared_input_replay_reproduces_recorded_operations(recorded_trace):
    directory, _, _ = recorded_trace
    report, tensors = replay_layer(directory / "raw/layer-00.pt", "cpu")
    assert tensors
    for case in report["common_inputs"].values():
        assert all(case["native_replays_exact"].values())
        assert case["common_input_controls"]["pool_k_exact"]
        assert case["common_input_controls"]["pool_v_exact"]
        assert case["stages"]["precast_attention"]["paged_vs_fp64"]["attention_tolerance_ratio"] < 1


def test_independent_fp64_reference_preserves_grouped_head_mapping():
    query = torch.zeros((1, 32, 1, 64), dtype=torch.float16)
    key = torch.zeros((1, 4, 16, 64), dtype=torch.float16)
    value = torch.arange(4, dtype=torch.float16)[None, :, None, None].expand(1, 4, 16, 64)
    actual = fp64_attention(query, key, value)
    expected = torch.arange(4).repeat_interleave(8).double()[None, :, None, None].expand_as(query)
    assert torch.equal(actual["precast_attention"], expected)
    assert torch.equal(
        actual["probabilities"], torch.full((1, 32, 1, 16), 1 / 16, dtype=torch.float64)
    )


def test_fp64_rounding_preserves_values_on_both_sides_of_an_fp16_midpoint():
    lower, upper = -0.0028858184814453125, -0.0028839111328125
    midpoint = (lower + upper) / 2
    inputs = torch.tensor([midpoint - 1e-10, midpoint, midpoint + 1e-10], dtype=torch.float64)
    expected = torch.tensor([lower, upper, upper], dtype=torch.float16)
    assert torch.equal(round_to_fp16(inputs), expected)
    assert inputs[0].float().item() == midpoint


def test_fp64_mismatch_metric_uses_direct_rounding_at_the_captured_boundary():
    exact = torch.tensor([-0.002884864910819074], dtype=torch.float64)
    paged = torch.tensor([-0.0028848648071289062], dtype=torch.float32)
    reference = torch.tensor([-0.00288486503995955], dtype=torch.float32)
    assert error_against_fp64(paged, exact)["fp16_mismatches"] == 1
    assert error_against_fp64(reference, exact)["fp16_mismatches"] == 0


def test_normalize_first_stage_is_accumulated_values_instead_of_probabilities():
    from types import SimpleNamespace

    events = [
        {"operation": "aten._softmax.default", "output": {"value": torch.zeros(4, 8, 16)}},
        {"operation": "aten.div.Tensor", "output": {"value": torch.ones(4, 8, 16)}},
        {"operation": "aten.add_.Tensor", "output": {"value": torch.full((4, 8, 64), 2.0)}},
    ]
    stages = trace.attention_stages(SimpleNamespace(math=[events]), 0, paged=True)
    assert stages["precast_attention"]["value"].shape == (4, 8, 64)
    assert stages["probabilities"]["value"].shape == (4, 8, 16)


def test_single_scalar_control_preserves_other_values_and_restores_the_function(monkeypatch):
    expected = torch.arange(2048, dtype=torch.float16).reshape(1, 32, 1, 64)

    def native(*args, **kwargs):
        return expected.clone()

    monkeypatch.setattr(decoder, "paged_attention", native)
    coordinate = (0, 21, 0, 60)
    with replace_attention_scalar(1, coordinate, expected, -0.5) as observation:
        actual = decoder.paged_attention(layer_index=1)
        assert observation["calls"] == observation["changed_elements"] == 1
        assert actual[coordinate] == -0.5
        assert observation["other_elements_exact"]
    assert decoder.paged_attention is native


def test_scalar_control_rejects_wrong_captured_output_and_restores_after_failure(monkeypatch):
    expected = torch.zeros((1, 32, 1, 64), dtype=torch.float16)

    def native(*args, **kwargs):
        return expected + 1

    monkeypatch.setattr(decoder, "paged_attention", native)
    with (
        pytest.raises(AssertionError, match="retained trace"),
        replace_attention_scalar(1, (0, 21, 0, 60), expected, 0),
    ):
        decoder.paged_attention(layer_index=1)
    assert decoder.paged_attention is native


def test_real_decoder_scalar_control_proves_sham_and_prefix_controls(recorded_trace):
    directory, captured, weights = recorded_trace
    first = captured["first_fp16_difference"]
    assert first is not None and first["operation"] == "attention_output"
    trace_dir = directory / "control-source"
    trace_dir.mkdir()
    shutil.copyfile(directory / "trace.json", trace_dir / "trace.json")
    layer = first["layer"]
    shutil.copyfile(directory / f"raw/layer-{layer:02}.pt", trace_dir / f"layer-{layer:02}.pt")
    guard = runpy.run_path(str(trace.ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
    result = run_control(weights, guard, directory / "baseline", trace_dir, directory / "control")
    assert result["valid_control"], result
    assert all(result["controls"].values())
    assert result["arms"]["sham"]["intervention"]["changed_elements"] == 0
    assert result["arms"]["single_scalar"]["intervention"]["changed_elements"] == 1
    assert result["prefix_stages_measured"] > 0
    for arm in result["arms"].values():
        assert arm["trace"]["complete"]
        assert arm["appended_writes_match_local_projections"]
        assert arm["cleanup"]["owned_blocks"] == arm["cleanup"]["reserved_blocks"] == 0
