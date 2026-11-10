import runpy

import pytest
import torch
from scripts import trace_two_token_decode as trace

from pagedcore import decoder
from pagedcore.weights import LayerWeights, TinyLlamaWeights

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def recorded_trace(tmp_path_factory):
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
    report = trace.trace_case(weights, guard, directory)
    return directory, report


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
    directory, report = recorded_trace
    assert "error" not in report, report.get("error")
    assert report["complete"] and all(report["instrumentation_controls"].values())
    assert len(report["comparisons"]) == 22 * len(trace.STAGES)
    assert report["kv_measurement"]["status"] == "complete"
    assert report["kv_measurement"]["measured_pairs"] == 44
    assert report["cleanup"]["owned_blocks"] == report["cleanup"]["reserved_blocks"] == 0
    reads = [r for r in report["comparisons"] if r["operation"] in ("k_read", "v_read")]
    assert len(reads) == 44
    for row in reads:
        assert row["paged_layout"]["shape"] == [2, 4, 64]
        assert row["reference_layout"]["shape"] == [1, 4, 2, 64]
        assert row["pool_read"]["shares_pool_storage"]
        assert row["pool_read"]["appended_position"] == 1
        assert row["appended_matches_local_projection"]
        assert "history" in row and "appended" in row
    saved = torch.load(directory / "raw/layer-00.pt", weights_only=True)
    assert any(e.get("operands") for e in saved["paged_attention_math"])
    if report["first_difference"]["operation"] == "query_key_dot":
        investigation = report["first_operation_investigation"]
        assert investigation["native_replays_exact"]
        assert all(investigation["operand_controls"].values())


def test_first_exact_difference_is_reported_inside_tolerance(recorded_trace):
    directory, _ = recorded_trace
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
    directory, _ = recorded_trace
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
