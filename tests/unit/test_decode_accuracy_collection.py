import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pagedcore.accuracy import ACCURACY_POLICY, accuracy_tolerances, valid_accuracy_report

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("lengths", "case"),
    [((15,), "single"), ((37, 73, 111, 145), "held_out")],
)
def test_numeric_failure_retains_all_rows_kv_and_cleanup(
    tmp_path, monkeypatch, forbid_dense_kv_history, lengths, case
):
    path = Path(__file__).parents[1] / "gpu/test_decoder_parity.py"
    spec = importlib.util.spec_from_file_location("decode_collection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allocators = []
    prefilled = []
    real_allocator = module.KVAllocator

    def allocator(pool):
        result = real_allocator(pool)
        allocators.append(result)
        return result

    def cache(length):
        return SimpleNamespace(
            length=length,
            layers=tuple((torch.zeros(1, 4, length, 64).half(),) * 2 for _ in range(22)),
        )

    def prefill(inputs, weights):
        return torch.tensor([[[0.0, 1.0]]]).half(), cache(inputs.shape[1])

    def prefill_paged(inputs, weights, allocator, request):
        prefilled.append(inputs[0].tolist())
        values = torch.zeros(22, inputs.shape[1], 4, 64).half()
        allocator.append(request, values, values)
        return prefill(inputs, weights)[0]

    def decode(tokens, previous, weights, *, fp32_attention):
        return torch.tensor([[[0.0, 1.0]]]).half(), cache(previous.length + 1)

    def decode_paged(tokens, weights, allocator, requests, *, attention_backend):
        for request in requests:
            values = torch.zeros(22, 1, 4, 64).half()
            allocator.append(request, values, values)
        # This exceeds the revised logit budget while keeping the same greedy choice.
        return torch.tensor([[[0.03, 1.0]]]).half().repeat(len(requests), 1, 1)

    monkeypatch.setattr(module, "KVAllocator", allocator)
    monkeypatch.setattr(module.decoder_module, "prefill", prefill)
    monkeypatch.setattr(module.decoder_module, "decode", decode)
    monkeypatch.setattr(module, "prefill_paged", prefill_paged)
    monkeypatch.setattr(module, "decode_paged", decode_paged)
    monkeypatch.setenv("PAGEDCORE_ACCURACY_DIR", str(tmp_path))
    report = {
        "policy": ACCURACY_POLICY,
        "tolerances": accuracy_tolerances(),
        "case": case,
        "lengths": list(lengths),
        "steps": 3,
        "comparisons": [],
    }
    weights = SimpleNamespace(embedding=torch.empty(1), lm_head=torch.empty(2, 1))
    module._assert_pinned_paged_decode(
        (None, weights), lengths, forbid_dense_kv_history, accuracy=report
    )
    rows = len(lengths) * 3
    assert [len(ids) for ids in prefilled] == list(lengths)
    if case == "held_out":
        base = next(
            c["input_ids"] for c in module.FIXTURES["prompts"] if c["name"] == "maximum_output"
        )
        assert prefilled[-1][:49] == base
        assert prefilled[-1][49:97] == base[1:]
    assert len(report["comparisons"]) == rows
    assert [row["attended_tokens"] for row in report["comparisons"][: len(lengths)]] == [
        length + 1 for length in lengths
    ]
    assert all(row["kv_measurement_complete"] for row in report["comparisons"])
    assert all(row["max_kv_tolerance_ratio"] == 0 for row in report["comparisons"])
    assert not valid_accuracy_report(report, lengths)
    assert len(list((tmp_path / case).glob("step-*.pt"))) == rows
    kv_paths = list((tmp_path / case).glob("kv-step-*.pt"))
    assert len(kv_paths) == rows
    assert torch.load(kv_paths[0], weights_only=True)[0]["actual"].dtype == torch.float16
    assert allocators[0].capacity().active_slots == 0
    assert allocators[0].capacity().owned_blocks == 0
