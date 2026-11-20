from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch

import pagedcore.flashinfer_backend as backend_module
from pagedcore.engine_types import KVReservation, RequestId
from pagedcore.flashinfer_backend import FlashInferAttention
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    wrapper = Mock()
    wrapper.run.return_value = torch.ones((1, 32, 64), dtype=torch.float16)
    # Exercise adapter orchestration on CPU; this is not native CUDA evidence.
    monkeypatch.setattr(
        backend_module, "_load_wrapper", lambda device: (torch.empty(8, dtype=torch.uint8), wrapper)
    )
    return FlashInferAttention(device="cpu"), wrapper


def cache_at_boundary():
    pool = torch.full((22, 2, 4, 16, 4, 64), torch.nan, dtype=torch.float16)
    allocator = KVAllocator(pool)
    request, blocker = RequestId("target"), RequestId("blocker")
    history = torch.randn((22, 16, 4, 64), dtype=torch.float16)
    allocator.create(KVReservation(request, 2))
    allocator.append(request, history, -history)
    allocator.create(KVReservation(blocker, 1))
    allocator.append(blocker, history + 10, history + 20)
    return pool, allocator, request


def test_native_adapter_reads_staged_nonadjacent_page_without_publishing_it(native):
    backend, wrapper = native
    pool, allocator, request = cache_at_boundary()
    counts, before = allocator.capacity(), pool.clone()
    query = torch.randn(1, 32, 1, 64).half()
    with pytest.raises(RuntimeError, match="abort"), allocator.append_decode((request,)):
        with pytest.raises(RuntimeError, match="unwritten"):
            allocator.paged_layer_view(request, layer_index=0)
        key, value = torch.full((4, 64), 3).half(), torch.full((4, 64), 7).half()
        allocator.write_decode_layer(request, 0, key, value)
        actual = backend(query, allocator, request, layer_index=0)
        assert actual.shape == query.shape and torch.all(actual == 1)
        args, options = wrapper.plan.call_args
        assert [item.tolist() for item in args] == [[0, 2], [0, 2], [1]]
        assert all(item.dtype == torch.int32 for item in args)
        assert options == {
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 64,
            "page_size": 16,
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.float16,
            "kv_data_type": torch.float16,
            "sm_scale": 0.125,
        }
        native_query, (native_key, native_value) = wrapper.run.call_args.args
        torch.testing.assert_close(native_query, query.reshape(1, 32, 64), atol=0, rtol=0)
        assert native_query.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
        for view in (native_key, native_value):
            assert view.shape == (4, 16, 4, 64) and view.is_contiguous()
            assert view.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        torch.testing.assert_close(native_key[2, 0], key, atol=0, rtol=0)
        torch.testing.assert_close(native_value[2, 0], value, atol=0, rtol=0)
        assert allocator.block_table(request) == (0,) and allocator.cached_length(request) == 16
        assert allocator.capacity() == counts
        raise RuntimeError("abort")
    assert allocator.block_table(request) == (0,) and allocator.cached_length(request) == 16
    assert allocator.capacity() == counts
    torch.testing.assert_close(pool[:, :, :2], before[:, :, :2], atol=0, rtol=0, equal_nan=True)
    view = allocator.paged_layer_view(request, layer_index=0)
    assert view.blocks == (0,) and view.length == view.last_page_tokens == 16
    backend(query, allocator, request, layer_index=0)
    assert [item.tolist() for item in wrapper.plan.call_args.args] == [[0, 1], [0], [16]]


def test_native_adapter_propagates_kernel_error_and_rejects_invalid_output(native):
    backend, wrapper = native
    _, allocator, request = cache_at_boundary()
    query = torch.zeros((1, 32, 1, 64), dtype=torch.float16)
    wrapper.run.side_effect = RuntimeError("native kernel failure")
    with pytest.raises(RuntimeError, match="native kernel failure"):
        backend(query, allocator, request, layer_index=0)
    wrapper.run.side_effect = None
    wrapper.run.return_value = torch.zeros((1, 32, 64), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="invalid decode output"):
        backend(query, allocator, request, layer_index=0)


def test_native_adapter_rejects_legacy_rounding_and_wrong_worker(native):
    backend, wrapper = native
    _, allocator, request = cache_at_boundary()
    query = torch.zeros((1, 32, 1, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="eager rounding"):
        backend(query, allocator, request, layer_index=0, eager_rounding=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend, query, allocator, request, layer_index=0)
        with pytest.raises(RuntimeError, match="owning worker"):
            future.result()
    wrapper.plan.assert_not_called()


def test_native_backend_cannot_silently_fall_back_on_cpu():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        FlashInferAttention(device="cpu")


@pytest.mark.parametrize("fault", ["architecture", "torch", "missing", "version"])
def test_native_backend_rejects_unvalidated_environment_before_allocation(monkeypatch, fault):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: (8, 0) if fault == "architecture" else (7, 5),
    )
    monkeypatch.setattr(torch, "__version__", "2.8.0" if fault == "torch" else "2.7.1+cu126")

    def installed(name):
        if fault == "missing":
            raise backend_module.PackageNotFoundError(name)
        return "0.7.4"

    monkeypatch.setattr(backend_module, "version", installed)
    allocation = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(torch, "empty", allocation)
    message = {
        "architecture": "SM75",
        "torch": "frozen torch",
        "missing": "install flashinfer",
        "version": "expected flashinfer",
    }[fault]
    with pytest.raises(RuntimeError, match=message):
        FlashInferAttention()
    allocation.assert_not_called()


def test_borrowed_layer_rejects_empty_invalid_and_unwritten_layers():
    pool = torch.zeros((22, 2, 2, 16, 4, 64), dtype=torch.float16)
    allocator, request = KVAllocator(pool), RequestId("empty")
    allocator.create(KVReservation(request, 1))
    with pytest.raises(ValueError, match="nonempty"):
        allocator.paged_layer_view(request, layer_index=0)
    for index in (-1, 22, True):
        with pytest.raises(ValueError, match="layer_index"):
            allocator.paged_layer_view(request, layer_index=index)
    history = torch.zeros((22, 1, 4, 64), dtype=torch.float16)
    allocator.append(request, history, history)
    with (
        pytest.raises(RuntimeError, match="every cache layer"),
        allocator.append_decode((request,)),
    ):
        allocator.write_decode_layer(request, 0, history[0, 0], history[0, 0])
        assert allocator.paged_layer_view(request, layer_index=0).length == 2
        with pytest.raises(RuntimeError, match="unwritten"):
            allocator.paged_layer_view(request, layer_index=1)
