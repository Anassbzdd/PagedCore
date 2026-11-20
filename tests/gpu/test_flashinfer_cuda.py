"""Opt-in native kernel evidence; missing dependencies are errors, not skips."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = [pytest.mark.gpu, pytest.mark.flashinfer]


def independent_attention(query, key, value):
    # Independent CPU/full-history oracle, used only outside decode/its guard.
    query, key, value = (tensor.cpu().double() for tensor in (query, key, value))
    heads = []
    for head in range(32):
        scores = (query[0, head, 0] * key[:, head // 8]).sum(-1) / 8
        probabilities = (scores - scores.max()).exp()
        probabilities /= probabilities.sum()
        heads.append((probabilities[:, None] * value[:, head // 8]).sum(0))
    return torch.stack(heads).reshape(1, 32, 1, 64)


@pytest.mark.parametrize(
    "length,scale", [(1, 1), (15, 1), (16, 1), (17, 1), (49, 1), (2048, 1), (49, 32)]
)
def test_flashinfer_reads_fragmented_pages_against_fp64(
    decode_attention_backend, forbid_dense_kv_history, length, scale
):
    backend = decode_attention_backend
    assert backend is not None
    pages = (length + 15) // 16
    pool = torch.full(
        (22, 2, 2 * pages + 1, 16, 4, 64), torch.nan, dtype=torch.float16, device="cuda:0"
    )
    allocator = KVAllocator(pool)
    target, blocker = RequestId("target"), RequestId("unrelated")
    query = (torch.randn((1, 32, 1, 64), device=pool.device) * scale).half()
    key = (torch.randn((length, 4, 64), device=pool.device) * scale).half()
    value = torch.randn_like(key)
    allocator.create(KVReservation(target, pages))
    allocator.create(KVReservation(blocker, pages))
    sentinel = torch.full((22, 16, 4, 64), 100, device=pool.device, dtype=pool.dtype)
    for start in range(0, length, 16):
        allocator.append(
            target,
            key[None, start : start + 16].expand(22, -1, -1, -1),
            value[None, start : start + 16].expand(22, -1, -1, -1),
        )
        allocator.append(blocker, sentinel, -sentinel)
    assert allocator.block_table(target) == tuple(range(0, 2 * pages, 2))
    expected = independent_attention(query, key, value)
    before, counts, pointer = pool.clone(), allocator.capacity(), pool.data_ptr()
    try:
        with forbid_dense_kv_history(pool):
            actual = backend(query, allocator, target, layer_index=0)
        torch.cuda.synchronize()
        assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(pool, before, atol=0, rtol=0, equal_nan=True)
        assert pool.data_ptr() == pointer and allocator.capacity() == counts
    finally:
        allocator.release_all(target)
        allocator.release_all(blocker)
        assert allocator.capacity() == CapacityCounts(2 * pages + 1, 2 * pages + 1, 0, 0, 0, 0)


def test_flashinfer_staged_page_and_worker_stream_ownership(
    decode_attention_backend, forbid_dense_kv_history
):
    backend = decode_attention_backend
    assert backend is not None
    pool = torch.full((22, 2, 3, 16, 4, 64), torch.nan, dtype=torch.float16, device="cuda:0")
    allocator, target = KVAllocator(pool), RequestId("staged")
    allocator.create(KVReservation(target, 2))
    key = torch.randn((22, 17, 4, 64), dtype=torch.float16, device=pool.device)
    value = torch.randn_like(key)
    query = torch.randn((1, 32, 1, 64), dtype=torch.float16, device=pool.device)
    allocator.append(target, key[:, :16], value[:, :16])
    before, counts = pool.clone(), allocator.capacity()
    expected = independent_attention(query, key[0], value[0])
    with pytest.raises(RuntimeError, match="injected"), allocator.append_decode((target,)):
        allocator.write_decode_layer(target, 0, key[0, 16], value[0, 16])
        with forbid_dense_kv_history(pool):
            actual = backend(query, allocator, target, layer_index=0)
        torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-3, rtol=2e-3)
        assert allocator.cached_length(target) == 16
        raise RuntimeError("injected rollback")
    assert allocator.capacity() == counts and allocator.block_table(target) == (0,)
    torch.testing.assert_close(pool[:, :, 0], before[:, :, 0], atol=0, rtol=0)
    # Queue stream misuse is rejected before handing any pages to the kernel.
    with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="owning worker"):
        backend(query, allocator, target, layer_index=0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend, query, allocator, target, layer_index=0)
        with pytest.raises(RuntimeError, match="owning worker"):
            future.result()
    # Retry overwrites unpublished data and commits only after all layers succeed.
    with allocator.append_decode((target,)):
        for layer in range(22):
            allocator.write_decode_layer(target, layer, key[layer, 16], value[layer, 16])
            backend(query, allocator, target, layer_index=layer)
    torch.cuda.synchronize()
    assert allocator.cached_length(target) == 17
    allocator.release_all(target)
    assert allocator.capacity() == CapacityCounts(3, 3, 0, 0, 0, 0)
