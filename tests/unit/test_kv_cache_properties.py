from dataclasses import dataclass, field
from math import ceil
from unittest.mock import patch

import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator

pytestmark = pytest.mark.unit


@dataclass
class _ExpectedCache:
    credits: int
    tokens: list[int] = field(default_factory=list)
    blocks: tuple[int, ...] = ()


def _tagged_kv(tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.tensor(tokens, dtype=torch.float16)[None, :, None, None].expand(22, -1, 4, 64)
    return key, -key - 1


class _AllocatorStateMachine(RuleBasedStateMachine):
    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.pool = torch.full((22, 2, blocks, 16, 4, 64), torch.nan, dtype=torch.float16)
        self.pointer = self.pool.untyped_storage().data_ptr()
        self.allocator = KVAllocator(self.pool)
        self.total = blocks
        self.free = set(range(blocks))
        self.requests: dict[RequestId, _ExpectedCache] = {}
        self.next_token = 0

    @rule(slot=st.integers(0, 3), credits=st.integers(1, 9))
    def create(self, slot: int, credits: int) -> None:
        request = RequestId(str(slot))
        reservation = KVReservation(request, credits)
        if request in self.requests:
            with pytest.raises(ValueError, match="already has a KV cache"):
                self.allocator.create(reservation)
        elif credits > self.total:
            with pytest.raises(ValueError, match="total KV pool capacity"):
                self.allocator.create(reservation)
        elif credits + sum(cache.credits for cache in self.requests.values()) > self.total:
            with pytest.raises(RuntimeError, match="insufficient unreserved blocks"):
                self.allocator.create(reservation)
        else:
            self.allocator.create(reservation)
            self.requests[request] = _ExpectedCache(credits)

    @precondition(lambda self: bool(self.requests))
    @rule(
        data=st.data(),
        tokens=st.sampled_from((1, 2, 15, 16, 17, 31, 32, 33)),
        fail_at=st.one_of(st.just(0), st.integers(1, 6)),
    )
    def append(self, data: st.DataObject, tokens: int, fail_at: int) -> None:
        request = data.draw(st.sampled_from(tuple(self.requests)), label="request")
        expected = self.requests[request]
        tags = list(range(self.next_token, self.next_token + tokens))
        self.next_token += tokens
        key, value = _tagged_kv(tags)
        length = len(expected.tokens) + tokens
        if length > expected.credits * 16:
            with pytest.raises(RuntimeError, match="exceeds request reservation"):
                self.allocator.append(request, key, value)
            return

        if fail_at:
            copies = 2 * len({position // 16 for position in range(len(expected.tokens), length)})
            fail_at = min(fail_at, copies)
            copy = torch.Tensor.copy_
            calls = 0

            def failing_copy(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
                nonlocal calls
                calls += 1
                if calls == fail_at:
                    raise RuntimeError("injected copy failure")
                return copy(destination, source)

            with (
                patch.object(torch.Tensor, "copy_", failing_copy),
                pytest.raises(RuntimeError, match="injected copy failure"),
            ):
                self.allocator.append(request, key, value)
            assert calls == fail_at
            return

        self.allocator.append(request, key, value)
        table = self.allocator.block_table(request)
        assert len(table) == ceil(length / 16)
        assert table[: len(expected.blocks)] == expected.blocks
        added = table[len(expected.blocks) :]
        assert len(set(added)) == len(added)
        assert set(added) <= self.free
        # Record the chosen free blocks without prescribing a free-list ordering.
        self.free.difference_update(added)
        expected.blocks = table
        expected.tokens.extend(tags)

    @rule(slot=st.integers(0, 3))
    def release(self, slot: int) -> None:
        request = RequestId(str(slot))
        if request not in self.requests:
            with pytest.raises(KeyError):
                self.allocator.release_all(request)
        else:
            self.allocator.release_all(request)
            self.free.update(self.requests.pop(request).blocks)

    @invariant()
    def matches_reference(self) -> None:
        owned = sum(ceil(len(cache.tokens) / 16) for cache in self.requests.values())
        reserved = sum(cache.credits for cache in self.requests.values())
        live = sum(len(cache.tokens) for cache in self.requests.values())
        assert self.allocator.capacity() == CapacityCounts(
            self.total, self.total - owned, owned, reserved, len(self.requests), live
        )
        assert set(self.allocator._sequences) == set(self.requests)
        assert len(self.allocator._free_blocks) == len(self.free)
        assert set(self.allocator._free_blocks) == self.free
        owners = {
            block: request for request, cache in self.requests.items() for block in cache.blocks
        }
        assert len(owners) == owned
        assert set(owners).isdisjoint(self.free)
        assert set(owners) | self.free == set(range(self.total))
        assert self.allocator._block_owners == owners
        assert self.pool.untyped_storage().data_ptr() == self.pointer

        for request, cache in self.requests.items():
            assert self.allocator._sequences[request].reservation == KVReservation(
                request, cache.credits
            )
            assert self.allocator.cached_length(request) == len(cache.tokens)
            assert self.allocator.block_table(request) == cache.blocks
            assert len(cache.blocks) == ceil(len(cache.tokens) / 16)
            for logical, physical in enumerate(cache.blocks):
                tags = cache.tokens[logical * 16 : (logical + 1) * 16]
                key, value = _tagged_kv(tags)
                actual_key, actual_value = self.allocator.read_block(request, logical)
                torch.testing.assert_close(actual_key, key, atol=0, rtol=0)
                torch.testing.assert_close(actual_value, value, atol=0, rtol=0)
                torch.testing.assert_close(
                    self.pool[:, 0, physical, : len(tags)], key, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    self.pool[:, 1, physical, : len(tags)], value, atol=0, rtol=0
                )

    def teardown(self) -> None:
        for request in list(self.requests):
            self.allocator.release_all(request)
            self.free.update(self.requests.pop(request).blocks)
            self.matches_reference()
        assert self.allocator.capacity() == CapacityCounts(self.total, self.total, 0, 0, 0, 0)


@pytest.mark.parametrize("blocks", [1, 2, 4, 8])
def test_allocator_randomized_operations_match_independent_model(blocks: int) -> None:
    run_state_machine_as_test(
        lambda: _AllocatorStateMachine(blocks),
        settings=settings(
            max_examples=50,
            stateful_step_count=40,
            deadline=None,
            derandomize=True,
            database=None,
        ),
    )
