from dataclasses import FrozenInstanceError, fields, replace
from itertools import pairwise
from pathlib import Path
from typing import assert_type

import pytest

from pagedcore.engine_types import (
    BlockId,
    BlockTable,
    CapacityCounts,
    EngineError,
    ErrorCode,
    FinishReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationFailed,
    GenerationTerminal,
    KVReservation,
    MonotonicTimestamp,
    RequestCounts,
    RequestId,
    RequestState,
    RequestTiming,
    TokenEvent,
    TransportStatus,
    TransportTerminal,
    validate_request_transition,
)


def test_generation_completion_survives_later_delivery_failure() -> None:
    request_id = RequestId("request-1")
    generation = GenerationCompleted(request_id, MonotonicTimestamp(100), 3, FinishReason.LENGTH)
    transport = TransportTerminal(
        request_id,
        MonotonicTimestamp(120),
        TransportStatus.FAILED,
        EngineError(ErrorCode.SLOW_CONSUMER, "Stream write deadline exceeded."),
    )

    assert generation.state is RequestState.COMPLETED
    assert generation.finish_reason is FinishReason.LENGTH
    assert transport.status is TransportStatus.FAILED
    with pytest.raises(FrozenInstanceError):
        generation.state = RequestState.CANCELLED  # type: ignore[misc,assignment]


def test_immediate_eos_has_no_token_timestamps() -> None:
    terminal = GenerationCompleted(RequestId("eos"), MonotonicTimestamp(20), 0, FinishReason.EOS)
    timing = RequestTiming(
        arrival=MonotonicTimestamp(0),
        admission=MonotonicTimestamp(10),
        generation_terminal=terminal.terminal_at,
    )

    assert terminal.output_tokens == 0
    assert timing.arrival == 0
    assert timing.first_token is None
    assert timing.last_token is None
    assert timing.transport_terminal is None


def test_disconnect_can_precede_generation_terminal() -> None:
    timing = RequestTiming(
        arrival=MonotonicTimestamp(0),
        transport_terminal=MonotonicTimestamp(10),
        generation_terminal=MonotonicTimestamp(20),
    )
    terminal = GenerationCancelled(
        RequestId("pending"), MonotonicTimestamp(20), 0, ErrorCode.CLIENT_DISCONNECTED
    )

    assert timing.admission is None
    assert timing.transport_terminal == 10
    assert terminal.reason is ErrorCode.CLIENT_DISCONNECTED
    assert terminal.state is RequestState.CANCELLED


@pytest.mark.parametrize(
    "terminal",
    [
        GenerationCompleted(RequestId("done"), MonotonicTimestamp(1), 1, FinishReason.LENGTH),
        GenerationCancelled(
            RequestId("cancel"), MonotonicTimestamp(1), 1, ErrorCode.SERVER_SHUTDOWN
        ),
        GenerationFailed(
            RequestId("fail"),
            MonotonicTimestamp(1),
            1,
            EngineError(ErrorCode.WORKER_FAILURE, "Generation failed."),
        ),
    ],
)
def test_terminal_variants_have_distinct_payloads(terminal: GenerationTerminal) -> None:
    if isinstance(terminal, GenerationCompleted):
        assert terminal.state is RequestState.COMPLETED
        assert not hasattr(terminal, "error")
        assert not hasattr(terminal, "reason")
    elif isinstance(terminal, GenerationCancelled):
        assert terminal.state is RequestState.CANCELLED
        assert not hasattr(terminal, "finish_reason")
    else:
        assert terminal.state is RequestState.FAILED
        assert terminal.error.code is ErrorCode.WORKER_FAILURE
        assert not hasattr(terminal, "finish_reason")


def test_reservation_is_independent_of_physical_assignment() -> None:
    reservation = KVReservation(RequestId("request-1"), maximum_block_credits=3)
    table: BlockTable = (BlockId(42), BlockId(7))
    counts = RequestCounts(
        prompt_tokens=17, max_new_tokens=32, output_tokens=1, cached_token_slots=17
    )
    capacity = CapacityCounts(
        10, 8, 2, reservation.maximum_block_credits, 1, counts.cached_token_slots
    )

    assert table[16 // 16] == 7
    assert reservation.maximum_block_credits > len(table)
    assert capacity.free_blocks + capacity.owned_blocks == capacity.total_blocks
    assert counts.cached_token_slots == counts.prompt_tokens


def test_measurement_schemas_do_not_retain_token_payloads_or_histories() -> None:
    forbidden = {"prompt", "text_delta", "token_id", "token_ids", "token_times"}
    for record_type in (RequestTiming, RequestCounts, CapacityCounts):
        assert forbidden.isdisjoint(field.name for field in fields(record_type))

    event = TokenEvent(RequestId("request-1"), 0, 123, MonotonicTimestamp(10))
    assert_type(event.request_id, RequestId)
    assert_type(event.published_at, MonotonicTimestamp)
    with pytest.raises(FrozenInstanceError):
        event.index = 1  # type: ignore[misc]


def test_error_codes_match_public_contract() -> None:
    constants = Path(__file__).resolve().parents[2] / "docs" / "CONSTANTS.md"
    documented = constants.read_text(encoding="utf-8").split("## Stable error codes", 1)[1]
    codes = {
        line.split("|")[-2].strip().strip("`")
        for line in documented.splitlines()
        if line.startswith("|") and "`" in line
    }

    assert {code.value for code in ErrorCode} == codes
    assert {reason.value for reason in FinishReason} == {"eos", "length"}


@pytest.mark.parametrize("decode", [False, True])
def test_successful_lifecycle_including_completion_during_prefill(decode: bool) -> None:
    states = [
        RequestState.RECEIVED,
        RequestState.VALIDATED,
        RequestState.PENDING,
        RequestState.ADMITTED,
        RequestState.PREFILLING,
    ]
    if decode:
        states.append(RequestState.DECODING)
    states.append(RequestState.COMPLETED)

    for current, next_state in pairwise(states):
        validate_request_transition(current, next_state)


@pytest.mark.parametrize(
    "current",
    [
        RequestState.RECEIVED,
        RequestState.VALIDATED,
        RequestState.PENDING,
        RequestState.ADMITTED,
        RequestState.PREFILLING,
        RequestState.DECODING,
    ],
)
@pytest.mark.parametrize("terminal", [RequestState.CANCELLED, RequestState.FAILED])
def test_cancellation_and_failure_at_every_nonterminal_state(
    current: RequestState, terminal: RequestState
) -> None:
    validate_request_transition(current, terminal)


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (RequestState.RECEIVED, RequestState.PENDING),
        (RequestState.VALIDATED, RequestState.ADMITTED),
        (RequestState.PENDING, RequestState.PREFILLING),
        (RequestState.PENDING, RequestState.DECODING),
        (RequestState.PENDING, RequestState.COMPLETED),
        (RequestState.ADMITTED, RequestState.DECODING),
        (RequestState.ADMITTED, RequestState.COMPLETED),
        (RequestState.PREFILLING, RequestState.PENDING),
        (RequestState.DECODING, RequestState.PREFILLING),
        (RequestState.DECODING, RequestState.DECODING),
    ],
)
def test_lifecycle_rejects_skipped_stages_reversals_and_self_transitions(
    current: RequestState, next_state: RequestState
) -> None:
    with pytest.raises(ValueError, match="Illegal request transition"):
        validate_request_transition(current, next_state)


@pytest.mark.parametrize(
    "terminal", [RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED]
)
@pytest.mark.parametrize("next_state", list(RequestState))
def test_terminal_state_cannot_be_resolved_again(
    terminal: RequestState, next_state: RequestState
) -> None:
    with pytest.raises(ValueError, match="Illegal request transition"):
        validate_request_transition(terminal, next_state)


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        ("pending", RequestState.ADMITTED),
        (RequestState.PENDING, "admitted"),
        (None, RequestState.ADMITTED),
        (RequestState.PENDING, None),
    ],
)
def test_transition_requires_enum_values(current: object, next_state: object) -> None:
    with pytest.raises(ValueError, match="require RequestState values"):
        validate_request_transition(current, next_state)  # type: ignore[arg-type]


@pytest.mark.parametrize("credits", [0, -1, True, False, 1.5, "3"])
def test_reservation_requires_positive_integer_credits(credits: object) -> None:
    with pytest.raises(ValueError, match="maximum_block_credits must be a positive integer"):
        KVReservation(RequestId("request-1"), credits)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        (0, 0, 0, 0, 0, 0),
        (8, 8, 0, 0, 0, 0),
        (8, 8, 0, 3, 1, 0),
        (8, 6, 2, 3, 1, 17),
        (8, 0, 8, 8, 2, 128),
    ],
)
def test_capacity_accepts_empty_reserved_and_full_pool_snapshots(
    values: tuple[int, int, int, int, int, int],
) -> None:
    capacity = CapacityCounts(*values)
    assert capacity.free_blocks + capacity.owned_blocks == capacity.total_blocks
    assert capacity.owned_blocks <= capacity.reserved_blocks <= capacity.total_blocks


@pytest.mark.parametrize(
    "name",
    [
        "total_blocks",
        "free_blocks",
        "owned_blocks",
        "reserved_blocks",
        "active_slots",
        "live_cached_token_slots",
    ],
)
@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1"])
def test_capacity_rejects_negative_or_noninteger_counts(name: str, value: object) -> None:
    capacity = CapacityCounts(8, 6, 2, 3, 1, 17)
    with pytest.raises(ValueError, match=f"{name} must be a nonnegative integer"):
        replace(capacity, **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("free", "owned", "reserved", "message"),
    [
        (6, 3, 3, "free_blocks \\+ owned_blocks must equal total_blocks"),
        (6, 2, 1, "owned_blocks <= reserved_blocks <= total_blocks must hold"),
        (6, 2, 9, "owned_blocks <= reserved_blocks <= total_blocks must hold"),
    ],
)
def test_capacity_rejects_missing_blocks_and_overcommitted_credits(
    free: int, owned: int, reserved: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CapacityCounts(8, free, owned, reserved, 1, 17)
