from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NewType, TypeAlias

RequestId = NewType("RequestId", str)
BlockId = NewType("BlockId", int)
# Values use time.monotonic_ns() in one server process, never wall or client time.
MonotonicTimestamp = NewType("MonotonicTimestamp", int)
BlockTable: TypeAlias = tuple[BlockId, ...]


class RequestState(StrEnum):
    RECEIVED = "received"
    VALIDATED = "validated"
    PENDING = "pending"
    ADMITTED = "admitted"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_REQUEST_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.RECEIVED: frozenset(
        {RequestState.VALIDATED, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.VALIDATED: frozenset(
        {RequestState.PENDING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.PENDING: frozenset(
        {RequestState.ADMITTED, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.ADMITTED: frozenset(
        {RequestState.PREFILLING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.PREFILLING: frozenset(
        {
            RequestState.DECODING,
            RequestState.COMPLETED,
            RequestState.CANCELLED,
            RequestState.FAILED,
        }
    ),
    RequestState.DECODING: frozenset(
        {RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.COMPLETED: frozenset(),
    RequestState.CANCELLED: frozenset(),
    RequestState.FAILED: frozenset(),
}


def validate_request_transition(current: RequestState, next_state: RequestState) -> None:
    if not isinstance(current, RequestState) or not isinstance(next_state, RequestState):
        raise ValueError("Request transitions require RequestState values")
    if next_state not in _REQUEST_TRANSITIONS[current]:
        raise ValueError(f"Illegal request transition: {current.value} -> {next_state.value}")


class FinishReason(StrEnum):
    EOS = "eos"
    LENGTH = "length"


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    REQUEST_BODY_TOO_LARGE = "request_body_too_large"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    KV_CAPACITY_EXCEEDED = "kv_capacity_exceeded"
    PENDING_QUEUE_FULL = "pending_queue_full"
    PREPROCESSING_CAPACITY_EXCEEDED = "preprocessing_capacity_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    SLOW_CONSUMER = "slow_consumer"
    WORKER_FAILURE = "worker_failure"
    SERVER_SHUTDOWN = "server_shutdown"
    CLIENT_DISCONNECTED = "client_disconnected"


CancellationReason: TypeAlias = Literal[
    ErrorCode.CLIENT_DISCONNECTED, ErrorCode.SLOW_CONSUMER, ErrorCode.SERVER_SHUTDOWN
]


@dataclass(frozen=True, slots=True)
class EngineError:
    """Messages must be safe to expose and never echo request content or secrets."""

    code: ErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """Transient mailbox payload; publication counts even with an empty text delta."""

    request_id: RequestId
    index: int
    token_id: int
    published_at: MonotonicTimestamp


@dataclass(frozen=True, slots=True)
class GenerationCompleted:
    request_id: RequestId
    terminal_at: MonotonicTimestamp
    output_tokens: int
    finish_reason: FinishReason
    state: Literal[RequestState.COMPLETED] = field(default=RequestState.COMPLETED, init=False)


@dataclass(frozen=True, slots=True)
class GenerationCancelled:
    request_id: RequestId
    terminal_at: MonotonicTimestamp
    output_tokens: int
    reason: CancellationReason
    state: Literal[RequestState.CANCELLED] = field(default=RequestState.CANCELLED, init=False)


@dataclass(frozen=True, slots=True)
class GenerationFailed:
    request_id: RequestId
    terminal_at: MonotonicTimestamp
    output_tokens: int
    error: EngineError
    state: Literal[RequestState.FAILED] = field(default=RequestState.FAILED, init=False)


# Delivered outside the token mailbox; completion does not imply resources are released.
GenerationTerminal: TypeAlias = GenerationCompleted | GenerationCancelled | GenerationFailed


class TransportStatus(StrEnum):
    CLOSED = "closed"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TransportTerminal:
    """Delivery outcome, independent of an already resolved generation outcome."""

    request_id: RequestId
    terminal_at: MonotonicTimestamp
    status: TransportStatus
    error: EngineError | None = None


@dataclass(frozen=True, slots=True)
class RequestTiming:
    arrival: MonotonicTimestamp
    admission: MonotonicTimestamp | None = None
    first_token: MonotonicTimestamp | None = None
    last_token: MonotonicTimestamp | None = None
    generation_terminal: MonotonicTimestamp | None = None
    transport_terminal: MonotonicTimestamp | None = None


@dataclass(frozen=True, slots=True)
class RequestCounts:
    prompt_tokens: int
    max_new_tokens: int
    output_tokens: int = 0
    cached_token_slots: int = 0


@dataclass(frozen=True, slots=True)
class KVReservation:
    """Maximum capacity credits granted at admission, including already owned blocks.

    Credits do not assign physical blocks. The admitted request also holds one
    active slot until worker cleanup; pending requests hold neither resource.
    Cleanup returns only acquired blocks, credits and slots, once per acquisition,
    after GPU accesses are safely ordered before reuse. Duplicate cleanup signals
    must not release again; a terminal outcome or transport closure is not cleanup.
    """

    request_id: RequestId
    maximum_block_credits: int

    def __post_init__(self) -> None:
        credits = self.maximum_block_credits
        if not isinstance(credits, int) or isinstance(credits, bool) or credits <= 0:
            raise ValueError("maximum_block_credits must be a positive integer")


@dataclass(frozen=True, slots=True)
class CapacityCounts:
    """Snapshot at a consistent worker boundary, including requests awaiting cleanup.

    Reserved credits overlap free capacity; owned is also called occupied in metrics.
    Counts alone cannot prove per-request ownership or that resources were released once.
    """

    total_blocks: int
    free_blocks: int
    owned_blocks: int
    reserved_blocks: int
    active_slots: int
    live_cached_token_slots: int

    def __post_init__(self) -> None:
        for name, value in (
            ("total_blocks", self.total_blocks),
            ("free_blocks", self.free_blocks),
            ("owned_blocks", self.owned_blocks),
            ("reserved_blocks", self.reserved_blocks),
            ("active_slots", self.active_slots),
            ("live_cached_token_slots", self.live_cached_token_slots),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.free_blocks + self.owned_blocks != self.total_blocks:
            raise ValueError("free_blocks + owned_blocks must equal total_blocks")
        if not self.owned_blocks <= self.reserved_blocks <= self.total_blocks:
            raise ValueError("owned_blocks <= reserved_blocks <= total_blocks must hold")
