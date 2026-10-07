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
    """Server clock only; absent events are None, including first/last token on zero output.

    Arrival is pending acceptance; admission grants credits and an active slot.
    Token times mark successful mailbox publication, not client receipt. Generation
    and transport terminal times need not coincide or occur in that order.
    """

    arrival: MonotonicTimestamp
    admission: MonotonicTimestamp | None = None
    first_token: MonotonicTimestamp | None = None
    last_token: MonotonicTimestamp | None = None
    generation_terminal: MonotonicTimestamp | None = None
    transport_terminal: MonotonicTimestamp | None = None


@dataclass(frozen=True, slots=True)
class RequestCounts:
    """Prompt includes special tokens; output counts mailbox publications, not delivery.

    Stopping EOS is excluded; ignored EOS and empty text deltas count as output.
    Cached slots count valid KV positions, not prompt plus all selected tokens.
    """

    prompt_tokens: int
    max_new_tokens: int
    output_tokens: int = 0
    cached_token_slots: int = 0


@dataclass(frozen=True, slots=True)
class KVReservation:
    """Maximum capacity credits granted at admission, including already owned blocks.

    Credits do not assign physical blocks. The admitted request also holds one
    active slot until worker cleanup; pending requests hold neither resource.
    """

    request_id: RequestId
    maximum_block_credits: int


@dataclass(frozen=True, slots=True)
class CapacityCounts:
    """Reserved credits overlap free capacity; owned is also called occupied in metrics."""

    total_blocks: int
    free_blocks: int
    owned_blocks: int
    reserved_blocks: int
    active_slots: int
    live_cached_token_slots: int
