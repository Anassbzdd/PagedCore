"""Pure internal timing arithmetic, independent of model and transport execution."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from pagedcore.engine_types import MonotonicTimestamp, RequestTiming


@dataclass(frozen=True, slots=True)
class InternalLatencies:
    ttft_seconds: float | None
    itl_seconds: tuple[float, ...]
    tpot_seconds: float | None
    generation_seconds: float | None
    stream_seconds: float | None


def analyze_internal_timing(
    timing: RequestTiming,
    token_times: tuple[MonotonicTimestamp, ...],
) -> InternalLatencies:
    """Token times are mailbox publications, including empty text deltas, on one clock."""
    values = (
        timing.arrival,
        timing.admission,
        *token_times,
        timing.generation_terminal,
        timing.transport_terminal,
    )
    present = [value for value in values if value is not None]
    endpoints = (timing.first_token, timing.last_token)
    if any(
        type(value) is not int or value < 0 for value in (*present, *endpoints) if value is not None
    ):
        raise ValueError("timestamps must be nonnegative integer nanoseconds")
    if present != sorted(present):
        raise ValueError("internal timestamps must be ordered on one monotonic clock")
    first = token_times[0] if token_times else None
    last = token_times[-1] if token_times else None
    if timing.first_token != first or timing.last_token != last:
        raise ValueError("token timing endpoints disagree with publications")

    def elapsed(end: MonotonicTimestamp | None) -> float | None:
        return None if end is None else (end - timing.arrival) / 1_000_000_000

    gaps = tuple((right - left) / 1_000_000_000 for left, right in pairwise(token_times))
    return InternalLatencies(
        elapsed(first),
        gaps,
        sum(gaps) / len(gaps) if gaps else None,
        elapsed(timing.generation_terminal),
        elapsed(timing.transport_terminal),
    )
