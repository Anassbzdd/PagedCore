from dataclasses import replace
from typing import Any

import pytest

from pagedcore.engine_types import MonotonicTimestamp, RequestTiming
from pagedcore.measurements import analyze_internal_timing

pytestmark = pytest.mark.unit


def ns(milliseconds: int) -> MonotonicTimestamp:
    return MonotonicTimestamp(milliseconds * 1_000_000)


def test_synthetic_internal_timing_separates_generation_from_delivery() -> None:
    timing = RequestTiming(ns(0), ns(20), ns(50), ns(90), ns(95), ns(110))
    result = analyze_internal_timing(timing, (ns(50), ns(70), ns(90)))
    assert result.ttft_seconds == 0.05
    assert result.itl_seconds == (0.02, 0.02)
    assert result.tpot_seconds == 0.02
    assert result.generation_seconds == 0.095
    assert result.stream_seconds == 0.11


@pytest.mark.parametrize("count", [0, 1])
def test_undefined_token_latencies_are_missing(count: int) -> None:
    times = (ns(10),) if count else ()
    endpoint = ns(10) if count else None
    result = analyze_internal_timing(
        RequestTiming(ns(0), ns(1), endpoint, endpoint, ns(20), ns(30)), times
    )
    assert result.ttft_seconds == (0.01 if count else None)
    assert result.itl_seconds == ()
    assert result.tpot_seconds is None


def test_equal_publication_times_and_unfinished_generation() -> None:
    result = analyze_internal_timing(RequestTiming(ns(5), ns(6), ns(10), ns(10)), (ns(10), ns(10)))
    assert result.itl_seconds == (0.0,)
    assert result.tpot_seconds == 0.0
    assert result.generation_seconds is None
    assert result.stream_seconds is None


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("generation_terminal", [None, ns(40)])
def test_disconnect_can_precede_worker_terminal(
    active: bool, generation_terminal: MonotonicTimestamp | None
) -> None:
    times = (ns(20), ns(22)) if active else ()
    timing = RequestTiming(
        arrival=ns(5),
        admission=ns(10) if active else None,
        first_token=times[0] if times else None,
        last_token=times[-1] if times else None,
        generation_terminal=generation_terminal,
        transport_terminal=ns(25),
    )

    result = analyze_internal_timing(timing, times)

    assert result.stream_seconds == 0.02
    assert result.generation_seconds == (None if generation_terminal is None else 0.035)
    assert result.ttft_seconds == (0.015 if active else None)
    assert result.itl_seconds == ((0.002,) if active else ())
    assert result.tpot_seconds == (0.002 if active else None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arrival", ns(-1)),
        ("admission", ns(60)),
        ("first_token", ns(51)),
        ("last_token", None),
        ("generation_terminal", ns(80)),
        ("transport_terminal", ns(-1)),
    ],
)
def test_inconsistent_timestamps_fail(field: str, value: Any) -> None:
    timing = RequestTiming(ns(0), ns(20), ns(50), ns(90), ns(95), ns(110))
    with pytest.raises(ValueError):
        analyze_internal_timing(replace(timing, **{field: value}), (ns(50), ns(70), ns(90)))


def test_transport_cannot_terminate_before_arrival() -> None:
    timing = RequestTiming(arrival=ns(5), transport_terminal=ns(4))
    with pytest.raises(ValueError):
        analyze_internal_timing(timing, ())


@pytest.mark.parametrize("value", [True, 1.5, "25"])
def test_transport_timestamp_requires_integer_nanoseconds(value: Any) -> None:
    timing = RequestTiming(arrival=ns(0), transport_terminal=value)
    with pytest.raises(ValueError, match="nonnegative integer nanoseconds"):
        analyze_internal_timing(timing, ())
