"""Characterization for vendor_retry.vendor_wait_seconds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.utils.vendor_retry import vendor_wait_seconds

FUTURE = "Fri, 01 Jan 2100 00:00:00 GMT"
PAST = "Mon, 01 Jan 1990 00:00:00 GMT"
FUTURE_NAIVE = "Fri, 01 Jan 2100 00:00:00 -0000"


def _exc(**attrs):
    exc = Exception("x")
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


class _Headers:
    """Headers whose .get misses but keys() holds mixed-case names; logs calls."""

    def __init__(self, data):
        self.data = data
        self.log: list[tuple[str, str]] = []

    def get(self, name):
        self.log.append(("get", name))
        return None

    def keys(self):
        self.log.append(("keys", ""))
        return list(self.data)

    def __getitem__(self, key):
        self.log.append(("item", key))
        return self.data[key]


def test_no_hints_returns_none():
    assert vendor_wait_seconds(_exc()) is None


def test_retry_after_ms_beats_everything():
    exc = _exc(headers={"retry-after-ms": "1500", "retry-after": "9"}, retry_after=7)
    assert vendor_wait_seconds(exc) == 1.5


def test_ms_from_later_source_beats_attr_and_seconds():
    exc = _exc(
        headers={"retry-after": "9"},
        response=SimpleNamespace(headers={"retry-after-ms": 250}),
        retry_after=7,
    )
    assert vendor_wait_seconds(exc) == 0.25


def test_attr_beats_seconds_header():
    assert vendor_wait_seconds(_exc(headers={"retry-after": "9"}, retry_after=7)) == 7


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), "abc", None])
def test_invalid_numbers_are_ignored(bad):
    exc = _exc(headers={"retry-after-ms": bad, "retry-after": "3"}, retry_after=bad)
    assert vendor_wait_seconds(exc) == 3.0


def test_seconds_header_first_source_with_value_wins():
    exc = _exc(
        headers={"retry-after": "abc"},
        response=SimpleNamespace(headers={"retry-after": "4"}),
    )
    # "abc" is a non-empty string that is not a date: skipped, next source wins
    assert vendor_wait_seconds(exc) == 4.0


def test_http_date_future_naive_and_aware_and_past():
    assert vendor_wait_seconds(_exc(headers={"retry-after": FUTURE})) > 1e9
    assert vendor_wait_seconds(_exc(headers={"retry-after": FUTURE_NAIVE})) > 1e9
    assert vendor_wait_seconds(_exc(headers={"retry-after": PAST})) == 0.0
    assert vendor_wait_seconds(_exc(headers={"retry-after": f"  {PAST} "})) == 0.0


def test_blank_and_non_string_raw_value_do_not_stop_the_scan():
    exc = _exc(
        headers={"retry-after": "   "},
        response=SimpleNamespace(headers={"retry-after": ["x"]}),
        litellm_response_headers={"retry-after": "6"},
    )
    assert vendor_wait_seconds(exc) == 6.0


def test_unparseable_date_falls_through_to_later_source():
    exc = _exc(
        headers={"retry-after": "not a date at all"},
        responseHeaders={"Retry-After": "8"},
    )
    assert vendor_wait_seconds(exc) == 8.0


def test_case_insensitive_header_scan_and_call_sequence():
    h = _Headers({"Retry-After": "2"})
    assert vendor_wait_seconds(_exc(headers=h)) == 2.0
    # ms: get, keys; then seconds pass: get, keys, item
    assert h.log == [
        ("get", "retry-after-ms"),
        ("keys", ""),
        ("get", "retry-after"),
        ("keys", ""),
        ("item", "Retry-After"),
    ]


def test_ms_found_stops_before_seconds_pass():
    h = _Headers({"Retry-After-Ms": "100", "Retry-After": "2"})
    assert vendor_wait_seconds(_exc(headers=h)) == 0.1
    assert h.log == [
        ("get", "retry-after-ms"),
        ("keys", ""),
        ("item", "Retry-After-Ms"),
    ]


def test_attr_read_once_and_only_after_ms_scan():
    class Probe(Exception):
        reads = 0

        @property
        def retry_after(self):
            Probe.reads += 1
            return 5

    assert vendor_wait_seconds(Probe()) == 5
    assert Probe.reads == 1
