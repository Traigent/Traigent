"""Characterization of ContentPolicy._events (span-event egress policy)."""

from __future__ import annotations

from opentelemetry.sdk.trace import Event

from traigent.observability.otel.policy import ContentPolicy

SECRET = "sk-abcdefghijklmnopqrstuvwxyz0123456789"  # pragma: allowlist secret


def _ev(name, attributes=None, ts=123):
    return Event(name, attributes, ts)


def _summary(result):
    events, dropped = result
    return [(e.name, dict(e.attributes or {}), e.timestamp) for e in events], dropped


class TestRestrictedModes:
    P = ContentPolicy("metadata")

    def test_unknown_event_dropped_and_counted(self):
        out = self.P._events([_ev("gen_ai.user.message", {"content": "x"})], "metadata")
        assert _summary(out) == ([], 1)

    def test_none_name_dropped(self):
        assert _summary(self.P._events([_ev(None)], "metadata")) == ([], 1)

    def test_exception_event_keeps_only_allowed_key(self):
        out = self.P._events(
            [
                _ev(
                    "exception",
                    {
                        "exception.type": "ValueError",
                        "exception.message": "customer text",
                        "exception.stacktrace": "trace",
                    },
                )
            ],
            "metadata",
        )
        assert _summary(out) == (
            [("exception", {"exception.type": "ValueError"}, 123)],
            0,
        )

    def test_string_length_boundary_128(self):
        ok = self.P._events(
            [_ev("exception", {"exception.type": "a" * 128})], "metadata"
        )
        bad = self.P._events(
            [_ev("exception", {"exception.type": "a" * 129})], "metadata"
        )
        assert _summary(ok)[0][0][1] == {"exception.type": "a" * 128}
        assert _summary(bad) == ([("exception", {}, 123)], 0)

    def test_rejected_values_leave_empty_attrs_but_keep_event(self):
        for value in [
            "",
            None,
            5,
            b"x",
            "line\nbreak",
            "tab\there",
            "del\x7fx",
            SECRET,
        ]:
            out = self.P._events(
                [_ev("exception", {"exception.type": value})], "metadata"
            )
            assert _summary(out) == ([("exception", {}, 123)], 0), value

    def test_none_attributes_tolerated(self):
        out = self.P._events([_ev("exception", None)], "metadata")
        assert _summary(out) == ([("exception", {}, 123)], 0)

    def test_empty_and_none_inputs(self):
        assert self.P._events([], "metadata") == ((), 0)
        assert self.P._events(None, "metadata") == ((), 0)

    def test_order_and_drop_count_over_mixed_events(self):
        out = self.P._events(
            [
                _ev("drop-me"),
                _ev("exception", {"exception.type": "A"}, 1),
                _ev("also-drop"),
                _ev("exception", {"exception.type": "B"}, 2),
            ],
            "metadata",
        )
        assert _summary(out) == (
            [
                ("exception", {"exception.type": "A"}, 1),
                ("exception", {"exception.type": "B"}, 2),
            ],
            2,
        )

    def test_non_record_mode_argument_governs(self):
        policy = ContentPolicy("record")
        out = policy._events([_ev("custom", {"k": "v"})], "metadata")
        assert _summary(out) == ([], 1)


class TestRecordMode:
    P = ContentPolicy("record")

    def test_any_event_kept_with_scrubbed_name_and_attrs(self):
        out = self.P._events([_ev(f"{SECRET} x", {"k": "v", "n": 1})], "record")
        assert _summary(out) == (
            [("[REDACTED:api_key] x", {"k": "v", "n": 1}, 123)],
            0,
        )

    def test_unsupported_attribute_counted_as_dropped(self):
        out = self.P._events([_ev("e", {"k": "v", "o": object()})], "record")
        assert _summary(out) == ([("e", {"k": "v"}, 123)], 1)

    def test_drops_accumulate_across_events(self):
        out = self.P._events(
            [_ev("a", {"o": object()}), _ev("b", {"p": object(), "q": object()})],
            "record",
        )
        assert [e.name for e in out[0]] == ["a", "b"]
        assert out[1] == 3

    def test_empty_or_non_string_name_becomes_event(self):
        for name in ["", None, 5]:
            out = self.P._events([_ev(name, {"k": "v"})], "record")
            assert _summary(out) == ([("event", {"k": "v"}, 123)], 0), name

    def test_name_capped_at_256(self):
        out = self.P._events([_ev("a" * 300)], "record")
        assert out[0][0].name == "a" * 256
        out = self.P._events([_ev("a" * 255)], "record")
        assert out[0][0].name == "a" * 255

    def test_none_attributes_tolerated(self):
        out = self.P._events([_ev("e", None)], "record")
        assert _summary(out) == ([("e", {}, 123)], 0)

    def test_unknown_event_not_dropped_in_record_mode(self):
        out = self.P._events([_ev("gen_ai.user.message", {"content": "hi"})], "record")
        assert _summary(out)[1] == 0
        assert len(out[0]) == 1


class _SpyEvent(Event):
    """Event whose accessors record the order in which they are read."""

    def __init__(self, name, attributes, timestamp, log, fail_on=None):
        self._n, self._a, self._t = name, attributes, timestamp
        self._log = log
        self._fail_on = fail_on

    def _touch(self, field, value):
        self._log.append(field)
        if self._fail_on == field:
            raise RuntimeError(f"boom-{field}")
        return value

    @property
    def name(self):
        return self._touch("name", self._n)

    @property
    def attributes(self):
        return self._touch("attributes", self._a)

    @property
    def timestamp(self):
        return self._touch("timestamp", self._t)


class TestAccessOrder:
    def test_restricted_mode_reads_name_then_attributes_then_name_then_timestamp(self):
        log: list[str] = []
        ev = _SpyEvent("exception", {"exception.type": "E"}, 5, log)
        log.clear()
        events, dropped = ContentPolicy("metadata")._events([ev], "metadata")
        assert log == ["name", "attributes", "name", "timestamp"]
        assert dropped == 0
        assert events[0].attributes == {"exception.type": "E"}

    def test_attributes_failure_surfaces_before_name_is_read_again(self):
        log: list[str] = []
        ev = _SpyEvent("exception", {"exception.type": "E"}, 5, log, "attributes")
        log.clear()
        try:
            ContentPolicy("metadata")._events([ev], "metadata")
        except RuntimeError as exc:
            assert str(exc) == "boom-attributes"
        else:
            raise AssertionError("expected failure")
        assert log == ["name", "attributes"]

    def test_unknown_event_reads_only_name(self):
        log: list[str] = []
        ev = _SpyEvent("other", {"a": "b"}, 5, log)
        log.clear()
        assert ContentPolicy("metadata")._events([ev], "metadata") == ((), 1)
        assert log == ["name"]
