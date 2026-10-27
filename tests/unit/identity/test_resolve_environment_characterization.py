"""Characterization of deployment environment resolution: precedence, warnings, reads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest

from traigent.identity import workload
from traigent.identity.workload import (
    EnvironmentResolution,
    EnvironmentTooLongError,
    resolve_environment,
)

Events = list[tuple[str, str]]

_LEGACY = "TRAIGENT_ENV is deprecated; set TRAIGENT_ENVIRONMENT instead."
_UNDECLARED = (
    "No deployment environment declared; using 'default'. Set "
    "TRAIGENT_ENVIRONMENT (or OTEL_RESOURCE_ATTRIBUTES "
    "deployment.environment.name) to label traces."
)
_INVALID = (
    "Ignoring unknown environment label in %s (not a known deployment "
    "label); falling back to the next environment source."
)


class FakeLogger:
    """Records warning calls as (message, args)."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, tuple[Any, ...]]] = []
        self.raise_on: str | None = None

    def warning(self, message: str, *args: Any) -> None:
        self.warnings.append((message, args))
        if self.raise_on == message:
            raise RuntimeError("log sink failed")


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> FakeLogger:
    fake = FakeLogger()
    monkeypatch.setattr(workload, "logger", fake)
    monkeypatch.setattr(workload, "_warned", set())
    return fake


class TracedEnv(Mapping[str, str]):
    """A mapping recording ``get`` reads, optionally raising for one key."""

    def __init__(
        self,
        data: dict[str, str],
        events: Events,
        raise_on: dict[str, BaseException] | None = None,
    ) -> None:
        self._data = data
        self._events = events
        self._raise_on = raise_on or {}

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        self._events.append(("get", key))
        if key in self._raise_on:
            raise self._raise_on[key]
        return self._data.get(key, default)


def _res(env: str, source: str, warning: str | None) -> EnvironmentResolution:
    return EnvironmentResolution(env, cast(Any, source), cast(Any, warning))


def test_explicit_label_is_trimmed_lowercased_and_not_read_from_env(
    log: FakeLogger,
) -> None:
    events: Events = []
    env = TracedEnv({"TRAIGENT_ENVIRONMENT": "production"}, events)
    assert resolve_environment("  My-Label ", env) == _res("my-label", "explicit", None)
    assert events == []
    assert log.warnings == []


def test_explicit_label_is_free_text_not_limited_to_known_names(
    log: FakeLogger,
) -> None:
    assert resolve_environment("whatever goes", {}) == _res(
        "whatever goes", "explicit", None
    )


def test_explicit_length_boundary_counts_trimmed_text(log: FakeLogger) -> None:
    at_cap = "a" * 64
    assert resolve_environment(at_cap, {}) == _res(at_cap, "explicit", None)
    assert resolve_environment("  " + at_cap + "  ", {}) == _res(
        at_cap, "explicit", None
    )
    with pytest.raises(EnvironmentTooLongError) as info:
        resolve_environment(at_cap + "a", {})
    assert str(info.value) == "environment must be at most 64 characters"
    with pytest.raises(EnvironmentTooLongError):
        resolve_environment(" " + at_cap + "A ", {})


def test_too_long_explicit_raises_before_reading_the_env(log: FakeLogger) -> None:
    events: Events = []
    with pytest.raises(EnvironmentTooLongError):
        resolve_environment("a" * 65, TracedEnv({}, events))
    assert events == []


@pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
def test_empty_explicit_counts_as_unset(log: FakeLogger, empty: str) -> None:
    env = {"TRAIGENT_ENVIRONMENT": "staging"}
    assert resolve_environment(empty, env) == _res(
        "staging", "traigent_environment", None
    )


def test_env_none_uses_process_environment_but_empty_mapping_does_not(
    log: FakeLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "production")
    assert resolve_environment() == _res("production", "traigent_environment", None)
    assert resolve_environment(None, None) == _res(
        "production", "traigent_environment", None
    )
    assert resolve_environment(None, {}) == _res(
        "default", "default", "undeclared_default"
    )


def test_precedence_order_and_reads(log: FakeLogger) -> None:
    data = {
        "TRAIGENT_ENVIRONMENT": "dev",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment.name=test",
        "ENVIRONMENT": "staging",
        "TRAIGENT_ENV": "ci",
    }
    events: Events = []
    assert resolve_environment(None, TracedEnv(data, events)) == _res(
        "dev", "traigent_environment", None
    )
    assert events == [("get", "TRAIGENT_ENVIRONMENT")]

    del data["TRAIGENT_ENVIRONMENT"]
    events.clear()
    assert resolve_environment(None, TracedEnv(data, events)) == _res(
        "test", "otel_resource", None
    )
    assert events == [
        ("get", "TRAIGENT_ENVIRONMENT"),
        ("get", "OTEL_RESOURCE_ATTRIBUTES"),
    ]

    del data["OTEL_RESOURCE_ATTRIBUTES"]
    events.clear()
    assert resolve_environment(None, TracedEnv(data, events)) == _res(
        "staging", "environment", None
    )
    assert [e[1] for e in events] == [
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
    ]

    del data["ENVIRONMENT"]
    events.clear()
    assert resolve_environment(None, TracedEnv(data, events)) == _res(
        "ci", "traigent_env_legacy", "legacy_env_var_deprecated"
    )
    assert [e[1] for e in events] == [
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
    ]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ("deployment.environment.name=staging", "staging"),
        ("a=b, deployment.environment.name = test ,c=d", "test"),
        ("x=y,deployment.environment.name=dev", "dev"),
        ("deployment.environment.name=stag%69ng", "staging"),
        ("deployment.environment.name=a,deployment.environment.name=dev", None),
    ],
)
def test_otel_attribute_parsing(
    log: FakeLogger, attributes: str, expected: str | None
) -> None:
    result = resolve_environment(None, {"OTEL_RESOURCE_ATTRIBUTES": attributes})
    if expected is None:
        assert result == _res("default", "default", "invalid_env_value_ignored")
    else:
        assert result == _res(expected, "otel_resource", None)


@pytest.mark.parametrize("attributes", ["", "noequals", "other.key=prod", "=prod"])
def test_otel_attribute_without_environment_key_is_skipped(
    log: FakeLogger, attributes: str
) -> None:
    result = resolve_environment(None, {"OTEL_RESOURCE_ATTRIBUTES": attributes})
    assert result == _res("default", "default", "undeclared_default")
    assert log.warnings == [(_UNDECLARED, ())]


def test_values_are_normalized_before_known_label_check(log: FakeLogger) -> None:
    assert resolve_environment(None, {"ENVIRONMENT": "  PRODUCTION "}) == _res(
        "production", "environment", None
    )


def test_unknown_value_is_ignored_with_value_free_warning(log: FakeLogger) -> None:
    env = {"TRAIGENT_ENVIRONMENT": "Secret-Label", "ENVIRONMENT": "staging"}
    assert resolve_environment(None, env) == _res(
        "staging", "environment", "invalid_env_value_ignored"
    )
    assert log.warnings == [(_INVALID, ("TRAIGENT_ENVIRONMENT",))]


def test_empty_env_value_is_unknown_not_unset(log: FakeLogger) -> None:
    result = resolve_environment(
        None, {"TRAIGENT_ENVIRONMENT": "", "ENVIRONMENT": "dev"}
    )
    assert result == _res("dev", "environment", "invalid_env_value_ignored")
    assert log.warnings == [(_INVALID, ("TRAIGENT_ENVIRONMENT",))]


def test_every_unknown_rung_warns_once_and_default_carries_ignored_flag(
    log: FakeLogger,
) -> None:
    env = {
        "TRAIGENT_ENVIRONMENT": "x1",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment.name=x2",
        "ENVIRONMENT": "x3",
        "TRAIGENT_ENV": "x4",
    }
    expected_default = _res("default", "default", "invalid_env_value_ignored")
    assert resolve_environment(None, env) == expected_default
    assert log.warnings == [
        (_INVALID, ("TRAIGENT_ENVIRONMENT",)),
        (_INVALID, ("OTEL_RESOURCE_ATTRIBUTES",)),
        (_INVALID, ("ENVIRONMENT",)),
        (_INVALID, ("TRAIGENT_ENV",)),
        (_UNDECLARED, ()),
    ]
    log.warnings.clear()
    assert resolve_environment(None, env) == expected_default
    assert log.warnings == []


def test_legacy_rung_warns_once_and_reports_deprecation(log: FakeLogger) -> None:
    env = {"TRAIGENT_ENV": "dev"}
    expected = _res("dev", "traigent_env_legacy", "legacy_env_var_deprecated")
    assert resolve_environment(None, env) == expected
    assert log.warnings == [(_LEGACY, ())]
    assert resolve_environment(None, env) == expected
    assert log.warnings == [(_LEGACY, ())]


def test_ignored_flag_outranks_legacy_deprecation_in_the_result(
    log: FakeLogger,
) -> None:
    env = {"ENVIRONMENT": "nope", "TRAIGENT_ENV": "dev"}
    assert resolve_environment(None, env) == _res(
        "dev", "traigent_env_legacy", "invalid_env_value_ignored"
    )
    assert log.warnings == [(_INVALID, ("ENVIRONMENT",)), (_LEGACY, ())]


def test_non_legacy_accepted_rung_does_not_emit_legacy_warning(log: FakeLogger) -> None:
    resolve_environment(None, {"ENVIRONMENT": "dev", "TRAIGENT_ENV": "dev"})
    assert log.warnings == []


def test_undeclared_default_warns_once(log: FakeLogger) -> None:
    assert resolve_environment(None, {}) == _res(
        "default", "default", "undeclared_default"
    )
    assert log.warnings == [(_UNDECLARED, ())]
    resolve_environment(None, {})
    assert log.warnings == [(_UNDECLARED, ())]


def test_default_read_trace_covers_all_four_variables(log: FakeLogger) -> None:
    events: Events = []
    resolve_environment(None, TracedEnv({}, events))
    assert events == [
        ("get", "TRAIGENT_ENVIRONMENT"),
        ("get", "OTEL_RESOURCE_ATTRIBUTES"),
        ("get", "ENVIRONMENT"),
        ("get", "TRAIGENT_ENV"),
    ]


@pytest.mark.parametrize(
    "key",
    ["TRAIGENT_ENVIRONMENT", "OTEL_RESOURCE_ATTRIBUTES", "ENVIRONMENT", "TRAIGENT_ENV"],
)
def test_raising_env_read_propagates_and_stops_scanning(
    log: FakeLogger, key: str
) -> None:
    events: Events = []
    order = [
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
    ]
    env = TracedEnv({}, events, raise_on={key: RuntimeError("read " + key)})
    with pytest.raises(RuntimeError, match="read " + key):
        resolve_environment(None, env)
    assert [e[1] for e in events] == order[: order.index(key) + 1]
    assert log.warnings == []


def test_raise_once_read_keeps_earlier_warning_state(log: FakeLogger) -> None:
    data = {"TRAIGENT_ENVIRONMENT": "bad"}
    state = {"n": 0}

    class Flaky(TracedEnv):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "ENVIRONMENT":
                state["n"] += 1
                if state["n"] == 1:
                    raise RuntimeError("once")
            return super().get(key, default)

    env = Flaky(data, [])
    with pytest.raises(RuntimeError, match="once"):
        resolve_environment(None, env)
    assert log.warnings == [(_INVALID, ("TRAIGENT_ENVIRONMENT",))]
    assert resolve_environment(None, env) == _res(
        "default", "default", "invalid_env_value_ignored"
    )
    # the invalid-label warning for the first variable is not repeated.
    assert log.warnings == [(_INVALID, ("TRAIGENT_ENVIRONMENT",)), (_UNDECLARED, ())]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_from_reads_propagate(
    log: FakeLogger, exc: type[BaseException]
) -> None:
    env = TracedEnv({}, [], raise_on={"ENVIRONMENT": exc()})
    with pytest.raises(exc):
        resolve_environment(None, env)


def test_failing_log_sink_propagates_after_marking_warned(log: FakeLogger) -> None:
    log.raise_on = _LEGACY
    env = {"TRAIGENT_ENV": "dev"}
    with pytest.raises(RuntimeError, match="log sink failed"):
        resolve_environment(None, env)
    log.raise_on = None
    assert resolve_environment(None, env) == _res(
        "dev", "traigent_env_legacy", "legacy_env_var_deprecated"
    )
    assert log.warnings == [(_LEGACY, ())]


def test_failing_invalid_label_sink_prevents_later_reads(log: FakeLogger) -> None:
    log.raise_on = _INVALID
    events: Events = []
    env = TracedEnv({"TRAIGENT_ENVIRONMENT": "bad"}, events)
    with pytest.raises(RuntimeError, match="log sink failed"):
        resolve_environment(None, env)
    assert events == [("get", "TRAIGENT_ENVIRONMENT")]


def test_warnings_never_contain_the_ignored_value(log: FakeLogger) -> None:
    resolve_environment(None, {"ENVIRONMENT": "super-secret-token-value"})
    flat = repr(log.warnings)
    assert "super-secret-token-value" not in flat
