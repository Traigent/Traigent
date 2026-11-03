"""Characterization of the response-rule trigger validator, pinned before extraction.

Full issue lists for the outcome-trigger and the monitor-trigger shapes, boundary
equality for ``unknown_for`` hours, and the exact attribute read sequence per shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.runtime.model import (
    Issue,
    ResponseRule,
    RuntimeRequirement,
    _validate_trigger,
)

_WHERE = "rules[3]"
_INVALID = "invalid_requirement"
_HOURS_RANGE = "runtime_trigger_hours_out_of_range"


def _requirement(*rules: ResponseRule, window_hours: int = 6) -> RuntimeRequirement:
    return RuntimeRequirement(
        name="r",
        latency_threshold_ms=100,
        environment="prod",
        max_violation_rate=0.1,
        window_hours=window_hours,
        rules=tuple(rules),
    )


def _target(**overrides: Any) -> ResponseRule:
    fields: dict[str, Any] = {
        "id": "fix",
        "event": "episode_opened",
        "action": "optimize",
        "deploy": "if_promoted",
    }
    fields.update(overrides)
    return ResponseRule(**fields)


def _issues(requirement: Any, rule: Any) -> list[Issue]:
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    return issues


def _outcome(event: str = "optimize_failed", **overrides: Any) -> ResponseRule:
    fields: dict[str, Any] = {
        "id": "after",
        "event": event,
        "action": "notify",
        "trigger_rule": "fix",
    }
    fields.update(overrides)
    return ResponseRule(**fields)


_TRIGGER = Issue(_INVALID, "rules[3].trigger", "error")
_TRIGGER_EVENT = Issue(_INVALID, "rules[3].trigger.event", "error")
_TRIGGER_HOURS = Issue(_INVALID, "rules[3].trigger.hours", "error")
_EVENT = Issue(_INVALID, "rules[3].event", "error")


# --------------------------------------------------------------- outcome triggers


@pytest.mark.parametrize(
    "event",
    ["optimize_no_candidate", "optimize_failed", "deploy_refused", "deploy_failed"],
)
def test_valid_outcome_trigger_has_no_issues(event: str) -> None:
    rule = _outcome(event)
    assert _issues(_requirement(_target(), rule), rule) == []


@pytest.mark.parametrize(
    "event",
    [
        "episode_opened",
        "unknown_for",
        "optimize",
        "Optimize_failed",
        "",
        "deploy_failed ",
    ],
)
def test_outcome_trigger_with_a_non_outcome_event_is_invalid(event: str) -> None:
    rule = _outcome(event)
    assert _issues(_requirement(_target(), rule), rule) == [_TRIGGER]


def test_outcome_trigger_to_an_unknown_rule_is_invalid() -> None:
    rule = _outcome(trigger_rule="missing")
    assert _issues(_requirement(_target(), rule), rule) == [_TRIGGER]


def test_outcome_trigger_with_no_rules_at_all_is_invalid() -> None:
    rule = _outcome()
    assert _issues(_requirement(), rule) == [_TRIGGER]


def test_outcome_trigger_on_itself_is_invalid() -> None:
    rule = _outcome(
        id="fix", trigger_rule="fix", action="optimize", deploy="if_promoted"
    )
    assert _issues(_requirement(rule), rule) == [_TRIGGER]


@pytest.mark.parametrize("action", ["notify", "open_next_step", "", "Optimize"])
def test_outcome_trigger_to_a_non_optimize_rule_is_invalid(action: str) -> None:
    rule = _outcome()
    assert _issues(_requirement(_target(action=action), rule), rule) == [_TRIGGER]


@pytest.mark.parametrize("deploy", [None, "never", "", "if_promoted "])
@pytest.mark.parametrize("event", ["deploy_refused", "deploy_failed"])
def test_deploy_outcome_needs_a_target_that_deploys_when_promoted(
    event: str, deploy: str | None
) -> None:
    rule = _outcome(event)
    assert _issues(_requirement(_target(deploy=deploy), rule), rule) == [_TRIGGER_EVENT]


@pytest.mark.parametrize("event", ["optimize_no_candidate", "optimize_failed"])
@pytest.mark.parametrize("deploy", [None, "never", "if_promoted"])
def test_optimize_outcome_does_not_depend_on_the_target_deploy_mode(
    event: str, deploy: str | None
) -> None:
    rule = _outcome(event)
    assert _issues(_requirement(_target(deploy=deploy), rule), rule) == []


def test_deploy_outcome_with_matching_deploy_mode_is_valid() -> None:
    rule = _outcome("deploy_failed")
    assert _issues(_requirement(_target(deploy="if_promoted"), rule), rule) == []


def test_first_invalid_condition_hides_the_deploy_mode_issue() -> None:
    rule = _outcome("deploy_failed")
    target = _target(action="notify", deploy="never")
    assert _issues(_requirement(target, rule), rule) == [_TRIGGER]


@pytest.mark.parametrize("hours", [0, 1, 6, 100, -1, "6", True, 2.5])
def test_hours_on_an_outcome_trigger_are_invalid(hours: Any) -> None:
    rule = _outcome(hours=hours)
    assert _issues(_requirement(_target(), rule), rule) == [_TRIGGER_HOURS]


def test_outcome_trigger_issues_are_ordered_trigger_then_hours() -> None:
    rule = _outcome("episode_opened", hours=3)
    assert _issues(_requirement(_target(), rule), rule) == [_TRIGGER, _TRIGGER_HOURS]


def test_deploy_mode_issue_comes_before_the_hours_issue() -> None:
    rule = _outcome("deploy_failed", hours=3)
    target = _target(deploy="never")
    assert _issues(_requirement(target, rule), rule) == [_TRIGGER_EVENT, _TRIGGER_HOURS]


def test_outcome_trigger_never_runs_the_monitor_trigger_checks() -> None:
    # an event valid only for monitor triggers: reported as an invalid trigger, once,
    # and no `.event` issue or hours-range issue is added
    rule = _outcome("unknown_for", hours=1)
    assert _issues(_requirement(_target(), rule, window_hours=6), rule) == [
        _TRIGGER,
        _TRIGGER_HOURS,
    ]


def test_duplicate_ids_resolve_to_the_last_rule() -> None:
    first = _target(id="fix", action="notify")
    last = _target(id="fix", action="optimize")
    rule = _outcome()
    assert _issues(_requirement(first, last, rule), rule) == []
    assert _issues(_requirement(last, first, rule), rule) == [_TRIGGER]


def test_unhashable_trigger_rule_raises_type_error_without_issues() -> None:
    class Unhashable:
        id = "after"
        event = "optimize_failed"
        trigger_rule = ["fix"]
        hours = None

    issues: list[Issue] = []
    with pytest.raises(TypeError):
        _validate_trigger(_requirement(_target()), Unhashable(), _WHERE, issues)
    assert issues == []


def test_event_with_scripted_equality_is_judged_by_that_equality() -> None:
    class AlwaysEqual(str):
        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

        __hash__ = str.__hash__

    rule = _outcome(AlwaysEqual("junk"))
    # membership in the outcome events holds, and in the deploy events too
    assert _issues(_requirement(_target(deploy="if_promoted"), rule), rule) == []
    assert _issues(_requirement(_target(deploy="never"), rule), rule) == [
        _TRIGGER_EVENT
    ]


# --------------------------------------------------------------- monitor triggers


@pytest.mark.parametrize(
    "event", ["episode_opened", "episode_resolved", "episode_withdrawn"]
)
def test_monitor_trigger_events_are_valid_without_hours(event: str) -> None:
    rule = ResponseRule(id="r", event=event, action="notify")
    assert _issues(_requirement(rule), rule) == []


@pytest.mark.parametrize(
    "event",
    ["optimize_failed", "deploy_failed", "bogus", "", "Episode_opened", "UNKNOWN_FOR"],
)
def test_unknown_monitor_event_is_invalid(event: str) -> None:
    rule = ResponseRule(id="r", event=event, action="notify")
    assert _issues(_requirement(rule), rule) == [_EVENT]


def test_unknown_monitor_event_with_hours_reports_event_then_hours() -> None:
    rule = ResponseRule(id="r", event="bogus", action="notify", hours=4)
    assert _issues(_requirement(rule), rule) == [_EVENT, _TRIGGER_HOURS]


@pytest.mark.parametrize(
    "event", ["episode_opened", "episode_resolved", "episode_withdrawn"]
)
@pytest.mark.parametrize("hours", [0, 1, 6, -3, "x", True, 1.5])
def test_hours_on_a_non_unknown_for_event_are_invalid(event: str, hours: Any) -> None:
    rule = ResponseRule(id="r", event=event, action="notify", hours=hours)
    assert _issues(_requirement(rule), rule) == [_TRIGGER_HOURS]


@pytest.mark.parametrize("hours", [None, 0, -1, True, False, 1.0, 6.0, "6", [6], 2.5])
def test_unknown_for_needs_a_real_positive_int(hours: Any) -> None:
    rule = ResponseRule(id="r", event="unknown_for", action="notify", hours=hours)
    assert _issues(_requirement(rule), rule) == [_TRIGGER_HOURS]


@pytest.mark.parametrize("hours", [1, 2, 5])
def test_unknown_for_hours_below_the_window_are_out_of_range(hours: int) -> None:
    rule = ResponseRule(id="r", event="unknown_for", action="notify", hours=hours)
    assert _issues(_requirement(rule, window_hours=6), rule) == [
        Issue(_HOURS_RANGE, "rules[3].trigger.hours", "error")
    ]


@pytest.mark.parametrize("hours", [6, 7, 24, 1000])
def test_unknown_for_hours_at_or_above_the_window_are_valid(hours: int) -> None:
    rule = ResponseRule(id="r", event="unknown_for", action="notify", hours=hours)
    assert _issues(_requirement(rule, window_hours=6), rule) == []


def test_unknown_for_boundary_follows_the_requirement_window() -> None:
    rule = ResponseRule(id="r", event="unknown_for", action="notify", hours=12)
    assert _issues(_requirement(rule, window_hours=12), rule) == []
    assert _issues(_requirement(rule, window_hours=13), rule) == [
        Issue(_HOURS_RANGE, "rules[3].trigger.hours", "error")
    ]


def test_unknown_for_without_hours_does_not_consult_the_window() -> None:
    class NoWindow:
        rules: tuple[ResponseRule, ...] = ()

        @property
        def window_hours(self) -> int:
            raise AssertionError("window must not be read")

    rule = ResponseRule(id="r", event="unknown_for", action="notify", hours=0)
    assert _issues(NoWindow(), rule) == [_TRIGGER_HOURS]


# ----------------------------------------------------------- attribute read traces


class _Obj:
    """Records each attribute read as ``<name>.<attr>``; values come from a mapping."""

    def __init__(self, log: list[str], name: str, **values: Any) -> None:
        self._log = log
        self._name = name
        self._values = values

    def __getattr__(self, item: str) -> Any:
        self._log.append(f"{self._name}.{item}")
        value = self._values[item]
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, _Script):
            return value.read()
        return value


class _Script:
    """One value per read; an exception instance is raised instead of returned."""

    def __init__(self, *values: Any) -> None:
        self._remaining = list(values)

    def read(self) -> Any:
        value = self._remaining.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _scripted(*values: Any) -> _Script:
    return _Script(*values)


def _trace_pair(
    log: list[str],
    *,
    rule: dict[str, Any],
    target: dict[str, Any] | None = None,
    window_hours: Any = 6,
) -> tuple[_Obj, _Obj]:
    target_obj = _Obj(
        log,
        "T",
        **(target or {"id": "fix", "action": "optimize", "deploy": "if_promoted"}),
    )
    rule_obj = _Obj(log, "X", **rule)
    requirement = _Obj(
        log, "R", rules=[target_obj, rule_obj], window_hours=window_hours
    )
    return requirement, rule_obj


def test_read_sequence_for_a_valid_deploy_outcome_trigger() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": "fix",
            "event": "deploy_failed",
            "hours": None,
        },
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == []
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.trigger_rule",
        "X.event",
        "T.id",
        "X.id",
        "T.action",
        "X.event",
        "T.deploy",
        "X.hours",
    ]


def test_read_sequence_stops_at_the_first_failed_outcome_condition() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={"id": "x", "trigger_rule": "gone", "event": "deploy_failed", "hours": 5},
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_TRIGGER, _TRIGGER_HOURS]
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.trigger_rule",
        "X.event",
        "X.hours",
    ]


def test_read_sequence_for_a_non_outcome_event_short_circuits() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": "fix",
            "event": "episode_opened",
            "hours": None,
        },
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_TRIGGER]
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.trigger_rule",
        "X.event",
        "X.hours",
    ]


def test_read_sequence_for_a_plain_monitor_event() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": None,
            "event": "episode_opened",
            "hours": None,
        },
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == []
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.event",
        "X.event",
        "X.hours",
    ]


def test_read_sequence_for_unknown_for_with_valid_hours() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={"id": "x", "trigger_rule": None, "event": "unknown_for", "hours": 9},
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == []
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.event",
        "X.event",
        "X.event",
        "X.hours",
        "R.window_hours",
    ]


def test_read_sequence_for_a_bad_monitor_event_with_hours() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={"id": "x", "trigger_rule": None, "event": "bogus", "hours": 2},
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_EVENT, _TRIGGER_HOURS]
    assert log == [
        "R.rules",
        "T.id",
        "X.id",
        "X.trigger_rule",
        "X.event",
        "X.event",
        "X.event",
        "X.hours",
    ]


def test_scripted_event_reads_are_consumed_one_per_use_in_the_outcome_shape() -> None:
    log: list[str] = []
    # reads: not-in-outcome check, then the deploy-event check
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": "fix",
            "event": _scripted("optimize_failed", "deploy_failed"),
            "hours": None,
        },
        target={"id": "fix", "action": "optimize", "deploy": "never"},
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_TRIGGER_EVENT]


def test_scripted_event_reads_are_consumed_one_per_use_in_the_monitor_shape() -> None:
    log: list[str] = []
    # reads: not-in-EVENTS, != unknown_for, == unknown_for
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": None,
            "event": _scripted("bogus", "unknown_for", "episode_opened"),
            "hours": 3,
        },
    )
    issues: list[Issue] = []
    _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_TRIGGER_HOURS]


def test_raising_trigger_rule_propagates_before_anything_is_reported() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": RuntimeError("trigger_rule failed"),
            "event": "e",
        },
    )
    issues: list[Issue] = []
    with pytest.raises(RuntimeError, match="trigger_rule failed"):
        _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == []
    assert log == ["R.rules", "T.id", "X.id", "X.trigger_rule"]


def test_raising_hours_after_the_trigger_issue_keeps_the_reported_issue() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": "gone",
            "event": "episode_opened",
            "hours": RuntimeError("hours failed"),
        },
    )
    issues: list[Issue] = []
    with pytest.raises(RuntimeError, match="hours failed"):
        _validate_trigger(requirement, rule, _WHERE, issues)
    assert issues == [_TRIGGER]


def test_raising_rules_accessor_propagates_first() -> None:
    log: list[str] = []
    rule = _Obj(log, "X", trigger_rule="fix", event="optimize_failed", hours=None)
    requirement = _Obj(log, "R", rules=RuntimeError("rules failed"))
    issues: list[Issue] = []
    with pytest.raises(RuntimeError, match="rules failed"):
        _validate_trigger(requirement, rule, _WHERE, issues)
    assert log == ["R.rules"]
    assert issues == []


def test_rules_are_read_once_even_when_the_trigger_is_a_monitor_trigger() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": None,
            "event": "episode_opened",
            "hours": None,
        },
    )
    _validate_trigger(requirement, rule, _WHERE, [])
    assert log.count("R.rules") == 1
    assert log[0] == "R.rules"


def test_raise_once_target_action_is_not_retried() -> None:
    log: list[str] = []
    requirement, rule = _trace_pair(
        log,
        rule={
            "id": "x",
            "trigger_rule": "fix",
            "event": "optimize_failed",
            "hours": None,
        },
        target={
            "id": "fix",
            "action": _scripted(RuntimeError("action failed")),
            "deploy": "never",
        },
    )
    with pytest.raises(RuntimeError, match="action failed"):
        _validate_trigger(requirement, rule, _WHERE, [])


def test_target_is_compared_with_none_never_truth_tested() -> None:
    class Falsy:
        id = "fix"
        action = "optimize"
        deploy = "if_promoted"

        def __bool__(self) -> bool:
            raise AssertionError("target truth must not be evaluated")

    class Requirement:
        window_hours = 6

        def __init__(self, rules: list[Any]) -> None:
            self.rules = rules

    rule = _outcome("deploy_failed")
    requirement = Requirement([Falsy(), rule])
    assert _issues(requirement, rule) == []
