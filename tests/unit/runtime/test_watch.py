"""watch: best-effort snapshot diffing with an honest coverage line."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

import traigent
from traigent.runtime.model import RuntimeEvent
from traigent.runtime.watch import poll_once

from .conftest import AGENT, PROJECT, FakeBackend, breach

LIST = ("GET", f"/agents/{AGENT}/production-breaches")


def _declare(mode: str = "alert", hook=None, **kw):
    kwargs = {
        "latency_ms": 2000,
        "max_violation_rate": 0.05,
        "on_breach": mode,
        "on_event": hook,
    }
    if mode == "fix":
        kwargs["max_fix_cost_usd"] = 1.0
    kwargs.update(kw)

    @traigent.requirement(**kwargs)
    def fn() -> None: ...

    return list(fn.requirements)


def _poll(reqs):
    return poll_once(reqs, project_id=PROJECT, agent_id=AGENT)


def test_open_resolve_and_withdraw_transitions_emit_once(backend: FakeBackend) -> None:
    seen: list[RuntimeEvent] = []
    reqs = _declare("alert", hook=seen.append)

    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    first = _poll(reqs)
    assert [(e.event, e.severity, e.breach_id) for e in first.events] == [
        ("episode_opened", "warning", "b1")
    ]
    assert first.coverage == "complete" and seen == first.events
    assert _poll(reqs).events == []  # unchanged snapshot: nothing new

    done = breach("b1", outcome="resolved", closed_at="2026-10-21T05:00:00Z")
    backend.on(*LIST, 200, {"items": [done], "has_more": False})
    resolved = _poll(reqs).events
    assert [(e.event, e.severity, e.outcome) for e in resolved] == [
        ("episode_resolved", "info", "resolved")
    ]

    backend.on(*LIST, 200, {"items": [breach("b2")], "has_more": False})
    _poll(reqs)
    gone = breach(
        "b2",
        outcome="withdrawn",
        withdrawn_reason="requirement_retired",
        closed_at="2026-10-21T06:00:00Z",
    )
    backend.on(*LIST, 200, {"items": [gone], "has_more": False})
    withdrawn = _poll(reqs).events
    assert [(e.event, e.withdrawn_reason) for e in withdrawn] == [
        ("episode_withdrawn", "requirement_retired")
    ]


def test_breach_first_seen_already_closed_emits_nothing(backend: FakeBackend) -> None:
    closed = breach("old", outcome="resolved", closed_at="2026-10-20T00:00:00Z")
    backend.on(*LIST, 200, {"items": [closed], "has_more": False})
    assert _poll(_declare()).events == []


def test_log_policy_reports_info_and_breaches_of_other_keys_are_ignored(
    backend: FakeBackend,
) -> None:
    backend.on(
        *LIST, 200, {"items": [breach("a"), breach("other", latency_threshold_ms=9999)]}
    )
    events = _poll(_declare("log")).events
    assert [(e.breach_id, e.severity) for e in events] == [("a", "info")]


@pytest.mark.parametrize(
    ("rows", "has_more", "coverage"),
    [
        (1, True, "incomplete (has_more)"),
        (1, False, "complete"),
        (99, None, "complete"),
        (100, None, "unknown (100 rows, no has_more)"),
        (100, False, "complete"),
        (100, True, "incomplete (has_more)"),
    ],
)
def test_coverage_line(
    backend: FakeBackend, rows: int, has_more, coverage: str
) -> None:
    items = [breach(f"b{i}") for i in range(rows)]
    payload = (
        {"items": items} if has_more is None else {"items": items, "has_more": has_more}
    )
    backend.on(*LIST, 200, payload)
    assert _poll(_declare()).coverage == coverage


def test_hook_exception_is_logged_not_raised(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_e):
        raise RuntimeError("hook failed")

    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    with caplog.at_level(logging.INFO, logger="traigent.runtime"):
        result = _poll(_declare("alert", hook=boom))
    assert len(result.events) == 1
    assert any("on_event_hook_failed" in r.getMessage() for r in caplog.records)
    assert any(
        r.levelno == logging.WARNING and "episode_opened" in r.getMessage()
        for r in caplog.records
    )


def test_backend_failure_is_reported_not_raised(backend: FakeBackend) -> None:
    backend.on(*LIST, 500, {"error_code": "internal"})
    result = _poll(_declare())
    assert result.error == "backend_error" and result.coverage.startswith("unknown")


def test_state_persists_across_processes(backend: FakeBackend, state_dir: Path) -> None:
    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    assert len(_poll(_declare()).events) == 1
    assert (state_dir / "runtime").exists()
    assert _poll(_declare()).events == []  # a fresh requirement object, same state dir


def test_hook_failure_log_never_carries_the_exception_text(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    sentinel = "SECRET_PROMPT_TEXT_9c1"

    def boom(_e):
        raise RuntimeError(sentinel)

    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    with caplog.at_level(logging.DEBUG, logger="traigent.runtime"):
        _poll(_declare("alert", hook=boom))
    assert sentinel not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_event_log_line_never_carries_the_requirement_name(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    canary = "CANARY_REQ_NAME_7f2e"
    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    with caplog.at_level(logging.DEBUG, logger="traigent.runtime"):
        result = _poll(_declare("alert", name=canary))
    assert [e.event for e in result.events] == ["episode_opened"]
    assert "episode_opened" in caplog.text
    assert canary not in caplog.text


def test_async_on_event_hook_is_awaited(backend: FakeBackend) -> None:
    seen: list[RuntimeEvent] = []

    async def hook(event: RuntimeEvent) -> None:
        await asyncio.sleep(0)
        seen.append(event)

    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    result = _poll(_declare("alert", hook=hook))
    assert seen == result.events and len(seen) == 1


def test_async_on_event_hook_failure_is_fixed_code_without_text(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    sentinel = "SECRET_ASYNC_TEXT_44b1"

    async def boom(_event: RuntimeEvent) -> None:
        await asyncio.sleep(0)
        raise RuntimeError(sentinel)

    backend.on(*LIST, 200, {"items": [breach("b1")], "has_more": False})
    with caplog.at_level(logging.DEBUG, logger="traigent.runtime"):
        result = _poll(_declare("alert", hook=boom))
    assert len(result.events) == 1
    assert any(
        r.levelno == logging.WARNING and "on_event_hook_failed" in r.getMessage()
        for r in caplog.records
    )
    assert sentinel not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
