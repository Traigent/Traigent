"""``watch``: best-effort snapshot reporting of breach transitions.

This is not an event feed.  The backend returns a bounded breach list, so a
transition that happens and drops out of the snapshot while ``watch`` is
offline can be missed; every poll therefore reports a coverage line.  Durable
production alerts are the backend webhook (a later slice).

Delivery is at-least-once per observed transition: an event is emitted BEFORE
the state is saved, so a crash between the two re-emits it on the next poll.
Emission is never duplicated by two pollers: the read, diff, emit and write of
one poll run under an exclusive lock on the watch state, so a second poller on
the same state directory waits (bounded) and then sees the first one's saved state.

A poll whose hook calls ``poll_once`` for the same agent and state would wait on
itself; it raises ``watch_reentry`` instead.

If saving the state fails (the poll result carries ``persisted=False`` and
``watch_state_not_persisted`` is logged) the next poll replays the same
transitions: a failed save can duplicate delivery, not only a crash.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import (
    E_STATE,
    SEVERITIES,
    RuntimeEvent,
    RuntimeRequirement,
    RuntimeRequirementError,
    normalise_environment,
)

LOGGER_NAME = "traigent.runtime"
BREACH_LIST_CAP = 100
COVERAGE_COMPLETE = "complete"
COVERAGE_INCOMPLETE = "incomplete (has_more)"
COVERAGE_UNKNOWN = "unknown (100 rows, no has_more)"
COVERAGE_UNAVAILABLE = "unknown (read failed)"

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.ERROR}


@dataclass
class PollResult:
    events: list[RuntimeEvent] = field(default_factory=list)
    coverage: str = COVERAGE_COMPLETE
    rows: int = 0
    error: str | None = None
    persisted: bool = True


def coverage_for(rows: int, has_more: bool | None) -> str:
    if has_more is True:
        return COVERAGE_INCOMPLETE
    if has_more is None and rows >= BREACH_LIST_CAP:
        return COVERAGE_UNKNOWN
    return COVERAGE_COMPLETE


def _severity(requirement: RuntimeRequirement, event: str) -> str | None:
    """Highest notify severity declared for ``event``; ``None`` when no rule reports it."""
    found = [
        r.severity
        for r in requirement.rules
        if r.event == event and r.action == "notify" and r.severity in SEVERITIES
    ]
    if not found:
        return None
    return str(max(found, key=SEVERITIES.index))


def _owner(
    requirements: Sequence[RuntimeRequirement], row: dict[str, Any]
) -> RuntimeRequirement | None:
    env = row.get("environment")
    for req in requirements:
        key = req.key()
        if (
            row.get("kind") == key["kind"]
            and row.get("latency_threshold_ms") == key["latency_threshold_ms"]
            and isinstance(env, str)
            and normalise_environment(env) == key["environment"]
        ):
            return req
    return None


async def _await(awaitable: Awaitable[object]) -> object:
    return await awaitable


def _run_hook(outcome: object) -> None:
    """Await a hook's awaitable to completion; refuse when a loop already runs here."""
    if not inspect.isawaitable(outcome):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await(outcome))  # type: ignore[arg-type]
        return
    if inspect.iscoroutine(outcome):
        outcome.close()  # never leave an un-awaited coroutine behind
    raise RuntimeError("on_event awaitable cannot run inside a running event loop")


def _emit(logger: logging.Logger, req: RuntimeRequirement, event: RuntimeEvent) -> None:
    # Log the event kind and the breach id only: the requirement name is customer text.
    logger.log(
        _LEVELS.get(event.severity, logging.INFO),
        "%s breach=%s severity=%s",
        event.event,
        event.breach_id,
        event.severity,
    )
    if req.on_event is not None:
        try:
            _run_hook(req.on_event(event))
        except Exception:  # noqa: BLE001 - a customer hook never breaks the watcher
            logger.warning("on_event_hook_failed event=%s", event.event)


def poll_once(
    requirements: Sequence[RuntimeRequirement],
    *,
    project_id: str,
    agent_id: str,
    client: RuntimeClient | None = None,
    state_dir: Path | None = None,
) -> PollResult:
    """One snapshot: diff against local state, emit transitions, report coverage."""
    logger = logging.getLogger(LOGGER_NAME)
    state = state_dir or manifest.default_state_dir()
    api = client or RuntimeClient(project_id)
    try:
        with manifest.watch_lock(state, agent_id):
            return _poll_locked(requirements, api, state, agent_id, logger)
    except (OSError, RuntimeRequirementError) as exc:
        if isinstance(exc, RuntimeRequirementError) and exc.code != E_STATE:
            raise
        logger.warning("watch state could not be locked")
        return PollResult(coverage=COVERAGE_UNAVAILABLE, error="local_state_invalid")


def _poll_locked(
    requirements: Sequence[RuntimeRequirement],
    api: RuntimeClient,
    state: Path,
    agent_id: str,
    logger: logging.Logger,
) -> PollResult:
    try:
        rows, has_more = api.list_breaches(agent_id)
        saved = manifest.read_watch_state(state, agent_id)
    except RuntimeRequirementError as exc:
        logger.warning("watch read failed: %s", exc.code)
        return PollResult(coverage=COVERAGE_UNAVAILABLE, error=exc.code)

    result = PollResult(coverage=coverage_for(len(rows), has_more), rows=len(rows))
    seen: dict[str, Any] = saved["breaches"]
    for row in rows:
        breach_id = row.get("breach_id")
        req = _owner(requirements, row)
        if not isinstance(breach_id, str) or req is None:
            continue
        outcome = row.get("outcome")
        closed = row.get("closed_at") is not None or outcome is not None
        previous = seen.get(breach_id, {}).get("status")
        if not closed:
            status, event_name = "open", "episode_opened"
        elif outcome == "withdrawn":
            status, event_name = "withdrawn", "episode_withdrawn"
        else:
            status, event_name = "resolved", "episode_resolved"
        # An unseen, already-closed breach is recorded without an event.
        emit = (previous is None and status == "open") or (
            previous == "open" and status != "open"
        )
        seen[breach_id] = {"status": status}
        if not emit:
            continue
        severity = _severity(req, event_name)
        if severity is None:
            continue
        event = RuntimeEvent(
            event=event_name,
            breach_id=breach_id,
            requirement=req.name,
            severity=severity,
            opened_at=_text(row.get("opened_at")),
            closed_at=_text(row.get("closed_at")),
            outcome=_text(outcome),
            withdrawn_reason=_text(row.get("withdrawn_reason")),
        )
        result.events.append(event)
        _emit(logger, req, event)
    try:
        manifest.write_watch_state(state, agent_id, saved)
    except (OSError, RuntimeRequirementError):
        result.persisted = False
        logger.warning("watch_state_not_persisted")
    logger.info("coverage: %s", result.coverage)
    return result


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
