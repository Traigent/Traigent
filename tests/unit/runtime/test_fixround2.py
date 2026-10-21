"""Regression tests for fix round 2 (leases, fencing, state versioning, locks, episodes)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import timedelta
from pathlib import Path

import pytest

import traigent
from traigent.runtime import manifest
from traigent.runtime.apply import apply_requirement
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.respond import respond
from traigent.runtime.watch import poll_once

from .conftest import AGENT, PROJECT, FakeBackend, breach, head
from .test_reactive_invariants import (
    BREACHES,
    Clock,
    _fn_and_req,
    _kw,
    CREATE,
    LIST,
    NEXT,
    RESTORE,
)


@pytest.fixture
def live(backend: FakeBackend) -> FakeBackend:
    backend.on(*CREATE, 201, head())
    backend.on(*LIST, 200, {"items": [head()]})
    backend.on(*BREACHES, 200, {"items": [breach("b1")], "has_more": False})
    backend.on(*NEXT, 200, RESTORE)
    return backend


def _code(exc: pytest.ExceptionInfo[RuntimeRequirementError]) -> str:
    return str(exc.value.code)


# --- 1: a running controller keeps its lease; outcome writes are fenced -----------------


def test_heartbeat_keeps_a_running_reservation_alive(state_dir: Path) -> None:
    now = manifest.utcnow()
    old = (now - timedelta(hours=30)).isoformat()
    manifest.reserve_occurrence(state_dir, AGENT, "k", {"started_at": old})
    # control half: with no heartbeat the 30h-old start is an expired lease
    assert manifest.read_occurrence(state_dir, AGENT, "k", now=now)["status"] == (
        "failed"
    )
    assert manifest.update_occurrence(
        state_dir, AGENT, "k", {"heartbeat_at": now.isoformat()}, attempt=1
    )
    assert manifest.read_occurrence(state_dir, AGENT, "k", now=now)["status"] == (
        "started"
    )


async def test_respond_renews_heartbeat_while_optimize_runs(
    live: FakeBackend,
    state_dir: Path,
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    lease = timedelta(seconds=0.4)
    seen: list[str] = []

    async def slow(**_kw: object) -> object:
        await asyncio.sleep(0.9)  # > 2 leases
        seen.append(
            manifest.read_occurrence(state_dir, AGENT, "b1|fix|0", lease=lease)[
                "status"
            ]
        )
        return type("R", (), {"optimization_id": "r"})()

    fn.optimize = slow
    await respond(fn, req, **_kw(state_dir, occurrence_lease=lease))
    assert seen == ["started"]


def test_stale_attempt_cannot_write_an_outcome(state_dir: Path) -> None:
    assert manifest.reserve_occurrence(state_dir, AGENT, "k", {"run_id": None})
    manifest.update_occurrence(state_dir, AGENT, "k", {"status": "failed"}, attempt=1)
    assert manifest.reserve_occurrence(state_dir, AGENT, "k", {}, attempt=2)
    # attempt 1's controller finally reports: fenced out
    assert not manifest.update_occurrence(
        state_dir, AGENT, "k", {"run_id": "old", "finished": True}, attempt=1
    )
    record = manifest.read_occurrence(state_dir, AGENT, "k")
    assert record["attempts"] == 2 and not record.get("finished")


async def test_respond_drops_a_fenced_out_outcome(
    live: FakeBackend,
    state_dir: Path,
    caplog,
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    caplog.set_level(logging.WARNING, logger="traigent.runtime")

    async def superseded(**_kw: object) -> object:
        path = manifest._occ_path(state_dir, AGENT)
        doc = json.loads(path.read_text())
        doc["b1|fix|0"]["attempts"] = 2  # a newer approved attempt took over
        path.write_text(json.dumps(doc))
        return type("R", (), {"optimization_id": "r"})()

    fn.optimize = superseded
    # Fenced out mid-run means the lease is lost: the run is stopped, never reported finished.
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert (err.value.code, err.value.path) == ("hold", "lease")
    assert not manifest.read_occurrence(state_dir, AGENT, "b1|fix|0").get("finished")


# --- 2 + 7: versioned local state ---------------------------------------------------------


def test_state_written_by_an_earlier_version_is_refused(state_dir: Path) -> None:
    state_dir.mkdir()
    (state_dir / "tvl-token-map.json").write_text(
        json.dumps({"format": "tvl-token-map/1", "tokens": []})
    )
    with pytest.raises(RuntimeRequirementError) as err:
        manifest.ensure_tokens(state_dir, [("n", "x")])
    assert _code(err) == "state_version_unsupported"


def test_marker_with_an_older_version_is_refused(state_dir: Path) -> None:
    state_dir.mkdir()
    (state_dir / manifest.STATE_MARKER).write_text(
        json.dumps({"sdk": "python", "state_version": 1})
    )
    with pytest.raises(RuntimeRequirementError) as err:
        manifest.read_occurrence(state_dir, AGENT, "k")
    assert _code(err) == "state_version_unsupported"


# --- 3: one SDK per state directory ----------------------------------------------------


def test_directory_owned_by_the_js_sdk_is_refused(state_dir: Path) -> None:
    state_dir.mkdir()
    (state_dir / manifest.STATE_MARKER).write_text(
        json.dumps({"sdk": "js", "state_version": manifest.STATE_VERSION})
    )
    with pytest.raises(RuntimeRequirementError) as err:
        manifest.reserve_occurrence(state_dir, AGENT, "k", {})
    assert _code(err) == "state_owned_by_other_sdk"


def test_first_use_writes_the_owner_marker_and_default_dir_names_the_sdk(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert manifest.reserve_occurrence(state_dir, AGENT, "k", {})
    marker = json.loads((state_dir / manifest.STATE_MARKER).read_text())
    assert marker == {"sdk": "python", "state_version": manifest.STATE_VERSION}
    monkeypatch.delenv(manifest.STATE_DIR_ENV, raising=False)
    assert manifest.default_state_dir().name == "python"


# --- 5': approval expiring while the reservation waits -------------------------------------


async def test_expiry_during_reservation_releases_it_and_does_not_start(
    live: FakeBackend,
    state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    clock = Clock()
    real = manifest.reserve_occurrence

    def slow_reserve(*a: object, **k: object) -> bool:
        won = real(*a, **k)  # type: ignore[arg-type]
        clock.now += timedelta(hours=25)  # the lock wait outlived the approval
        return bool(won)

    monkeypatch.setattr(manifest, "reserve_occurrence", slow_reserve)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir, clock=clock))
    assert _code(err) == "approval_expired"
    fn.optimize.assert_not_awaited()
    # a never-started fix consumed nothing
    assert manifest.read_occurrence(state_dir, AGENT, "b1|fix|0") is None


def test_ledger_lock_wait_is_bounded(state_dir: Path) -> None:
    path = manifest._occ_path(state_dir, AGENT)
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with manifest._ledger_lock(state_dir, path):
            started.set()
            release.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    started.wait(5)
    try:
        with pytest.raises(RuntimeRequirementError) as err:
            with manifest._ledger_lock(state_dir, path, timeout=0.2):
                pass
        assert err.value.code == "state_lock_timeout"
    finally:
        release.set()
        thread.join(5)


# --- 8: re-entrant poll -------------------------------------------------------------------


def test_poll_from_inside_the_hook_is_refused_not_deadlocked(
    backend: FakeBackend,
) -> None:
    backend.on(
        *BREACHES,
        200,
        {
            "items": [
                {
                    "breach_id": "b1",
                    "kind": "latency_violation_rate",
                    "latency_threshold_ms": 2000,
                    "environment": "production",
                    "outcome": None,
                    "closed_at": None,
                    "opened_at": "2026-10-21T01:00:00Z",
                }
            ],
            "has_more": False,
        },
    )
    inner: list[object] = []

    def hook(_event: object) -> None:
        try:
            poll_once(reqs, project_id=PROJECT, agent_id=AGENT)
        except RuntimeRequirementError as exc:
            inner.append(exc.code)

    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="alert", on_event=hook
    )
    def fn() -> None: ...

    reqs = list(fn.requirements)
    outer = poll_once(reqs, project_id=PROJECT, agent_id=AGENT)
    assert inner == ["watch_reentry"] and len(outer.events) == 1


# --- 9': episode-wide exclusion ---------------------------------------------------------------


def test_second_rule_on_the_same_episode_is_refused_while_the_first_runs(
    state_dir: Path,
) -> None:
    assert manifest.reserve_occurrence(state_dir, AGENT, "b1|fix|0", {})
    with pytest.raises(RuntimeRequirementError) as err:
        manifest.reserve_occurrence(state_dir, AGENT, "b1|repair|0", {})
    assert _code(err) == "episode_fix_running"
    # another episode is independent; a finished run does not block
    assert manifest.reserve_occurrence(state_dir, AGENT, "b2|repair|0", {})
    manifest.update_occurrence(
        state_dir, AGENT, "b1|fix|0", {"finished": True}, attempt=1
    )
    assert manifest.reserve_occurrence(state_dir, AGENT, "b1|repair|0", {})


# control: passes without the fix (the old code had no episode exclusion at all)
def test_a_stale_heartbeat_does_not_block_the_episode(state_dir: Path) -> None:
    now = manifest.utcnow()
    old = (now - timedelta(hours=30)).isoformat()
    manifest.reserve_occurrence(state_dir, AGENT, "b1|fix|0", {"started_at": old})
    assert manifest.reserve_occurrence(state_dir, AGENT, "b1|repair|0", {}, now=now)


# --- 10: a failed save is visible -----------------------------------------------------------------


def test_save_failure_reports_persisted_false(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    backend.on(*BREACHES, 200, {"items": [], "has_more": False})
    caplog.set_level(logging.WARNING, logger="traigent.runtime")

    def boom(*_a: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manifest, "write_watch_state", boom)

    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05, on_breach="alert")
    def fn() -> None: ...

    result = poll_once(list(fn.requirements), project_id=PROJECT, agent_id=AGENT)
    assert result.persisted is False and "watch_state_not_persisted" in caplog.text
