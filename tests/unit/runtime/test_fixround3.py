"""Regression tests for fix round 3 (fresh lease clock, heartbeat cleanup, active-only writes)."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from traigent.runtime import manifest
from traigent.runtime.apply import apply_requirement
from traigent.runtime.respond import respond

from .conftest import AGENT, PROJECT, FakeBackend, breach, head
from .test_reactive_invariants import (
    BREACHES,
    CREATE,
    LIST,
    NEXT,
    RESTORE,
    _fn_and_req,
    _kw,
)


@pytest.fixture
def live(backend: FakeBackend) -> FakeBackend:
    backend.on(*CREATE, 201, head())
    backend.on(*LIST, 200, {"items": [head()]})
    backend.on(*BREACHES, 200, {"items": [breach("b1")], "has_more": False})
    backend.on(*NEXT, 200, RESTORE)
    return backend


async def test_lease_starts_after_the_lock_is_acquired(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    lease = timedelta(seconds=0.4)
    path = manifest._occ_path(state_dir, AGENT)
    held = threading.Event()
    released: list[object] = []

    def hold() -> None:
        with manifest._ledger_lock(state_dir, path):
            held.set()
            time.sleep(0.5)  # longer than the lease
            released.append(manifest.utcnow())

    seen: dict[str, object] = {}

    async def optimize(**_kw: object) -> object:
        record = manifest.read_occurrence(state_dir, AGENT, "b1|fix|0", lease=lease)
        seen.update(record or {})
        return type("R", (), {"optimization_id": "r"})()

    fn.optimize = optimize
    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(5)
    try:
        await respond(fn, req, **_kw(state_dir, occurrence_lease=lease))
    finally:
        thread.join(5)
    stamp = released[0]
    assert seen["started_at"] >= stamp.isoformat()  # type: ignore[union-attr,operator]
    assert seen["heartbeat_at"] >= stamp.isoformat()  # type: ignore[union-attr,operator]
    assert seen["status"] == "started"  # not expired for a second controller


async def test_cancelling_respond_leaves_no_heartbeat_task(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    lease = timedelta(seconds=0.4)
    entered = asyncio.Event()

    async def hang(**_kw: object) -> object:
        entered.set()
        await asyncio.sleep(30)
        return None

    fn.optimize = hang
    task = asyncio.create_task(
        respond(fn, req, **_kw(state_dir, occurrence_lease=lease))
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    leftovers = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task()
        and "_heartbeat" in repr(t.get_coro())
        and not t.done()
    ]
    assert leftovers == []
    await asyncio.sleep(0.6)  # no beat renews it: the lease expires
    record = manifest.read_occurrence(state_dir, AGENT, "b1|fix|0", lease=lease)
    assert record is not None and record["status"] == "failed"


def test_late_writes_of_an_expired_rolled_back_attempt_are_dropped(
    state_dir: Path,
) -> None:
    t0 = manifest.utcnow()
    lease = timedelta(seconds=10)
    assert manifest.reserve_occurrence(
        state_dir, AGENT, "k", {"run_id": None}, now=t0, lease=lease
    )
    later = t0 + timedelta(seconds=30)  # attempt 1 expired
    assert manifest.reserve_occurrence(
        state_dir, AGENT, "k", {}, attempt=2, now=later, lease=lease
    )
    assert manifest.release_occurrence(state_dir, AGENT, "k", attempt=2)
    # the restored record is failed with attempts == 1; attempt 1's late writes drop
    assert not manifest.update_occurrence(
        state_dir, AGENT, "k", {"heartbeat_at": later.isoformat()}, attempt=1
    )
    assert not manifest.update_occurrence(
        state_dir, AGENT, "k", {"run_id": "old", "finished": True}, attempt=1
    )
    record = manifest.read_occurrence(state_dir, AGENT, "k", now=later, lease=lease)
    assert record["status"] == "failed" and not record.get("finished")
