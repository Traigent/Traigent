"""Regression tests for fix round 4 (fail-closed lock, exact numbers, token-map lock, guard order)."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import traigent
from traigent.runtime import manifest
from traigent.runtime.apply import _equal, profile_diff
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.apply import apply_requirement
from traigent.runtime.client import RuntimeClient
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


def test_missing_flock_fails_closed_before_the_protected_section(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    monkeypatch.setattr(manifest, "fcntl", None)
    entered: list[bool] = []
    with pytest.raises(RuntimeRequirementError) as err:
        with manifest._ledger_lock(state_dir, state_dir / "x.json"):
            entered.append(True)
    assert err.value.code == "state_lock_unavailable"
    assert entered == []
    with pytest.raises(RuntimeRequirementError) as err2:
        manifest.ensure_tokens(state_dir, [("requirement", "a")])
    assert err2.value.code == "state_lock_unavailable"
    assert not (state_dir / "tvl-token-map.json").exists()
    with pytest.raises(RuntimeRequirementError) as err3:
        manifest.reserve_occurrence(state_dir, AGENT, "b|fix|0", {"run_id": None})
    assert err3.value.code == "state_lock_unavailable"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (1e-12, 1e-14, False),
        (1e-10, 1e-12, False),
        (1e-12, 1e-12, True),
        (1, 1.0, True),
        (0.05, 0.05000000001, False),
    ],
)
def test_number_equality_is_exact_after_canonical_normalisation(
    a: float, b: float, same: bool
) -> None:
    assert _equal(a, b) is same
    req = SimpleNamespace(profile=lambda: {"false_alarm_budget": a})
    assert (profile_diff(req, {"false_alarm_budget": b}) == []) is same  # type: ignore[arg-type]


def test_concurrent_token_minting_loses_no_mapping(state_dir: Path) -> None:
    manifest.check_state_dir(
        state_dir, create=True
    )  # marker race is separate from this test
    workers = 12
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def mint(i: int) -> None:
        try:
            barrier.wait(10)
            manifest.ensure_tokens(state_dir, [("requirement", f"name{i}")])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=mint, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    doc = manifest.ensure_tokens(state_dir, [], persist=False)
    locals_ = {t["local"] for t in doc["tokens"]}
    assert locals_ == {f"name{i}" for i in range(workers)}


async def test_non_optimizable_function_consumes_nothing(state_dir: Path) -> None:
    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="fix", max_fix_cost_usd=2.0
    )
    def plain() -> None: ...

    assert not hasattr(plain, "optimize")
    asked: list[object] = []
    kwargs = _kw(state_dir, confirm=lambda f: asked.append(f) or True)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(plain, plain.requirements[0], **kwargs)
    assert err.value.code == "function_not_optimizable"
    assert asked == []  # no approval requested
    assert not list(state_dir.rglob("*occurr*")) if state_dir.exists() else True


class _Clock:
    def __init__(self) -> None:
        self.now: datetime = manifest.utcnow()

    def __call__(self) -> datetime:
        return self.now


def _live(backend: FakeBackend, *ids: str, step: dict | None = None) -> FakeBackend:
    backend.on(*CREATE, 201, head())
    backend.on(*LIST, 200, {"items": [head()]})
    backend.on(
        *BREACHES,
        200,
        {"items": [breach(i) for i in ids or ("b1",)], "has_more": False},
    )
    backend.on(*NEXT, 200, step or RESTORE)
    return backend


def _applied(state_dir: Path):
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    return fn, req


def _occ(state_dir: Path, key: str = "b1|fix|0") -> dict:
    return (
        manifest.read_occurrence(state_dir, AGENT, key, lease=timedelta(days=3650))
        or {}
    )


@pytest.mark.backend_online
async def test_sync_run_outliving_the_lease_is_not_recorded_finished(
    backend: FakeBackend, state_dir: Path
) -> None:
    _live(backend)
    fn, req = _applied(state_dir)
    clock = _Clock()

    async def optimize(**_kw: object) -> object:
        clock.now += timedelta(seconds=5)  # the loop was blocked past the lease
        return type("R", (), {"optimization_id": "late"})()

    fn.optimize = optimize
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            **_kw(state_dir, clock=clock, occurrence_lease=timedelta(seconds=0.4)),
        )
    assert (err.value.code, err.value.path) == ("hold", "lease")
    assert not _occ(state_dir).get("finished")
    assert not _occ(state_dir).get("run_id")


@pytest.mark.backend_online
async def test_refused_heartbeat_cancels_the_run(
    backend: FakeBackend, state_dir: Path
) -> None:
    _live(backend)
    fn, req = _applied(state_dir)
    clock = _Clock()
    cancelled: list[bool] = []

    async def optimize(**_kw: object) -> object:
        clock.now += timedelta(seconds=5)  # lease lapses while the run is in flight
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return None

    fn.optimize = optimize
    with pytest.raises(RuntimeRequirementError) as err:
        await asyncio.wait_for(
            respond(
                fn,
                req,
                **_kw(state_dir, clock=clock, occurrence_lease=timedelta(seconds=0.04)),
            ),
            10,
        )
    assert (err.value.code, err.value.path) == ("hold", "lease")
    assert cancelled == [True]
    assert not _occ(state_dir).get("finished")


def test_heartbeat_does_not_resurrect_an_expired_lease(state_dir: Path) -> None:
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    assert manifest.reserve_occurrence(
        state_dir, AGENT, "b1|fix|0", {"run_id": None}, attempt=1, clock=lambda: t0
    )
    lease = timedelta(seconds=1)
    late = t0 + timedelta(seconds=10)
    assert (
        manifest.heartbeat_occurrence(
            state_dir, AGENT, "b1|fix|0", attempt=1, lease=lease, clock=lambda: late
        )
        is None
    )
    assert _occ(state_dir)["heartbeat_at"] != late.isoformat()
    soon = t0 + timedelta(milliseconds=500)
    assert (
        manifest.heartbeat_occurrence(
            state_dir, AGENT, "b1|fix|0", attempt=1, lease=lease, clock=lambda: soon
        )
        == soon
    )


def test_heartbeat_samples_the_clock_after_the_lock_wait(state_dir: Path) -> None:
    t0 = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    lease = timedelta(seconds=1)
    assert manifest.reserve_occurrence(
        state_dir, AGENT, "b1|fix|0", {"run_id": None}, attempt=1, clock=lambda: t0
    )
    path = manifest._occ_path(state_dir, AGENT)
    holding, late = threading.Event(), threading.Event()
    results: list[datetime | None] = []

    def clock() -> datetime:  # before expiry until the lock holder lets go
        return t0 + (
            timedelta(seconds=10) if late.is_set() else timedelta(milliseconds=500)
        )

    def hold() -> None:
        with manifest._ledger_lock(state_dir, path):
            holding.set()
            threading.Event().wait(0.3)
            late.set()  # expiry passes while the beat is still waiting for the lock

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(5)
    beat = threading.Thread(
        target=lambda: results.append(
            manifest.heartbeat_occurrence(
                state_dir, AGENT, "b1|fix|0", attempt=1, lease=lease, clock=clock
            )
        )
    )
    beat.start()
    holder.join(10)
    beat.join(10)
    assert results == [None]  # judged at the post-wait time, so refused
    assert _occ(state_dir)["heartbeat_at"] == t0.isoformat()


@pytest.mark.backend_online
async def test_heartbeat_errors_stop_counting_as_ownership_after_the_deadline(
    backend: FakeBackend, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live(backend)
    fn, req = _applied(state_dir)
    clock = _Clock()

    def broken(*_a: object, **_k: object) -> datetime:
        raise OSError("disk")

    monkeypatch.setattr(manifest, "heartbeat_occurrence", broken)
    cancelled: list[bool] = []

    async def optimize(**_kw: object) -> object:
        clock.now += timedelta(seconds=5)  # past the last verified deadline
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return None

    fn.optimize = optimize
    with pytest.raises(RuntimeRequirementError) as err:
        await asyncio.wait_for(
            respond(
                fn,
                req,
                **_kw(state_dir, clock=clock, occurrence_lease=timedelta(seconds=0.04)),
            ),
            10,
        )
    assert (err.value.code, err.value.path) == ("hold", "lease")
    assert cancelled == [True]


@pytest.mark.backend_online
async def test_failed_final_ownership_check_never_authorises_success(
    backend: FakeBackend, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live(backend)
    fn, req = _applied(state_dir)

    def broken(*_a: object, **_k: object) -> datetime:
        raise OSError("disk")

    monkeypatch.setattr(manifest, "heartbeat_occurrence", broken)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert (err.value.code, err.value.path) == ("hold", "lease")
    assert not _occ(state_dir).get("finished")


@pytest.mark.backend_online
async def test_a_raising_run_is_drained_even_when_the_lease_was_lost(
    backend: FakeBackend, state_dir: Path
) -> None:
    import gc

    _live(backend)
    fn, req = _applied(state_dir)
    clock = _Clock()
    unretrieved: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _l, ctx: unretrieved.append(ctx)
    )

    async def optimize(**_kw: object) -> object:
        clock.now += timedelta(seconds=5)  # blocked past the lease, then fails
        raise RuntimeError("provider down")

    fn.optimize = optimize
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            **_kw(state_dir, clock=clock, occurrence_lease=timedelta(seconds=0.4)),
        )
    assert (err.value.code, err.value.path) == ("hold", "lease")
    gc.collect()
    await asyncio.sleep(0)
    assert unretrieved == []


@pytest.mark.backend_online
async def test_final_write_fencing_drops_the_stale_outcome(
    backend: FakeBackend, state_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    import json
    import logging

    _live(backend)
    fn, req = _applied(state_dir)
    caplog.set_level(logging.WARNING, logger="traigent.runtime")
    real = manifest.update_occurrence

    def superseded(
        state, agent, key, info, **kw
    ):  # a newer attempt takes over just before the write
        if info.get("finished"):
            path = manifest._occ_path(state, agent)
            doc = json.loads(path.read_text())
            doc[key]["attempts"] = 2
            path.write_text(json.dumps(doc))
        return real(state, agent, key, info, **kw)

    monkeypatch.setattr(manifest, "update_occurrence", superseded)
    await respond(fn, req, **_kw(state_dir))
    assert "stale_outcome_dropped" in caplog.text
    record = _occ(state_dir)
    assert record["attempts"] == 2
    assert not record.get("finished")


@pytest.mark.backend_online
async def test_named_breach_is_the_one_fixed(
    backend: FakeBackend, state_dir: Path
) -> None:
    named = {
        "step": {"code": "RESTORE_PRODUCTION_REQUIREMENT"},
        "basis": {"production_breach": {"open_breach_ref": "b2"}},
    }
    _live(backend, "b1", "b2", step=named)
    fn, req = _applied(state_dir)
    result = await respond(fn, req, **_kw(state_dir))
    assert result.breach_id == "b2"


@pytest.mark.backend_online
async def test_unnamed_breach_ref_holds(backend: FakeBackend, state_dir: Path) -> None:
    other = {
        "step": {"code": "RESTORE_PRODUCTION_REQUIREMENT"},
        "basis": {"production_breach": {"open_breach_ref": "b9"}},
    }
    _live(backend, "b1", "b2", step=other)
    fn, req = _applied(state_dir)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert (err.value.code, err.value.path) == ("hold", "breach")


@pytest.mark.backend_online
def test_malformed_list_item_fails_closed(backend: FakeBackend) -> None:
    backend.on(*LIST, 200, {"items": [head(), "junk"]})
    backend.on(*BREACHES, 200, {"items": [breach("b1"), 7], "has_more": False})
    client = RuntimeClient(PROJECT)
    for call in (client.list_requirements, lambda: client.list_breaches(AGENT)):
        with pytest.raises(RuntimeRequirementError) as err:
            call()
        assert (err.value.code, err.value.path) == ("backend_error", "response")


@pytest.mark.backend_online
async def test_cancelled_run_is_marked_failed_and_cancellation_propagates(
    backend: FakeBackend, state_dir: Path
) -> None:
    _live(backend)
    fn, req = _applied(state_dir)
    started = asyncio.Event()

    async def optimize(**_kw: object) -> object:
        started.set()
        await asyncio.sleep(30)
        return None

    fn.optimize = optimize
    task = asyncio.ensure_future(respond(fn, req, **_kw(state_dir)))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _occ(state_dir)["status"] == "failed"
