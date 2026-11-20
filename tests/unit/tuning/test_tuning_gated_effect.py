"""One gated step for every effect: the order of authenticate, recheck, one clock
sample, deadline check and effect; the clock built at operation entry; the persisted
instant; and the actions this release runs."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.controller import Controller, traigent_optimize_fn
from traigent.tuning.errors import E_UNSUPPORTED_IN_RELEASE, TuningError

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    T0,
    FakeOptimizer,
    _deploy_awaiting,
    _holdout,
    at,
    controller,
    e_ref,
    head,
    make,
    module,
    state,
)
from .test_tuning_effect_time import _clock, _scope_end


def _platform(hook: Any) -> ApproverAuthority:
    return ApproverAuthority(frozenset(APPROVERS), mode="platform", authenticator=hook)


def _seconds(moment: Any) -> int:
    return int((moment - T0).total_seconds())


def _refusals(c: Controller) -> list[str]:
    return [str(f.get("reason")) for f in c.controller_facts()]


def _started(c: Controller) -> list[Any]:
    return [
        ln
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "started"
    ]


class _FakeSdkFunction:
    """Just enough of an optimized function for the adapter to copy and run."""

    def __init__(self, started: list[int]) -> None:
        self._csm = object()
        self._started = started

    def _initialize_state_and_wrapper(self) -> None:
        self._csm = object()

    def _load_dataset(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def optimize(self, **kwargs: Any) -> Any:
        self._started.append(1)
        return {"candidates": []}


@contextlib.contextmanager
def _artifact_lock_held() -> Iterator[None]:
    controller_module._ARTIFACT_ENV_LOCK.acquire()
    try:
        yield
    finally:
        with contextlib.suppress(RuntimeError):
            controller_module._ARTIFACT_ENV_LOCK.release()


def _wait_for_marker(tmp_path: Path) -> None:
    deadline = time.monotonic() + 20
    folder = tmp_path / "store" / "attempts"
    while time.monotonic() < deadline:
        if list(folder.glob("*.json")):
            time.sleep(0.2)  # the step is now waiting for the artifact lock
            return
        time.sleep(0.02)
    raise AssertionError("the step never reached the start marker")


# -- S: the scope ends while another optimization holds the artifact lock ----------
def test_a_scope_that_ends_while_waiting_for_the_artifact_lock_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    started: list[int] = []
    c = controller(
        tmp_path,
        traigent_optimize_fn(_FakeSdkFunction(started)),
        now=at(0),
    )
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    reports: list[Any] = []
    with _artifact_lock_held():
        worker = threading.Thread(target=lambda: reports.append(c.step()))
        worker.start()
        _wait_for_marker(tmp_path)
        clock[0] = _scope_end() + timedelta(seconds=1)  # expires during the wait
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert started == [] and _started(c) == []
    assert reports[0].refused and reports[0].refused[0].endswith(":scope_expired")


def test_a_scope_that_ends_during_the_authenticator_starts_no_injected_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    calls = [0]

    def hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] == 2:  # the final authenticator call
            clock[0] = _scope_end() + timedelta(seconds=1)
        return True

    c = controller(tmp_path, opt, now=at(0), authority=_platform(hook))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    report = c.step()
    assert opt.calls == [] and _started(c) == []
    assert report.refused and report.refused[0].endswith(":scope_expired")


# -- S and R: a revocation during the history read is seen -----------------------------
def test_an_approver_revoked_during_the_history_read_does_not_start_the_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    revoked = [False]
    c = controller(
        tmp_path,
        opt,
        now=at(0),
        authority=_platform(lambda approver, context: not revoked[0]),
    )
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = Controller._last_persisted_instant

    def revoke_during_read(self: Controller) -> int:
        last = real(self)
        revoked[0] = True  # the last history read of the preparation
        return last

    monkeypatch.setattr(Controller, "_last_persisted_instant", revoke_during_read)
    report = c.step(at(10))
    assert opt.calls == [] and _started(c) == []
    assert report.refused and report.refused[0].endswith(":approver_not_authorized")


# -- R: the serving code version is read after the final authenticator -----------------
# -- R: the instant is the one sampled AFTER the final authenticator -------------------
# -- the clock starts at operation entry -----------------------------------------------
def test_a_lock_wait_advances_the_first_scheduling_read_of_a_fixed_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer(), now=at(0))
    ticks = [100.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: ticks[0])
    real = Controller._locked

    @contextlib.contextmanager
    def slow_lock(self: Controller) -> Iterator[None]:
        with real(self):
            ticks[0] += 90.0  # waited this long behind another writer
            yield

    monkeypatch.setattr(Controller, "_locked", slow_lock)
    c.step(at(10))
    tick = next(ln for ln in c._lines() if ln["category"] == "tick")
    assert tick["instant"] == _seconds(at(10)) + 90


@pytest.mark.parametrize("entry", ["step", "record_window"])
def test_a_fixed_instant_is_refused_by_every_journal_writing_entry_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    c = _deploy_awaiting(tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "production")
    c.authority = _platform(lambda approver, context: True)
    calls = {
        "step": lambda: c.step(at(31)),
        "record_window": lambda: c.record_window({}, at(31)),
    }
    before = (tmp_path / "store" / "history.jsonl").read_bytes()
    with pytest.raises(TuningError) as err:
        calls[entry]()
    assert err.value.code == "E_FIXED_CLOCK_IN_PRODUCTION"
    assert (tmp_path / "store" / "history.jsonl").read_bytes() == before


# -- a persisted instant ahead of the clock refuses ------------------------------------
def test_a_spend_is_refused_when_the_journal_is_ahead_of_the_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    calls = [0]

    def hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] == 2:  # the final authenticator call: the clock is set back
            clock[0] = at(39)
        return True

    c = controller(tmp_path, opt, now=at(0), authority=_platform(hook))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    report = c.step()
    assert opt.calls == [] and _started(c) == []
    assert report.refused and report.refused[0].endswith(":persisted_time_ahead")


# -- only the actions this release runs ------------------------------------------------
def test_a_rule_with_an_action_this_release_does_not_run_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    doc = module()
    doc["responses"][0]["rules"][0]["do"] = "open_next_step"
    st = state()
    st.tvl = doc
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "store", FakeOptimizer())
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE
    assert "action" in err.value.detail
