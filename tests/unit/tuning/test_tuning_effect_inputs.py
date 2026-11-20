"""The clock and the authority are read last; unsupported rules are refused; the policy
is a private snapshot; a fixed clock is a simulation clock only; abandonment frees an
approved attempt that never got a marker."""

from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.controller import Controller
from traigent.tuning.errors import E_UNSUPPORTED_IN_RELEASE, TuningError

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
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


def _finished(c: Controller) -> list[str]:
    return [
        ln["record"]["completion"]
        for ln in c._lines()
        if ln["category"] == "optimizations" and ln["record"].get("fact") == "finished"
    ]


# -- the clock is sampled after every read and staging step --------------------------
def test_a_scope_that_ends_during_the_final_history_read_does_not_start_the_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    c = controller(tmp_path, opt, now=at(0))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    real_last = Controller._last_persisted_instant

    def slow_last(self: Controller) -> int:
        last = real_last(self)  # the final history read of the preparation
        clock[0] = _scope_end() + timedelta(seconds=1)
        return last

    monkeypatch.setattr(Controller, "_last_persisted_instant", slow_last)
    report = c.step()
    assert opt.calls == []
    assert report.refused and report.refused[0].endswith(":scope_expired")
    assert _finished(c) == []


def test_a_release_whose_head_staging_crosses_expiry_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    clock = _clock(monkeypatch, at(32))
    real = controller_module._stage_json

    def slow(path: Path, value: Any) -> str:
        staged = real(path, value)
        clock[0] = clock[0] + timedelta(days=8)  # staging took past the 7 day expiry
        return staged

    monkeypatch.setattr(controller_module, "_stage_json", slow)
    outcome = c.submit_acceptance(record)
    assert (outcome.result, outcome.reason) == ("refused", "acceptance_not_usable")
    assert head(tmp_path)["generation"] == 0
    assert not list((tmp_path / "store").glob(".head.json.*.tmp"))  # nothing left over


# -- the platform authenticator is asked again at the effect -------------------------
def test_an_authenticator_that_revokes_before_the_spend_stops_the_optimizer(
    tmp_path: Path,
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
    real_write = controller_module._write_json

    def write(path: Path, value: Any) -> None:
        real_write(path, value)
        if value.get("state") == "started":
            revoked[0] = True  # revoked after the approval, before the effect

    controller_module._write_json = write
    try:
        report = c.step(at(10))
    finally:
        controller_module._write_json = real_write
    assert opt.calls == []
    assert report.refused and report.refused[0].endswith(":approver_not_authorized")
    assert _finished(c) == []  # nothing started: the attempt stays open
    assert not list((tmp_path / "store" / "attempts").glob("*.json"))


def test_an_authenticator_that_revokes_before_the_head_moves_stops_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    revoked = [False]
    c.authority = _platform(lambda approver, context: not revoked[0])
    real = controller_module._append_line

    def append(path: Path, value: Any) -> None:
        real(path, value)
        if isinstance(value, dict) and value.get("fact") == "commit_intent":
            revoked[0] = True  # revoked while the intent was being journaled

    monkeypatch.setattr(controller_module, "_append_line", append)
    outcome = c.submit_acceptance(record, at(31))
    assert (outcome.result, outcome.reason) == ("refused", "approver_not_authorized")
    assert head(tmp_path)["generation"] == 0


# -- a granted approval that never got a marker is abandonable -----------------------
def _crash_after_approval(c: Controller, monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(self: Controller, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated crash after the approval was journaled")

    monkeypatch.setattr(Controller, "_scope_deadline", crash)
    with pytest.raises(RuntimeError):
        c.step(at(10))
    monkeypatch.undo()


def _held(c: Controller) -> list[Any]:
    return [f for f in c._derive()["facts"] if f["fact"] == "held"]


def test_an_approved_attempt_without_a_marker_holds_later_schedules_until_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    _crash_after_approval(c, monkeypatch)
    restarted = controller(tmp_path, FakeOptimizer())
    restarted.step(at(41))
    assert "on_E6" in {f["occurrence"]["rule"] for f in _held(restarted)}


def test_abandoning_records_an_approved_attempt_that_has_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    _crash_after_approval(c, monkeypatch)
    opt = FakeOptimizer()
    restarted = controller(tmp_path, opt)
    restarted.step(at(10), abandon_interrupted=True)
    assert _finished(restarted) == ["cancelled"]
    restarted.step(at(41))  # the next schedule opens; the freed lock does not hold it
    facts = restarted._derive()["facts"]
    held_rules = {f["occurrence"]["rule"] for f in _held(restarted)}
    opened_rules = {
        f["occurrence"]["rule"] for f in facts if f["fact"] == "attempt_opened"
    }
    assert "on_E6" not in held_rules and "on_E6" in opened_rules
    assert opt.calls == []  # cancelled, never invoked


def test_abandoning_does_not_cancel_an_attempt_approved_in_the_same_step(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10), abandon_interrupted=True)
    assert len(opt.calls) == 1
    assert _finished(c) == ["normal"]


# -- unsupported rule conditions are refused at construction -------------------------
@pytest.mark.parametrize(
    "extra",
    [
        {"when": {"monitor": "quality_prod"}},
        {"repeat_after_hours": 24},
        {"max_repeats": 2},
        {"only_while": {"monitor": "quality_prod", "episode": "open"}},
        {"unless": {"monitor": "quality_prod"}},
        {"persist": True},
        {"target": "quality_prod"},
        {"cooldown_hours": 12},
        {"some_future_key": 1},
    ],
)
def test_a_rule_condition_this_release_does_not_run_is_refused(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    doc = module()
    doc["responses"][0]["rules"][0].update(extra)
    st = state()
    st.tvl = doc  # the policy builder only knows the builder shape
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "store", FakeOptimizer())
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE


def test_a_trigger_shape_this_release_does_not_run_is_refused(tmp_path: Path) -> None:
    doc = module()
    doc["responses"][0]["rules"][0]["trigger"]["schedule"]["jitter_hours"] = 1
    st = state()
    st.tvl = doc
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "store", FakeOptimizer())
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE


# -- the policy is a private snapshot -----------------------------------------------
def test_changing_the_callers_policy_after_construction_changes_nothing_executed(
    tmp_path: Path,
) -> None:
    doc = module()
    st = state(doc)
    c = make(st, tmp_path / "store", FakeOptimizer())
    original = copy.deepcopy(doc)
    doc["responses"][0]["rules"].clear()  # the caller's dictionary
    doc["responses"][0]["rules"].append({"id": "x", "when": 1})
    st.tvl["tvars"].clear()
    assert c.step(at(10)).awaiting_approval == [e_ref("E1")]
    assert c._module["responses"] == original["responses"]


def test_changing_the_exposed_state_does_not_change_the_executed_policy(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    assert c.state.tvl is not c._module
    c.state.tvl["responses"][0]["rules"].clear()
    c.state.tvl["tvars"].clear()
    assert c.step(at(10)).awaiting_approval == [e_ref("E1")]
    assert c._module["responses"][0]["rules"]


# -- a fixed clock is for simulation only -------------------------------------------
def test_a_fixed_instant_is_refused_on_the_effect_paths_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = _deploy_awaiting(tmp_path, opt)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    monkeypatch.setenv("ENVIRONMENT", "production")
    c.authority = _platform(lambda approver, context: True)
    with pytest.raises(TuningError) as err:
        c.step(at(31))
    assert err.value.code == "E_FIXED_CLOCK_IN_PRODUCTION"
    with pytest.raises(TuningError) as err:
        c.submit_acceptance(record, at(31))
    assert err.value.code == "E_FIXED_CLOCK_IN_PRODUCTION"
    assert head(tmp_path)["generation"] == 0


def test_an_unset_environment_counts_as_production_for_a_fixed_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with pytest.raises(TuningError):
        controller_module._effect_clock(at(1))


def test_a_fixed_instant_outside_production_keeps_advancing_with_real_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: now[0])
    clock = controller_module._effect_clock(at(1))
    assert clock() == at(1)
    now[0] += 90.0  # a long lock wait
    assert clock() == at(1) + timedelta(seconds=90)


def test_the_live_clock_is_read_afresh_when_no_instant_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _clock(monkeypatch, at(5))
    read = controller_module._effect_clock(None)
    assert read() == at(5)
    clock[0] = at(6)
    assert read() == at(6)
