"""Hardening: authenticator everywhere, effect time under the lock,
journal retries, recovery against later history, schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import (
    FailureRate,
    ProposeFix,
    Tuning,
)
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.controller import (
    Controller,
)
from traigent.tuning.errors import (
    E_APPROVER_NOT_AUTHORIZED,
    E_UNSUPPORTED_IN_RELEASE,
    TuningError,
)
from traigent.tuning.policy import Every
from traigent.tuning.state import attach_tuning

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    NEW,
    OLD,
    T0,
    FakeOptimizer,
    _deploy_awaiting,
    _holdout,
    at,
    controller,
    e_ref,
    head,
    make,
    state,
)


def _platform(answer: list[bool]) -> ApproverAuthority:
    return ApproverAuthority(
        frozenset(APPROVERS),
        mode="platform",
        authenticator=lambda approver, context: answer[0],
    )


# -- 1. the authenticator decides every authority decision ----------------------------
def test_a_platform_authenticator_returning_false_refuses_run_approval(
    tmp_path: Path,
) -> None:
    answer = [False]
    c = make(state(), tmp_path / "a", FakeOptimizer(), authority=_platform(answer))
    with pytest.raises(TuningError) as err:
        c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    assert err.value.code == E_APPROVER_NOT_AUTHORIZED


def test_authentication_is_rechecked_when_an_approved_run_is_about_to_start(
    tmp_path: Path,
) -> None:
    answer = [True]
    opt = FakeOptimizer()
    c = make(state(), tmp_path / "a", opt, authority=_platform(answer))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    answer[0] = False  # the platform no longer vouches for the approver
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]


# -- 6. standing mode is not part of this release ---------------------------------------
def test_standing_on_a_rule_that_deploys_if_promoted_is_refused(tmp_path: Path) -> None:
    st = state()
    for response in st.tvl["responses"]:
        for rule in response["rules"]:
            if rule["id"] == "on_E1":
                rule["approval"] = "standing"
                rule["deploy"] = "if_promoted"
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "s", FakeOptimizer())
    assert err.value.code == E_UNSUPPORTED_IN_RELEASE


def test_status_names_the_human_who_approved_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = make(state(), tmp_path / "s", FakeOptimizer())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    real = Controller._append

    def crash(self: Controller, category: str, record: Any, *a: Any, **k: Any) -> None:
        if record and record.get("fact") == "finished":
            raise RuntimeError("simulated crash before recording")
        real(self, category, record, *a, **k)

    monkeypatch.setattr(Controller, "_append", crash)
    with pytest.raises(RuntimeError):
        c.step(at(10))
    monkeypatch.undo()
    entries = [a for a in c.status()["open_attempts"] if a["occurrence"] == e_ref("E1")]
    assert entries and entries[0]["approval"] == "approved by approver-1"


# -- 7. a modern schedule anchor needs no explicit snapshot ----------------------------
def test_every_with_a_modern_anchor_validates_without_a_module_context() -> None:
    def fn(q: str) -> str:
        return q

    fn.configuration_space = {"model": ["a", "b"]}  # type: ignore[attr-defined]
    policy = Tuning(
        keep=[FailureRate("customer:c", "0.1", name="q")],
        on=[Every(1, "2026-11-20T12:00:00Z")],
        fix=ProposeFix(),
    )
    st = attach_tuning(fn, "agent-1", policy)
    codes = [d.code for d in st.diagnostics]
    assert "schedule_anchor_out_of_range" not in codes
    assert st.validated is True


# -- opaque references are matched whole: a trailing newline is not accepted ---------
def test_an_approver_with_a_trailing_newline_is_refused(tmp_path: Path) -> None:
    from traigent.tuning.controller import E_INVALID_RECORD

    c = make(state(), tmp_path / "a", FakeOptimizer(), authority=_platform([True]))
    with pytest.raises(TuningError) as err:
        c.approve_optimization(e_ref("E1"), "approver-1\n", at(10))
    assert err.value.code == E_INVALID_RECORD


def test_an_agent_or_environment_with_a_trailing_newline_is_refused() -> None:
    from traigent.tuning.controller import E_INVALID_RECORD, default_store_dir

    for args in (("agent\n", None), ("agent", "production\n")):
        with pytest.raises(TuningError) as err:
            default_store_dir(*args)
        assert err.value.code == E_INVALID_RECORD
