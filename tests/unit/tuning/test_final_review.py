"""Hardening: authenticator everywhere, effect time under the lock,
journal retries, recovery against concurrent ingestion, standing mode, schedules."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import (
    FailureRate,
    ProposeFix,
    Tuning,
)
from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.controller import (
    E_HEAD_DIVERGED,
    E_STANDING_DEPLOY_UNSUPPORTED,
    Controller,
)
from traigent.tuning.errors import E_APPROVER_NOT_AUTHORIZED, TuningError
from traigent.tuning.policy import Every
from traigent.tuning.state import attach_tuning

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    NEW,
    NEWER,
    OLD,
    FakeOptimizer,
    _crash_install,
    _deploy_awaiting,
    _holdout,
    _standing_state,
    at,
    controller,
    e_ref,
    event,
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
def test_a_platform_authenticator_returning_false_refuses_run_approval_and_grants(
    tmp_path: Path,
) -> None:
    answer = [False]
    c = make(state(), tmp_path / "a", FakeOptimizer(), authority=_platform(answer))
    c.record_event(event("E1", NEW["model"], 9), at(10))
    with pytest.raises(TuningError) as err:
        c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    assert err.value.code == E_APPROVER_NOT_AUTHORIZED

    g = make(
        _standing_state(), tmp_path / "b", FakeOptimizer(), authority=_platform(answer)
    )
    with pytest.raises(TuningError) as err:
        g.record_standing_grant(
            "G1",
            "tuning/on_new_model",
            1,
            at(10) + timedelta(days=1),
            "issuer-1",
            at(10),
        )
    assert err.value.code == E_APPROVER_NOT_AUTHORIZED


def test_authentication_is_rechecked_when_an_approved_run_is_about_to_start(
    tmp_path: Path,
) -> None:
    answer = [True]
    opt = FakeOptimizer()
    c = make(state(), tmp_path / "a", opt, authority=_platform(answer))
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    answer[0] = False  # the platform no longer vouches for the approver
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]


# -- 2. the effect time is read after the lock is held --------------------------------
def test_an_acceptance_that_expires_while_waiting_for_the_lock_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    clock = [at(32)]  # usable now
    real_utc = controller_module._utc
    monkeypatch.setattr(
        controller_module,
        "_utc",
        lambda value: real_utc(value) if value is not None else clock[0],
    )
    outcome: list[Any] = []
    with c._locked():  # another writer (an optimization) holds the store
        worker = threading.Thread(
            target=lambda: outcome.append(c.submit_acceptance(record))
        )
        worker.start()
        time.sleep(0.5)  # the caller is now waiting for the lock
        clock[0] = at(31) + timedelta(days=8)  # ... and the acceptance expires
    worker.join(timeout=30)
    assert outcome and outcome[0].result != "written"
    assert head(tmp_path)["generation"] == 0 and head(tmp_path)["config"] == OLD


# -- 3. an abandoned attempt never disables the recovery of its retry -----------------
def test_a_retry_after_an_abandoned_commit_is_recovered_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = _crash_install(monkeypatch)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))  # intent 1, head not installed
    monkeypatch.setattr(Controller, "_install_head", real)
    controller(tmp_path, FakeOptimizer())  # recovery abandons intent 1

    retry = controller(tmp_path, FakeOptimizer())
    _crash_install(monkeypatch, after=True)
    with pytest.raises(RuntimeError):
        retry.submit_acceptance(record, at(32))  # the identical record, intent 2
    monkeypatch.setattr(Controller, "_install_head", real)
    assert head(tmp_path)["generation"] == 1

    restarted = controller(tmp_path, FakeOptimizer())
    assert restarted._tvl_generation(restarted._derive()) == 1  # no divergence
    restarted.step(at(33))  # E_HEAD_DIVERGED would raise here
    commits = [
        json.loads(ln.partition("\t")[2])
        for ln in (tmp_path / "store" / "commits.jsonl").read_text().splitlines()
    ]
    assert [e["fact"] for e in commits] == [
        "commit_intent",
        "commit_abandoned",
        "commit_intent",
        "commit_done",
    ]
    assert len({e["commit"] for e in commits}) == 2


# -- 4. recovery against concurrent ingestion -----------------------------------------
def _crash_after_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Controller, dict[str, Any]]:
    """A second controller opened before the crash, and the acceptance that crashes."""
    c = _deploy_awaiting(tmp_path)
    other = controller(tmp_path, FakeOptimizer())  # opened before the crash
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = _crash_install(monkeypatch, after=True)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))
    monkeypatch.setattr(Controller, "_install_head", real)
    assert head(tmp_path)["generation"] == 1
    return other, record


def test_ingestion_by_another_controller_recovers_the_commit_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other, _ = _crash_after_install(tmp_path, monkeypatch)
    late = at(31) + timedelta(days=8)  # past the acceptance's expiry
    other.record_event(event("E2", NEWER["model"], 31 + 8 * 24 - 1), late)

    restarted = controller(tmp_path, FakeOptimizer())
    assert restarted._tvl_generation(restarted._derive()) == 1
    restarted.step(late)  # no E_HEAD_DIVERGED
    acceptances = [
        json.loads(ln.partition("\t")[2])
        for ln in (tmp_path / "store" / "history.jsonl").read_text().splitlines()
        if '"acceptances"' in ln
    ]
    assert [a["instant"] for a in acceptances] == [31 * 3600]
    facts = [f for f in restarted._derive()["facts"] if f["fact"] == "acceptance"]
    assert [f["result"] for f in facts] == ["written"]  # never rejected for expiry


def test_history_that_grew_past_the_crash_still_takes_the_acceptance_at_its_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other, _ = _crash_after_install(tmp_path, monkeypatch)
    late = at(31) + timedelta(days=8)
    monkeypatch.setattr(Controller, "_recover_commits", lambda self: None)
    other.record_event(
        event("E2", NEWER["model"], 31 + 8 * 24 - 1), late
    )  # no recovery
    monkeypatch.undo()

    restarted = controller(tmp_path, FakeOptimizer())  # recovery splices it in
    assert restarted._tvl_generation(restarted._derive()) == 1
    restarted.step(late)  # no divergence
    lines = [
        json.loads(ln.partition("\t")[2])
        for ln in (tmp_path / "store" / "history.jsonl").read_text().splitlines()
    ]
    instants = [ln["instant"] for ln in lines]
    assert instants == sorted(instants)  # the later event is ordered after it
    accepted = [ln for ln in lines if ln["category"] == "acceptances"]
    assert [a["instant"] for a in accepted] == [31 * 3600]
    facts = [f for f in restarted._derive()["facts"] if f["fact"] == "acceptance"]
    assert [f["result"] for f in facts] == ["written"]


def test_an_unreconcilable_recovery_is_loud_and_leaves_the_journal_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, record = _crash_after_install(tmp_path, monkeypatch)
    (tmp_path / "store" / "head.json").write_text(  # a head the relation cannot reach
        json.dumps({**head(tmp_path), "generation": 1}), encoding="utf-8"
    )
    path = tmp_path / "store" / "commits.jsonl"
    before = path.read_text()
    monkeypatch.setattr(
        Controller,
        "_derive",
        lambda self, extra=None: {"facts": [], "head_generations": []},
    )
    monkeypatch.setattr(
        controller_module,
        "conditions_trace",
        lambda *a, **k: {"facts": [], "head_generations": []},
    )
    with pytest.raises(TuningError) as err:
        Controller(
            state(), tmp_path / "store", FakeOptimizer(), authorized_approvers=APPROVERS
        )
    assert err.value.code == E_HEAD_DIVERGED
    assert path.read_text() == before  # not marked done


# -- 6. standing mode ------------------------------------------------------------------
def test_standing_on_a_rule_that_deploys_if_promoted_is_refused(tmp_path: Path) -> None:
    st = _standing_state()
    for response in st.tvl["responses"]:
        for rule in response["rules"]:
            if rule["id"] == "on_new_model":
                rule["deploy"] = "if_promoted"
    with pytest.raises(TuningError) as err:
        make(st, tmp_path / "s", FakeOptimizer())
    assert err.value.code == E_STANDING_DEPLOY_UNSUPPORTED


def test_status_says_which_standing_grant_approved_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = make(_standing_state(), tmp_path / "s", FakeOptimizer())
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(10) + timedelta(days=1), "issuer-1", at(10)
    )
    c.record_event(event("E1", NEW["model"], 9), at(10))
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
    assert entries and entries[0]["approval"] == "approved by standing grant G1"


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
