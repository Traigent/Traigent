"""Regression tests for the reactive-requirements invariant review (C5-C14, INV-33)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import traigent
from traigent.runtime import manifest
from traigent.runtime.apply import apply_requirement
from traigent.runtime.model import (
    ResponseRule,
    RuntimeRequirement,
    RuntimeRequirementError,
    build_rules,
    canonical_json,
)
from traigent.runtime.respond import respond
from traigent.runtime.tvl_fragment import parse_fragment
from traigent.runtime.watch import poll_once

from .conftest import AGENT, FIXTURES, PROJECT, FakeBackend, breach, head

CREATE = ("POST", "/production-requirements")
LIST = ("GET", "/production-requirements")
BREACHES = ("GET", f"/agents/{AGENT}/production-breaches")
NEXT = ("GET", f"/agent-readiness/{AGENT}/next-step")
RESTORE = {"step": {"code": "RESTORE_PRODUCTION_REQUIREMENT"}}
SHARED_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "runtime_policy_digests.json"
)


@pytest.fixture
def live(backend: FakeBackend) -> FakeBackend:
    backend.on(*CREATE, 201, head())
    backend.on(*LIST, 200, {"items": [head()]})
    backend.on(*BREACHES, 200, {"items": [breach("b1")], "has_more": False})
    backend.on(*NEXT, 200, RESTORE)
    return backend


def _fn_and_req():
    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="fix", max_fix_cost_usd=2.0
    )
    def fn() -> None: ...

    fn.optimize = AsyncMock(return_value=SimpleNamespace(optimization_id="r"))
    return fn, fn.requirements[0]


def _kw(state_dir: Path, **extra):
    return {
        "project_id": PROJECT,
        "agent_id": AGENT,
        "confirm": lambda _f: True,
        "interactive": True,
        "state_dir": state_dir,
        **extra,
    }


class Clock:
    def __init__(self) -> None:
        self.now: datetime = manifest.utcnow()

    def __call__(self) -> datetime:
        return self.now


# --- C7: expiry re-checked at the last moment --------------------------------------------


async def test_c7_approval_expiring_during_second_fresh_round_holds(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    clock = Clock()
    calls: list[int] = []

    def next_step(_body):
        calls.append(1)
        if len(calls) == 2:  # the second fresh-check round
            clock.now += timedelta(hours=25)
        return 200, RESTORE

    live.routes[NEXT] = next_step
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir, clock=clock))
    assert err.value.code == "approval_expired" and len(calls) == 2
    fn.optimize.assert_not_awaited()


# --- C6: every bound field compared on read-back -------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_policy_digest", "0" * 64),
        ("breach_id", "other"),
        ("rule_id", "other"),
        ("occurrence", 7),
        ("cost_ceiling_usd", 99.0),
        ("attempt", 5),
    ],
)
async def test_c6_tampered_approval_field_holds(
    live: FakeBackend,
    state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    original = manifest.write_approval

    def write_then_overwrite(*args, **kwargs):
        record = original(*args, **kwargs)
        path = manifest._approval_path(state_dir, AGENT)
        doc = json.loads(path.read_text())
        (key,) = doc
        doc[key][field] = value
        path.write_text(json.dumps(doc))
        return record

    monkeypatch.setattr(manifest, "write_approval", write_then_overwrite)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert (err.value.code, err.value.path) == ("hold", "approval")
    fn.optimize.assert_not_awaited()


# --- C5: attempt generation ----------------------------------------------------------------


async def test_c5_pre_failure_approval_cannot_authorize_post_failure_retry(
    live: FakeBackend, state_dir: Path
) -> None:
    fn_a, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    fn_a.optimize = AsyncMock(side_effect=RuntimeError("boom"))
    fn_b, _ = _fn_and_req()

    def confirm_b(_facts) -> bool:
        # Caller A starts, fails and releases the occurrence while B's person decides.
        outcome: list[str] = []

        def run_a() -> None:
            try:
                asyncio.run(respond(fn_a, req, **_kw(state_dir)))
            except RuntimeRequirementError as exc:
                outcome.append(exc.code)

        worker = threading.Thread(target=run_a)
        worker.start()
        worker.join(30)
        assert outcome == ["optimize_failed"]
        return True

    # B's approval is issued from a ledger that had no failure, so it must not be usable now.
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn_b, req, **_kw(state_dir, confirm=confirm_b))
    assert (err.value.code, err.value.path) == ("hold", "approval")
    fn_b.optimize.assert_not_awaited()
    record = manifest.read_occurrence(state_dir, AGENT, "b1|fix|0")
    assert record["status"] == "failed" and record["attempts"] == 1


async def test_c5_retry_after_failure_with_a_new_approval_is_attempt_two(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    fn.optimize = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeRequirementError):
        await respond(fn, req, **_kw(state_dir))
    fn.optimize = AsyncMock(return_value=SimpleNamespace(optimization_id="again"))
    result = await respond(fn, req, **_kw(state_dir))
    assert result.run_id == "again"
    approvals = json.loads(manifest._approval_path(state_dir, AGENT).read_text())
    assert approvals["b1|fix|0"]["attempt"] == 2
    assert manifest.read_occurrence(state_dir, AGENT, "b1|fix|0")["attempts"] == 2


# --- INV-33: lease ---------------------------------------------------------------------------


async def test_inv33_stale_reservation_is_read_as_failed_and_needs_a_new_approval(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    clock = Clock()
    key = "b1|fix|0"
    assert manifest.reserve_occurrence(
        state_dir, AGENT, key, {"started_at": clock().isoformat(), "run_id": None}
    )
    with pytest.raises(RuntimeRequirementError) as err:  # inside the lease: blocks
        await respond(fn, req, **_kw(state_dir, clock=clock))
    assert err.value.code == "occurrence_already_used"

    clock.now += timedelta(hours=25)  # the controller crashed; lease expired
    asked: list[int] = []

    def confirm(_f) -> bool:
        asked.append(1)
        return True

    result = await respond(fn, req, **_kw(state_dir, clock=clock, confirm=confirm))
    assert result.run_id == "r" and asked == [1]
    assert manifest.read_occurrence(state_dir, AGENT, key, now=clock())["attempts"] == 2


async def test_inv33_lease_is_overridable_and_a_run_id_is_never_stale(
    state_dir: Path,
) -> None:
    now = manifest.utcnow()
    old = (now - timedelta(hours=2)).isoformat()
    manifest.reserve_occurrence(state_dir, AGENT, "k", {"started_at": old})
    manifest.reserve_occurrence(state_dir, AGENT, "done", {"started_at": old})
    manifest.update_occurrence(state_dir, AGENT, "done", {"run_id": "r1"}, attempt=1)
    short = timedelta(hours=1)
    assert (
        manifest.read_occurrence(state_dir, AGENT, "k", now=now, lease=short)["status"]
        == "failed"
    )
    assert manifest.read_occurrence(state_dir, AGENT, "k", now=now)["status"] == (
        "started"
    )
    assert (
        manifest.read_occurrence(state_dir, AGENT, "done", now=now, lease=short)[
            "status"
        ]
        == "started"
    )


# --- C9: breach reference ---------------------------------------------------------------------


def _step_with_ref(ref: str | None) -> dict:
    return {
        **RESTORE,
        "basis": {"production_breach": {"open_breach_ref": ref} if ref else None},
    }


async def test_c9_step_for_another_breach_holds(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    live.on(*NEXT, 200, _step_with_ref("someone-else"))
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert (err.value.code, err.value.path) == ("hold", "breach")
    fn.optimize.assert_not_awaited()


# control: passes without the fix (the old code already took the first breach)
async def test_c9_matching_reference_proceeds_even_with_two_open_breaches(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    rows = {"items": [breach("b1"), breach("b2")], "has_more": False}
    live.on(*BREACHES, 200, rows)
    live.on(*NEXT, 200, _step_with_ref("b1"))
    result = await respond(fn, req, **_kw(state_dir))
    assert result.breach_id == "b1"


async def test_c9_no_reference_and_two_open_breaches_is_ambiguous(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    live.on(*BREACHES, 200, {"items": [breach("b1"), breach("b2")], "has_more": False})
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **_kw(state_dir))
    assert err.value.code == "ambiguous_breach"
    fn.optimize.assert_not_awaited()


# --- C14: one optimize rule ----------------------------------------------------------------------


def test_c14_two_episode_opened_optimize_rules_are_refused() -> None:
    _fn, req = _fn_and_req()
    two = dataclasses.replace(
        req,
        rules=req.rules
        + (ResponseRule("fix2", "episode_opened", "optimize", None, "human"),),
    )
    codes = [(i.code, i.path) for i in two.validate() if i.severity == "error"]
    assert ("multiple_optimize_rules", "rules[5]") in codes
    assert not [i for i in req.validate() if i.severity == "error"]


async def test_c14_respond_refuses_two_optimize_rules(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    two = dataclasses.replace(
        req,
        rules=req.rules
        + (ResponseRule("fix2", "episode_opened", "optimize", None, "human"),),
    )
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, two, **_kw(state_dir))
    assert err.value.code == "multiple_optimize_rules"


# --- occurrence numbering from 0 -------------------------------------------------------------------


async def test_first_occurrence_is_numbered_zero(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    result = await respond(fn, req, **_kw(state_dir))
    assert result.occurrence == 0
    assert manifest.read_occurrence(state_dir, AGENT, "b1|fix|0") is not None


# --- TVL: notify severity defaults to critical -----------------------------------------------------


def test_notify_rule_without_severity_defaults_to_critical() -> None:
    doc = yaml.safe_load((FIXTURES / "latency.tvl.yml").read_text())
    del doc["responses"][0]["rules"][0]["severity"]
    req = parse_fragment(doc, max_fix_cost_usd=1.0)
    assert req.rules[0].action == "notify" and req.rules[0].severity == "critical"


# --- C11: one poller at a time on a state dir ---------------------------------------------------------


def test_c11_two_concurrent_pollers_emit_an_episode_once(
    backend: FakeBackend,
) -> None:
    backend.on(*BREACHES, 200, {"items": [breach("b1")], "has_more": False})
    seen: list[str] = []

    def slow_hook(event) -> None:
        seen.append(event.breach_id)
        time.sleep(0.4)

    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="alert", on_event=slow_hook
    )
    def fn() -> None: ...

    reqs = list(fn.requirements)
    results: list[object] = []

    def poll() -> None:
        results.append(poll_once(reqs, project_id=PROJECT, agent_id=AGENT))

    first = threading.Thread(target=poll)
    second = threading.Thread(target=poll)
    first.start()
    time.sleep(0.15)
    second.start()
    first.join(10)
    second.join(10)
    assert seen == ["b1"]


# control: passes without the fix (the old code had no lock to leak)
def test_c11_lock_released_after_a_poll(backend: FakeBackend, caplog) -> None:
    backend.on(*BREACHES, 200, {"items": [], "has_more": False})
    caplog.set_level(logging.INFO, logger="traigent.runtime")

    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05, on_breach="alert")
    def fn() -> None: ...

    reqs = list(fn.requirements)
    for _ in range(2):  # a second poll would block forever if the lock leaked
        poll_once(reqs, project_id=PROJECT, agent_id=AGENT)


# --- C13: canonical numbers + shared cross-SDK digests ---------------------------------------------------


def _shared() -> dict:
    return json.loads(SHARED_FIXTURE.read_text())


def test_c13_canonical_number_encoding() -> None:
    for case in _shared()["numbers"]:
        assert canonical_json(case["value"]) == case["canonical"], case
    assert canonical_json({"a": 1.0, "b": [True, None, -0.0]}) == (
        '{"a":1,"b":[true,null,0]}'
    )


def _from_declaration(decl: dict) -> RuntimeRequirement:
    spec = dict(decl)
    on_breach = spec.pop("on_breach")
    return RuntimeRequirement(**spec, on_breach=on_breach, rules=build_rules(on_breach))


def test_c13_shared_policy_digest_fixture() -> None:
    cases = _shared()["declarations"]
    assert len(cases) >= 6
    for case in cases:
        req = _from_declaration(case["declaration"])
        assert canonical_json(req.to_doc()) == case["canonical_json"], case["id"]
        assert req.policy_digest() == case["policy_digest"], case["id"]


# --- small branches of the new helpers -----------------------------------------------------------------


def test_canonical_json_refuses_non_finite_and_unknown_types() -> None:
    with pytest.raises(ValueError):
        canonical_json(float("nan"))
    with pytest.raises(TypeError):
        canonical_json({"a": object()})


def test_approval_match_is_type_strict() -> None:
    record = {"occurrence": 0, "breach_id": "b", "cost_ceiling_usd": 2.0}
    assert manifest.approval_matches(record, occurrence=0, cost_ceiling_usd=2)
    assert not manifest.approval_matches(record, occurrence=False)  # bool is not 0
    assert not manifest.approval_matches(record, breach_id=1)
    assert not manifest.approval_matches(record, rule_id="missing")


def test_unreadable_lease_timestamp_fails_closed(state_dir: Path) -> None:
    manifest.reserve_occurrence(state_dir, AGENT, "k", {"started_at": "garbage"})
    later = manifest.utcnow() + timedelta(days=30)
    assert manifest.read_occurrence(state_dir, AGENT, "k", now=later)["status"] == (
        "started"
    )
    naive = {"status": "started", "started_at": "2020-01-01T00:00:00", "attempts": 1}
    assert manifest.effective_record(naive)["status"] == "failed"
    assert manifest.failed_attempts(None) == 0


def test_reservation_with_a_stale_attempt_holds(state_dir: Path) -> None:
    assert manifest.reserve_occurrence(state_dir, AGENT, "k", {"run_id": None})
    manifest.update_occurrence(state_dir, AGENT, "k", {"status": "failed"}, attempt=1)
    with pytest.raises(RuntimeRequirementError) as err:
        manifest.reserve_occurrence(state_dir, AGENT, "k", {}, attempt=1)
    assert (err.value.code, err.value.path) == ("hold", "approval")
    assert manifest.reserve_occurrence(state_dir, AGENT, "k", {}, attempt=2)


def test_c11_unlockable_state_is_a_closed_unavailable_result(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    from contextlib import contextmanager

    @contextmanager
    def broken(*_a, **_k):
        raise PermissionError("no lock")
        yield

    monkeypatch.setattr(manifest, "watch_lock", broken)

    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05, on_breach="alert")
    def fn() -> None: ...

    result = poll_once(
        list(fn.requirements),
        project_id=PROJECT,
        agent_id=AGENT,
        state_dir=state_dir,
    )
    assert result.error == "local_state_invalid" and result.events == []
