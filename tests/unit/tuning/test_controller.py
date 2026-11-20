"""The local loop controller drives the SLICE scenario end to end (fake optimizer)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

import traigent
from traigent.cli.main import cli
from traigent.config.context import config_context
from traigent.observability import observe
from traigent.tuning import validator
from traigent.tuning.controller import (
    E_ATTEMPT_IN_PROGRESS,
    E_NO_DEPLOY_AWAITING,
    Controller,
    HeadLoader,
    config_fingerprint,
    occurrence_ref,
)
from traigent.tuning.errors import TuningError, ValidatorUnavailable
from traigent.tuning.policy import Tuning
from traigent.tuning.state import TuningState

pytestmark = pytest.mark.skipif(
    not validator.validator_available(),
    reason="TVL toolchain with 1.7 support not installed",
)

SLICE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "tuning"
    / "external-slice-trace.yml"
)
T0 = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
OLD = {"model": "anthropic/claude-haiku-4-5"}
NEW = {"model": "anthropic/claude-haiku-5-5"}


def _ref(event_id: str) -> str:
    """The deploy reference of an external occurrence, scope_ref from TVL itself."""
    return occurrence_ref(
        {
            "response": "tuning",
            "rule": "on_new_model",
            "scope_kind": "external",
            "scope": {"issuer": "P", "event_id": event_id},
            "j": 0,
        }
    )


E1_REF = _ref("E1")
E6_REF = _ref("E6")


def at(hours: int) -> datetime:
    return T0 + timedelta(hours=hours)


def module() -> dict[str, Any]:
    case = yaml.safe_load(SLICE.read_text(encoding="utf-8"))
    return copy.deepcopy(case["module"])


def state(doc: dict[str, Any] | None = None, *, validated: bool = True) -> TuningState:
    doc = doc or module()
    return TuningState(
        agent="agent-1", policy=Tuning.from_tvl(doc), tvl=doc, validated=validated
    )


def event(event_id: str, model: str, issued_at: int) -> dict[str, Any]:
    return {
        "kind": "model_released",
        "event_id": event_id,
        "issuer": "P",
        "issued_at": issued_at,
        "catalogue_revision": 1,
        "subject": {
            "model_ref": model,
            "provider_family": "anthropic",
            "tier": "small",
        },
        "authenticated": True,
    }


class FakeOptimizer:
    """Stands in for ``fn.optimize``; records every call."""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def __call__(self, space: dict[str, Any]) -> Any:
        self.calls.append(space)
        if self.result is not None:
            return self.result
        return {
            "candidates": [
                {
                    "config": {"model": space["model"][-1]},
                    "legacy_decision": "Promote",
                    "monitor_verdicts": {"quality_gate": "MET"},
                }
            ]
        }


def controller(tmp_path: Path, optimize: Any, **kw: Any) -> Controller:
    return Controller(
        state(), tmp_path / "store", optimize, initial_config=OLD, now=T0, **kw
    )


def head(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "store" / "head.json").read_text(encoding="utf-8"))


def to_deploy_awaiting(c: Controller, ref: str, hour: int) -> None:
    c.approve_optimization(ref, "approver-1", at(hour))
    c.step(at(hour))


def test_slice_end_to_end(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    assert head(tmp_path)["generation"] == 0

    # E1 -> attempt opened, but nothing runs without a recorded approval.
    c.record_event(event("E1", NEW["model"], 9), at(10))
    report = c.step(at(10))
    assert report.awaiting_approval == [E1_REF]
    assert opt.calls == []

    # approval -> optimize runs once with the admitted search domain.
    to_deploy_awaiting(c, E1_REF, 10)
    assert len(opt.calls) == 1
    assert NEW["model"] in opt.calls[0]["model"]
    status = c.status()
    assert [d["occurrence"] for d in status["pending_deploys"]] == [E1_REF]
    assert head(tmp_path)["generation"] == 0  # nothing deployed yet

    # human acceptance at generation 0 -> head generation 1, the new model.
    outcome = c.accept(E1_REF, "approver-1", 0, now=at(31))
    assert (outcome.result, outcome.generation) == ("written", 1)
    assert head(tmp_path)["generation"] == 1
    assert head(tmp_path)["config"] == NEW

    # redelivered E1 does nothing.
    c.record_event(event("E1", NEW["model"], 9), at(32))
    c.step(at(32))
    assert len(opt.calls) == 1 and head(tmp_path)["generation"] == 1

    # a stale acceptance (generation 0) for a new deploy is refused; head unchanged.
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    to_deploy_awaiting(c, E6_REF, 40)
    assert len(opt.calls) == 2
    stale = c.accept(E6_REF, "approver-1", 0, now=at(41))
    assert (stale.result, stale.reason) == ("refused", "stale_generation")
    assert head(tmp_path)["generation"] == 1 and head(tmp_path)["config"] == NEW

    # re-running step with an unchanged history is a no-op.
    files = ("history.jsonl", "head.json", "heads.jsonl")
    before = {f: (tmp_path / "store" / f).read_bytes() for f in files}
    report = c.step(at(41))
    assert not report.changed and len(opt.calls) == 2
    assert before == {f: (tmp_path / "store" / f).read_bytes() for f in files}


def test_crash_between_optimize_and_record_never_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(E1_REF, "approver-1", at(10))

    real = Controller._append

    def crash(self: Controller, category: str, record: Any, instant: int) -> None:
        if record and record.get("fact") == "finished":
            raise RuntimeError("simulated crash before recording")
        real(self, category, record, instant)

    monkeypatch.setattr(Controller, "_append", crash)
    with pytest.raises(RuntimeError):
        c.step(at(10))
    monkeypatch.undo()
    assert len(opt.calls) == 1

    restarted = controller(tmp_path, opt)  # a new process over the same store
    restarted.step(at(11))
    assert len(opt.calls) == 1
    assert [d["occurrence"] for d in restarted.status()["pending_deploys"]] == [E1_REF]


def test_interrupted_attempt_is_not_started_twice(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(E1_REF, "approver-1", at(10))
    marker_dir = tmp_path / "store" / "attempts"
    c.step(at(10))  # runs and finishes
    marker = next(marker_dir.glob("*.json"))
    data = json.loads(marker.read_text(encoding="utf-8"))
    # A store holding only an in-progress marker (crash mid-optimize): refuse.
    c2 = Controller(state(), tmp_path / "other", opt, initial_config=OLD, now=T0)
    c2.record_event(event("E1", NEW["model"], 9), at(10))
    c2.approve_optimization(E1_REF, "approver-1", at(10))
    (tmp_path / "other" / "attempts").mkdir()
    (tmp_path / "other" / "attempts" / marker.name).write_text(
        json.dumps(
            {"attempt": "A1", "occurrence": data["occurrence"], "state": "started"}
        ),
        encoding="utf-8",
    )
    calls = len(opt.calls)
    with pytest.raises(TuningError) as err:
        c2.step(at(10))
    assert err.value.code == E_ATTEMPT_IN_PROGRESS and len(opt.calls) == calls
    report = c2.step(at(11), abandon_interrupted=True)
    assert report.finished == ["A1"] and len(opt.calls) == calls


def test_candidate_without_promotion_evidence_never_deploys(tmp_path: Path) -> None:
    opt = FakeOptimizer(result={"best_config": NEW})  # a bare result, no gate evidence
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, E1_REF, 10)
    assert len(opt.calls) == 1
    assert c.status()["pending_deploys"] == []
    with pytest.raises(TuningError) as err:
        c.accept(E1_REF, "approver-1", 0, now=at(31))
    assert err.value.code == E_NO_DEPLOY_AWAITING
    assert head(tmp_path)["generation"] == 0


def test_acceptance_with_wrong_contract_digest_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, E1_REF, 10)
    record = c.build_acceptance(E1_REF, "approver-1", 0, now=at(31))
    record["contract_digest"] = "sha256:" + "d" * 64
    outcome = c.submit_acceptance(record, at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "binding_mismatch")
    assert head(tmp_path)["generation"] == 0


def test_unauthorized_approver_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer(), authorized_approvers={"boss"})
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, E1_REF, 10)
    outcome = c.accept(E1_REF, "intern", 0, now=at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "unauthorized")
    assert head(tmp_path)["generation"] == 0


def test_invalid_record_never_lands_in_history(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    with pytest.raises(TuningError):
        c.record_event({"kind": "model_released"}, at(10))
    assert not (tmp_path / "store" / "history.jsonl").exists()


def test_every_executing_path_refuses_without_a_validated_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidatorUnavailable):
        Controller(state(validated=False), tmp_path / "s", None)
    c = controller(tmp_path, FakeOptimizer())
    c.state.validated = False
    for call in (
        lambda: c.step(at(1)),
        lambda: c.record_event(event("E1", "m", 0), at(1)),
        lambda: c.record_window({}, at(1)),
        lambda: c.approve_optimization(E1_REF, "a", at(1)),
        lambda: c.accept(E1_REF, "a", 0, now=at(1)),
        lambda: c.status(),
    ):
        with pytest.raises(ValidatorUnavailable):
            call()


def test_observe_follow_head_sees_the_new_config_and_stamps_generation(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    store = tmp_path / "store"
    client = MagicMock()
    seen: list[Any] = []

    @observe(
        "classify", client=client, agent="agent-1", follow="head", tuning_store=store
    )
    def classify() -> str:
        seen.append(dict(traigent.get_config()))
        return "ok"

    def stamp() -> dict[str, Any]:
        return client.start_trace.call_args.kwargs["metadata"]

    classify()
    assert seen[-1] == OLD
    assert stamp()["agent"] == "agent-1" and stamp()["head_generation"] == 0

    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, E1_REF, 10)
    c.accept(E1_REF, "approver-1", 0, now=at(31))

    classify()
    assert seen[-1] == NEW
    assert stamp()["head_generation"] == 1
    assert HeadLoader(store).generation() == 1
    assert config_context.get() != NEW  # scoped to the call


def test_fingerprint_is_the_tvl_canonical_digest() -> None:
    assert len(config_fingerprint(OLD)) == 64


def test_cli_drives_the_loop(tmp_path: Path) -> None:
    runner = CliRunner()
    doc = tmp_path / "module.yml"
    doc.write_text(yaml.safe_dump(module()), encoding="utf-8")
    ev = tmp_path / "e1.json"
    ev.write_text(json.dumps(event("E1", NEW["model"], 9)), encoding="utf-8")
    common = ["--tvl", str(doc), "--agent", "agent-1", "--store", str(tmp_path / "s")]

    def run(*args: str) -> Any:
        return runner.invoke(cli, ["tune", *args, *common])

    first = run("event", str(ev), "--now", "2026-11-20T10:00:00Z")
    assert first.exit_code == 0, first.output
    step = run("step", "--now", "2026-11-20T10:00:00Z")
    assert json.loads(step.output)["awaiting_approval"] == [E1_REF]
    status = run("status")
    assert json.loads(status.output)["open_attempts"][0]["state"] == "awaiting_approval"
    approve = run("approve", E1_REF, "--approver", "approver-1")
    assert approve.exit_code == 0, approve.output
    missing = run("accept", E1_REF, "--generation", "0", "--approver", "approver-1")
    assert missing.exit_code != 0  # no optimize_fn via --tvl: the step cannot run
