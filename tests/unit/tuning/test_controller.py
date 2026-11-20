"""The local loop controller drives the SLICE scenario end to end (fake optimizer)."""

from __future__ import annotations

import copy
import functools
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
from traigent.tuning.controller import (
    E_ACCEPTANCE_CONFLICT,
    E_ATTEMPT_IN_PROGRESS,
    E_INVALID_GRANT,
    E_NO_DEPLOY_AWAITING,
    E_STORE_CORRUPT,
    E_STORE_FORMAT,
    Controller,
    HeadLoader,
    config_fingerprint,
    occurrence_ref,
    traigent_optimize_fn,
)
from traigent.tuning.errors import TuningError, ValidatorUnavailable
from traigent.tuning.policy import Tuning
from traigent.tuning.state import TuningState

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


@functools.cache
def e_ref(event_id: str) -> str:
    """Resolved lazily: TVL is only imported once a test runs (and was not skipped)."""
    return _ref(event_id)


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
    assert report.awaiting_approval == [e_ref("E1")]
    assert opt.calls == []

    # approval -> optimize runs once with the admitted search domain.
    to_deploy_awaiting(c, e_ref("E1"), 10)
    assert len(opt.calls) == 1
    assert NEW["model"] in opt.calls[0]["model"]
    status = c.status()
    assert [d["occurrence"] for d in status["pending_deploys"]] == [e_ref("E1")]
    assert head(tmp_path)["generation"] == 0  # nothing deployed yet

    # human acceptance at generation 0 -> head generation 1, the new model.
    outcome = c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert (outcome.result, outcome.generation) == ("written", 1)
    assert head(tmp_path)["generation"] == 1
    assert head(tmp_path)["config"] == NEW

    # redelivered E1 does nothing.
    c.record_event(event("E1", NEW["model"], 9), at(32))
    c.step(at(32))
    assert len(opt.calls) == 1 and head(tmp_path)["generation"] == 1

    # a stale acceptance (generation 0) for a new deploy is refused; head unchanged.
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    to_deploy_awaiting(c, e_ref("E6"), 40)
    assert len(opt.calls) == 2
    stale = c.accept(e_ref("E6"), "approver-1", 0, now=at(41))
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
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))

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
    assert [d["occurrence"] for d in restarted.status()["pending_deploys"]] == [
        e_ref("E1")
    ]


def test_interrupted_attempt_is_not_started_twice(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    marker_dir = tmp_path / "store" / "attempts"
    c.step(at(10))  # runs and finishes
    marker = next(marker_dir.glob("*.json"))
    data = json.loads(marker.read_text(encoding="utf-8"))
    # A store holding only an in-progress marker (crash mid-optimize): refuse.
    c2 = Controller(state(), tmp_path / "other", opt, initial_config=OLD, now=T0)
    c2.record_event(event("E1", NEW["model"], 9), at(10))
    c2.approve_optimization(e_ref("E1"), "approver-1", at(10))
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
    to_deploy_awaiting(c, e_ref("E1"), 10)
    assert len(opt.calls) == 1
    assert c.status()["pending_deploys"] == []
    with pytest.raises(TuningError) as err:
        c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert err.value.code == E_NO_DEPLOY_AWAITING
    assert head(tmp_path)["generation"] == 0


def test_acceptance_with_wrong_contract_digest_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    record["contract_digest"] = "sha256:" + "d" * 64
    outcome = c.submit_acceptance(record, at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "binding_mismatch")
    assert head(tmp_path)["generation"] == 0


def test_unauthorized_approver_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer(), authorized_approvers={"boss"})
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    outcome = c.accept(e_ref("E1"), "intern", 0, now=at(31))
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
        lambda: c.approve_optimization(e_ref("E1"), "a", at(1)),
        lambda: c.accept(e_ref("E1"), "a", 0, now=at(1)),
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
    to_deploy_awaiting(c, e_ref("E1"), 10)
    c.accept(e_ref("E1"), "approver-1", 0, now=at(31))

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
    assert json.loads(step.output)["awaiting_approval"] == [e_ref("E1")]
    status = run("status")
    assert json.loads(status.output)["open_attempts"][0]["state"] == "awaiting_approval"
    approve = run("approve", e_ref("E1"), "--approver", "approver-1")
    assert approve.exit_code == 0, approve.output
    missing = run(
        "accept", e_ref("E1"), "--generation", "0", "--approver", "approver-1"
    )
    assert missing.exit_code != 0  # no optimize_fn via --tvl: the step cannot run


# ---------------------------------------------------------------------------------
# second-model review fixes
# ---------------------------------------------------------------------------------
SENTINEL = "SENTINEL-PROMPT-TEXT-7f3a"


def _deploy_awaiting(tmp_path: Path, opt: Any = None, **kw: Any) -> Controller:
    c = controller(tmp_path, opt or FakeOptimizer(), **kw)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    return c


def test_conflicting_acceptance_replay_never_changes_the_written_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = Controller._write_head

    def crash(self: Controller, trace: Any) -> Any:
        raise RuntimeError("simulated crash before the head write")

    monkeypatch.setattr(Controller, "_write_head", crash)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))  # recorded, head not yet written
    monkeypatch.setattr(Controller, "_write_head", real)

    conflicting = {**record, "candidate_fingerprint": config_fingerprint(OLD)}
    with pytest.raises(TuningError) as err:
        c.submit_acceptance(conflicting, at(32))
    assert err.value.code == E_ACCEPTANCE_CONFLICT
    lines = (tmp_path / "store" / "history.jsonl").read_text().splitlines()
    assert sum('"acceptances"' in ln for ln in lines) == 1  # the replay never landed

    restarted = controller(tmp_path, FakeOptimizer())
    assert restarted.step(at(33)).head_written == 1
    assert head(tmp_path)["config"] == NEW  # the exact record, not the replay


def test_follow_head_trace_metadata_is_closed(tmp_path: Path) -> None:
    secret = {"model": "m", "prompt": SENTINEL}
    Controller(state(), tmp_path / "store", None, initial_config=secret, now=T0)
    client = MagicMock()
    seen: list[Any] = []

    @observe(
        "classify",
        client=client,
        agent="agent-1",
        follow="head",
        tuning_store=tmp_path / "store",
    )
    def classify() -> str:
        seen.append(dict(traigent.get_config()))
        return "ok"

    classify()
    assert seen == [secret]  # the call itself still runs on the head config
    emitted = json.dumps(client.mock_calls, default=repr)
    assert SENTINEL not in emitted
    metadata = client.start_trace.call_args.kwargs["metadata"]
    assert metadata["agent"] == "agent-1" and metadata["head_generation"] == 0
    assert metadata["head_fingerprint"] == config_fingerprint(secret)
    assert "traigent_active_config" not in metadata


def _standing_state() -> TuningState:
    doc = module()
    for response in doc["responses"]:
        for rule in response["rules"]:
            if rule["id"] == "on_new_model":
                rule["approval"] = "standing"
    # the policy builder only knows human approval; the controller reads ``tvl``
    return TuningState(
        agent="agent-1", policy=Tuning.from_tvl(module()), tvl=doc, validated=True
    )


def _standing(tmp_path: Path, opt: Any) -> Controller:
    return Controller(
        _standing_state(), tmp_path / "store", opt, initial_config=OLD, now=T0
    )


def test_standing_rule_optimizes_only_under_a_recorded_grant(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]

    with pytest.raises(TuningError) as err:  # expiry beyond 30 days
        c.record_standing_grant(
            "G0", "tuning/on_new_model", 1, at(10) + timedelta(days=31), "owner", at(10)
        )
    assert err.value.code == E_INVALID_GRANT

    c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(days=30), "owner", at(10)
    )
    c.step(at(10))
    c.step(at(10))  # a retry of the same occurrence does not consume again
    assert len(opt.calls) == 1
    assert c.accept(e_ref("E1"), "approver-1", 0, now=at(31)).result == "written"

    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    report = c.step(at(40))  # the grant allowed one occurrence
    assert len(opt.calls) == 1 and report.awaiting_approval == [e_ref("E6")]


def test_expired_standing_grant_does_not_authorize(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(10) + timedelta(hours=2), "owner", at(10)
    )
    c.record_event(event("E1", NEW["model"], 19), at(20))
    c.step(at(20))
    assert opt.calls == []


class _Live:
    """A real mock-mode ``@traigent.optimize`` function and what it serves."""

    @staticmethod
    def build(**extra: Any) -> Any:
        @traigent.optimize(
            configuration_space={"model": [OLD["model"], NEW["model"]]},
            objectives=["accuracy"],
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
            offline=True,
            max_trials=2,
            default_config=dict(OLD),
            **extra,
        )
        def live(q: str) -> str:
            return "a"

        return live


def test_real_adapter_runs_optimization_without_changing_the_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    live = _Live.build()
    before = (dict(live.current_config), live.get_best_config(), live._state)
    result = traigent_optimize_fn(live, algorithm="grid", max_trials=2)(
        {"model": [OLD["model"], NEW["model"]]}
    )
    assert result.trials  # the optimization really ran
    assert (dict(live.current_config), live.get_best_config(), live._state) == before
    assert len(live._optimization_history) == 0


def test_head_write_rechecks_the_approver_at_effect_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path, authorized_approvers={"boss"})
    real = Controller._write_head

    def crash(self: Controller, trace: Any) -> Any:
        raise RuntimeError("simulated crash before the head write")

    monkeypatch.setattr(Controller, "_write_head", crash)
    with pytest.raises(RuntimeError):
        c.accept(e_ref("E1"), "boss", 0, now=at(31))  # derived, not yet written
    monkeypatch.setattr(Controller, "_write_head", real)

    revoked = controller(tmp_path, FakeOptimizer(), authorized_approvers={"other"})
    assert revoked.step(at(32)).head_written is None
    assert head(tmp_path)["generation"] == 0  # the head file never moved
    assert [f["fact"] for f in revoked.controller_facts()] == ["head_write_refused"]

    restored = controller(tmp_path, FakeOptimizer(), authorized_approvers={"boss"})
    assert restored.step(at(33)).head_written == 1
    assert head(tmp_path)["config"] == NEW


def test_expired_scope_blocks_a_late_approved_optimization(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    late = at(10) + timedelta(days=31)
    c.approve_optimization(e_ref("E1"), "approver-1", late)
    report = c.step(late)
    assert opt.calls == [] and report.started == []
    assert report.refused and report.refused[0].endswith(":scope_expired")
    assert [f["reason"] for f in c.controller_facts()] == ["scope_expired"]
    assert not list((tmp_path / "store" / "attempts").glob("*.json"))


@pytest.mark.parametrize("failing", ["head.json", "heads.jsonl", "meta.json"])
def test_init_crash_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    from traigent.tuning import controller as module_

    real_write, real_append = module_._write_json, module_._append_line

    def write(path: Path, value: Any) -> None:
        if path.name == failing:
            raise RuntimeError("simulated crash during init")
        real_write(path, value)

    def append(path: Path, value: Any) -> None:
        if path.name == failing:
            raise RuntimeError("simulated crash during init")
        real_append(path, value)

    monkeypatch.setattr(module_, "_write_json", write)
    monkeypatch.setattr(module_, "_append_line", append)
    with pytest.raises(RuntimeError):
        controller(tmp_path, FakeOptimizer())
    monkeypatch.undo()

    c = controller(tmp_path, FakeOptimizer())  # the next start finishes the init
    assert head(tmp_path) == {
        "config": OLD,
        "fingerprint": config_fingerprint(OLD),
        "generation": 0,
    }
    assert c.status()["head"]["generation"] == 0
    assert len((tmp_path / "store" / "heads.jsonl").read_text().splitlines()) == 1


def test_torn_history_tail_is_truncated_and_earlier_lines_verified(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    path = tmp_path / "store" / "history.jsonl"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 25])  # a crash mid-append of the last line

    restarted = controller(tmp_path, FakeOptimizer())
    assert len(path.read_text().splitlines()) == 1  # torn tail truncated
    assert len(restarted.status()["open_attempts"]) == 1  # E1 survives
    restarted.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    assert len(path.read_text().splitlines()) == 2


def test_damaged_earlier_line_is_corruption_not_a_torn_tail(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    path = tmp_path / "store" / "history.jsonl"
    first, second = path.read_text().splitlines()
    path.write_text(first.replace("E1", "EX") + "\n" + second + "\n")
    with pytest.raises(TuningError) as err:
        controller(tmp_path, FakeOptimizer())
    assert err.value.code == E_STORE_CORRUPT


def test_required_env_makes_a_missing_toolchain_fail_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.tuning import conftest

    monkeypatch.setattr(conftest.validator, "validator_available", lambda: False)
    monkeypatch.delenv(conftest.REQUIRED_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        conftest.check_toolchain()
    monkeypatch.setenv(conftest.REQUIRED_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        conftest.check_toolchain()


def _grant_fixture(tmp_path: Path) -> tuple[Controller, FakeOptimizer]:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(hours=2), "owner", at(10)
    )
    c.record_event(event("E1", NEW["model"], 9), at(10))
    return c, opt


@pytest.mark.parametrize("crash_at", ["before_optimizer", "before_approval_record"])
def test_recovery_revalidates_an_expired_standing_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: str
) -> None:
    c, opt = _grant_fixture(tmp_path)
    if crash_at == "before_optimizer":
        c.optimize_fn = None  # the approval persists, then the run never starts
        with pytest.raises(TuningError):
            c.step(at(11))
    else:
        real = Controller._append

        def crash(self: Controller, category: str, record: Any, instant: int) -> None:
            if record and record.get("fact") == "approval":
                raise RuntimeError("simulated crash before the approval record")
            real(self, category, record, instant)

        monkeypatch.setattr(Controller, "_append", crash)
        with pytest.raises(RuntimeError):
            c.step(at(11))  # the grant use is persisted, the approval is not
        monkeypatch.undo()

    restarted = Controller(
        _standing_state(), tmp_path / "store", opt, initial_config=OLD, now=T0
    )
    report = restarted.step(at(20))  # the grant expired at hour 12
    assert opt.calls == [] and report.started == []
    if crash_at == "before_optimizer":
        assert report.refused and report.refused[0].endswith(":grant_expired")
        reasons = [f["reason"] for f in restarted.controller_facts()]
        assert reasons == ["grant_expired"]
    else:  # the used-reference shortcut no longer revives an expired grant
        assert report.awaiting_approval == [e_ref("E1")]
    assert not list((tmp_path / "store" / "attempts").glob("*.json"))


def test_newline_terminated_bad_record_is_corruption_never_truncated(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    path = tmp_path / "store" / "history.jsonl"
    first, second = path.read_text().splitlines()
    path.write_text(first + "\n" + second.replace("E6", "EX") + "\n")  # committed
    before = path.read_bytes()
    with pytest.raises(TuningError) as err:
        controller(tmp_path, FakeOptimizer())
    assert err.value.code == E_STORE_CORRUPT
    assert path.read_bytes() == before


def test_legacy_plain_json_store_is_refused_untouched(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    store = tmp_path / "store"
    for name in ("history.jsonl", "heads.jsonl"):
        path = store / name
        lines = [ln.partition("\t")[2] for ln in path.read_text().splitlines()]
        path.write_text("\n".join(lines))  # old format, even without a final newline
    before = {n: (store / n).read_bytes() for n in ("history.jsonl", "heads.jsonl")}
    with pytest.raises(TuningError) as err:
        controller(tmp_path, FakeOptimizer())
    assert err.value.code == E_STORE_FORMAT
    assert str(store) in str(err.value) and "re-initialize" in str(err.value)
    assert before == {n: (store / n).read_bytes() for n in before}


def test_candidate_run_never_leaks_into_the_live_best_config_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.delenv("TRAIGENT_RESULTS_FOLDER", raising=False)
    monkeypatch.delenv("TRAIGENT_OPTIMIZATION_LOG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    live = _Live.build()
    candidate_dir = tmp_path / "store" / "candidates" / "attempt-1"
    result = traigent_optimize_fn(live, algorithm="grid", max_trials=2)(
        {"model": [OLD["model"], NEW["model"]]}, candidate_dir=candidate_dir
    )
    assert result.trials
    # the candidate's artifacts exist, inside the store
    assert list(candidate_dir.rglob("best_config*.json"))
    # nothing landed where a live wrapper looks (cwd, home, results folder)
    assert not list(tmp_path.glob(".traigent/**/best_config*.json"))
    assert not list((tmp_path / "home").rglob("best_config*.json"))
    assert "TRAIGENT_RESULTS_FOLDER" not in __import__("os").environ
    # a fresh live wrapper does not pick the candidate up
    fresh = _Live.build(auto_load_best=True)
    assert dict(fresh.current_config) == OLD
