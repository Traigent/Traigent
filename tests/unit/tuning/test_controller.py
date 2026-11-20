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
from traigent.tuning.authority import ApproverAuthority, EventVerifier, sign_event
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
from traigent.tuning.errors import (
    E_APPROVER_NOT_AUTHORIZED,
    E_NO_APPROVER_AUTHORITY,
    TuningError,
    ValidatorUnavailable,
)
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
NEWER = {"model": "anthropic/claude-haiku-5-6"}
SECRET = b"test-issuer-secret"  # pragma: allowlist secret
ISSUER_ENV = "TRAIGENT_TEST_ISSUER_P"
APPROVERS = frozenset({"approver-1", "issuer-1", "boss"})
ROWS = 1000
KIT: dict[str, Any] = {}


def verifier() -> EventVerifier:
    return EventVerifier({"P": SECRET})


def _skill_agent(x: int) -> str:
    """A synthetic ticket router: each model gets a different share of rows wrong."""
    from traigent.config.context import config_context

    model = (config_context.get() or {}).get("model")
    right = "billing" if x % 2 == 0 else "support"
    wrong = {
        OLD["model"]: x % 4 >= 2,  # half
        NEW["model"]: x % 34 == 33,  # about 3%
    }.get(model, False)
    return ("support" if right == "billing" else "billing") if wrong else right


def _route_check(example: Any, output: Any) -> str:
    return "pass" if output == example["output"] else "fail"


@pytest.fixture(autouse=True)
def _holdout(tmp_path: Path) -> None:
    path = tmp_path / "kit-holdout.jsonl"
    rows = [
        {"input": {"x": i}, "output": "billing" if i % 2 == 0 else "support"}
        for i in range(ROWS)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    KIT["holdout"] = path


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
    doc = copy.deepcopy(case["module"])
    # the local evidence measures accuracy (exact match on the holdout)
    doc["objectives"] = [{"name": "accuracy", "direction": "maximize"}]
    doc["promotion_policy"]["min_effect"] = {"accuracy": 0.02}
    return doc


def kit(policy: Tuning) -> Tuning:
    """Wire the customer-side promotion evidence: checker, labels, holdout."""
    policy.checkers = {"customer:ticket_route_check": _route_check}
    policy.checker_labels = {"customer:ticket_route_check": ("pass", "fail")}
    policy.holdout = KIT["holdout"]
    return policy


def state(doc: dict[str, Any] | None = None, *, validated: bool = True) -> TuningState:
    doc = doc or module()
    st = TuningState(
        agent="agent-1",
        policy=kit(Tuning.from_tvl(doc)),
        tvl=doc,
        validated=validated,
    )
    st.optimized = _skill_agent
    return st


def event(
    event_id: str, model: str, issued_hour: int, *, secret: bytes | None = SECRET
) -> dict[str, Any]:
    """A signed ``model_released`` envelope (``issued_hour``: hours after the origin)."""
    envelope: dict[str, Any] = {
        "kind": "model_released",
        "event_id": event_id,
        "issuer": "P",
        "issued_at": issued_hour * 3600,
        "catalogue_revision": 1,
        "subject": {
            "model_ref": model,
            "provider_family": "anthropic",
            "tier": "small",
        },
    }
    if secret is not None:
        envelope["signature"] = sign_event(envelope, secret)
    return envelope


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
                    # the optimizer proposes a config; the controller measures it
                    "config": {"model": space["model"][-1]},
                }
            ]
        }


def make(st: TuningState, store: Path, optimize: Any, **kw: Any) -> Controller:
    """A controller with the test authority and trusted issuer configured."""
    kw.setdefault("authorized_approvers", APPROVERS)
    kw.setdefault("event_verifier", verifier())
    kw.setdefault("initial_config", OLD)
    kw.setdefault("now", T0)
    return Controller(st, store, optimize, **kw)


def controller(tmp_path: Path, optimize: Any, **kw: Any) -> Controller:
    return make(state(), tmp_path / "store", optimize, **kw)


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
    c2 = make(state(), tmp_path / "other", opt)
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
    opt = FakeOptimizer(result={"best_config": NEW})  # a bare result
    c = controller(tmp_path, opt)
    c.state.policy.holdout = None  # no holdout: nothing to measure the candidate on
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    assert len(opt.calls) == 1
    assert c.status()["pending_deploys"] == []
    with pytest.raises(TuningError) as err:
        c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert err.value.code == E_NO_DEPLOY_AWAITING
    assert head(tmp_path)["generation"] == 0
    assert any(f.get("reason") == "holdout_missing" for f in c.controller_facts())


def test_acceptance_with_wrong_contract_digest_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    record["contract_digest"] = "sha256:" + "d" * 64
    outcome = c.submit_acceptance(record, at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "binding_mismatch")
    assert head(tmp_path)["generation"] == 0


def test_unauthorized_approver_is_refused_on_both_decisions(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer(), authorized_approvers={"boss"})
    c.record_event(event("E1", NEW["model"], 9), at(10))
    with pytest.raises(TuningError) as err:  # Approve run
        c.approve_optimization(e_ref("E1"), "intern", at(10))
    assert err.value.code == E_APPROVER_NOT_AUTHORIZED
    c.approve_optimization(e_ref("E1"), "boss", at(10))
    c.step(at(10))
    outcome = c.accept(e_ref("E1"), "intern", 0, now=at(31))  # Accept release
    # local-trusted only vouches for listed operators: an unlisted one is unknown
    assert (outcome.result, outcome.reason) == ("invalid", "unauthenticated")
    assert head(tmp_path)["generation"] == 0


def test_platform_authenticated_but_unlisted_approver_is_unauthorized(
    tmp_path: Path,
) -> None:
    platform = ApproverAuthority(
        frozenset({"boss"}), mode="platform", authenticator=lambda who, _ctx: True
    )
    c = controller(
        tmp_path, FakeOptimizer(), authority=platform, authorized_approvers=None
    )
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "boss", at(10))
    c.step(at(10))
    outcome = c.accept(e_ref("E1"), "intern", 0, now=at(31))
    assert (outcome.result, outcome.reason) == ("invalid", "unauthorized")
    assert c.accept(e_ref("E1"), "boss", 0, now=at(32)).result == "written"


def test_platform_authority_without_an_authenticator_is_refused() -> None:
    with pytest.raises(TuningError):
        ApproverAuthority(frozenset({"boss"}), mode="platform")


def test_authenticator_failure_never_authenticates(tmp_path: Path) -> None:
    def broken(_who: str, _ctx: Any) -> bool:
        raise RuntimeError("platform unreachable")

    platform = ApproverAuthority(
        frozenset({"boss"}), mode="platform", authenticator=broken
    )
    assert platform.authenticated("boss", {}) is False


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


def _cli_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, list[str]]:
    monkeypatch.setenv(ISSUER_ENV, SECRET.decode())
    doc = tmp_path / "module.yml"
    doc.write_text(yaml.safe_dump(module()), encoding="utf-8")
    auth = tmp_path / "authority.yml"
    auth.write_text(
        yaml.safe_dump(
            {
                "mode": "local-trusted",
                "approvers": ["approver-1"],
                "event_issuers": {"P": {"secret_env": ISSUER_ENV}},
            }
        ),
        encoding="utf-8",
    )
    common = ["--tvl", str(doc), "--agent", "agent-1", "--store", str(tmp_path / "s")]
    return auth, common


def test_cli_drives_the_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    auth, common = _cli_fixture(tmp_path, monkeypatch)
    ev = tmp_path / "e1.json"
    ev.write_text(json.dumps(event("E1", NEW["model"], 9)), encoding="utf-8")

    def run(*args: str, authority: bool = True) -> Any:
        extra = ["--authority", str(auth)] if authority else []
        return runner.invoke(cli, ["tune", *args, *common, *extra])

    first = run("event", str(ev), "--now", "2026-11-20T10:00:00Z")
    assert first.exit_code == 0, first.output
    assert json.loads(first.output)["authenticated"] is True
    step = run("step", "--now", "2026-11-20T10:00:00Z")
    assert json.loads(step.output)["awaiting_approval"] == [e_ref("E1")]
    status = json.loads(run("status").output)
    assert status["open_attempts"][0]["state"] == "awaiting_approval"
    view = status["open_attempts"][0]["approve_run"]  # what Approve run shows
    assert view["trigger"]["external"] == "model_released"
    assert "spend_limit" in view

    # no authority configured: neither human decision can be given
    for args in (
        ("approve", e_ref("E1"), "--approver", "approver-1"),
        ("accept", e_ref("E1"), "--generation", "0", "--approver", "approver-1"),
    ):
        refused = run(*args, authority=False)
        assert refused.exit_code != 0 and E_NO_APPROVER_AUTHORITY in refused.output
    unlisted = run("approve", e_ref("E1"), "--approver", "intern")
    assert unlisted.exit_code != 0 and E_APPROVER_NOT_AUTHORIZED in unlisted.output

    approve = run("approve", e_ref("E1"), "--approver", "approver-1")
    assert approve.exit_code == 0, approve.output
    assert json.loads(approve.output)["authentication"] == "local-trusted"
    missing = run(
        "accept", e_ref("E1"), "--generation", "0", "--approver", "approver-1"
    )
    assert missing.exit_code != 0  # no optimize_fn via --tvl: nothing awaits


def test_cli_unauthenticated_event_fires_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    auth, common = _cli_fixture(tmp_path, monkeypatch)
    forged = event("E1", NEW["model"], 9, secret=None)
    forged["authenticated"] = True  # the file's own claim is never read
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    for extra in (["--authority", str(auth)], []):
        result = runner.invoke(
            cli,
            [
                "tune",
                "event",
                str(path),
                *common,
                *extra,
                "--now",
                "2026-11-20T10:00:00Z",
            ],
        )
        assert result.exit_code == 0
        assert (
            json.loads(result.output[result.output.index("{") :])["authenticated"]
            is False
        )
    step = runner.invoke(
        cli,
        [
            "tune",
            "step",
            *common,
            "--authority",
            str(auth),
            "--now",
            "2026-11-20T10:00:00Z",
        ],
    )
    assert json.loads(step.output)["awaiting_approval"] == []


def test_cli_surfaces_the_tvl_diagnostic_for_release_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release rollback is deferred out of TVL 1.7: the TVL code is shown unchanged."""
    auth, common = _cli_fixture(tmp_path, monkeypatch)
    doc = module()
    doc["responses"][0]["rules"].append(
        {
            "id": "undo",
            "trigger": {"monitor": "quality_prod", "event": "recovery_overdue"},
            "do": "rollback",
            "target": "release_rollback_target",
            "approval": "human",
        }
    )
    path = tmp_path / "rollback.yml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "tune",
            "step",
            "--tvl",
            str(path),
            "--agent",
            "agent-1",
            "--store",
            str(tmp_path / "r"),
        ],
    )
    assert result.exit_code != 0
    assert "release_rollback_deferred" in result.output
    assert "E_TVL_VALIDATION_FAILED" in result.output


# ---------------------------------------------------------------------------------
# second-model review fixes
# ---------------------------------------------------------------------------------
SENTINEL = "SENTINEL-PROMPT-TEXT-7f3a"


def _deploy_awaiting(tmp_path: Path, opt: Any = None, **kw: Any) -> Controller:
    c = controller(tmp_path, opt or FakeOptimizer(), **kw)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    return c


def _crash_install(monkeypatch: pytest.MonkeyPatch, *, after: bool = False) -> Any:
    """Make the head install die (``after``: right after head.json is written)."""
    real = Controller._install_head

    def crash(self: Controller, head_: Any) -> None:
        if after:
            real(self, head_)
        raise RuntimeError("simulated crash at the head install")

    monkeypatch.setattr(Controller, "_install_head", crash)
    return real


def test_conflicting_acceptance_replay_never_changes_the_written_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = _crash_install(monkeypatch)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))  # intent journalled, head not installed
    monkeypatch.setattr(Controller, "_install_head", real)

    conflicting = {**record, "candidate_fingerprint": config_fingerprint(OLD)}
    with pytest.raises(TuningError) as err:
        c.submit_acceptance(conflicting, at(32))
    assert err.value.code == E_ACCEPTANCE_CONFLICT
    lines = (tmp_path / "store" / "history.jsonl").read_text().splitlines()
    assert not any('"acceptances"' in ln for ln in lines)  # nothing landed twice

    outcome = c.submit_acceptance(record, at(33))  # the exact record, still valid
    assert (outcome.result, outcome.generation) == ("written", 1)
    assert head(tmp_path)["config"] == NEW  # the exact record, not the replay


def test_physical_writer_checks_expiry_authority_and_generation_at_the_write(
    tmp_path: Path,
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    current = head(tmp_path)
    instant = c._instant(at(31))
    assert c._physical_refusal(record, instant, current) is None
    week = 7 * 86400
    assert c._physical_refusal(record, instant + week, current) == (
        "acceptance_not_usable"  # t >= expires_at
    )
    assert c._physical_refusal(record, instant - 3600, current) == (
        "acceptance_not_usable"  # t < decided_at
    )
    assert c._physical_refusal(record, instant, {**current, "generation": 3}) == (
        "stale_generation"
    )
    assert c._physical_refusal(
        record, instant, {**current, "fingerprint": "f" * 64}
    ) == ("rollback_target_mismatch")
    marker = next((tmp_path / "store" / "attempts").glob("*.json"))
    data = json.loads(marker.read_text())
    (gate,) = data["gates"].values()
    gate["incumbent_generation"] = 7  # the gate measured some other incumbent
    marker.write_text(json.dumps(data), encoding="utf-8")
    assert c._physical_refusal(record, instant, current) == "gate_incumbent_stale"
    c.authority = ApproverAuthority(frozenset({"someone-else"}))  # revoked
    assert c._physical_refusal(record, instant, current) == "approver_not_authorized"


def test_write_the_relation_allows_but_the_head_refuses_is_journalled_not_consumed(
    tmp_path: Path,
) -> None:
    c = _deploy_awaiting(tmp_path)
    marker = next((tmp_path / "store" / "attempts").glob("*.json"))
    data = json.loads(marker.read_text())
    data["configs"] = {}  # the candidate configuration is no longer retrievable
    marker.write_text(json.dumps(data), encoding="utf-8")
    outcome = c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert (outcome.result, outcome.reason) == ("refused", "candidate_config_missing")
    assert head(tmp_path)["generation"] == 0
    lines = (tmp_path / "store" / "history.jsonl").read_text().splitlines()
    assert not any('"acceptances"' in ln for ln in lines)  # relation never moved
    assert [f["fact"] for f in c.controller_facts()][-1] == "head_write_refused"
    c.step(at(32))  # relation and head.json agree: no E_HEAD_DIVERGED
    assert len(c.status()["pending_deploys"]) == 1  # still awaiting a decision


def test_crash_before_install_never_installs_an_expired_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = _crash_install(monkeypatch)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))
    monkeypatch.setattr(Controller, "_install_head", real)

    restarted = controller(tmp_path, FakeOptimizer())  # eight days later
    late = at(31) + timedelta(days=8)
    report = restarted.step(late)
    assert report.head_written is None
    assert head(tmp_path)["generation"] == 0 and head(tmp_path)["config"] == OLD
    assert restarted._tvl_generation(restarted._derive()) == 0  # no divergence
    outcome = restarted.submit_acceptance(record, late)  # expired: never installed
    assert (outcome.result, outcome.reason) == ("invalid", "expired")
    assert head(tmp_path)["generation"] == 0
    reasons = [f.get("reason") for f in restarted.controller_facts()]
    assert "interrupted_before_install" in reasons


def test_crash_after_install_is_completed_at_the_original_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    real = _crash_install(monkeypatch, after=True)
    with pytest.raises(RuntimeError):
        c.submit_acceptance(record, at(31))  # head.json moved, history not yet
    monkeypatch.setattr(Controller, "_install_head", real)
    assert head(tmp_path)["generation"] == 1

    restarted = controller(tmp_path, FakeOptimizer())  # recovery runs at start
    assert restarted._tvl_generation(restarted._derive()) == 1  # same generation
    assert head(tmp_path)["config"] == NEW
    acceptances = [
        json.loads(ln.partition("\t")[2])
        for ln in (tmp_path / "store" / "history.jsonl").read_text().splitlines()
        if '"acceptances"' in ln
    ]
    assert [a["instant"] for a in acceptances] == [c._instant(at(31))]
    replay = restarted.submit_acceptance(record, at(40))  # exact redelivery
    assert (replay.result, replay.generation) == ("written", 1)
    assert head(tmp_path)["generation"] == 1  # nothing written twice


def test_outcome_is_the_committed_result(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    outcome = c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    assert (outcome.result, outcome.generation) == ("written", 1)
    assert head(tmp_path)["generation"] == 1 == outcome.generation
    done = [
        json.loads(ln.partition("\t")[2])["fact"]
        for ln in (tmp_path / "store" / "commits.jsonl").read_text().splitlines()
    ]
    assert done == ["commit_intent", "commit_done"]


def test_a_reissued_acceptance_after_an_unauthorized_attempt_is_not_a_conflict(
    tmp_path: Path,
) -> None:
    c = _deploy_awaiting(tmp_path, authorized_approvers={"approver-1", "boss"})
    first = c.accept(e_ref("E1"), "intern", 0, now=at(31))
    assert (first.result, first.reason) == ("invalid", "unauthenticated")
    second = c.accept(e_ref("E1"), "boss", 0, now=at(32))  # new approver, new id
    assert (second.result, second.generation) == ("written", 1)


def test_a_reissued_acceptance_after_expiry_is_not_a_conflict(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    old = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31), nonce="n1")
    expired = c.submit_acceptance(old, at(31) + timedelta(days=8))
    assert (expired.result, expired.reason) == ("invalid", "expired")
    fresh = c.build_acceptance(
        e_ref("E1"), "approver-1", 0, now=at(31) + timedelta(days=9), nonce="n2"
    )
    assert fresh["acceptance_id"] != old["acceptance_id"]
    assert c.submit_acceptance(fresh, at(31) + timedelta(days=9)).result == "written"


def test_acceptance_id_depends_on_approver_instant_and_nonce(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    base = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31), nonce="n")
    same = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31), nonce="n")
    assert same["acceptance_id"] == base["acceptance_id"]
    for other in (
        c.build_acceptance(e_ref("E1"), "boss", 0, now=at(31), nonce="n"),
        c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(32), nonce="n"),
        c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31), nonce="m"),
    ):
        assert other["acceptance_id"] != base["acceptance_id"]


def test_follow_head_trace_metadata_is_closed(tmp_path: Path) -> None:
    secret = {"model": "m", "prompt": SENTINEL}
    make(state(), tmp_path / "store", None, initial_config=secret)
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
                rule["deploy"] = "never"  # standing + if_promoted is refused
    # the policy builder only knows human approval; the controller reads ``tvl``
    st = state()
    st.tvl = doc
    return st


def _standing(tmp_path: Path, opt: Any) -> Controller:
    return make(_standing_state(), tmp_path / "store", opt)


def test_standing_rule_optimizes_only_under_a_recorded_grant(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]

    with pytest.raises(TuningError) as err:  # expiry beyond 30 days
        c.record_standing_grant(
            "G0",
            "tuning/on_new_model",
            1,
            at(10) + timedelta(days=31),
            "issuer-1",
            at(10),
        )
    assert err.value.code == E_INVALID_GRANT

    c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(days=30), "issuer-1", at(10)
    )
    c.step(at(10))
    c.step(at(10))  # a retry of the same occurrence does not consume again
    assert len(opt.calls) == 1

    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    report = c.step(at(40))  # the grant allowed one occurrence
    assert len(opt.calls) == 1 and report.awaiting_approval == [e_ref("E6")]


def test_expired_standing_grant_does_not_authorize(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(10) + timedelta(hours=2), "issuer-1", at(10)
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


def test_result_after_scope_expiry_is_recorded_never_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late result or approval for a scope-expired attempt is recorded, never acted on.

    The relation closes the attempt (``scope_expired``) and answers the late record with
    an ``attempt_fact`` (``ignored(scope_closed)``); the controller must not crash on it,
    must not re-run the optimizer, and must not offer or write the candidate.
    """
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

    late = at(10) + timedelta(days=31)
    restarted = controller(tmp_path, opt)
    restarted.step(late)
    assert len(opt.calls) == 1
    assert restarted.status()["pending_deploys"] == []
    assert head(tmp_path)["config"] == OLD

    restarted._append(
        "optimizations",
        {"attempt": "A1", "fact": "approval", "state": "granted"},
        int((late - T0).total_seconds()),
    )
    restarted.step(late)
    facts = restarted._derive()["facts"]
    assert any(
        f["fact"] == "attempt_closed"
        and f["attempt"] == "A1"
        and f["reason"] == "scope_expired"
        for f in facts
    )
    assert any(
        f["fact"] == "attempt_fact" and f["result"] == "ignored(scope_closed)"
        for f in facts
    )
    assert len(opt.calls) == 1
    assert restarted.status()["pending_deploys"] == []
    assert head(tmp_path)["config"] == OLD


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
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(hours=2), "issuer-1", at(10)
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

    restarted = make(_standing_state(), tmp_path / "store", opt)
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


def test_candidate_config_mutated_during_evaluation_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())

    def mutating(_attempt: str, config: Any, _incumbent: Any) -> None:
        config["opts"]["threshold"] = 2  # the fingerprint was taken at threshold 1
        return None

    monkeypatch.setattr(c, "_measure_candidate", mutating)
    original = {"model": NEW["model"], "opts": {"threshold": 1}}
    completion, candidates, configs, gates = c._map_result(
        "a1", {"best_config": original}
    )
    assert candidates == [{"evidence_valid": False}]
    assert configs == {} and gates == {}
    assert original["opts"]["threshold"] == 1  # the caller's object is untouched


# ---------------------------------------------------------------------------------
# clock: whole seconds, the relation's own, never in the future
# ---------------------------------------------------------------------------------
def _history(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(ln.partition("\t")[2])
        for ln in (tmp_path / "store" / "history.jsonl").read_text().splitlines()
    ]


def test_history_instants_are_whole_seconds_after_the_origin(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    moment = T0 + timedelta(hours=10, minutes=37, seconds=12)
    c.record_event(event("E1", NEW["model"], 9), moment)
    assert [ln["instant"] for ln in _history(tmp_path)] == [10 * 3600 + 37 * 60 + 12]
    assert c.origin == T0  # UTC midnight: windows of 24 h line up with calendar days


def test_acceptance_is_recorded_at_the_real_second_never_in_the_future(
    tmp_path: Path,
) -> None:
    c = _deploy_awaiting(tmp_path)
    moment = at(31) + timedelta(minutes=29, seconds=41)  # not on an hour
    assert c.accept(e_ref("E1"), "approver-1", 0, now=moment).result == "written"
    real = int((moment - T0).total_seconds())
    instants = [ln["instant"] for ln in _history(tmp_path)]
    assert max(instants) == real  # an hour-ceil clock would say 32 h
    assert instants == sorted(instants)


def test_acceptance_usable_until_the_exact_second_of_expiry(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    expiry = at(31) + timedelta(days=7)
    just_before = c.submit_acceptance(record, expiry - timedelta(seconds=1))
    assert just_before.result == "written"  # decided_at <= t < expires_at


def test_acceptance_is_expired_at_the_exact_second_of_expiry(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    outcome = c.submit_acceptance(record, at(31) + timedelta(days=7))
    assert (outcome.result, outcome.reason) == ("invalid", "expired")
    assert head(tmp_path)["generation"] == 0


def test_a_window_that_has_not_ended_is_refused_not_moved_forward(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    window = {
        "monitor": "quality_prod",
        "start": 0,
        "end": 86400,
        "verdict": "MET",
        "head_generation": 0,
    }
    with pytest.raises(TuningError):
        c.record_window(window, at(10))  # ends at hour 24
    assert not (tmp_path / "store" / "history.jsonl").exists()
    assert c.record_window(window, at(25)) == 25 * 3600


def test_a_later_clock_never_moves_recorded_time_backwards(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.record_event(
        event("E2", "anthropic/claude-haiku-5-6", 9), at(9)
    )  # clock set back
    assert [ln["instant"] for ln in _history(tmp_path)] == [10 * 3600, 10 * 3600]


def test_step_adds_a_tick_only_when_time_passing_derives_something(
    tmp_path: Path,
) -> None:
    doc = module()
    doc["responses"][0]["rules"].append(
        {
            "id": "weekly",
            "trigger": {
                "schedule": {"every_days": 7, "anchor": "2026-11-21T00:00:00Z"}
            },
            "do": "notify",
        }
    )
    st = state()
    st.tvl = doc  # the policy builder only knows the builder shape
    c = make(st, tmp_path / "store", FakeOptimizer())
    c.step(at(1))
    c.step(at(2))
    assert not (tmp_path / "store" / "history.jsonl").exists()  # nothing fired
    c.step(at(30))  # past the anchor: the first tick fires
    assert [ln["category"] for ln in _history(tmp_path)] == ["tick"]
    c.step(at(31))
    assert len(_history(tmp_path)) == 1


# ---------------------------------------------------------------------------------
# authority: explicit, checked at the effect, never self-declared
# ---------------------------------------------------------------------------------
def test_without_an_authority_nobody_can_approve_or_accept(tmp_path: Path) -> None:
    c = Controller(
        state(), tmp_path / "store", FakeOptimizer(), initial_config=OLD, now=T0
    )
    for call in (
        lambda: c.approve_optimization(e_ref("E1"), "approver-1", at(1)),
        lambda: c.submit_acceptance({}, at(1)),
        lambda: c.record_standing_grant(
            "G1", "tuning/on_new_model", 1, at(1) + timedelta(days=1), "issuer-1", at(1)
        ),
    ):
        with pytest.raises(TuningError) as err:
            call()
        assert err.value.code == E_NO_APPROVER_AUTHORITY


def test_local_trusted_is_labelled_and_refused_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    approvals = json.loads((tmp_path / "store" / "approvals.json").read_text())
    assert approvals[e_ref("E1")]["authentication"] == "local-trusted"
    c.accept(e_ref("E1"), "approver-1", 0, now=at(31))
    line = [ln for ln in _history(tmp_path) if ln["category"] == "acceptances"][0]
    assert line["meta"] == {"authentication": "local-trusted"}

    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(TuningError) as err:
        c.approve_optimization(e_ref("E6"), "approver-1", at(32))
    assert err.value.code == "E_LOCAL_AUTHORITY_IN_PRODUCTION"


def test_an_event_is_authenticated_by_the_verifier_never_by_its_own_flag(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    forged = event("E1", NEW["model"], 9, secret=None)
    forged["authenticated"] = True  # a file's own claim
    assert c.record_event(forged, at(10)).authenticated is False
    wrong_key = event("E2", NEW["model"], 9, secret=b"another-secret")
    assert c.record_event(wrong_key, at(10)).authenticated is False
    tampered = event("E3", NEW["model"], 9)
    tampered["subject"]["model_ref"] = "anthropic/claude-haiku-9-9"  # after signing
    assert c.record_event(tampered, at(10)).authenticated is False
    stranger = event("E4", NEW["model"], 9)
    stranger["issuer"] = "mallory"
    assert c.record_event(stranger, at(10)).authenticated is False
    assert c.step(at(10)).awaiting_approval == []  # nothing fired
    facts = [f for f in c._derive()["facts"] if f["fact"] == "event"]
    assert {f["result"] for f in facts} == {"rejected(unauthenticated)"}
    # a later authentic copy of a rejected id is still admitted (RFC 0011 section 2.13)
    assert c.record_event(event("E1", NEW["model"], 9), at(10)).authenticated is True
    assert c.step(at(10)).awaiting_approval == [e_ref("E1")]


def test_without_a_verifier_no_event_counts(tmp_path: Path) -> None:
    c = make(state(), tmp_path / "store", FakeOptimizer(), event_verifier=None)
    c.event_verifier = None
    assert c.record_event(event("E1", NEW["model"], 9), at(10)).authenticated is False
    assert c.step(at(10)).awaiting_approval == []


def test_a_revoked_approver_cannot_start_a_run_that_was_approved(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.authority = ApproverAuthority(frozenset({"someone-else"}))  # authority changed
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]
    c.authority = ApproverAuthority(APPROVERS)
    c.step(at(10))
    assert len(opt.calls) == 1


def test_standing_grant_needs_a_listed_issuer_now_and_at_every_use(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    with pytest.raises(TuningError) as err:
        c.record_standing_grant(
            "G1",
            "tuning/on_new_model",
            2,
            at(10) + timedelta(days=1),
            "stranger",
            at(10),
        )
    assert err.value.code == E_APPROVER_NOT_AUTHORIZED
    grant = c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(days=1), "issuer-1", at(10)
    )
    assert grant["covers"] == "optimize" and grant["authentication"] == "local-trusted"
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.authority = ApproverAuthority(frozenset({"someone-else"}))  # issuer revoked
    assert c.step(at(10)).awaiting_approval == [e_ref("E1")] and opt.calls == []
    c.authority = ApproverAuthority(APPROVERS)
    c.step(at(10))
    assert len(opt.calls) == 1


def test_a_standing_grant_covers_the_optimization_only_it_never_moves_the_head(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(10) + timedelta(days=1), "issuer-1", at(10)
    )
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.step(at(10))
    assert len(opt.calls) == 1
    assert head(tmp_path)["generation"] == 0  # the grant never moved the head
    assert c.status()["pending_deploys"] == []  # deploy: never, nothing to accept


def test_a_grant_never_allows_more_runs_than_its_count(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = _standing(tmp_path, opt)
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 1, at(10) + timedelta(days=2), "issuer-1", at(10)
    )
    for index, hour in enumerate((10, 40, 70)):
        c.record_event(
            event(f"E{index}", f"anthropic/claude-haiku-5-{index}", hour - 1), at(hour)
        )
        c.step(at(hour))
    assert len(opt.calls) == 1  # one use, one run


# ---------------------------------------------------------------------------------
# no holdout bypass: every promotion path is measured here
# ---------------------------------------------------------------------------------
def test_supplied_verdicts_are_ignored_the_candidate_is_measured(
    tmp_path: Path,
) -> None:
    """An optimizer returning its own Promote/MET for a config that is no better."""
    liar = FakeOptimizer(
        result={
            "candidates": [
                {
                    "config": dict(OLD),  # the incumbent itself
                    "legacy_decision": "Promote",
                    "monitor_verdicts": {"quality_gate": "MET"},
                }
            ]
        }
    )
    c = controller(tmp_path, liar)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    assert c.status()["pending_deploys"] == []  # measured: not better than itself
    assert any(f["fact"] == "external_verdicts_ignored" for f in c.controller_facts())
    (marker,) = (tmp_path / "store" / "attempts").glob("*.json")
    (cand,) = json.loads(marker.read_text())["finished_record"]["candidates"]
    assert cand["legacy_decision"] == "NoDecision"


def test_supplied_verdicts_without_a_holdout_never_promote(tmp_path: Path) -> None:
    liar = FakeOptimizer(
        result={
            "candidates": [
                {
                    "config": dict(NEW),
                    "legacy_decision": "Promote",
                    "monitor_verdicts": {"quality_gate": "MET"},
                }
            ]
        }
    )
    c = controller(tmp_path, liar)
    c.state.policy.holdout = None
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    assert c.status()["pending_deploys"] == []
    assert any(f.get("reason") == "holdout_missing" for f in c.controller_facts())


def test_the_gate_names_the_incumbent_it_measured(tmp_path: Path) -> None:
    c = _deploy_awaiting(tmp_path)
    (marker,) = (tmp_path / "store" / "attempts").glob("*.json")
    (gate,) = json.loads(marker.read_text())["gates"].values()
    assert gate["incumbent_fingerprint"] == config_fingerprint(OLD)
    assert gate["incumbent_generation"] == 0
    view = c.status()["pending_deploys"][0]["accept_release"]
    assert view["incumbent"] == {
        "fingerprint": config_fingerprint(OLD),
        "generation": 0,
    }
    assert view["gate_incumbent_current"] is True
    # a different incumbent changes the gate digest (it is part of what was signed)
    other = dict(gate)
    other["incumbent_generation"] = 1
    other.pop("gate_digest")
    from traigent.tuning.controller import _digest

    assert (
        "sha256:" + _digest({k: gate[k] for k in gate if k in _GATE_KEYS})
        == gate["gate_digest"]
    )
    assert (
        "sha256:"
        + _digest(
            {**{k: gate[k] for k in gate if k in _GATE_KEYS}, "incumbent_generation": 1}
        )
        != gate["gate_digest"]
    )


_GATE_KEYS = {
    "format",
    "contract_digest",
    "fingerprint",
    "incumbent_fingerprint",
    "incumbent_generation",
    "legacy_decision",
    "monitors",
    "decision",
    "holdout_digest",
}


def test_digests_come_from_the_deploys_own_attempt_not_a_fingerprint_scan(
    tmp_path: Path,
) -> None:
    """The same configuration measured twice, on two holdouts: each deploy binds ITS own."""
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    to_deploy_awaiting(c, e_ref("E1"), 10)
    reject = c.accept(e_ref("E1"), "approver-1", 0, "reject", now=at(11))
    assert reject.result == "reject_recorded"
    first = json.loads(
        next((tmp_path / "store" / "attempts").glob("*.json")).read_text()
    )
    first_digest = first["finished_record"]["candidates"][0]["holdout_digest"]

    rows = [  # a refreshed holdout: same kind of rows, other content
        {"input": {"x": i}, "output": "billing" if i % 2 == 0 else "support"}
        for i in range(ROWS + 10)
    ]
    KIT["holdout"].write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    c.record_event(event("E7", NEW["model"], 40), at(41))  # the SAME candidate config
    to_deploy_awaiting(c, e_ref("E7"), 41)
    record = c.build_acceptance(e_ref("E7"), "approver-1", 0, now=at(42))
    assert (
        record["holdout_digest"] != first_digest
    )  # the second attempt's, not the first's
    assert c.submit_acceptance(record, at(42)).result == "written"


def test_the_status_shows_both_human_decisions(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.step(at(10))
    attempt = c.status()["open_attempts"][0]
    run = attempt["approve_run"]  # Approve run
    assert run["trigger"]["external"] == "model_released"
    assert run["spend_limit"] is None  # honest: this controller has no limit set
    assert run["admits"]["domain"][-1] == NEW["model"]
    to_deploy_awaiting(c, e_ref("E1"), 10)
    release = c.status()["pending_deploys"][0]["accept_release"]  # Accept release
    assert release["incumbent"]["generation"] == 0
    assert release["candidate"]["fingerprint"] == config_fingerprint(NEW)
    assert release["holdout"]["rows"] == ROWS
    assert release["holdout"]["counts"]["quality_gate"]["n"] == ROWS
    assert release["holdout"]["prior_uses"] == 0
    assert release["gate"]["decision"] == "Promote"
    assert release["observed_generation"] == 0
