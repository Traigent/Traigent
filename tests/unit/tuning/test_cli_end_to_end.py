"""End to end through the real ``traigent tune`` CLI (observe over optimize, mock mode)."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.tuning.controller import _read_lines

WITNESS = """
from unittest.mock import MagicMock

import traigent
from traigent.observability import observe
from traigent.tuning.policy import Every, FailureRate, ModelReleased, OnBreach, ProposeFix, Tuning

# synthetic: a newer model answers harder documents ("level N" needs skill >= N)
SKILL = {"anthropic/claude-haiku-4-5": 0, "anthropic/claude-sonnet-4-6": 1,
         "anthropic/claude-haiku-5-5": 2, "anthropic/claude-haiku-5-6": 3}
EVAL = [{"input": {"doc": f"level {i} invoice"}, "output": "billing"} for i in range(4)]
TRACE_CLIENT = MagicMock()


def ticket_route_check(example, output):
    return "fail" if not output else "pass"  # only an empty answer is a failure


policy = Tuning(
    keep=[FailureRate(checker="customer:ticket_route_check", at_most="0.08")],
    on=[TRIGGER, OnBreach()],
    fix=ProposeFix(approval="human", deploy="if_promoted"),
    recover="since_head_move",
    checkers={"customer:ticket_route_check": ticket_route_check},
    checker_labels={"customer:ticket_route_check": ["pass", "fail"]},
    holdout=HOLDOUT_ARG,
)


@observe(client=TRACE_CLIENT, agent="ticket-router", tune=policy,
         follow="head", tuning_store="./store")
@traigent.optimize(
    configuration_space={"model": ["anthropic/claude-haiku-4-5",
                                   "anthropic/claude-sonnet-4-6",
                                   "anthropic/claude-haiku-5-5"]},
    objectives=["accuracy"], eval_dataset=EVAL, offline=True, max_trials=4,
    algorithm="grid",
    default_config={"model": "anthropic/claude-haiku-4-5"},
)
def classify(doc: str) -> str:
    model = (traigent.get_config() or {}).get("model")
    level = int(doc.split()[1])
    return "billing" if SKILL.get(model, 0) >= level else "support"
"""


def _authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    path = tmp_path / "authority.yml"
    path.write_text("mode: local-trusted\napprovers: [alice]\n", encoding="utf-8")
    return ["--authority", str(path)]


def _holdout_text() -> str:
    rows = [
        {"input": {"doc": f"level {i % 4} invoice {i}"}, "output": "billing"}
        for i in range(80)
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def _write_agent(
    tmp_path: Path,
    name: str,
    holdout: bool = True,
    all_fail: bool = False,
    external: bool = False,
) -> None:
    arg = '"holdout.jsonl"' if holdout else "None"
    midnight = datetime.now(UTC).strftime("%Y-%m-%dT00:00:00Z")  # the store's origin
    trigger = (
        'ModelReleased(family="anthropic", tier=["small"], admit="model")'
        if external
        else f'Every(30, "{midnight}")'
    )
    (tmp_path / f"{name}.py").write_text(
        WITNESS.replace("HOLDOUT_ARG", arg)
        .replace("TRIGGER,", f"{trigger},")
        .replace(
            'return "fail" if not output else "pass"',
            "return 'fail'" if all_fail else 'return "fail" if not output else "pass"',
        ),
        encoding="utf-8",
    )
    if holdout:
        (tmp_path / "holdout.jsonl").write_text(_holdout_text(), encoding="utf-8")


def _run(args: list[str], ok: bool = True) -> str:
    result = CliRunner().invoke(cli, ["tune", *args])
    if ok:
        assert result.exit_code == 0, result.output
    return result.output


def _json(text: str) -> dict:
    return json.loads(text[text.index("{") :])


def test_cli_runs_optimizer_behind_observe_without_evidence_never_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path, "agent_e2e2", holdout=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e2", None)
    ctl = [
        "--target",
        "agent_e2e2.py:classify",
        "--store",
        "./store",
        "--agent",
        "ticket-router",
        *_authority(tmp_path, monkeypatch),
    ]
    try:
        occ = _json(_run(["step", *ctl]))["awaiting_approval"][0]
        _run(["approve", occ, "--approver", "alice", *ctl])
        report = _json(_run(["step", *ctl]))  # E_NO_OPTIMIZER before the fix
        assert report["finished"]
        assert _json(_run(["status", *ctl]))["pending_deploys"] == []
        facts = _read_lines(tmp_path / "store" / "controller.jsonl")
        assert any(
            f.get("code") == "E_NO_PROMOTION_EVIDENCE"
            and f["reason"] == "holdout_missing"
            for f in facts
        )
    finally:
        sys.modules.pop("agent_e2e2", None)


def test_cli_failing_quality_on_the_holdout_rejects_and_never_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path, "agent_e2e3", all_fail=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e3", None)
    ctl = [
        "--target",
        "agent_e2e3.py:classify",
        "--store",
        "./store",
        "--agent",
        "ticket-router",
        *_authority(tmp_path, monkeypatch),
    ]
    try:
        occ = _json(_run(["step", *ctl]))["awaiting_approval"][0]
        _run(["approve", occ, "--approver", "alice", *ctl])
        assert _json(_run(["step", *ctl]))["finished"]
        assert _json(_run(["status", *ctl]))["pending_deploys"] == []
        finished = [
            json.loads(p.read_text())["finished_record"]
            for p in (tmp_path / "store" / "attempts").glob("*.json")
        ]
        (cand,) = finished[0]["candidates"]
        assert list(cand["monitor_verdicts"].values()) == ["BREACHED"]
        assert cand["holdout_digest"].startswith("sha256:")
    finally:
        sys.modules.pop("agent_e2e3", None)


def test_cli_runs_the_spend_loop_through_observe_over_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path, "agent_e2e")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e", None)
    ctl = [
        "--target",
        "agent_e2e.py:classify",
        "--store",
        "./store",
        "--agent",
        "ticket-router",
        *_authority(tmp_path, monkeypatch),
    ]
    try:
        assert "wrote" in _run(["compile", "agent_e2e.py:classify", "-o", "p.tvl.yml"])
        _run(["check", "p.tvl.yml"])
        occ = _json(_run(["step", *ctl]))["awaiting_approval"][0]
        approve_text = _run(["status", "--text", *ctl])  # what Approve run shows
        assert "Approve run" in approve_text and "trigger:" in approve_text
        assert "spend limit: " in approve_text
        # the head is seeded from the optimized function's default config
        head = json.loads((tmp_path / "store" / "head.json").read_text())
        assert head["config"] == {"model": "anthropic/claude-haiku-4-5"}
        _run(["approve", occ, "--approver", "alice", *ctl])
        _run(["step", *ctl])  # optimize runs once (mock mode); the candidate waits
        status = _json(_run(["status", *ctl]))
        assert status["head"]["generation"] == 0
        assert len(status["pending_deploys"]) == 1
        text = _run(["status", "--text", *ctl])  # the release evidence
        for needle in (
            "Awaiting platform-signed release",
            "incumbent:  generation 0",
            "candidate:",
            "holdout:    80 rows",
            "gate:       Promote",
            "observed generation: 0",
        ):
            assert needle in text, text
        finished = [
            json.loads(p.read_text())["finished_record"]
            for p in (tmp_path / "store" / "attempts").glob("*.json")
        ]
        (cand,) = finished[0]["candidates"]
        assert cand["legacy_decision"] == "Promote"  # from real holdout counts
        assert list(cand["monitor_verdicts"].values()) == ["MET"]
        # the accept command refuses and the head stays where it was
        before = (tmp_path / "store" / "head.json").read_bytes()
        out = _run(
            ["accept", occ, "--generation", "0", "--approver", "alice", *ctl], ok=False
        )
        assert "E_UNSUPPORTED_IN_RELEASE" in out
        _run(["step", *ctl])
        assert _json(_run(["status", *ctl]))["head"]["generation"] == 0
        assert (tmp_path / "store" / "head.json").read_bytes() == before
        # a followed call stamps the closed head metadata only
        import agent_e2e

        agent_e2e.classify("level 0 invoice")
        md = agent_e2e.TRACE_CLIENT.start_trace.call_args.kwargs["metadata"]
        assert md["head_generation"] == 0
        assert not any(
            "model" in str(k) or "anthropic" in str(v) for k, v in md.items()
        )
    finally:
        sys.modules.pop("agent_e2e", None)
        shutil.rmtree(tmp_path / "store", ignore_errors=True)


def test_cli_refuses_external_events_and_external_triggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path, "agent_e2e4", external=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e4", None)
    ctl = [
        "--target",
        "agent_e2e4.py:classify",
        "--store",
        "./store",
        "--agent",
        "ticket-router",
        *_authority(tmp_path, monkeypatch),
    ]
    try:
        # the policy still compiles and checks: TVL specifies the trigger
        assert "wrote" in _run(["compile", "agent_e2e4.py:classify", "-o", "p.tvl.yml"])
        _run(["check", "p.tvl.yml"])
        (tmp_path / "e1.json").write_text(
            '{"kind": "model_released"}', encoding="utf-8"
        )
        for command in (["event", "e1.json"], ["step"]):
            out = _run([*command, *ctl], ok=False)
            assert "E_UNSUPPORTED_IN_RELEASE" in out
        assert not (tmp_path / "store" / "history.jsonl").exists()
    finally:
        sys.modules.pop("agent_e2e4", None)
        shutil.rmtree(tmp_path / "store", ignore_errors=True)
