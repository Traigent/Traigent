"""End to end through the real ``traigent tune`` CLI (observe over optimize, mock mode)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.tuning.controller import _read_lines

WITNESS = """
from unittest.mock import MagicMock

import traigent
from traigent.observability import observe
from traigent.tuning.policy import FailureRate, ModelReleased, OnBreach, ProposeFix, Tuning

# synthetic: a newer model answers harder documents ("level N" needs skill >= N)
SKILL = {"anthropic/claude-haiku-4-5": 0, "anthropic/claude-sonnet-4-6": 1,
         "anthropic/claude-haiku-5-5": 2, "anthropic/claude-haiku-5-6": 3}
EVAL = [{"input": {"doc": f"level {i} invoice"}, "output": "billing"} for i in range(4)]
TRACE_CLIENT = MagicMock()


def doc_label_check(example, output):
    return "fail" if not output else "pass"  # only an empty answer is a failure


policy = Tuning(
    keep=[FailureRate(checker="customer:doc_label_check", at_most="0.08")],
    on=[ModelReleased(family="anthropic", tier=["small", "medium"], admit="model"),
        OnBreach()],
    fix=ProposeFix(approval="human", deploy="if_promoted"),
    recover="since_head_move",
    checkers={"customer:doc_label_check": doc_label_check},
    holdout=HOLDOUT_ARG,
)


@observe(client=TRACE_CLIENT, agent="doc-classifier", tune=policy,
         follow="head", tuning_store="./store")
@traigent.optimize(
    configuration_space={"model": ["anthropic/claude-haiku-4-5",
                                   "anthropic/claude-sonnet-4-6"]},
    objectives=["accuracy"], eval_dataset=EVAL, offline=True, max_trials=4,
    default_config={"model": "anthropic/claude-haiku-4-5"},
)
def classify(doc: str) -> str:
    model = (traigent.get_config() or {}).get("model")
    level = int(doc.split()[1])
    return "billing" if SKILL.get(model, 0) >= level else "support"
"""
EVENT = {
    "kind": "model_released",
    "event_id": "E1",
    "issuer": "P",
    "issued_at": 9,
    "catalogue_revision": 1,
    "subject": {
        "model_ref": "anthropic/claude-haiku-5-5",
        "provider_family": "anthropic",
        "tier": "small",
    },
    "authenticated": True,
}


def _holdout_text() -> str:
    rows = [
        {"input": {"doc": f"level {i % 4} invoice {i}"}, "output": "billing"}
        for i in range(80)
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def _write_agent(
    tmp_path: Path, name: str, holdout: bool = True, all_fail: bool = False
) -> None:
    arg = '"holdout.jsonl"' if holdout else "None"
    (tmp_path / f"{name}.py").write_text(
        WITNESS.replace("HOLDOUT_ARG", arg).replace(
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
    (tmp_path / "e1.json").write_text(json.dumps(EVENT), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e2", None)
    ctl = [
        "--target",
        "agent_e2e2.py:classify",
        "--store",
        "./store",
        "--agent",
        "doc-classifier",
    ]
    try:
        _run(["event", "e1.json", *ctl])
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
    (tmp_path / "e1.json").write_text(json.dumps(EVENT), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e3", None)
    ctl = [
        "--target",
        "agent_e2e3.py:classify",
        "--store",
        "./store",
        "--agent",
        "doc-classifier",
    ]
    try:
        _run(["event", "e1.json", *ctl])
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


def test_cli_loop_closes_through_observe_over_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    _write_agent(tmp_path, "agent_e2e")
    (tmp_path / "e1.json").write_text(json.dumps(EVENT), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_e2e", None)
    ctl = [
        "--target",
        "agent_e2e.py:classify",
        "--store",
        "./store",
        "--agent",
        "doc-classifier",
    ]
    try:
        assert "wrote" in _run(["compile", "agent_e2e.py:classify", "-o", "p.tvl.yml"])
        _run(["check", "p.tvl.yml"])
        _run(["event", "e1.json", *ctl])
        occ = _json(_run(["step", *ctl]))["awaiting_approval"][0]
        # the head is seeded from the optimized function's default config
        head = json.loads((tmp_path / "store" / "head.json").read_text())
        assert head["config"] == {"model": "anthropic/claude-haiku-4-5"}
        _run(["approve", occ, "--approver", "alice", *ctl])
        _run(["step", *ctl])  # optimize runs once (mock mode); deploy now awaits
        status = _json(_run(["status", *ctl]))
        assert status["head"]["generation"] == 0
        assert len(status["pending_deploys"]) == 1
        finished = [
            json.loads(p.read_text())["finished_record"]
            for p in (tmp_path / "store" / "attempts").glob("*.json")
        ]
        (cand,) = finished[0]["candidates"]
        assert cand["legacy_decision"] == "Promote"  # from real holdout counts
        assert list(cand["monitor_verdicts"].values()) == ["MET"]
        accepted = _json(
            _run(["accept", occ, "--generation", "0", "--approver", "alice", *ctl])
        )
        assert accepted["result"] == "written"
        _run(["step", *ctl])
        status = _json(_run(["status", *ctl]))
        assert status["head"]["generation"] == 1
        head = json.loads((tmp_path / "store" / "head.json").read_text())
        assert head["generation"] == 1
        assert (
            head["config"]["model"] == "anthropic/claude-haiku-5-5"
        )  # the admitted model
        before = (tmp_path / "store" / "head.json").read_bytes()
        _run(["event", "e1.json", *ctl])
        _run(["step", *ctl])
        assert (tmp_path / "store" / "head.json").read_bytes() == before
        # a newer admitted model opens a new deploy (measured against the new head);
        # accepting it at generation 0 (the head has since moved) is refused as stale.
        e6 = {
            **EVENT,
            "event_id": "E6",
            "issued_at": 10,
            "subject": {**EVENT["subject"], "model_ref": "anthropic/claude-haiku-5-6"},
        }
        (tmp_path / "e6.json").write_text(json.dumps(e6), encoding="utf-8")
        _run(["event", "e6.json", *ctl])
        occ6 = _json(_run(["step", *ctl]))["awaiting_approval"][0]
        _run(["approve", occ6, "--approver", "alice", *ctl])
        _run(["step", *ctl])
        assert len(_json(_run(["status", *ctl]))["pending_deploys"]) == 1
        stale = _run(
            ["accept", occ6, "--generation", "0", "--approver", "alice", *ctl], ok=False
        )
        assert "stale_generation" in stale
        assert _json(_run(["status", *ctl]))["head"]["generation"] == 1
        # a followed call stamps the closed head metadata only
        import agent_e2e

        agent_e2e.classify("level 0 invoice")
        md = agent_e2e.TRACE_CLIENT.start_trace.call_args.kwargs["metadata"]
        assert md["head_generation"] == 1
        assert not any(
            "model" in str(k) or "anthropic" in str(v) for k, v in md.items()
        )
    finally:
        sys.modules.pop("agent_e2e", None)
        shutil.rmtree(tmp_path / "store", ignore_errors=True)
