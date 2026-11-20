"""End to end through the real ``traigent tune`` CLI (observe over optimize, mock mode)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from traigent.cli.main import cli

WITNESS = """
from unittest.mock import MagicMock

import traigent
from traigent.observability import observe
from traigent.tuning.policy import FailureRate, ModelReleased, OnBreach, ProposeFix, Tuning

EVAL = [
    {"input": {"doc": "invoice 123"}, "output": "billing"},
    {"input": {"doc": "reset my password"}, "output": "support"},
    {"input": {"doc": "refund request"}, "output": "billing"},
]
TRACE_CLIENT = MagicMock()
policy = Tuning(
    keep=[FailureRate(checker="customer:doc_label_check", at_most="0.08")],
    on=[ModelReleased(family="anthropic", tier=["small", "medium"], admit="model"),
        OnBreach()],
    fix=ProposeFix(approval="human", deploy="if_promoted"),
    recover="since_head_move",
)


@observe(client=TRACE_CLIENT, agent="doc-classifier", tune=policy,
         follow="head", tuning_store="./store")
@traigent.optimize(
    configuration_space={"model": ["anthropic/claude-haiku-4-5",
                                   "anthropic/claude-sonnet-4-6"]},
    objectives=["accuracy"], eval_dataset=EVAL, offline=True, max_trials=2,
    default_config={"model": "anthropic/claude-haiku-4-5"},
)
def classify(doc: str) -> str:
    _ = (traigent.get_config() or {}).get("model")
    return "billing" if "invoice" in doc else "support"
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


def _run(args: list[str], ok: bool = True) -> str:
    result = CliRunner().invoke(cli, ["tune", *args])
    if ok:
        assert result.exit_code == 0, result.output
    return result.output


def _json(text: str) -> dict:
    return json.loads(text[text.index("{") :])


def _with_gate_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach promotion-gate evidence for the admitted model to the real optimize result.

    The local controller never promotes without evidence (absence never promotes) and
    the SDK has no OptimizationResult -> evidence bridge here yet, so the full loop
    test supplies the evidence a promotion gate would.
    """
    from traigent.tuning import controller as ctl_mod

    real = ctl_mod.traigent_optimize_fn

    def adapt(fn, **kw):
        run = real(fn, **kw)

        def wrapped(space, *, candidate_dir=None):
            run(space, candidate_dir=candidate_dir)  # the real mock-mode optimization
            return {
                "candidates": [
                    {
                        "config": {"model": space["model"][-1]},
                        "legacy_decision": "Promote",
                        "monitor_verdicts": dict.fromkeys(_promotion_monitors(), "MET"),
                    }
                ]
            }

        wrapped.accepts_candidate_dir = True
        return wrapped

    monkeypatch.setattr(ctl_mod, "traigent_optimize_fn", adapt)


def _promotion_monitors() -> list[str]:
    doc = yaml.safe_load(Path("p.tvl.yml").read_text(encoding="utf-8"))
    return [m["name"] for m in doc.get("monitors", []) if m.get("phase") == "promotion"]


def test_cli_runs_optimizer_behind_observe_without_evidence_never_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_e2e2.py").write_text(WITNESS, encoding="utf-8")
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
    finally:
        sys.modules.pop("agent_e2e2", None)


def test_cli_loop_closes_through_observe_over_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_gate_evidence(monkeypatch)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_e2e.py").write_text(WITNESS, encoding="utf-8")
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
        accepted = _json(
            _run(["accept", occ, "--generation", "0", "--approver", "alice", *ctl])
        )
        assert accepted["result"] == "written"
        _run(["step", *ctl])
        status = _json(_run(["status", *ctl]))
        assert status["head"]["generation"] == 1
        head = json.loads((tmp_path / "store" / "head.json").read_text())
        assert head["generation"] == 1
        assert head["config"].get("model")
        before = (tmp_path / "store" / "head.json").read_bytes()
        _run(["event", "e1.json", *ctl])
        _run(["step", *ctl])
        assert (tmp_path / "store" / "head.json").read_bytes() == before
        # a second admitted model opens a new deploy; accepting it at generation 0
        # (the head has since moved) is refused as stale and the head stays put.
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
        stale = _run(
            ["accept", occ6, "--generation", "0", "--approver", "alice", *ctl], ok=False
        )
        assert "stale_generation" in stale
        assert _json(_run(["status", *ctl]))["head"]["generation"] == 1
        # a followed call stamps the closed head metadata only
        import agent_e2e

        agent_e2e.classify("invoice 1")
        md = agent_e2e.TRACE_CLIENT.start_trace.call_args.kwargs["metadata"]
        assert md["head_generation"] == 1
        assert not any(
            "model" in str(k) or "anthropic" in str(v) for k, v in md.items()
        )
    finally:
        sys.modules.pop("agent_e2e", None)
        shutil.rmtree(tmp_path / "store", ignore_errors=True)
