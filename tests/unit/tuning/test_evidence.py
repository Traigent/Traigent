"""Local promotion evidence: counts from a holdout, verdicts from TVL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning.evidence import (
    EvidenceUnavailable,
    build_promotion_evidence,
    holdout_digest,
    monitor_counts,
)

MODULE: dict[str, Any] = {
    "objectives": [{"name": "accuracy", "direction": "maximize"}],
    "promotion_policy": {"alpha": 0.05, "min_effect": {"accuracy": 0.02}},
    "requirements": [
        {
            "name": "failure_rate",
            "kind": "rate_bound",
            "unit": "invocation",
            "max_rate": 0.08,
            "violation": {"checker": "customer:c", "label_in": ["fail"]},
        }
    ],
    "monitors": [
        {
            "name": "gate",
            "requirement": "failure_rate",
            "phase": "promotion",
            "evidence": {"source": "evaluation_set"},
            "decision": {
                "method": "clopper_pearson_rational_v2",
                "confidence": 0.95,
                "false_alarm_budget": 0.05,
            },
        }
    ],
}


def _holdout(tmp_path: Path, n: int = 60) -> Path:
    path = tmp_path / "h.jsonl"
    rows = [{"input": {"x": i}, "output": i % 2} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _agent(x: int) -> int:
    from traigent.config.context import config_context

    cfg = config_context.get() or {}
    right = x % 2
    # the incumbent is wrong on half the documents (so paired differences vary)
    return right if cfg.get("good") or x % 4 < 2 else 1 - right


def _build(tmp_path: Path, **over: Any) -> Any:
    args: dict[str, Any] = {
        "fn": _agent,
        "checkers": {"customer:c": lambda ex, out: "pass"},
        "holdout": _holdout(tmp_path),
        "candidate_config": {"good": True},
        "incumbent_config": {"good": False},
    }
    args.update(over)
    return build_promotion_evidence(MODULE, **args)


def test_counts_from_checker_labels_errors_are_unresolved() -> None:
    examples = [{"input": {}}] * 4
    runs = [(True, "a"), (True, "b"), (False, None), (True, "c")]

    def checker(_ex: Any, out: Any) -> str:
        if out == "c":
            raise ValueError("boom")
        return "fail" if out == "a" else "pass"

    assert monitor_counts(checker, ["fail"], examples, runs) == {
        "n": 4,
        "k": 1,
        "u": 2,
        "pl": 0,
        "complete": True,
    }


def test_real_counts_promote_and_carry_the_holdout_digest(tmp_path: Path) -> None:
    evidence = _build(tmp_path)
    assert evidence.legacy_decision == "Promote"
    assert evidence.monitor_verdicts == {"gate": "MET"}
    assert evidence.holdout_digest == holdout_digest(tmp_path / "h.jsonl")


def test_breached_monitor_reports_breached(tmp_path: Path) -> None:
    evidence = _build(tmp_path, checkers={"customer:c": lambda ex, out: "fail"})
    assert evidence.monitor_verdicts == {"gate": "BREACHED"}


def test_checker_errors_make_the_verdict_unknown_never_met(tmp_path: Path) -> None:
    def broken(_ex: Any, _out: Any) -> str:
        raise RuntimeError

    evidence = _build(tmp_path, checkers={"customer:c": broken})
    assert evidence.monitor_verdicts["gate"].startswith("UNKNOWN(")


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"holdout": None}, "holdout_missing"),
        ({"checkers": {}}, "checker_missing"),
        ({"fn": None}, "agent_function_missing"),
    ],
)
def test_missing_inputs_raise_a_closed_reason(
    tmp_path: Path, over: dict[str, Any], reason: str
) -> None:
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, **over)
    assert err.value.reason == reason


def test_the_same_config_is_not_a_promotion(tmp_path: Path) -> None:
    evidence = _build(tmp_path, incumbent_config={"good": True})
    assert evidence.legacy_decision == "NoDecision"
