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

from .conftest import tvl_free

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


LABELS = {"customer:c": ["pass", "fail"]}


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
        "checker_labels": LABELS,
        "holdout": _holdout(tmp_path),
        "candidate_config": {"good": True},
        "incumbent_config": {"good": False},
    }
    args.update(over)
    return build_promotion_evidence(MODULE, **args)


@tvl_free
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
@tvl_free
def test_missing_inputs_raise_a_closed_reason(
    tmp_path: Path, over: dict[str, Any], reason: str
) -> None:
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, **over)
    assert err.value.reason == reason


def test_the_same_config_is_not_a_promotion(tmp_path: Path) -> None:
    evidence = _build(tmp_path, incumbent_config={"good": True})
    assert evidence.legacy_decision == "NoDecision"


# -- review blockers -------------------------------------------------------------


def test_parameter_injection_sees_the_candidate_config(tmp_path: Path) -> None:
    from traigent.config.providers import ParameterBasedProvider

    seen: list[str] = []

    def agent(x: int, config: Any = None) -> int:
        seen.append(config.model)
        return x % 2 if config.model == "B" or x % 4 < 2 else 1 - x % 2

    wrapped = ParameterBasedProvider().inject_config(agent, {"model": "A"}, "config")
    evidence = _build(
        tmp_path,
        fn=wrapped,
        candidate_config={"model": "B"},
        incumbent_config={"model": "A"},
    )
    assert evidence.legacy_decision == "Promote"
    assert seen.count("B") == 60 and seen.count("A") == 60


@tvl_free
def test_among_requirements_are_refused_not_counted(tmp_path: Path) -> None:
    module = json.loads(json.dumps(MODULE))
    module["requirements"][0]["among"] = {"checker": "customer:pop", "label_in": ["in"]}
    path = tmp_path / "h.jsonl"
    rows = [{"input": {"x": i}, "output": i % 2} for i in range(1000)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceUnavailable) as err:
        build_promotion_evidence(
            module,
            fn=_agent,
            checkers={
                "customer:c": lambda ex, out: (
                    "fail" if ex["input"]["x"] < 10 else "pass"
                )
            },
            checker_labels=LABELS,
            holdout=path,
            candidate_config={"good": True},
            incumbent_config={"good": False},
        )
    assert err.value.reason == "requirement_among_unsupported"


def test_each_invocation_and_checker_call_gets_independent_copies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "h.jsonl"
    rows = [{"input": {"items": [1, 2, 3]}, "output": 3} for _ in range(40)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    seen: list[int] = []

    def agent(items: list[int]) -> int:
        seen.append(len(items))
        return items.pop()

    def checker(ex: Any, out: Any) -> str:
        ex["input"]["items"].clear()
        return "pass"

    build_promotion_evidence(
        MODULE,
        fn=agent,
        checkers={"customer:c": checker},
        checker_labels=LABELS,
        holdout=path,
        candidate_config={"good": True},
        incumbent_config={"good": False},
    )
    assert set(seen) == {3}


@tvl_free
def test_digest_and_examples_come_from_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.tuning import evidence as ev

    path = _holdout(tmp_path)
    original = path.read_bytes()
    swapped = b'{"input": {"x": 1}, "output": 0}\n' * 60
    reads = iter([original, swapped, swapped])
    monkeypatch.setattr(Path, "read_bytes", lambda self: next(reads))
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: next(reads).decode("utf-8")
    )
    digest, examples = ev.read_holdout(path)
    import hashlib

    assert digest == "sha256:" + hashlib.sha256(original).hexdigest()
    assert [e["input"]["x"] for e in examples] == list(range(60))


def test_function_mutating_nested_config_cannot_change_the_candidate(
    tmp_path: Path,
) -> None:
    config = {"opts": {"threshold": 1}}

    def agent(x: int) -> int:
        from traigent.config.context import config_context

        cfg = config_context.get() or {}
        cfg["opts"]["threshold"] += 1
        return x % 2

    _build(
        tmp_path,
        fn=agent,
        candidate_config=config,
        incumbent_config={"opts": {"threshold": 1}},
    )
    assert config == {"opts": {"threshold": 1}}


# -- evidence soundness ------------------------------------------------------------


def test_a_raised_call_is_unresolved_not_a_zero_score(tmp_path: Path) -> None:
    """A transient candidate failure must neither score 0.0 nor sink the candidate."""
    from traigent.config.context import config_context

    def flaky(x: int) -> int:
        cfg = config_context.get() or {}
        if cfg.get("good") and x == 7:
            raise RuntimeError("provider timeout")
        return x % 2 if cfg.get("good") or x % 4 < 2 else 1 - x % 2

    evidence = _build(tmp_path, fn=flaky)
    assert evidence.unresolved_rows == 1 and evidence.n == 60
    assert evidence.counts["gate"]["u"] == 1  # counted toward max_unresolved_fraction
    assert evidence.legacy_decision == "Promote"


def test_an_incumbent_outage_does_not_inflate_the_candidate(tmp_path: Path) -> None:
    """Rows the incumbent could not answer leave the pair; they are not free wins."""
    from traigent.config.context import config_context

    def down(x: int) -> int:
        cfg = config_context.get() or {}
        if not cfg.get("good") and x % 2 == 0:
            raise RuntimeError("provider outage")
        return x % 2

    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, fn=down)
    assert err.value.reason == "too_many_unresolved"


def test_a_checker_answer_outside_the_declared_labels_is_unresolved(
    tmp_path: Path,
) -> None:
    from traigent.tuning.evidence import E_CHECKER_LABEL_OUT_OF_SET

    def typo(ex: Any, out: Any) -> str:
        return "pasz" if ex["input"]["x"] == 3 else "pass"  # a typo is not a pass

    evidence = _build(tmp_path, checkers={"customer:c": typo})
    assert evidence.counts["gate"]["u"] == 1 and evidence.counts["gate"]["k"] == 0
    assert E_CHECKER_LABEL_OUT_OF_SET in evidence.errors


@tvl_free
def test_labels_must_be_declared_and_cover_the_violating_ones(tmp_path: Path) -> None:
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, checker_labels=None)
    assert err.value.reason == "checker_labels_missing"
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, checker_labels={"customer:c": ["pass"]})  # no "fail"
    assert err.value.reason == "checker_labels_inconsistent"


@tvl_free
def test_a_holdout_row_without_an_expected_output_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    rows = [{"input": {"x": i}} for i in range(60)]  # no "output"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, holdout=path)
    assert err.value.reason == "holdout_missing_expected_output"
    rows = [{"input": {"x": i}, "output": None} for i in range(60)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceUnavailable):  # a null expected output is no output
        _build(tmp_path, holdout=path)


def test_candidate_and_incumbent_are_interleaved_per_row(tmp_path: Path) -> None:
    from traigent.config.context import config_context

    order: list[tuple[int, bool]] = []

    def agent(x: int) -> int:
        cfg = config_context.get() or {}
        order.append((x, bool(cfg.get("good"))))
        return x % 2

    _build(tmp_path, fn=agent)
    assert len(order) == 120
    pairs = [order[i : i + 2] for i in range(0, 120, 2)]
    # each row is evaluated for both configurations back to back (paired) ...
    assert all(a[0] == b[0] and a[1] != b[1] for a, b in pairs)
    # ... and which side goes first alternates, so drift cannot favour one side
    assert {a[1] for a, _ in pairs} == {True, False}


def test_a_holdout_equal_to_the_optimization_eval_set_is_refused(
    tmp_path: Path,
) -> None:
    from traigent.tuning.evidence import eval_set_identity

    holdout = _holdout(tmp_path)
    rows = [json.loads(line) for line in holdout.read_text().splitlines()]
    # by file bytes
    digests, _ = eval_set_identity(str(holdout))
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, eval_set_digests=digests)
    assert err.value.reason == "holdout_is_optimization_eval_set"
    # by content, whatever the file layout (an in-memory eval set)
    digests, _ = eval_set_identity(rows)
    with pytest.raises(EvidenceUnavailable):
        _build(tmp_path, eval_set_digests=digests)
    # a different eval set is fine
    digests, _ = eval_set_identity([{"input": {"x": 999}, "output": 1}])
    assert _build(tmp_path, eval_set_digests=digests).legacy_decision == "Promote"
