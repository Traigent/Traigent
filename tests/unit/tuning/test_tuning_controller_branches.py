"""Characterization of controller branches the refactors of ``_gated_effect``,
``_advance`` and ``_map_result`` must keep: each test pins today's decision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from traigent.tuning.controller import OptimizerRun, StepReport

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    NEW,
    FakeOptimizer,
    _holdout,
    at,
    controller,
    e_ref,
)
from .test_tuning_effect_time import _clock
from .test_tuning_gated_effect import _platform, _started


# -- _gated_effect: an authenticator that raises never authenticates -----------------
def test_an_authenticator_that_raises_at_the_final_gate_refuses_the_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clock(monkeypatch, at(40))
    opt = FakeOptimizer()
    calls = [0]

    def hook(approver: str, context: Any) -> bool:
        calls[0] += 1
        if calls[0] == 3:  # the final authenticator call (1 approve, 2 step)
            raise RuntimeError("platform down")
        return True

    c = controller(tmp_path, opt, now=at(0), authority=_platform(hook))
    c.approve_optimization(e_ref("E1"), "approver-1", at(40))
    report = c.step()
    assert opt.calls == [] and _started(c) == []
    assert report.refused and report.refused[0].endswith(":approver_not_authorized")


# -- _advance ------------------------------------------------------------------------
def test_an_unapproved_attempt_whose_scope_lapsed_records_a_refusal_and_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.step(at(10))  # E1 opens its attempt and awaits approval
    monkeypatch.setattr(c, "_scope_requires_intervention", lambda occ, instant: False)
    report = StepReport()
    assert c._advance(c._derive(), lambda: at(10), report, False) is False
    assert report.refused == ["A1:scope_expired"] and opt.calls == []
    assert any(
        f.get("fact") == "optimize_refused"
        and f.get("attempt") == "A1"
        and f.get("reason") == "scope_expired"
        for f in c.controller_facts()
    )


def test_a_waiting_attempt_is_listed_once_on_a_reused_report(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.step(at(10))
    report = StepReport()
    for _ in range(2):
        assert c._advance(c._derive(), lambda: at(10), report, False) is False
    assert report.awaiting_approval == [e_ref("E1")]


# -- _map_result ---------------------------------------------------------------------
class _Obj:
    best_config = {"model": NEW["model"]}
    completion = "bogus"


def test_a_result_object_with_an_unknown_completion_maps_to_normal(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    completion, candidates, _configs, _gates = c._map_result("a1", _Obj())
    assert completion == "normal"
    assert len(candidates) == 1


def test_an_optimizer_run_is_unwrapped_and_a_mapping_completion_is_checked(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    run = OptimizerRun(
        result={"completion": "nope", "candidates": []}, loaded_dataset_digests=[]
    )
    assert c._map_result("a1", run) == ("normal", [], {}, {})


def test_a_candidate_without_a_config_is_invalid_evidence(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    _comp, candidates, configs, gates = c._map_result(
        "a1", {"candidates": [{"config": {}}, "not-a-mapping-config"]}
    )
    assert candidates == [{"evidence_valid": False}, {"evidence_valid": False}]
    assert configs == {} and gates == {}


def test_an_unknown_legacy_decision_is_treated_as_no_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())

    class _Evidence:
        legacy_decision = "Promote-Everything"
        monitor_verdicts: dict[str, str] = {}
        holdout_digest = None
        n = 0
        unresolved_rows = 0
        counts: dict[str, int] = {}
        errors: list[str] = []

    monkeypatch.setattr(c, "_measure_candidate", lambda *a, **k: _Evidence())
    _comp, candidates, configs, gates = c._map_result(
        "a1", {"best_config": {"model": NEW["model"]}}
    )
    assert candidates[0]["evidence_valid"] is True
    assert candidates[0]["legacy_decision"] == "NoDecision"
    assert len(configs) == 1 and len(gates) == 1


def test_a_mutated_candidate_config_records_a_controller_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())

    def mutating(
        _attempt: str, config: Any, _incumbent: Any, _loaded: Any = None
    ) -> None:
        config["x"] = 2
        return None

    monkeypatch.setattr(c, "_measure_candidate", mutating)
    c._map_result("a1", {"best_config": {"model": NEW["model"], "x": 1}})
    assert any(
        f.get("fact") == "candidate_config_changed" for f in c.controller_facts()
    )


def test_a_raising_authenticate_callable_denies_before_the_effect(
    tmp_path: Path,
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    effects: list[int] = []

    def boom() -> bool:
        raise RuntimeError("platform down")

    def effect(final_check: Any) -> int:
        effects.append(final_check())
        return 1

    result = c._gated_effect(
        clock=lambda: at(10),
        last_persisted=0,
        authenticate=boom,
        lapse_at=lambda _i: None,
        record=lambda _i: effects.append(-1),
        effect=effect,
    )
    assert result == (None, "approver_not_authorized", None)
    assert effects == []


def test_a_spend_that_cannot_be_judged_leaves_the_loop_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = controller(tmp_path, FakeOptimizer())
    c.step(at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))  # the grant is recorded; the spend runs or is refused
    monkeypatch.setattr(c, "_spend", lambda *a, **k: False)
    monkeypatch.setattr(c, "_approval_recorded", lambda attempt: True)
    monkeypatch.setattr(
        c,
        "_open_attempts",
        lambda facts: [
            {
                "attempt": "A9",
                "occurrence": {
                    "response": "tuning",
                    "rule": "on_E1",
                    "scope_kind": "schedule",
                    "scope": [0, 0],
                },
                "finished": False,
                "closed": False,
            }
        ],
    )
    monkeypatch.setattr(c, "_scope_requires_intervention", lambda occ, instant: True)
    monkeypatch.setattr("traigent.tuning.controller.occurrence_ref", lambda occ: "r")
    assert c._advance(c._derive(), lambda: at(10), StepReport(), False) is False
