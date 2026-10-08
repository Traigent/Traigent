"""#2481 (separable part): a safety violation on the last trial is reported.

When the trial budget ran out on the same trial whose result violated a
``safety_constraints`` entry, the run ended with ``max_trials_reached``: the
loop's budget check ran before the stop conditions, and the stop-condition
manager checked ``max_trials`` before the safety condition. The halt policy
itself (when "not yet shown safe" halts) is unchanged here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.api.safety import custom_safety, hallucination_rate
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.stop_conditions import MaxTrialsStopCondition
from traigent.core.types import TrialResult, TrialStatus


def _violating_trial() -> TrialResult:
    from datetime import UTC, datetime

    return TrialResult(
        trial_id="t1",
        config={},
        metrics={"safety_score": 0.0},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def test_manager_reports_safety_over_max_trials_when_both_fire() -> None:
    constraint = custom_safety(
        "must_pass", lambda config, metrics: metrics.get("safety_score", 0.0)
    ).above(0.5, min_samples=1, confidence=0.5)
    manager = StopConditionManager(
        max_trials=1,
        max_samples=None,
        samples_include_pruned=False,
        plateau_window=None,
        plateau_epsilon=None,
        objective_schema=None,
        metric_limit=None,
        metric_name=None,
        metric_include_pruned=False,
        safety_constraints=[constraint],
    )
    assert isinstance(manager.conditions[0], MaxTrialsStopCondition)

    assert manager.should_stop([_violating_trial()]) == (True, "safety_constraint")


def test_last_trial_violation_is_not_reported_as_max_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps({"input": {"q": f"q{i}"}, "output": "A"}) + "\n"
            for i in range(4)
        )
    )

    @traigent.optimize(
        evaluation=EvaluationOptions(
            eval_dataset=str(data),
            metric_functions={
                "accuracy": lambda output, expected, **_: 1.0,
                "hallucination_rate": lambda output, expected, **_: 0.5,
            },
        ),
        objectives=["accuracy"],
        configuration_space={"x": list(range(6))},
        offline=True,
        algorithm="grid",
        safety_constraints=[hallucination_rate().below(0.1, min_samples=4)],
    )
    def answer(q: str) -> str:
        traigent.get_config()
        return "A"

    result = answer.optimize_sync(max_trials=4)

    assert len(result.trials) == 4
    assert result.stop_reason == "safety_constraint"
