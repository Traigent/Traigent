"""OptimizationResult repr must not print tuned config values (#2973).

TrialResult and TrialError already redact their configs in repr. The dataclass
default repr of OptimizationResult printed best_config and the runner-up config
in full, and grew with every trial.
"""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)

TUNED_MARKER = "tuned-config-marker-a"


def _trial(index: int) -> TrialResult:
    return TrialResult(
        trial_id=f"trial-{index}",
        config={"system_prompt": TUNED_MARKER},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def _result(trial_count: int) -> OptimizationResult:
    return OptimizationResult(
        trials=[_trial(i) for i in range(trial_count)],
        best_config={"system_prompt": TUNED_MARKER},
        best_score=0.5,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        best_config_margin={
            "runner_up": {"system_prompt": TUNED_MARKER},
            "delta": 0.0,
        },
    )


def test_repr_does_not_print_best_config_or_runner_up() -> None:
    rendered = repr(_result(trial_count=2))

    assert TUNED_MARKER not in rendered
    assert "best_config='<redacted>'" in rendered


def test_str_matches_repr() -> None:
    result = _result(trial_count=2)

    assert str(result) == repr(result)


def test_repr_reports_counts_and_is_bounded_in_trial_count() -> None:
    small = repr(_result(trial_count=2))
    large = repr(_result(trial_count=40))

    assert "trials=40" in large
    assert "best_score=0.5" in large
    assert len(large) < 1000
    assert len(large) - len(small) < 200
