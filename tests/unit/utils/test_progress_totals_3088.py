"""Progress callbacks must not announce a made-up configuration total."""

from types import SimpleNamespace

from traigent.utils.callbacks import (
    DetailedProgressCallback,
    ProgressInfo,
    SimpleProgressCallback,
)

SPACE = {"m": ["a", "b", "c"], "t": (0.0, 1.0), "k": [1, 2, 3, 4]}


def _progress(total: int) -> ProgressInfo:
    return ProgressInfo(
        current_trial=1,
        total_trials=total,
        completed_trials=1,
        successful_trials=1,
        failed_trials=0,
        best_score=0.5,
        best_config=None,
        elapsed_time=0.0,
        estimated_remaining=None,
        current_algorithm="random",
    )


def test_simple_random_with_range_does_not_claim_a_product() -> None:
    lines: list[str] = []
    cb = SimpleProgressCallback(output=lines.append)
    cb.on_optimization_start(SPACE, ["accuracy"], "random")
    assert "12" not in lines[0]
    assert "unknown" in lines[0]
    cb.on_trial_start(0, {"m": "a"})
    assert lines[-1].startswith("[1/?]")


def test_simple_uses_orchestrator_total_after_first_trial() -> None:
    lines: list[str] = []
    cb = SimpleProgressCallback(output=lines.append)
    cb.on_optimization_start(SPACE, ["accuracy"], "random")
    cb.on_trial_start(0, {"m": "a"})
    trial = SimpleNamespace(status="completed", metrics={}, config={"m": "a"})
    cb.on_trial_complete(trial, _progress(5))
    assert cb.total_trials == 5
    assert any(line.startswith("[1/5]") for line in lines)


def test_simple_grid_of_lists_keeps_exact_size_capped_by_budget() -> None:
    lines: list[str] = []
    cb = SimpleProgressCallback(output=lines.append)
    cb.on_optimization_start({"m": ["a", "b", "c"], "k": [1, 2]}, ["accuracy"], "grid")
    assert "with 6 configurations" in lines[0]
    cb.on_trial_start(0, {"m": "a"})
    trial = SimpleNamespace(status="completed", metrics={}, config={"m": "a"})
    cb.on_trial_complete(trial, _progress(4))
    assert cb.total_trials == 4


def test_detailed_random_with_range_total_is_unknown(capsys) -> None:
    cb = DetailedProgressCallback()
    cb.on_optimization_start(SPACE, ["accuracy"], "random")
    out = capsys.readouterr().out
    assert cb.total_trials == 0
    assert "Total configurations to test: 12" not in out
    assert "unknown" in out


def test_detailed_grid_of_lists_keeps_exact_size(capsys) -> None:
    cb = DetailedProgressCallback()
    cb.on_optimization_start({"m": ["a", "b"], "k": [1, 2, 3]}, ["accuracy"], "grid")
    assert cb.total_trials == 6
    assert "Total configurations to test: 6" in capsys.readouterr().out


def test_detailed_grid_capped_by_budget_prints_the_budget(capsys) -> None:
    cb = DetailedProgressCallback()
    cb.on_optimization_start({"m": ["a", "b", "c"], "k": [1, 2]}, ["accuracy"], "grid")
    cb.on_trial_start(0, {"m": "a"})
    trial = SimpleNamespace(status="completed", metrics={}, config={"m": "a"})
    cb.on_trial_complete(trial, _progress(4))
    assert "Trial 1/4 completed" in capsys.readouterr().out
