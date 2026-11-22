"""Mid-run progress totals reflect a finite search space (Traigent#3207)."""

from __future__ import annotations

import asyncio
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.callbacks import ProgressBarCallback


def _run(tmp_path: Path, space: dict, algorithm: str, max_trials: int) -> str:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        '{"input": {"q": "a"}, "output": "a"}\n{"input": {"q": "b"}, "output": "b"}\n'
    )

    @traigent.optimize(
        configuration_space=space,
        evaluation=EvaluationOptions(eval_dataset=str(dataset)),
        execution=ExecutionOptions(offline=True),
        algorithm=algorithm,
        max_trials=max_trials,
    )
    def agent(q: str) -> str:
        traigent.get_config()
        return q

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        asyncio.run(agent.optimize(callbacks=[ProgressBarCallback()]))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _offline_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _denominators(out: str) -> set[str]:
    return set(re.findall(r"\(\d+/(\d+)\)", out))


def test_grid_progress_total_is_grid_size_not_max_trials_3207(tmp_path: Path) -> None:
    out = _run(tmp_path, {"a": [1, 2], "b": [3, 4]}, "grid", 8)
    assert _denominators(out) == {"4"}


def test_random_continuous_progress_total_stays_max_trials_3207(
    tmp_path: Path,
) -> None:
    out = _run(tmp_path, {"t": (0.0, 1.0)}, "random", 8)
    assert _denominators(out) == {"8"}


class _Opt(GridSearchOptimizer):
    """Grid optimizer whose cardinality is forced for the helper table."""

    def __init__(self, cardinality: object) -> None:
        self._forced = cardinality

    @property
    def config_space_cardinality(self) -> object:  # type: ignore[override]
        return self._forced


class _Repeater:
    """Non-deduplicating optimizer (like a remote client): never capped."""

    config_space_cardinality = 1


@pytest.mark.parametrize(
    ("cardinality", "max_trials", "extra", "expected"),
    [
        (4, 8, 0, 4),  # finite grid smaller than budget
        (4, 8, 1, 5),  # default_config baseline is an extra trial
        (4, 3, 0, 3),  # budget smaller than the grid
        (4, 5, 1, 5),
        (None, 8, 0, 8),  # continuous / unknown space
        (0, 8, 0, 8),  # empty space: leave unchanged
        (True, 8, 0, 8),  # bool is not a cardinality
        (4, None, 0, None),  # no budget: leave unchanged
    ],
)
def test_expected_total_trials_helper_3207(
    cardinality: object, max_trials: int | None, extra: int, expected: int | None
) -> None:
    assert (
        OptimizationOrchestrator._expected_total_trials(
            _Opt(cardinality),
            max_trials,
            extra_trials=extra,  # type: ignore[arg-type]
        )
        == expected
    )


def test_optimizer_that_may_repeat_is_not_capped_3207() -> None:
    assert (
        OptimizationOrchestrator._expected_total_trials(
            _Repeater(),
            8,  # type: ignore[arg-type]
        )
        == 8
    )
