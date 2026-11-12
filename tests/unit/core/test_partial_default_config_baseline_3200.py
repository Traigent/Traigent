"""#3200: a partial default_config must not make trial 1 miss tuned keys."""

from __future__ import annotations

import pytest

import traigent
from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _run(
    space, *, max_trials=3, constraints=None, default_config=None, algorithm="random"
):
    seen: list[dict] = []

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "x")], name="w"),
        objectives=["accuracy"],
        scoring_function=lambda p, e: 1.0,
        configuration_space=space,
        default_config=default_config,
        constraints=constraints,
        algorithm=algorithm,
        execution={"offline": True},
        injection_mode="parameter",
        config_param="cfg",
    )
    def agent(text, cfg=None):
        seen.append({k: cfg[k] for k in space})
        return "x"

    agent.optimize_sync(max_trials=max_trials, random_seed=7)
    return seen


def test_partial_defaults_trial_one_is_defaults_plus_deterministic_fills():
    seen = _run(
        {
            "model": Choices(["gpt-4o-mini", "gpt-4o", "gpt-4.1"]),  # no default
            "temperature": Range(0.0, 1.0, default=0.25),
            "k": IntRange(2, 9),  # no default -> lower bound
            "lr": LogRange(1e-4, 1e-1),  # no default -> lower bound
        }
    )
    assert seen, "no trial ran"
    assert seen[0] == {
        "model": "gpt-4o-mini",
        "temperature": 0.25,
        "k": 2,
        "lr": 1e-4,
    }


def test_stepped_range_without_default_fills_the_first_grid_point():
    seen = _run(
        {
            "model": Choices(["gpt-4o-mini", "gpt-4o"], default="gpt-4o"),
            "t": Range(0.2, 1.0, step=0.2),
        }
    )
    assert seen[0] == {"model": "gpt-4o", "t": 0.2}


def test_grid_search_still_evaluates_every_grid_point_plus_a_distinct_baseline():
    space = {
        "model": Choices(["gpt-4o-mini", "gpt-4o"]),
        "t": Range(0.0, 1.0, step=0.5, default=0.25),  # 0.25 is off the grid
    }
    seen = _run(space, algorithm="grid", max_trials=10)
    grid = {(m, t) for m in ("gpt-4o-mini", "gpt-4o") for t in (0.0, 0.5, 1.0)}
    assert len(grid) == 6
    assert seen[0] == {"model": "gpt-4o-mini", "t": 0.25}  # distinct baseline
    assert {(c["model"], c["t"]) for c in seen[1:]} == grid
    assert len(seen) == len(grid) + 1


def test_full_defaults_run_exactly_the_declared_baseline_first():
    seen = _run(
        {
            "model": Choices(["gpt-4o-mini", "gpt-4o"], default="gpt-4o"),
            "temperature": Range(0.0, 1.0, default=0.25),
        }
    )
    assert seen[0] == {"model": "gpt-4o", "temperature": 0.25}


def test_explicit_default_config_covering_all_keys_is_unchanged():
    seen = _run(
        {"model": Choices(["gpt-4o-mini", "gpt-4o"]), "temperature": Range(0.0, 1.0)},
        default_config={"model": "gpt-4o-mini", "temperature": 0.75},
    )
    assert seen[0] == {"model": "gpt-4o-mini", "temperature": 0.75}


def test_constraint_violating_completion_skips_the_baseline(caplog):
    def cool(cfg):
        return cfg["temperature"] <= 0.5

    with caplog.at_level("WARNING", logger="traigent.core.orchestrator"):
        seen = _run(
            {
                "model": Choices(["gpt-4o-mini", "gpt-4o"]),
                "temperature": Range(0.0, 1.0, default=0.9),
            },
            constraints=[cool],
            max_trials=3,
        )
    assert seen, "no trial ran"
    assert all(cfg["temperature"] <= 0.5 for cfg in seen)
    assert caplog.text.count("Skipping the default_config baseline") == 1


def _fake(space, default_config, constraints=()):
    from types import SimpleNamespace

    return SimpleNamespace(
        optimizer=SimpleNamespace(config_space=space),
        _default_config=default_config,
        _constraints_pre_eval=list(constraints),
    )


def test_unfillable_parameter_type_skips_the_baseline_with_a_named_warning(caplog):
    from traigent.core.orchestrator import OptimizationOrchestrator

    fake = _fake({"model": ["x"], "weird": {"type": "object"}}, {"model": "x"})
    with caplog.at_level("WARNING", logger="traigent.core.orchestrator"):
        assert OptimizationOrchestrator._complete_default_config(fake) is None
    assert "parameter 'weird'" in caplog.text
    assert caplog.text.count("Skipping the default_config baseline") == 1


def test_missing_default_config_raises_optimization_error_not_assert():
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.utils.exceptions import OptimizationError

    fake = _fake({"model": ["x"]}, None)
    with pytest.raises(
        OptimizationError,
        match="default config baseline requested without a default_config",
    ):
        OptimizationOrchestrator._complete_default_config(fake)


def test_completion_never_touches_the_optimizer():
    from traigent.core.orchestrator import OptimizationOrchestrator

    fake = _fake({"model": ["x", "y"], "t": (0.1, 0.9)}, {"model": "y"})
    # optimizer has no suggest_next_trial at all: completion cannot call it
    assert OptimizationOrchestrator._complete_default_config(fake) == {
        "model": "y",
        "t": 0.1,
    }
