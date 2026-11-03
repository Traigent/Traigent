"""#2846: W&B sweep goal, ConfigurationComparison and the UI plugin's
configuration ranking follow the objective's direction instead of assuming
higher-is-better."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from traigent.api.types import ConfigurationComparison
from traigent.integrations.observability.wandb import create_wandb_sweep_config

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "plugins" / "traigent-ui" / "traigent_ui" / "optimization_callbacks.py"
)


@pytest.mark.parametrize(
    ("objective", "goal"),
    [("cost", "minimize"), ("latency", "minimize"), ("accuracy", "maximize")],
)
def test_sweep_goal_follows_objective_direction(objective: str, goal: str) -> None:
    config = create_wandb_sweep_config("f", {"t": [0.1, 0.2]}, [objective])
    assert config["metric"]["goal"] == goal
    assert config["metric"]["name"] == f"best_metrics/{objective}"


def test_sweep_goal_accepts_declared_orientation_for_custom_metric() -> None:
    config = create_wandb_sweep_config(
        "f", {"t": [0.1]}, ["loss_x"], orientations={"loss_x": "minimize"}
    )
    assert config["metric"]["goal"] == "minimize"


def test_sweep_undeclared_custom_objective_raises() -> None:
    with pytest.raises(ValueError, match="no declared orientation"):
        create_wandb_sweep_config("f", {"t": [0.1]}, ["my_metric"])


def _comparison() -> ConfigurationComparison:
    return ConfigurationComparison(
        configurations=[{"id": "cheap"}, {"id": "dear"}],
        comparison_metrics={
            "cost": [0.001, 0.05],
            "accuracy": [0.80, 0.90],
            "latency": [120, 900],
            "custom": [1.0, 2.0],
        },
        statistical_tests={},
        significant_differences=[],
        confidence_level=0.95,
    )


def test_best_configuration_minimizes_cost_and_latency() -> None:
    comparison = _comparison()
    assert comparison.get_best_configuration("cost")[0] == 0
    assert comparison.get_best_configuration("latency")[0] == 0
    assert comparison.get_best_configuration("accuracy")[0] == 1


def test_best_configuration_orientation_override_and_custom_metric() -> None:
    comparison = _comparison()
    assert comparison.get_best_configuration("custom", orientation="minimize")[0] == 0
    assert comparison.get_best_configuration("custom", orientation="maximize")[0] == 1
    with pytest.raises(ValueError, match="no declared orientation"):
        comparison.get_best_configuration("custom")


@pytest.fixture(scope="module")
def callbacks():
    spec = importlib.util.spec_from_file_location(
        "_test_plugin_ui_callbacks_2846", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ranked(callbacks, tmp_path: Path, objective: str) -> list[str]:
    cb = callbacks.InvocationLoggingCallback(run_id="r", base_dir=tmp_path)
    cb.objectives = [objective]
    cb.config_performance = {
        "a": {
            "config": {"id": "a"},
            "invocation_count": 1,
            "metrics": [{objective: 0.001}],
        },
        "b": {
            "config": {"id": "b"},
            "invocation_count": 1,
            "metrics": [{objective: 0.1}],
        },
        "c": {
            "config": {"id": "c"},
            "invocation_count": 1,
            "metrics": [{objective: 0.01}],
        },
    }
    cb._update_metrics_file(None)
    data = json.loads(cb.metrics_file.read_text())
    return [row["config"]["id"] for row in data["configuration_analysis"]]


def test_ui_plugin_lists_cheapest_configuration_first_for_cost(callbacks, tmp_path):
    assert _ranked(callbacks, tmp_path, "cost") == ["a", "c", "b"]


def test_ui_plugin_lists_best_accuracy_first(callbacks, tmp_path):
    assert _ranked(callbacks, tmp_path, "accuracy") == ["b", "c", "a"]


def test_ui_plugin_leaves_unknown_direction_unsorted(callbacks, tmp_path):
    assert _ranked(callbacks, tmp_path, "my_metric") == ["a", "b", "c"]
