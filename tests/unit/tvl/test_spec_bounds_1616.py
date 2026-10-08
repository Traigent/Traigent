"""Traigent#1616 axes B and C: bounds checks the canonical TVL tooling enforces.

Both specs below are canonical-ERROR (``tvl/spec/examples/validation-phase5/
chance-constraint-invalid-threshold.tvl.yml`` and ``validation-phase3/
budget-invalid.tvl.yml``). Before the fix the SDK loaded them silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.tvl.models import ChanceConstraint, ExplorationBudgets
from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.exceptions import TVLValidationError

_BASE = """\
tvl:
  module: corp.validation.bounds_1616
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://datasets/validation/dev.parquet
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4", "claude-3"]
constraints:
  structural: []
objectives:
  - name: quality
    metric_ref: metrics.quality.v1
    direction: maximize
"""


def _write(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "spec.tvl.yml"
    path.write_text(_BASE + extra, encoding="utf-8")
    return path


def _policy(threshold: float) -> str:
    return f"""\
promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    quality: 0.01
  chance_constraints:
    - name: safety_violation
      threshold: {threshold}
      confidence: 0.95
"""


def _budgets(body: str) -> str:
    return f"""\
exploration:
  strategy:
    type: random
  budgets:
{body}
"""


# --- Axis B: chance-constraint threshold ------------------------------------


@pytest.mark.parametrize("threshold", [1.5, -0.1])
def test_chance_threshold_outside_unit_interval_is_rejected(
    tmp_path: Path, threshold: float
) -> None:
    with pytest.raises(TVLValidationError, match="threshold must be in"):
        load_tvl_spec(spec_path=_write(tmp_path, _policy(threshold)))


@pytest.mark.parametrize("threshold", [0.0, 0.05, 1.0])
def test_chance_threshold_inside_unit_interval_loads(
    tmp_path: Path, threshold: float
) -> None:
    artifact = load_tvl_spec(spec_path=_write(tmp_path, _policy(threshold)))
    assert artifact.promotion_policy is not None
    [cc] = artifact.promotion_policy.chance_constraints
    assert cc.threshold == threshold


def test_chance_threshold_nan_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        ChanceConstraint.from_dict(
            {"name": "x", "threshold": float("nan"), "confidence": 0.95}
        )


# --- Axis C: exploration budgets --------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "    max_trials: 0",
        "    max_trials: -3",
        "    max_spend_usd: -0.01",
        "    max_wallclock_s: 0",
    ],
)
def test_non_runnable_budget_is_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(TVLValidationError, match="Invalid budgets"):
        load_tvl_spec(spec_path=_write(tmp_path, _budgets(body)))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # Canonical types these fields as integers. A fraction used to be
        # truncated first, so 1.9 loaded as 1 and 0.5 was reported as "got 0".
        ("    max_trials: 1.9", "max_trials must be an integer, got 1.9"),
        ("    max_wallclock_s: 0.5", "max_wallclock_s must be an integer, got 0.5"),
        ("    max_trials: true", "max_trials must be an integer, got True"),
        # int(inf) raised OverflowError, which escaped the loader's wrapper.
        ("    max_wallclock_s: .inf", "max_wallclock_s must be an integer, got inf"),
    ],
)
def test_non_integer_budget_is_rejected_with_its_own_value(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(TVLValidationError, match=message):
        load_tvl_spec(spec_path=_write(tmp_path, _budgets(body)))


def test_integral_float_budget_loads(tmp_path: Path) -> None:
    # JSON Schema treats 3.0 as an integer, so canonical accepts it.
    body = "    max_trials: 3.0\n    max_wallclock_s: 60.0"
    artifact = load_tvl_spec(spec_path=_write(tmp_path, _budgets(body)))
    assert artifact.exploration_budgets == ExplorationBudgets(
        max_trials=3, max_wallclock_s=60
    )


def test_minimum_budgets_load(tmp_path: Path) -> None:
    body = "    max_trials: 1\n    max_spend_usd: 0\n    max_wallclock_s: 1"
    artifact = load_tvl_spec(spec_path=_write(tmp_path, _budgets(body)))
    assert artifact.exploration_budgets == ExplorationBudgets(
        max_trials=1, max_spend_usd=0.0, max_wallclock_s=1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_trials": 0},
        {"max_spend_usd": float("nan")},
        {"max_wallclock_s": -1},
    ],
)
def test_direct_construction_applies_the_same_bounds(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="exploration.budgets"):
        ExplorationBudgets(**kwargs)
