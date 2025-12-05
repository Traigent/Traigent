"""Tests for loading TVL specifications."""

from __future__ import annotations

from pathlib import Path

from traigent.tvl.spec_loader import load_tvl_spec

FIXTURE_SPEC = Path(
    "docs/tvl/tvl-website/client/public/examples/ch1_motivation_experiment.tvl.yml"
)


def test_loads_configuration_space_and_objectives() -> None:
    """Ensure the sample spec normalizes spaces/objectives."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)

    assert "model" in artifact.configuration_space
    assert artifact.configuration_space["temperature"] == (0.1, 0.8)
    assert artifact.objective_schema is not None
    assert [obj.name for obj in artifact.objective_schema.objectives] == [
        "answer_quality",
        "response_latency",
        "token_cost",
    ]
    assert artifact.budget.max_trials == 60
    assert artifact.metadata["spec_id"] == "rag-campus-orientation"


def test_environment_overrides_budget_and_space() -> None:
    """Environment overlays update the merged spec before normalization."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC, environment="finals_week")

    assert artifact.budget.max_trials == 90
    assert artifact.configuration_space["retrieval_depth"] == (3.0, 8.0)


def test_compiled_constraints_attach_metadata() -> None:
    """Constraints are converted to decorator callables with metadata."""

    artifact = load_tvl_spec(spec_path=FIXTURE_SPEC)
    assert artifact.constraints, "spec defines constraints"
    constraint = artifact.constraints[0]
    meta = getattr(constraint, "__tvl_constraint__", {})
    assert meta["id"] == "campus-hour-latency"
    assert isinstance(constraint({"response_latency_ms": 800}), bool)
