"""Contract tests for fail-closed objective direction resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)
from traigent.api.decorators import optimize
from traigent.core import objective_directions
from traigent.core.orchestrator_helpers import prepare_objectives
from traigent.utils.results_table import _find_best_per_objective, _get_objective_info


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("accuracy", "maximize"),
        ("success", "maximize"),
        ("success_rate", "maximize"),
        ("exact_match_default", "maximize"),
        ("cost", "minimize"),
        ("total_cost", "minimize"),
        ("cost_per_example_mean", "minimize"),
        ("input_cost", "minimize"),
        ("output_cost", "minimize"),
        ("latency", "minimize"),
        ("duration", "minimize"),
        ("response_time_ms", "minimize"),
        ("avg_response_time", "minimize"),
        ("avg_response_time_ms", "minimize"),
        ("execution_time_ms", "minimize"),
        ("error_rate", "minimize"),
        ("empty_output_rate", "minimize"),
        ("truncated_output_rate", "minimize"),
    ],
)
def test_sdk_owned_names_have_exact_canonical_orientation(
    name: str, expected: str
) -> None:
    schema = create_default_objectives([name])
    assert schema.get_orientation(name) == expected
    assert _get_objective_info([name]) == [(name, expected)]


@pytest.mark.parametrize(
    "name", ["accuracy_loss", "cost_savings", "time_to_first_success"]
)
def test_semantically_misleading_custom_names_require_declaration(name: str) -> None:
    with pytest.raises(ValueError, match="has no declared orientation") as error:
        create_default_objectives([name])

    message = str(error.value)
    assert "from traigent.core.objectives import create_default_objectives" in message
    assert f"create_default_objectives([{name!r}]" in message
    assert f"orientations={{{name!r}: 'minimize'}}" in message


def test_custom_name_explicit_orientation_wins() -> None:
    schema = create_default_objectives(
        ["plugin_quality"], orientations={"plugin_quality": "maximize"}
    )
    assert schema.get_orientation("plugin_quality") == "maximize"


@pytest.mark.parametrize("name", ["accuracy", "total_cost"])
def test_explicit_orientation_overrides_known_name(name: str) -> None:
    expected = "minimize" if name == "accuracy" else "maximize"
    schema = create_default_objectives([name], orientations={name: expected})
    assert schema.get_orientation(name) == expected


def test_invalid_explicit_orientation_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 'maximize' or 'minimize'"):
        create_default_objectives(
            ["custom_metric"], orientations={"custom_metric": "ascending"}
        )


def test_from_dict_rejects_missing_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="has no declared orientation"):
        ObjectiveDefinition.from_dict({"name": "plugin_quality", "weight": 1.0})


def test_from_dict_uses_canonical_default_and_preserves_explicit_override() -> None:
    defaulted = ObjectiveDefinition.from_dict({"name": "total_cost", "weight": 1.0})
    overridden = ObjectiveDefinition.from_dict(
        {"name": "total_cost", "orientation": "maximize", "weight": 1.0}
    )
    assert defaulted.orientation == "minimize"
    assert overridden.orientation == "maximize"


def test_prepare_objectives_does_not_swallow_missing_orientation() -> None:
    with pytest.raises(ValueError, match="has no declared orientation"):
        prepare_objectives(["plugin_quality"], None)


def test_results_table_object_without_orientation_uses_canonical_resolver() -> None:
    objectives = SimpleNamespace(objectives=[SimpleNamespace(name="total_cost")])
    assert _get_objective_info(objectives) == [("total_cost", "minimize")]


def test_results_table_rejects_unknown_bare_name() -> None:
    with pytest.raises(ValueError, match="has no declared orientation"):
        _get_objective_info(["plugin_quality"])


def test_results_table_observes_canonical_resolver_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: the table must not retain a private direction copy."""
    original = objective_directions.resolve_objective_orientation

    def mutated(name: str, explicit: str | None = None) -> str:
        if name == "accuracy" and explicit is None:
            return "minimize"
        return original(name, explicit)

    monkeypatch.setattr(objective_directions, "resolve_objective_orientation", mutated)
    assert _get_objective_info(["accuracy"]) == [("accuracy", "minimize")]


def test_results_table_does_not_treat_band_as_maximize() -> None:
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(
                name="response_length",
                orientation="band",
                weight=1.0,
                band={"target": [90.0, 110.0]},
            )
        ]
    )
    objective_info = _get_objective_info(schema)

    assert objective_info == [("response_length", "band")]
    assert _find_best_per_objective([], objective_info) == {"response_length": set()}


@pytest.mark.parametrize("offline", [False, True])
def test_public_decorator_rejects_unknown_bare_objective_early(
    offline: bool,
) -> None:
    with pytest.raises(ValueError, match="has no declared orientation"):

        @optimize(
            objectives=["plugin_quality"],
            configuration_space={"temperature": [0.0]},
            offline=offline,
            mock={"enabled": True},
        )
        def agent(prompt: str) -> str:
            return prompt


def test_public_decorator_preserves_declared_custom_orientation() -> None:
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name="plugin_quality", orientation="maximize", weight=1.0)]
    )

    @optimize(
        objectives=schema,
        configuration_space={"temperature": [0.0, 1.0]},
        offline=True,
        mock={"enabled": True},
    )
    def agent(prompt: str, temperature: float = 0.0) -> str:
        return prompt

    assert agent.objective_schema.get_orientation("plugin_quality") == "maximize"


def test_backend_wire_preserves_declared_custom_direction() -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    wire = OptimizationOrchestrator._session_objective_to_wire(
        ObjectiveDefinition(name="plugin_quality", orientation="minimize", weight=1.0)
    )
    assert wire == {
        "name": "plugin_quality",
        "orientation": "minimize",
        "weight": 1.0,
    }


def test_optimizer_preserves_custom_band_target_semantics() -> None:
    from traigent.api.types import TrialResult, TrialStatus
    from traigent.optimizers.random import RandomSearchOptimizer
    from traigent.tvl.models import BandTarget

    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(
                name="response_length",
                orientation="band",
                weight=1.0,
                band=BandTarget(low=90.0, high=110.0),
            )
        ]
    )
    optimizer = RandomSearchOptimizer(
        {"temperature": [0.0, 1.0]},
        ["response_length"],
        objective_schema=schema,
    )
    optimizer.update_best(
        TrialResult(
            trial_id="too-long",
            config={"temperature": 1.0},
            metrics={"response_length": 180.0},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=0.0,
        )
    )
    optimizer.update_best(
        TrialResult(
            trial_id="in-band",
            config={"temperature": 0.0},
            metrics={"response_length": 100.0},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=0.0,
        )
    )

    assert optimizer.objective_orientations == {"response_length": "band"}
    assert optimizer.best_config == {"temperature": 0.0}
    assert optimizer.best_score == 100.0

    optimizer.reset()
    optimizer.update_best(
        TrialResult(
            trial_id="second-run-in-band",
            config={"temperature": 1.0},
            metrics={"response_length": 105.0},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=0.0,
        )
    )

    assert optimizer.best_config == {"temperature": 1.0}
    assert optimizer.best_score == 105.0
