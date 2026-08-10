"""Regression coverage for the removed named strategy-preset selection API.

The epsilon / quality-floor / Pareto-frontier advisory *selection* rules
(``max_accuracy_then_cheapest_within_epsilon``, ``quality_floor_min_cost``,
``pareto_frontier``) were IP pulled back out of the SDK: that selection logic
is the technique and belongs in the backend, not in a package users can read.
``traigent.api.strategy_presets`` is deleted; nothing in the public surface may
still construct or expose a named preset.

Note: ``traigent.api.types.PresetSelection`` (the generic advisory-selection
*data container*, and ``OptimizationResult.preset_selection``) is deliberately
NOT covered here — it is retained so that results persisted by an older SDK
build can still be read back and displayed. Nothing in this build can ever
populate it with a live named-preset selection any more (see
``tests/unit/cli/test_results_commands.py::test_results_rerank_preset_option_is_gone``
for the CLI-side counterpart).
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

import traigent
import traigent.api
from traigent.api.types import ExampleResult, OptimizationStatus
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.registry import _OPTIMIZER_REGISTRY, register_optimizer

REMOVED_MODULES = ("traigent.api.strategy_presets",)

REMOVED_PRESET_NAMES = (
    "max_accuracy_then_cheapest_within_epsilon",
    "quality_floor_min_cost",
    "pareto_frontier",
)

REMOVED_ROOT_AND_API_EXPORTS = (
    "ADVISORY_SELECTION_NOTICE",
    "NormalizedStrategyPreset",
    "StrategyPresetError",
    "StrategyPresetValidationError",
    "UnknownStrategyPresetError",
    "VALID_PRESET_NAMES",
    "normalize_strategy_preset",
    "select_strategy_preset",
)


def assert_names_the_removal(message: str, preset_name: str) -> None:
    """Assert the message tells the user the *preset* is gone and what replaces it.

    Keyed to the substantive wording, not to the version number: "0.27.0" is
    owner-configurable, so pinning it would add brittleness without adding
    proof. What is pinned is the part a user acts on — that this name was a
    named strategy preset, that such presets were removed, and where to go
    instead (``algorithm=`` / ``objectives=``).

    The failure this exists to catch: a message like "strategy=... is not a
    valid optimizer; presets were removed in 0.27.0 ..." still contains
    "removed", so a bare ``"removed" in message`` check passes while the user
    is told their name is a bad optimizer rather than a retired feature.
    """
    lowered = message.lower()
    assert preset_name in message, message
    assert "named strategy preset" in lowered, message
    assert "removed" in lowered, message
    assert "algorithm=" in lowered, message
    assert "objectives=" in lowered, message


def test_strategy_presets_module_is_not_importable() -> None:
    for module_name in REMOVED_MODULES:
        module_message = f"{module_name} is still importable"
        assert importlib.util.find_spec(module_name) is None, module_message

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("traigent.api.strategy_presets")


def test_no_preset_symbol_is_exported_from_traigent_or_traigent_api() -> None:
    for export_name in REMOVED_ROOT_AND_API_EXPORTS:
        root_export_message = f"{export_name} is still in traigent.__all__"
        assert export_name not in traigent.__all__, root_export_message
        api_export_message = f"{export_name} is still in traigent.api.__all__"
        assert export_name not in traigent.api.__all__, api_export_message
        root_attribute_message = f"traigent still has attribute {export_name}"
        assert not hasattr(traigent, export_name), root_attribute_message
        api_attribute_message = f"traigent.api still has attribute {export_name}"
        assert not hasattr(traigent.api, export_name), api_attribute_message


@pytest.mark.parametrize("preset_name", REMOVED_PRESET_NAMES)
def test_removed_preset_names_raise_a_message_that_names_the_removal(
    preset_name: str,
) -> None:
    """A retired preset name is refused by name, not mistaken for an optimizer.

    Without this, ``strategy="quality_floor_min_cost"`` falls through to the
    deprecated optimizer alias: the user is first told to "use 'algorithm'
    instead" and then told the algorithm does not exist — an instruction that
    cannot be followed and never mentions that the feature was removed.

    "Names the removal" is checked as the wording a user acts on (this name
    was a named strategy preset, presets were removed, use ``algorithm=`` /
    ``objectives=``), not as the mere presence of the word "removed" — a
    message calling the name an invalid *optimizer* is exactly the regression
    this guards, and would otherwise slip through.
    """
    for raises in (
        lambda: traigent.optimize(strategy=preset_name),
        lambda: OptimizedFunction._resolve_runtime_strategy_argument(
            strategy=preset_name, strategy_params=None, algorithm=None
        ),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError) as excinfo:
                raises()

        message = str(excinfo.value)
        assert_names_the_removal(message, preset_name)
        # It must not send the user back to the name that just failed …
        assert f"algorithm={preset_name}" not in message
        # … and the deprecated-alias warning must not fire on the way out.
        assert not [
            entry for entry in caught if issubclass(entry.category, DeprecationWarning)
        ]


def test_only_the_three_retired_names_are_refused() -> None:
    """The refusal is by exact name, not by resemblance and not case-folded.

    A legal optimizer alias — including one whose name merely resembles a
    retired preset, and one that is a retired name in different casing — still
    takes the pre-existing deprecated-alias path and resolves, rather than
    being swept into the refusal. ``PARETO_FRONTIER`` is in this list because
    the registry is case-sensitive: it is a name a user can own (see
    ``test_a_registered_optimizer_named_like_a_preset_still_runs``), so the
    refusal must not claim it.
    """
    for strategy in (
        "GRID",
        "Grid_Search",
        "pareto_optimal",
        "quality_floor",
        "PARETO_FRONTIER",
        "Quality_Floor_Min_Cost",
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = OptimizedFunction._resolve_runtime_strategy_argument(
                strategy=strategy, strategy_params=None, algorithm=None
            )

        assert resolved == strategy
        assert [
            entry for entry in caught if issubclass(entry.category, DeprecationWarning)
        ], f"{strategy} lost the pre-existing deprecated-alias warning"


@pytest.fixture
def registered_uppercase_pareto_optimizer() -> Iterator[list[dict[str, Any]]]:
    """Register a user-owned optimizer literally named ``PARETO_FRONTIER``.

    ``register_optimizer`` accepts any casing and ``get_optimizer`` looks names
    up case-sensitively, so this is a legal, working name that has nothing to
    do with the retired ``pareto_frontier`` preset.
    """
    suggested: list[dict[str, Any]] = []

    class _UppercasePareto(GridSearchOptimizer):
        def suggest_next_trial(self, history: Any) -> dict[str, Any]:
            config = super().suggest_next_trial(history)
            suggested.append(dict(config))
            return config

    register_optimizer("PARETO_FRONTIER", _UppercasePareto)
    try:
        yield suggested
    finally:
        _OPTIMIZER_REGISTRY.pop("PARETO_FRONTIER", None)


def test_a_registered_optimizer_named_like_a_preset_still_runs(
    registered_uppercase_pareto_optimizer: list[dict[str, Any]],
) -> None:
    """The regression a case-folded refusal caused, pinned as an actual run.

    With folding, ``strategy="PARETO_FRONTIER"`` raised ``TypeError`` claiming
    the user's own registered optimizer was a retired preset — it could never
    run. Here the run completes and the registered optimizer is the one that
    produced the trials, proving the name reached the registry rather than the
    refusal.
    """

    async def evaluator(func: Any, config: Any, example: Any) -> ExampleResult:
        return ExampleResult(
            example_id="example-1",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
        )

    @traigent.optimize(
        configuration_space={"variant": ["a", "b"]},
        evaluation={
            "eval_dataset": [
                EvaluationExample(input_data={}, expected_output="answer"),
            ],
            "custom_evaluator": evaluator,
        },
        objectives=["accuracy"],
        max_trials=2,
    )
    def target() -> str:
        traigent.get_config()
        return "answer"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = target.optimize_sync(
            strategy="PARETO_FRONTIER", cost_approved=True, progress_bar=False
        )

    assert result.status is OptimizationStatus.COMPLETED
    # The trials exist and they came from the registered optimizer, not from
    # whatever the default would have been.
    assert len(result.trials) == 2
    optimizer_message = (
        "the registered PARETO_FRONTIER optimizer never suggested a trial"
    )
    assert registered_uppercase_pareto_optimizer, optimizer_message
    assert [trial.config for trial in result.trials] == (
        registered_uppercase_pareto_optimizer
    )
