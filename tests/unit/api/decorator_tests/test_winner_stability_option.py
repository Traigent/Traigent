"""ExecutionOptions.winner_stability_reps contract tests.

The opt-in post-selection winner rerun count: ``0`` (default) is off, values
are validated at construction, and a configured value is plumbed onto the
``OptimizedFunction`` (from where the orchestrator reads it). Measured-only —
this option carries no gating and no claim semantics.
"""

from __future__ import annotations

import pydantic
import pytest

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.core.optimized_function import OptimizedFunction


class TestWinnerStabilityRepsOption:
    def test_default_is_zero_off(self):
        assert ExecutionOptions().winner_stability_reps == 0

    def test_positive_value_accepted(self):
        assert ExecutionOptions(winner_stability_reps=3).winner_stability_reps == 3

    @pytest.mark.parametrize("value", [-1, 1001])
    def test_out_of_range_rejected_at_construction(self, value: int):
        with pytest.raises(pydantic.ValidationError) as exc_info:
            ExecutionOptions(winner_stability_reps=value)
        assert "winner_stability_reps" in str(exc_info.value)

    def test_non_int_rejected_at_construction(self):
        with pytest.raises(pydantic.ValidationError):
            ExecutionOptions(winner_stability_reps="three")

    def test_execution_bundle_plumbs_onto_optimized_function(self):
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            execution=ExecutionOptions(winner_stability_reps=3),
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert isinstance(test_func, OptimizedFunction)
        assert test_func.winner_stability_reps == 3

    def test_execution_dict_plumbs_onto_optimized_function(self):
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            execution={"winner_stability_reps": 2},
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert test_func.winner_stability_reps == 2

    def test_direct_decorator_kwarg_plumbs_onto_optimized_function(self):
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            winner_stability_reps=4,
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert test_func.winner_stability_reps == 4

    def test_direct_decorator_kwarg_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="winner_stability_reps"):

            @optimize(
                configuration_space={"model": ["cheap", "strong"]},
                objectives=["accuracy"],
                winner_stability_reps=-1,
            )
            def test_func(text: str, model: str = "cheap") -> str:
                return f"Response: {text} ({model})"

    def test_conflicting_direct_and_bundle_values_rejected(self):
        with pytest.raises(TypeError, match="winner_stability_reps"):

            @optimize(
                configuration_space={"model": ["cheap", "strong"]},
                objectives=["accuracy"],
                winner_stability_reps=2,
                execution=ExecutionOptions(winner_stability_reps=3),
            )
            def test_func(text: str, model: str = "cheap") -> str:
                return f"Response: {text} ({model})"

    def test_default_off_without_execution_options(self):
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert test_func.winner_stability_reps == 0

    def test_call_time_kwarg_is_rejected_not_silently_dead(self):
        """Decorator-only: .optimize(winner_stability_reps=...) hard-fails."""

        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        with pytest.raises(TypeError, match="winner_stability_reps"):
            test_func.optimize_sync(winner_stability_reps=3)
