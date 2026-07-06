"""Tests for decorator error handling and edge cases.

Tests error scenarios including:
- Invalid configurations
- Missing parameters
- Function execution errors
- Configuration conflicts
- Edge cases and boundary conditions
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from traigent.api.constraints import require
from traigent.api.decorators import ExecutionOptions, optimize
from traigent.api.parameter_ranges import Choices
from traigent.utils.exceptions import ConfigurationError, ValidationError

from .test_base import DecoratorTestBase


class TestConfigurationErrors(DecoratorTestBase):
    """Test configuration-related errors."""

    def _decorate_with_default_config(self, configuration_space, default_config):
        @optimize(
            configuration_space=configuration_space,
            default_config=default_config,
        )
        def test_func(text: str) -> str:
            return text

        return test_func

    def test_invalid_configuration_space_type(self):
        """Test error when configuration space is not a dict."""
        with pytest.raises(ValidationError):

            @optimize(configuration_space="invalid_type")  # Should be dict
            def test_func(text: str) -> str:
                return text

    def test_invalid_configuration_values(self):
        """Test error when configuration values are invalid."""
        with pytest.raises(ValidationError):

            @optimize(
                configuration_space={
                    "model": "gpt-4",  # Should be list
                    "temperature": 0.5,  # Should be list
                }
            )
            def test_func(text: str) -> str:
                return text

    def test_empty_configuration_lists(self):
        """Test error when configuration lists are empty."""
        with pytest.raises(ValidationError):

            @optimize(
                configuration_space={
                    "model": [],  # Empty list
                    "temperature": [],  # Empty list
                }
            )
            def test_func(text: str) -> str:
                return text

    def test_default_config_outside_categorical_domain_passes(self):
        """Default config may be a local baseline outside the tunable domain."""
        test_func = self._decorate_with_default_config(
            {"model": ["gpt-3.5", "gpt-4"]},
            {"model": "claude-2"},
        )

        assert hasattr(test_func, "optimize")

    def test_default_config_model_outside_space_passes(self):
        """Regression #1784: backend model repro is valid as a local baseline."""
        test_func = self._decorate_with_default_config(
            {"model": ["openrouter/deepseek/deepseek-chat"]},
            {"model": "openrouter/openai/gpt-4.1-mini"},
        )

        assert hasattr(test_func, "optimize")

    def test_default_config_categorical_uses_strict_type_membership(self):
        """Regression #1784: int 0 is not a strict member of float choice 0.0."""
        with pytest.raises(
            ConfigurationError,
            match="default_config\\['value'\\].*0.*type int.*declared type float",
        ):
            self._decorate_with_default_config({"value": [0.0]}, {"value": 0})

    def test_default_config_bool_categorical_uses_strict_type_membership(self):
        """Regression #1784: bool True is not a strict member of int choice 1."""
        with pytest.raises(
            ConfigurationError,
            match="default_config\\['flag'\\].*True.*type bool.*declared type int",
        ):
            self._decorate_with_default_config({"flag": [1]}, {"flag": True})

    def test_default_config_numeric_range_outside_domain_passes(self):
        """Default config ranges are allowed to describe local baselines."""
        test_func = self._decorate_with_default_config(
            {"temperature": (0.0, 1.0)}, {"temperature": 2.0}
        )

        assert hasattr(test_func, "optimize")

    def test_default_config_strict_member_passes(self):
        test_func = self._decorate_with_default_config({"value": [0.0]}, {"value": 0.0})

        assert hasattr(test_func, "optimize")

    def test_default_config_static_key_outside_space_is_ignored(self):
        test_func = self._decorate_with_default_config(
            {"model": ["openrouter/deepseek/deepseek-chat"]},
            {
                "model": "openrouter/deepseek/deepseek-chat",
                "static_timeout": 30,
            },
        )

        assert hasattr(test_func, "optimize")

    def test_invalid_objectives(self):
        """Test error when objectives are invalid."""
        with pytest.raises(ValidationError):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                objectives="accuracy",  # Should be list
            )
            def test_func(text: str) -> str:
                return text


class TestInjectionModeErrors(DecoratorTestBase):
    """Test injection mode related errors."""

    def test_invalid_injection_mode(self):
        """Test error with invalid injection mode."""
        with pytest.raises(ValueError):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="invalid_mode",  # Should be context/parameter/seamless
            )
            def test_func(text: str) -> str:
                return text

    def test_parameter_mode_missing_traigent_config(self):
        """Test parameter mode when function doesn't have traigent_config parameter."""
        # Parameter mode requires the config parameter to exist
        with pytest.raises(ConfigurationError):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                injection_mode="parameter",
            )
            def test_func(text: str) -> str:  # Missing config parameter
                return text

    def test_seamless_mode_parameter_conflict(self):
        """Test seamless mode with conflicting parameter names."""

        @optimize(
            configuration_space={
                "text": ["value1", "value2"]
            },  # Conflicts with function param
            injection_mode="seamless",
        )
        def test_func(text: str) -> str:
            return text

        # Should handle the conflict gracefully
        result = test_func("hello")
        assert result == "hello"


class TestFunctionExecutionErrors(DecoratorTestBase):
    """Test errors during function execution."""

    def test_function_raises_exception(self):
        """Test decorator handling when decorated function raises exception."""

        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        def failing_func(text: str) -> str:
            raise ValueError("Function failed")

        with pytest.raises(ValueError):
            failing_func("hello")

    def test_async_function_execution_error(self):
        """Test error handling in async decorated functions."""

        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        async def async_failing_func(text: str) -> str:
            raise RuntimeError("Async function failed")

        import asyncio

        with pytest.raises(RuntimeError):
            asyncio.run(async_failing_func("hello"))

    def test_function_with_invalid_return_type(self):
        """Test handling of functions with unexpected return types."""

        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        def invalid_return_func(text: str) -> str:
            return None  # Should return string

        # Should execute without crashing
        result = invalid_return_func("hello")
        assert result is None


class TestEdgeCases(DecoratorTestBase):
    """Test edge cases and boundary conditions."""

    def test_decorator_on_class_method(self):
        """Test decorator on class methods."""

        class TestClass:
            @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
            def method(self, text: str) -> str:
                return f"Method result: {text}"

        obj = TestClass()
        # Class method decoration may have issues with self parameter
        # Just test that the decorator can be applied without error
        assert hasattr(obj.method, "optimize")

        # For now, skip the actual execution test due to self parameter issues
        # This would need more complex handling in the provider

    def test_decorator_on_static_method(self):
        """Test decorator on static methods."""

        class TestClass:
            @staticmethod
            @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
            def static_method(text: str) -> str:
                return f"Static result: {text}"

        result = TestClass.static_method("hello")
        assert "Static result: hello" in result

    def test_decorator_with_property(self):
        """Test decorator interaction with property decorator."""

        class TestClass:
            def __init__(self):
                self._value = "initial"

            @property
            @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
            def value(self) -> str:
                return self._value

        # Properties with optimize decorator might not work as expected
        # This tests the error handling - verify class was created successfully
        obj = TestClass()
        assert hasattr(obj, "value")

    def test_multiple_decorators(self):
        """Test optimize decorator with other decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"[Wrapped] {result}"

            return wrapper

        @other_decorator
        @optimize(configuration_space={"model": ["gpt-3.5", "gpt-4"]})
        def decorated_func(text: str) -> str:
            return f"Result: {text}"

        result = decorated_func("hello")
        assert "[Wrapped]" in result
        assert "hello" in result

    def test_very_large_configuration_space(self):
        """Test with very large configuration space."""
        large_config = {
            f"param_{i}": list(range(10))
            for i in range(100)  # 100 parameters with 10 values each
        }

        @optimize(configuration_space=large_config)
        def large_config_func(text: str) -> str:
            return text

        # Should handle large configuration spaces
        result = large_config_func("hello")
        assert result == "hello"

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""

        @optimize(
            configuration_space={"prompt": ["Hello 世界", "Bonjour ñoño", "🚀 Rocket"]}
        )
        def unicode_func(text: str) -> str:
            return f"Unicode: {text}"

        result = unicode_func("测试")
        assert "测试" in result

    def test_recursive_decorated_function(self):
        """Test decorator on recursive functions."""

        @optimize(configuration_space={"depth": [1, 2, 3]})
        def recursive_func(n: int, depth: int = 2) -> int:
            if n <= 0:
                return 0
            return n + recursive_func(n - 1, depth)

        result = recursive_func(5)
        assert result == 15  # 5 + 4 + 3 + 2 + 1 + 0

    def test_generator_function(self):
        """Test decorator on generator functions."""

        @optimize(configuration_space={"batch_size": [10, 20, 50]})
        def generator_func(items: list, batch_size: int = 10):
            for i in range(0, len(items), batch_size):
                yield items[i : i + batch_size]

        items = list(range(25))
        batches = list(generator_func(items))
        assert len(batches) > 0
        assert all(isinstance(batch, list) for batch in batches)


class TestEnterpriseGatedFeatures(DecoratorTestBase):
    """Test enterprise-gated reps options are rejected at construction time.

    See issue #931: ``reps_per_trial`` and ``reps_aggregation`` were documented
    as public configuration but raised ``NotImplementedError`` late in
    ``_resolve_execution_bundle_options``. The contract is now enforced at the
    ``ExecutionOptions`` Pydantic boundary so users learn about the gate before
    the decorator runs.
    """

    def test_reps_per_trial_rejected_at_construction(self):
        """ExecutionOptions(reps_per_trial!=1) fails at construction."""
        with pytest.raises(PydanticValidationError) as exc:
            ExecutionOptions(reps_per_trial=3)

        message = str(exc.value)
        assert "reps_per_trial" in message
        assert "Traigent Enterprise" in message

    def test_reps_aggregation_rejected_at_construction(self):
        """ExecutionOptions(reps_aggregation!='mean') fails at construction."""
        with pytest.raises(PydanticValidationError) as exc:
            ExecutionOptions(reps_aggregation="median")

        message = str(exc.value)
        assert "reps_aggregation" in message
        assert "Traigent Enterprise" in message

    def test_reps_per_trial_dict_rejected_at_decoration(self):
        """Passing reps_per_trial via execution={...} fails at decoration."""
        with pytest.raises(PydanticValidationError) as exc:

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                execution={"reps_per_trial": 3},
            )
            def test_func(text: str) -> str:
                return text

        message = str(exc.value)
        assert "reps_per_trial" in message
        assert "Traigent Enterprise" in message

    def test_reps_aggregation_dict_rejected_at_decoration(self):
        """Passing reps_aggregation via execution={...} fails at decoration."""
        with pytest.raises(PydanticValidationError) as exc:

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                execution={"reps_aggregation": "median"},
            )
            def test_func(text: str) -> str:
                return text

        message = str(exc.value)
        assert "reps_aggregation" in message
        assert "Traigent Enterprise" in message

    def test_default_reps_values_work(self):
        """Test that default reps values (1, 'mean') work fine."""

        @optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            execution={"reps_per_trial": 1, "reps_aggregation": "mean"},
        )
        def test_func(text: str) -> str:
            return text

        # Should not raise - defaults are allowed
        result = test_func("hello")
        assert result == "hello"

    def test_default_reps_construction_succeeds(self):
        """ExecutionOptions() and explicit defaults must still construct."""
        # Implicit defaults
        ExecutionOptions()
        # Explicit defaults
        ExecutionOptions(reps_per_trial=1, reps_aggregation="mean")

    @pytest.mark.parametrize(
        ("field", "value", "option_field"),
        [
            ("hybrid_api_batch_size", 0, "batch_size"),
            ("hybrid_api_batch_parallelism", 0, "batch_parallelism"),
            ("hybrid_api_heartbeat_interval", 0.0, "heartbeat_interval"),
            ("hybrid_api_heartbeat_interval", -5.0, "heartbeat_interval"),
        ],
    )
    def test_hybrid_api_execution_options_reject_degenerate_values(
        self, field, value, option_field
    ):
        """Hybrid API execution controls must be positive."""
        with pytest.raises(PydanticValidationError) as exc:
            ExecutionOptions(**{field: value})

        assert option_field in str(exc.value)

    def test_max_trials_zero_rejected_at_decoration(self):
        """max_trials=0 fails at decoration instead of producing a no-op run."""
        with pytest.raises(ValueError, match="max_trials must be a positive integer"):

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                max_trials=0,
            )
            def test_func(text: str) -> str:
                return text

    def test_unsatisfiable_finite_constraints_rejected_at_decoration(self):
        """Contradictory finite-domain constraints fail before optimization starts."""
        model = Choices(["a", "b"], name="model")

        with pytest.raises(ValueError, match="constraints are unsatisfiable"):

            @optimize(
                configuration_space={"model": model},
                constraints=[require(model.equals("a")), require(model.equals("b"))],
            )
            def test_func(text: str) -> str:
                return text

    def test_satisfiable_finite_constraints_still_decorate(self):
        """A finite-domain constraint set with a valid config is unchanged."""
        model = Choices(["a", "b"], name="model")

        @optimize(
            configuration_space={"model": model},
            constraints=[require(model.equals("a"))],
        )
        def test_func(text: str) -> str:
            return text

        assert test_func("ok") == "ok"

    def test_reps_assignment_rejected_after_construction(self):
        """Assignment validation preserves the enterprise gate after construction."""
        options = ExecutionOptions()

        with pytest.raises(PydanticValidationError) as exc:
            options.reps_per_trial = 3

        message = str(exc.value)
        assert "reps_per_trial" in message
        assert "Traigent Enterprise" in message


class TestExecutionOptionsRejectsUnknownKeys(DecoratorTestBase):
    """Traigent#1723 (g1:F1): ExecutionOptions used ``extra="allow"`` and
    silently swallowed misspelled option keys into ``__pydantic_extra__``
    where nothing ever read them (a typo like ``algorithmn="grid"`` or
    ``offlne=True`` degraded the run to defaults with no error). It is now
    ``extra="forbid"`` with the still-supported legacy keys split out before
    the forbid check runs, so unknown keys raise and legacy keys keep working.
    """

    @pytest.mark.parametrize(
        ("kwargs", "bad_key"),
        [
            ({"algorithmn": "grid"}, "algorithmn"),
            ({"offlne": True}, "offlne"),
            ({"minimal_loging": False}, "minimal_loging"),
            ({"totally_made_up": 1}, "totally_made_up"),
        ],
    )
    def test_unknown_key_raises_naming_the_key(self, kwargs, bad_key):
        """A genuinely unknown/misspelled key raises ValidationError naming it."""
        with pytest.raises(PydanticValidationError) as exc:
            ExecutionOptions(**kwargs)

        message = str(exc.value)
        assert bad_key in message
        assert "extra" in message.lower() or "not permitted" in message.lower()

    def test_unknown_key_via_execution_dict_raises_at_decoration(self):
        """The same rejection fires through @optimize(execution={...})."""
        with pytest.raises(PydanticValidationError) as exc:

            @optimize(
                configuration_space={"model": ["gpt-3.5", "gpt-4"]},
                execution={"algorithmn": "grid"},
            )
            def test_func(text: str) -> str:
                return text

        assert "algorithmn" in str(exc.value)

    @pytest.mark.parametrize(
        ("legacy_key", "value"),
        [
            ("execution_mode", "hybrid_api"),
            ("privacy_enabled", True),
            ("cloud_fallback_policy", "auto"),
        ],
    )
    def test_legacy_scalar_keys_round_trip_to_stash(self, legacy_key, value):
        """Tolerated legacy keys are captured verbatim, not rejected."""
        options = ExecutionOptions(**{legacy_key: value})

        assert options.legacy_option_values.get(legacy_key) == value
        # Legacy keys must NOT leak into the model as real fields.
        assert legacy_key not in ExecutionOptions.model_fields

    def test_legacy_hybrid_api_key_round_trips_to_stash(self):
        """A flat hybrid_api_* key is tolerated and stashed, not rejected."""
        transport = object()
        options = ExecutionOptions(hybrid_api_transport=transport)

        assert options.legacy_option_values.get("hybrid_api_transport") is transport

    def test_multiple_legacy_keys_round_trip_together(self):
        """Several legacy keys survive together with public fields intact."""
        transport = object()
        options = ExecutionOptions(
            algorithm="grid",
            execution_mode="hybrid_api",
            privacy_enabled=True,
            cloud_fallback_policy="auto",
            hybrid_api_transport=transport,
        )

        assert options.algorithm == "grid"
        assert options.legacy_option_values == {
            "execution_mode": "hybrid_api",
            "privacy_enabled": True,
            "cloud_fallback_policy": "auto",
            "hybrid_api_transport": transport,
        }

    def test_legacy_stash_survives_unrelated_field_assignment(self):
        """Finding 1a: validate_assignment=True re-runs the wrap validator on
        every field set with a non-Mapping input; the legacy stash captured at
        construction must be carried forward, not silently cleared to {}."""
        transport = object()
        options = ExecutionOptions(
            execution_mode="hybrid_api",
            privacy_enabled=True,
            cloud_fallback_policy="auto",
            hybrid_api_transport=transport,
        )

        # Assign an unrelated PUBLIC field (goes through validate_assignment).
        options.minimal_logging = False

        assert options.minimal_logging is False
        assert options.legacy_option_values == {
            "execution_mode": "hybrid_api",
            "privacy_enabled": True,
            "cloud_fallback_policy": "auto",
            "hybrid_api_transport": transport,
        }

    def test_no_legacy_keys_yields_empty_stash(self):
        """A clean bundle has an empty legacy stash (byte-identical happy path)."""
        assert ExecutionOptions().legacy_option_values == {}
        assert ExecutionOptions(algorithm="grid").legacy_option_values == {}
