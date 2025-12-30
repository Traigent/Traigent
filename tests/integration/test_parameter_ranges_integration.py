"""Integration tests for SE-friendly parameter ranges with @optimize decorator.

Tests cover:
- Inline Range/Choices in decorator
- Inline tuple/list in decorator
- Mixed old + new syntax in configuration_space
- Inline overrides configuration_space entry
- Defaults flow to default_config
- Unknown kwarg raises TypeError (typo protection)
- End-to-end optimization with mock mode
"""

import os
from typing import Any

import pytest

# Set mock mode before importing traigent
os.environ["TRAIGENT_MOCK_MODE"] = "true"

import traigent
from traigent import Choices, IntRange, LogRange, Range, optimize
from traigent.core.optimized_function import OptimizedFunction


class TestDecoratorInlineRanges:
    """Test inline Range/Choices in @optimize decorator."""

    def test_inline_range_in_decorator(self):
        """Test Range can be used inline in decorator."""

        @optimize(
            eval_dataset="test.jsonl",
            temperature=Range(0.0, 2.0),
        )
        def my_func(query: str) -> str:
            return query

        assert isinstance(my_func, OptimizedFunction)
        assert "temperature" in my_func.configuration_space
        # Should be normalized to tuple
        assert my_func.configuration_space["temperature"] == (0.0, 2.0)

    def test_inline_int_range_in_decorator(self):
        """Test IntRange can be used inline in decorator."""

        @optimize(
            eval_dataset="test.jsonl",
            max_tokens=IntRange(100, 4096),
        )
        def my_func(query: str) -> str:
            return query

        assert "max_tokens" in my_func.configuration_space
        assert my_func.configuration_space["max_tokens"] == (100, 4096)

    def test_inline_log_range_in_decorator(self):
        """Test LogRange can be used inline in decorator."""

        @optimize(
            eval_dataset="test.jsonl",
            learning_rate=LogRange(1e-5, 1e-1),
        )
        def my_func(query: str) -> str:
            return query

        assert "learning_rate" in my_func.configuration_space
        config = my_func.configuration_space["learning_rate"]
        assert isinstance(config, dict)
        assert config["log"] is True

    def test_inline_choices_in_decorator(self):
        """Test Choices can be used inline in decorator."""

        @optimize(
            eval_dataset="test.jsonl",
            model=Choices(["gpt-4", "gpt-3.5-turbo"]),
        )
        def my_func(query: str) -> str:
            return query

        assert "model" in my_func.configuration_space
        assert my_func.configuration_space["model"] == ["gpt-4", "gpt-3.5-turbo"]

    def test_multiple_inline_params(self):
        """Test multiple inline parameters together."""

        @optimize(
            eval_dataset="test.jsonl",
            temperature=Range(0.0, 2.0),
            max_tokens=IntRange(100, 4096),
            model=Choices(["gpt-4", "gpt-3.5"]),
        )
        def my_func(query: str) -> str:
            return query

        assert len(my_func.configuration_space) == 3
        assert "temperature" in my_func.configuration_space
        assert "max_tokens" in my_func.configuration_space
        assert "model" in my_func.configuration_space


class TestDecoratorInlineTupleList:
    """Test inline tuple/list syntax (legacy syntax in inline position)."""

    def test_inline_tuple_in_decorator(self):
        """Test tuple range can be used inline in decorator."""

        @optimize(
            eval_dataset="test.jsonl",
            temperature=(0.0, 1.0),
        )
        def my_func(query: str) -> str:
            return query

        assert "temperature" in my_func.configuration_space
        assert my_func.configuration_space["temperature"] == (0.0, 1.0)

    def test_inline_list_raises_type_error(self):
        """Test that inline list raises TypeError (use Choices instead)."""
        # Lists are NOT treated as inline param definitions to catch typos.
        # Users should use Choices([...]) for categorical parameters.
        with pytest.raises(TypeError, match="Unknown keyword arguments"):

            @optimize(
                eval_dataset="test.jsonl",
                model=["gpt-4", "gpt-3.5"],
            )
            def my_func(query: str) -> str:
                return query


class TestConfigurationSpaceWithRanges:
    """Test Range/Choices in configuration_space dict."""

    def test_range_in_configuration_space(self):
        """Test Range in configuration_space dict."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={
                "temperature": Range(0.0, 2.0),
            },
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.configuration_space["temperature"] == (0.0, 2.0)

    def test_choices_in_configuration_space(self):
        """Test Choices in configuration_space dict."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={
                "model": Choices(["gpt-4", "gpt-3.5"]),
            },
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.configuration_space["model"] == ["gpt-4", "gpt-3.5"]

    def test_mixed_syntax_in_configuration_space(self):
        """Test mixed old and new syntax in configuration_space."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={
                "temperature": (0.0, 1.0),  # Old syntax
                "model": Choices(["gpt-4"]),  # New syntax
                "max_tokens": [100, 500],  # Old syntax
            },
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.configuration_space["temperature"] == (0.0, 1.0)
        assert my_func.configuration_space["model"] == ["gpt-4"]
        assert my_func.configuration_space["max_tokens"] == [100, 500]


class TestPrecedenceRules:
    """Test precedence: inline kwargs override configuration_space dict."""

    def test_inline_overrides_config_space(self):
        """Test inline params take precedence over configuration_space."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={"temp": (0.0, 1.0)},
            temp=Range(0.0, 2.0),  # Should win!
        )
        def my_func(query: str) -> str:
            return query

        # Inline should override
        assert my_func.configuration_space["temp"] == (0.0, 2.0)

    def test_inline_merges_with_config_space(self):
        """Test inline params merge with configuration_space."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={"model": ["gpt-4"]},
            temperature=Range(0.0, 1.0),
        )
        def my_func(query: str) -> str:
            return query

        assert "model" in my_func.configuration_space
        assert "temperature" in my_func.configuration_space


class TestDefaultHandling:
    """Test that default values flow to default_config."""

    def test_range_default_flows_to_default_config(self):
        """Test Range default populates default_config."""

        @optimize(
            eval_dataset="test.jsonl",
            temperature=Range(0.0, 2.0, default=0.7),
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.default_config is not None
        assert my_func.default_config.get("temperature") == 0.7

    def test_choices_default_flows_to_default_config(self):
        """Test Choices default populates default_config."""

        @optimize(
            eval_dataset="test.jsonl",
            model=Choices(["gpt-4", "gpt-3.5"], default="gpt-4"),
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.default_config is not None
        assert my_func.default_config.get("model") == "gpt-4"

    def test_explicit_default_config_takes_precedence(self):
        """Test explicit default_config overrides Range/Choices defaults."""

        @optimize(
            eval_dataset="test.jsonl",
            default_config={"temperature": 1.5},
            temperature=Range(0.0, 2.0, default=0.7),
        )
        def my_func(query: str) -> str:
            return query

        # Explicit default_config should win
        assert my_func.default_config["temperature"] == 1.5

    def test_multiple_defaults(self):
        """Test multiple parameters with defaults."""

        @optimize(
            eval_dataset="test.jsonl",
            temperature=Range(0.0, 2.0, default=0.7),
            model=Choices(["gpt-4", "gpt-3.5"], default="gpt-4"),
            max_tokens=IntRange(100, 4096, default=1000),
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.default_config["temperature"] == 0.7
        assert my_func.default_config["model"] == "gpt-4"
        assert my_func.default_config["max_tokens"] == 1000


class TestTypoProtection:
    """Test that unknown kwargs raise TypeError (typo protection)."""

    def test_unknown_kwarg_raises_type_error(self):
        """Test unknown kwarg raises clear error."""
        with pytest.raises(TypeError, match="Unknown keyword arguments"):

            @optimize(
                eval_dataset="test.jsonl",
                objectivs=["accuracy"],  # Typo!
            )
            def my_func(query: str) -> str:
                return query

    def test_unknown_string_kwarg_raises_type_error(self):
        """Test string value (not a param definition) raises error."""
        with pytest.raises(TypeError, match="Unknown keyword arguments"):

            @optimize(
                eval_dataset="test.jsonl",
                some_param="value",  # Not a valid param definition
            )
            def my_func(query: str) -> str:
                return query


class TestOptunaIntegration:
    """Test that parameter ranges work with Optuna optimizer."""

    def test_range_to_optuna_distribution(self):
        """Test Range converts to Optuna FloatDistribution."""
        from traigent.optimizers.optuna_utils import config_space_to_distributions

        config_space = {
            "temperature": Range(0.0, 2.0),
        }
        distributions = config_space_to_distributions(config_space)

        from optuna.distributions import FloatDistribution

        assert isinstance(distributions["temperature"], FloatDistribution)
        assert distributions["temperature"].low == 0.0
        assert distributions["temperature"].high == 2.0

    def test_int_range_to_optuna_distribution(self):
        """Test IntRange converts to Optuna IntDistribution."""
        from traigent.optimizers.optuna_utils import config_space_to_distributions

        config_space = {
            "max_tokens": IntRange(100, 4096),
        }
        distributions = config_space_to_distributions(config_space)

        from optuna.distributions import IntDistribution

        assert isinstance(distributions["max_tokens"], IntDistribution)
        assert distributions["max_tokens"].low == 100
        assert distributions["max_tokens"].high == 4096

    def test_log_range_to_optuna_distribution(self):
        """Test LogRange converts to Optuna FloatDistribution with log=True."""
        from traigent.optimizers.optuna_utils import config_space_to_distributions

        config_space = {
            "learning_rate": LogRange(1e-5, 1e-1),
        }
        distributions = config_space_to_distributions(config_space)

        from optuna.distributions import FloatDistribution

        assert isinstance(distributions["learning_rate"], FloatDistribution)
        assert distributions["learning_rate"].log is True

    def test_choices_to_optuna_distribution(self):
        """Test Choices converts to Optuna CategoricalDistribution."""
        from traigent.optimizers.optuna_utils import config_space_to_distributions

        config_space = {
            "model": Choices(["gpt-4", "gpt-3.5"]),
        }
        distributions = config_space_to_distributions(config_space)

        from optuna.distributions import CategoricalDistribution

        assert isinstance(distributions["model"], CategoricalDistribution)
        assert distributions["model"].choices == ("gpt-4", "gpt-3.5")

    def test_range_with_step_to_optuna(self):
        """Test Range with step converts correctly."""
        from traigent.optimizers.optuna_utils import config_space_to_distributions

        config_space = {
            "temperature": Range(0.0, 1.0, step=0.1),
        }
        distributions = config_space_to_distributions(config_space)

        assert distributions["temperature"].step == 0.1


class TestBackwardCompatibility:
    """Test backward compatibility with old tuple/list syntax."""

    def test_old_configuration_space_syntax(self):
        """Test old configuration_space syntax still works."""

        @optimize(
            eval_dataset="test.jsonl",
            configuration_space={
                "temperature": (0.0, 1.0),
                "model": ["gpt-4", "gpt-3.5"],
            },
        )
        def my_func(query: str) -> str:
            return query

        assert my_func.configuration_space["temperature"] == (0.0, 1.0)
        assert my_func.configuration_space["model"] == ["gpt-4", "gpt-3.5"]


class TestImportFromTraigent:
    """Test that classes can be imported from traigent package."""

    def test_import_range(self):
        from traigent import Range

        r = Range(0.0, 1.0)
        assert r.low == 0.0

    def test_import_int_range(self):
        from traigent import IntRange

        r = IntRange(1, 10)
        assert r.low == 1

    def test_import_log_range(self):
        from traigent import LogRange

        r = LogRange(0.01, 1.0)
        assert r.low == 0.01

    def test_import_choices(self):
        from traigent import Choices

        c = Choices(["a", "b"])
        assert len(c) == 2

    def test_import_parameter_range(self):
        from traigent import ParameterRange, Range

        assert isinstance(Range(0.0, 1.0), ParameterRange)
