"""Tests for constraint builder mixins.

Tests the NumericConstraintBuilderMixin and CategoricalConstraintBuilderMixin
to ensure all constraint methods work correctly for all ParameterRange types.
"""

from __future__ import annotations

from traigent.api.constraints import Condition
from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range


class TestNumericConstraintBuilders:
    """Tests for NumericConstraintBuilderMixin methods on Range/IntRange/LogRange."""

    # =========================================================================
    # Range Tests
    # =========================================================================

    def test_range_equals(self) -> None:
        """Test Range.equals() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.equals(0.5)

        assert isinstance(cond, Condition)
        assert cond.operator == "=="
        assert cond.value == 0.5
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.7) is False

    def test_range_not_equals(self) -> None:
        """Test Range.not_equals() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.not_equals(0.5)

        assert cond.operator == "!="
        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.5) is False

    def test_range_gt(self) -> None:
        """Test Range.gt() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.gt(0.5)

        assert cond.operator == ">"
        assert cond.evaluate(0.6) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.4) is False

    def test_range_gte(self) -> None:
        """Test Range.gte() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.gte(0.5)

        assert cond.operator == ">="
        assert cond.evaluate(0.6) is True
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.4) is False

    def test_range_lt(self) -> None:
        """Test Range.lt() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lt(0.5)

        assert cond.operator == "<"
        assert cond.evaluate(0.4) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.6) is False

    def test_range_lte(self) -> None:
        """Test Range.lte() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lte(0.5)

        assert cond.operator == "<="
        assert cond.evaluate(0.4) is True
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.6) is False

    def test_range_in_range(self) -> None:
        """Test Range.in_range() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.in_range(0.3, 0.7)

        assert cond.operator == "in_range"
        assert cond.value == (0.3, 0.7)
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.3) is True
        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.2) is False
        assert cond.evaluate(0.8) is False

    def test_range_is_in(self) -> None:
        """Test Range.is_in() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.is_in([0.0, 0.5, 1.0])

        assert cond.operator == "in"
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.3) is False

    def test_range_not_in(self) -> None:
        """Test Range.not_in() creates correct condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.not_in([0.0, 1.0])

        assert cond.operator == "not_in"
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.0) is False

    # =========================================================================
    # IntRange Tests
    # =========================================================================

    def test_int_range_equals(self) -> None:
        """Test IntRange.equals() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.equals(512)

        assert cond.operator == "=="
        assert cond.evaluate(512) is True
        assert cond.evaluate(256) is False

    def test_int_range_not_equals(self) -> None:
        """Test IntRange.not_equals() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.not_equals(512)

        assert cond.operator == "!="
        assert cond.evaluate(256) is True
        assert cond.evaluate(512) is False

    def test_int_range_gt(self) -> None:
        """Test IntRange.gt() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.gt(500)

        assert cond.operator == ">"
        assert cond.evaluate(600) is True
        assert cond.evaluate(500) is False
        assert cond.evaluate(400) is False

    def test_int_range_gte(self) -> None:
        """Test IntRange.gte() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.gte(500)

        assert cond.operator == ">="
        assert cond.evaluate(600) is True
        assert cond.evaluate(500) is True
        assert cond.evaluate(400) is False

    def test_int_range_lt(self) -> None:
        """Test IntRange.lt() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.lt(500)

        assert cond.operator == "<"
        assert cond.evaluate(400) is True
        assert cond.evaluate(500) is False
        assert cond.evaluate(600) is False

    def test_int_range_lte(self) -> None:
        """Test IntRange.lte() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.lte(500)

        assert cond.operator == "<="
        assert cond.evaluate(400) is True
        assert cond.evaluate(500) is True
        assert cond.evaluate(600) is False

    def test_int_range_in_range(self) -> None:
        """Test IntRange.in_range() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.in_range(256, 1024)

        assert cond.operator == "in_range"
        assert cond.evaluate(512) is True
        assert cond.evaluate(256) is True
        assert cond.evaluate(1024) is True
        assert cond.evaluate(128) is False
        assert cond.evaluate(2048) is False

    def test_int_range_is_in(self) -> None:
        """Test IntRange.is_in() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.is_in([256, 512, 1024])

        assert cond.operator == "in"
        assert cond.evaluate(512) is True
        assert cond.evaluate(300) is False

    def test_int_range_not_in(self) -> None:
        """Test IntRange.not_in() creates correct condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = tokens.not_in([256, 1024])

        assert cond.operator == "not_in"
        assert cond.evaluate(512) is True
        assert cond.evaluate(256) is False

    # =========================================================================
    # LogRange Tests
    # =========================================================================

    def test_log_range_equals(self) -> None:
        """Test LogRange.equals() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.equals(1e-3)

        assert cond.operator == "=="
        assert cond.evaluate(1e-3) is True
        assert cond.evaluate(1e-4) is False

    def test_log_range_not_equals(self) -> None:
        """Test LogRange.not_equals() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.not_equals(1e-3)

        assert cond.operator == "!="
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-3) is False

    def test_log_range_gt(self) -> None:
        """Test LogRange.gt() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.gt(1e-4)

        assert cond.operator == ">"
        assert cond.evaluate(1e-3) is True
        assert cond.evaluate(1e-4) is False
        assert cond.evaluate(1e-5) is False

    def test_log_range_gte(self) -> None:
        """Test LogRange.gte() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.gte(1e-4)

        assert cond.operator == ">="
        assert cond.evaluate(1e-3) is True
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-5) is False

    def test_log_range_lt(self) -> None:
        """Test LogRange.lt() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.lt(1e-3)

        assert cond.operator == "<"
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-3) is False
        assert cond.evaluate(1e-2) is False

    def test_log_range_lte(self) -> None:
        """Test LogRange.lte() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.lte(1e-3)

        assert cond.operator == "<="
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-3) is True
        assert cond.evaluate(1e-2) is False

    def test_log_range_in_range(self) -> None:
        """Test LogRange.in_range() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.in_range(1e-4, 1e-2)

        assert cond.operator == "in_range"
        assert cond.evaluate(1e-3) is True
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-2) is True
        assert cond.evaluate(1e-5) is False
        assert cond.evaluate(1e-1) is False

    def test_log_range_is_in(self) -> None:
        """Test LogRange.is_in() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.is_in([1e-5, 1e-4, 1e-3])

        assert cond.operator == "in"
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-2) is False

    def test_log_range_not_in(self) -> None:
        """Test LogRange.not_in() creates correct condition."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")
        cond = lr.not_in([1e-5, 1e-3])

        assert cond.operator == "not_in"
        assert cond.evaluate(1e-4) is True
        assert cond.evaluate(1e-3) is False


class TestCategoricalConstraintBuilders:
    """Tests for CategoricalConstraintBuilderMixin methods on Choices."""

    def test_choices_equals(self) -> None:
        """Test Choices.equals() creates correct condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.equals("gpt-4")

        assert isinstance(cond, Condition)
        assert cond.operator == "=="
        assert cond.evaluate("gpt-4") is True
        assert cond.evaluate("gpt-3.5") is False

    def test_choices_not_equals(self) -> None:
        """Test Choices.not_equals() creates correct condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.not_equals("gpt-4")

        assert cond.operator == "!="
        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("gpt-4") is False

    def test_choices_is_in(self) -> None:
        """Test Choices.is_in() creates correct condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.is_in(["gpt-4", "gpt-3.5"])

        assert cond.operator == "in"
        assert cond.evaluate("gpt-4") is True
        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("claude") is False

    def test_choices_not_in(self) -> None:
        """Test Choices.not_in() creates correct condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.not_in(["claude"])

        assert cond.operator == "not_in"
        assert cond.evaluate("gpt-4") is True
        assert cond.evaluate("claude") is False

    def test_choices_with_boolean_values(self) -> None:
        """Test Choices constraint methods with boolean values."""
        use_cache = Choices([True, False], name="use_cache")

        cond_eq = use_cache.equals(True)
        assert cond_eq.evaluate(True) is True
        assert cond_eq.evaluate(False) is False

        cond_in = use_cache.is_in([True])
        assert cond_in.evaluate(True) is True
        assert cond_in.evaluate(False) is False

    def test_choices_with_numeric_values(self) -> None:
        """Test Choices constraint methods with numeric values."""
        temp_choices = Choices([0.0, 0.5, 1.0], name="temperature")

        cond = temp_choices.is_in([0.0, 1.0])
        assert cond.evaluate(0.0) is True
        assert cond.evaluate(1.0) is True
        assert cond.evaluate(0.5) is False


class TestConstraintBuilderChaining:
    """Tests for combining constraint conditions using operators."""

    def test_and_condition(self) -> None:
        """Test chaining conditions with & operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.gte(0.3) & temp.lte(0.7)

        assert cond.evaluate_config({"temperature": 0.5}, {id(temp): "temperature"})
        assert cond.evaluate_config({"temperature": 0.3}, {id(temp): "temperature"})
        assert cond.evaluate_config({"temperature": 0.7}, {id(temp): "temperature"})
        assert not cond.evaluate_config({"temperature": 0.2}, {id(temp): "temperature"})
        assert not cond.evaluate_config({"temperature": 0.8}, {id(temp): "temperature"})

    def test_or_condition(self) -> None:
        """Test chaining conditions with | operator."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.equals("gpt-4") | model.equals("claude")

        assert cond.evaluate_config({"model": "gpt-4"}, {id(model): "model"})
        assert cond.evaluate_config({"model": "claude"}, {id(model): "model"})
        assert not cond.evaluate_config({"model": "gpt-3.5"}, {id(model): "model"})

    def test_not_condition(self) -> None:
        """Test negating conditions with ~ operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = ~temp.equals(0.5)

        assert cond.evaluate_config({"temperature": 0.7}, {id(temp): "temperature"})
        assert not cond.evaluate_config({"temperature": 0.5}, {id(temp): "temperature"})

    def test_implication(self) -> None:
        """Test creating implications with >> operator."""
        from traigent.api.constraints import Constraint

        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = model.equals("gpt-4") >> temp.lte(0.7)

        assert isinstance(constraint, Constraint)
        # When model is gpt-4, temperature must be <= 0.7
        var_names = {id(model): "model", id(temp): "temperature"}
        assert constraint.evaluate({"model": "gpt-4", "temperature": 0.5}, var_names)
        assert constraint.evaluate({"model": "gpt-4", "temperature": 0.7}, var_names)
        assert not constraint.evaluate(
            {"model": "gpt-4", "temperature": 0.9}, var_names
        )
        # When model is not gpt-4, constraint is satisfied
        assert constraint.evaluate({"model": "gpt-3.5", "temperature": 1.5}, var_names)
