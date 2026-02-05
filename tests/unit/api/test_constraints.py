"""Unit tests for the TVL constraint system.

Tests cover:
- Condition creation and evaluation
- Constraint creation and evaluation
- Builder pattern methods on Range/Choices
- Constraint to callable conversion
- ConfigSpace validation
- TVL export functionality
"""

from __future__ import annotations

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.constraints import (
    AndCondition,
    Condition,
    Constraint,
    ConstraintConflict,
    ConstraintScopeError,
    OrCondition,
    check_constraints_conflict,
    constraints_to_callables,
    explain_constraint_violation,
    implies,
    require,
    when,
)
from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range
from traigent.api.validation_protocol import (
    PythonConstraintValidator,
    SatStatus,
)


class TestCondition:
    """Tests for Condition class."""

    def test_condition_equality(self) -> None:
        """Test equality operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="==", value=0.5)

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.7) is False

    def test_condition_not_equal(self) -> None:
        """Test not-equal operator condition."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = Condition(_tvar=model, operator="!=", value="gpt-4")

        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("gpt-4") is False

    def test_condition_greater_than(self) -> None:
        """Test greater-than operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator=">", value=0.5)

        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.3) is False

    def test_condition_greater_equal(self) -> None:
        """Test greater-or-equal operator condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = Condition(_tvar=tokens, operator=">=", value=1000)

        assert cond.evaluate(1000) is True
        assert cond.evaluate(2000) is True
        assert cond.evaluate(500) is False

    def test_condition_less_than(self) -> None:
        """Test less-than operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="<", value=0.7)

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.7) is False
        assert cond.evaluate(0.9) is False

    def test_condition_less_equal(self) -> None:
        """Test less-or-equal operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="<=", value=0.7)

        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.9) is False

    def test_condition_in(self) -> None:
        """Test 'in' operator condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = Condition(_tvar=model, operator="in", value=("gpt-4", "claude"))

        assert cond.evaluate("gpt-4") is True
        assert cond.evaluate("claude") is True
        assert cond.evaluate("gpt-3.5") is False

    def test_condition_not_in(self) -> None:
        """Test 'not_in' operator condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = Condition(_tvar=model, operator="not_in", value=("gpt-4",))

        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("claude") is True
        assert cond.evaluate("gpt-4") is False

    def test_condition_in_range(self) -> None:
        """Test 'in_range' operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="in_range", value=(0.3, 0.7))

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.3) is True
        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.2) is False
        assert cond.evaluate(0.8) is False

    def test_condition_to_expression(self) -> None:
        """Test expression generation."""
        temp = Range(0.0, 2.0, name="temperature")

        cond_eq = Condition(_tvar=temp, operator="==", value=0.5)
        assert cond_eq.to_expression("temperature") == "temperature == 0.5"

        cond_lte = Condition(_tvar=temp, operator="<=", value=0.7)
        assert cond_lte.to_expression("temp") == "temp <= 0.7"

        cond_range = Condition(_tvar=temp, operator="in_range", value=(0.3, 0.7))
        assert cond_range.to_expression("t") == "(t >= 0.3) and (t <= 0.7)"


class TestBuilderMethods:
    """Tests for builder methods on ParameterRange classes."""

    def test_range_equals(self) -> None:
        """Test Range.equals() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.equals(0.5)

        assert isinstance(cond, Condition)
        assert cond.operator == "=="
        assert cond.value == 0.5
        assert cond.tvar is temp

    def test_range_not_equals(self) -> None:
        """Test Range.not_equals() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.not_equals(0.5)

        assert cond.operator == "!="
        assert cond.value == 0.5

    def test_range_gte(self) -> None:
        """Test Range.gte() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.gte(0.3)

        assert cond.operator == ">="
        assert cond.value == 0.3

    def test_range_lte(self) -> None:
        """Test Range.lte() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lte(0.7)

        assert cond.operator == "<="
        assert cond.value == 0.7

    def test_range_gt(self) -> None:
        """Test Range.gt() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.gt(0.5)

        assert cond.operator == ">"
        assert cond.value == 0.5

    def test_range_lt(self) -> None:
        """Test Range.lt() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lt(0.5)

        assert cond.operator == "<"
        assert cond.value == 0.5

    def test_range_in_range(self) -> None:
        """Test Range.in_range() builder method."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.in_range(0.3, 0.7)

        assert cond.operator == "in_range"
        assert cond.value == (0.3, 0.7)

    def test_int_range_builders(self) -> None:
        """Test IntRange builder methods."""
        tokens = IntRange(100, 4096, name="max_tokens")

        assert tokens.gte(1000).operator == ">="
        assert tokens.lte(2000).operator == "<="
        assert tokens.equals(1500).operator == "=="

    def test_log_range_builders(self) -> None:
        """Test LogRange builder methods."""
        lr = LogRange(1e-5, 1e-1, name="learning_rate")

        assert lr.gte(1e-4).operator == ">="
        assert lr.lte(1e-2).operator == "<="

    def test_choices_equals(self) -> None:
        """Test Choices.equals() builder method."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.equals("gpt-4")

        assert cond.operator == "=="
        assert cond.value == "gpt-4"

    def test_choices_not_equals(self) -> None:
        """Test Choices.not_equals() builder method."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = model.not_equals("gpt-4")

        assert cond.operator == "!="
        assert cond.value == "gpt-4"

    def test_choices_is_in(self) -> None:
        """Test Choices.is_in() builder method."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.is_in(["gpt-4", "claude"])

        assert cond.operator == "in"
        assert cond.value == ("gpt-4", "claude")

    def test_choices_not_in(self) -> None:
        """Test Choices.not_in() builder method."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = model.not_in(["gpt-4"])

        assert cond.operator == "not_in"
        assert cond.value == ("gpt-4",)


class TestConstraint:
    """Tests for Constraint class."""

    def test_constraint_implication(self) -> None:
        """Test implication constraint creation and evaluation."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = Constraint(
            when=model.equals("gpt-4"),
            then=temp.lte(0.7),
            description="GPT-4 needs low temp",
        )

        assert constraint.is_implication is True
        assert constraint.description == "GPT-4 needs low temp"

        # Use id() for identity-based lookup (avoids collision with identical values)
        var_names = {id(model): "model", id(temp): "temperature"}

        # When condition true, then must be true
        assert (
            constraint.evaluate({"model": "gpt-4", "temperature": 0.5}, var_names)
            is True
        )
        assert (
            constraint.evaluate({"model": "gpt-4", "temperature": 1.0}, var_names)
            is False
        )

        # When condition false, constraint is satisfied
        assert (
            constraint.evaluate({"model": "gpt-3.5", "temperature": 1.0}, var_names)
            is True
        )

    def test_constraint_standalone(self) -> None:
        """Test standalone expression constraint."""
        temp = Range(0.0, 2.0, name="temperature")

        constraint = Constraint(
            expr=temp.gte(0.1),
            description="Minimum temperature",
        )

        assert constraint.is_implication is False

        # Use id() for identity-based lookup (avoids collision with identical values)
        var_names = {id(temp): "temperature"}

        assert constraint.evaluate({"temperature": 0.5}, var_names) is True
        assert constraint.evaluate({"temperature": 0.05}, var_names) is False

    def test_constraint_validation_errors(self) -> None:
        """Test constraint validation."""
        temp = Range(0.0, 2.0, name="temperature")

        # Must have either implication or expression
        with pytest.raises(ValueError, match="requires either"):
            Constraint()

        # Cannot have both
        with pytest.raises(ValueError, match="cannot have both"):
            Constraint(
                when=temp.lte(0.5),
                then=temp.gte(0.1),
                expr=temp.gte(0.2),
            )


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_implies(self) -> None:
        """Test implies() factory function."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = implies(
            model.equals("gpt-4"),
            temp.lte(0.7),
            description="GPT-4 needs low temp",
            id="c1",
        )

        assert constraint.is_implication is True
        assert constraint.description == "GPT-4 needs low temp"
        assert constraint.id == "c1"

    def test_require(self) -> None:
        """Test require() factory function."""
        temp = Range(0.0, 2.0, name="temperature")

        constraint = require(
            temp.gte(0.1),
            description="Minimum temperature",
            id="c2",
        )

        assert constraint.is_implication is False
        assert constraint.description == "Minimum temperature"
        assert constraint.id == "c2"


class TestConstraintToCallable:
    """Tests for constraint to callable conversion."""

    def test_to_callable_with_names(self) -> None:
        """Test converting constraint to callable using tvar names."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))
        fn = constraint.to_callable()

        assert fn({"model": "gpt-4", "temperature": 0.5}) is True
        assert fn({"model": "gpt-4", "temperature": 1.0}) is False
        assert fn({"model": "gpt-3.5", "temperature": 1.0}) is True

    def test_to_callable_with_var_names(self) -> None:
        """Test converting constraint to callable with explicit var_names."""
        model = Choices(["gpt-4", "gpt-3.5"])
        temp = Range(0.0, 2.0)

        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))
        # Use id() for identity-based lookup (avoids collision with identical values)
        var_names = {id(model): "m", id(temp): "t"}
        fn = constraint.to_callable(var_names)

        assert fn({"m": "gpt-4", "t": 0.5}) is True
        assert fn({"m": "gpt-4", "t": 1.0}) is False

    def test_constraints_to_callables(self) -> None:
        """Test batch conversion of constraints to callables."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")
        tokens = IntRange(100, 4096, name="max_tokens")

        constraints = [
            implies(model.equals("gpt-4"), tokens.gte(1000)),
            implies(model.equals("gpt-3.5"), temp.lte(0.7)),
            require(temp.gte(0.1)),
        ]

        fns = constraints_to_callables(constraints)
        assert len(fns) == 3

        config = {"model": "gpt-4", "temperature": 0.5, "max_tokens": 2000}
        assert all(fn(config) for fn in fns)


class TestConfigSpace:
    """Tests for ConfigSpace class."""

    def test_create_config_space(self) -> None:
        """Test creating a ConfigSpace."""
        temp = Range(0.0, 2.0, name="temperature", unit="ratio")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
            description="Test config space",
        )

        assert len(space.tvars) == 2
        assert len(space.constraints) == 1
        assert space.description == "Test config space"

    def test_validate(self) -> None:
        """Test ConfigSpace validation."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        result = space.validate({"temperature": 0.5, "model": "gpt-4"})
        assert result.is_valid is True

        result = space.validate({"temperature": 1.0, "model": "gpt-4"})
        assert result.is_valid is False
        assert len(result.violations) == 1

    def test_from_decorator_args(self) -> None:
        """Test ConfigSpace.from_decorator_args factory method."""
        space = ConfigSpace.from_decorator_args(
            inline_params={
                "temperature": Range(0.0, 1.0),
                "model": ["gpt-4", "claude"],
            }
        )

        assert len(space.tvars) == 2
        assert "temperature" in space.tvars
        assert "model" in space.tvars
        assert isinstance(space.tvars["model"], Choices)

    def test_to_tvl_spec(self) -> None:
        """Test TVL spec export."""
        temp = Range(0.0, 2.0, name="temperature", unit="ratio")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        spec = space.to_tvl_spec()

        assert "tvars" in spec
        assert len(spec["tvars"]) == 2
        assert "constraints" in spec
        assert "structural" in spec["constraints"]


class TestPythonConstraintValidator:
    """Tests for PythonConstraintValidator."""

    def test_validate_config(self) -> None:
        """Test config validation."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraints = [implies(model.equals("gpt-4"), temp.lte(0.7))]
        # Use id() for identity-based lookup (avoids collision with identical values)
        var_names = {id(model): "model", id(temp): "temperature"}

        validator = PythonConstraintValidator()

        result = validator.validate_config(
            {"model": "gpt-4", "temperature": 0.5},
            constraints,
            var_names,
        )
        assert result.is_valid is True

        result = validator.validate_config(
            {"model": "gpt-4", "temperature": 1.0},
            constraints,
            var_names,
        )
        assert result.is_valid is False
        assert result.violations[0].constraint_index == 0

    def test_check_satisfiability_no_constraints(self) -> None:
        """Test satisfiability with no constraints."""
        validator = PythonConstraintValidator()
        result = validator.check_satisfiability({"temp": Range(0.0, 1.0)}, [])

        assert result.status == SatStatus.SAT

    def test_check_satisfiability_unknown(self) -> None:
        """Test satisfiability returns UNKNOWN for complex constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        constraints = [require(temp.gte(0.1))]

        validator = PythonConstraintValidator()
        result = validator.check_satisfiability({"temperature": temp}, constraints)

        assert result.status == SatStatus.UNKNOWN


class TestCompoundConditions:
    """Tests for future compound condition classes."""

    def test_and_condition_requires_two(self) -> None:
        """Test AndCondition requires at least 2 conditions."""
        temp = Range(0.0, 2.0, name="temperature")

        with pytest.raises(ValueError, match="at least 2"):
            AndCondition(conditions=(temp.lte(0.7),))

    def test_or_condition_requires_two(self) -> None:
        """Test OrCondition requires at least 2 conditions."""
        temp = Range(0.0, 2.0, name="temperature")

        with pytest.raises(ValueError, match="at least 2"):
            OrCondition(conditions=(temp.lte(0.7),))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_param_in_config_returns_false(self) -> None:
        """Test evaluation with missing parameter returns False (fail-closed).

        If a TVAR is properly registered (in var_names) but the config doesn't
        contain the parameter, the constraint should fail-closed (return
        False) rather than silently succeed.
        """
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        # temp is in var_names, but config is missing "temp"
        var_names = {id(temp): "temp"}

        # Should return False (fail-closed) because param missing from config
        result = constraint.evaluate({"other": 0.5}, var_names)
        assert result is False

    def test_constraint_with_none_tvar_name(self) -> None:
        """Test constraint with unnamed tvar uses fallback."""
        temp = Range(0.0, 2.0)  # No name
        constraint = require(temp.gte(0.1))

        # Should still work with explicit var_names (using id())
        var_names = {id(temp): "t"}
        result = constraint.evaluate({"t": 0.5}, var_names)
        assert result is True

    def test_frozen_condition(self) -> None:
        """Test that Condition is immutable."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="<=", value=0.7)

        with pytest.raises(AttributeError):
            cond.value = 0.8  # type: ignore[misc]

    def test_frozen_constraint(self) -> None:
        """Test that Constraint is immutable."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        with pytest.raises(AttributeError):
            constraint.description = "New description"  # type: ignore[misc]

    def test_identical_ranges_no_collision(self) -> None:
        """Test that identical Range objects don't collide in var_names.

        This tests the fix for the ParameterRange dict key collision issue
        where frozen dataclasses with identical values would collide.
        """
        # Two Range objects with identical values but different purposes
        range1 = Range(0.0, 1.0, name="alpha")
        range2 = Range(0.0, 1.0, name="beta")

        space = ConfigSpace(
            tvars={"alpha": range1, "beta": range2},
            constraints=[],
        )

        # Both should be present (no collision)
        assert len(space.var_names) == 2
        assert space.var_names[id(range1)] == "alpha"
        assert space.var_names[id(range2)] == "beta"

    def test_to_callable_warns_on_missing_name(self) -> None:
        """Test that to_callable warns when tvar has no name."""
        import warnings

        temp = Range(0.0, 2.0)  # No name!
        model = Choices(["a", "b"])  # No name!
        constraint = implies(model.equals("a"), temp.lte(0.7))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            constraint.to_callable()
            assert len(w) == 1
            assert "without names" in str(w[0].message)


class TestTupleToRangeConversion:
    """Tests for tuple-to-Range conversion in from_decorator_args."""

    def test_numeric_tuple_becomes_range(self) -> None:
        """Test that (low, high) tuple becomes Range, not Choices."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"temperature": (0.0, 1.0)}
        )

        assert "temperature" in space.tvars
        assert isinstance(space.tvars["temperature"], Range)
        assert space.tvars["temperature"].low == 0.0
        assert space.tvars["temperature"].high == 1.0

    def test_int_tuple_becomes_int_range(self) -> None:
        """Test that (int, int) tuple becomes IntRange."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"max_tokens": (100, 4096)}
        )

        assert "max_tokens" in space.tvars
        assert isinstance(space.tvars["max_tokens"], IntRange)
        assert space.tvars["max_tokens"].low == 100
        assert space.tvars["max_tokens"].high == 4096

    def test_list_becomes_choices(self) -> None:
        """Test that list becomes Choices."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"model": ["gpt-4", "gpt-3.5"]}
        )

        assert "model" in space.tvars
        assert isinstance(space.tvars["model"], Choices)

    def test_string_tuple_becomes_choices(self) -> None:
        """Test that tuple of strings becomes Choices, not Range."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"model": ("gpt-4", "gpt-3.5")}
        )

        assert "model" in space.tvars
        assert isinstance(space.tvars["model"], Choices)


class TestNormalizeConstraints:
    """Tests for normalize_constraints() function."""

    def test_empty_list(self) -> None:
        """Test normalizing empty constraint list."""
        from traigent.api.constraints import normalize_constraints

        result = normalize_constraints([])
        assert result == []

    def test_none_input(self) -> None:
        """Test normalizing None input."""
        from traigent.api.constraints import normalize_constraints

        result = normalize_constraints(None)
        assert result == []

    def test_pure_constraint_list(self) -> None:
        """Test normalizing list of Constraint objects."""
        from traigent.api.constraints import normalize_constraints

        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraints = [
            implies(model.equals("gpt-4"), temp.lte(0.7)),
            require(temp.gte(0.1)),
        ]

        result = normalize_constraints(constraints)

        assert len(result) == 2
        assert all(callable(fn) for fn in result)

        # Test the callables work
        config = {"model": "gpt-4", "temperature": 0.5}
        assert all(fn(config) for fn in result)

    def test_pure_callable_list(self) -> None:
        """Test normalizing list of pure callables (backward compat)."""
        from traigent.api.constraints import normalize_constraints

        constraints = [
            lambda cfg: cfg["temperature"] < 1.0,
            lambda cfg: cfg["max_tokens"] > 100,
        ]

        result = normalize_constraints(constraints)

        assert len(result) == 2
        assert all(callable(fn) for fn in result)

        # Callables should pass through unchanged
        config = {"temperature": 0.5, "max_tokens": 500}
        assert all(fn(config) for fn in result)

    def test_mixed_constraints_and_callables(self) -> None:
        """Test normalizing mixed list of Constraints and callables."""
        from traigent.api.constraints import normalize_constraints

        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraints = [
            implies(model.equals("gpt-4"), temp.lte(0.7)),  # Constraint
            lambda cfg: cfg["max_tokens"] < 4096,  # Callable
            require(temp.gte(0.1)),  # Constraint
        ]

        result = normalize_constraints(constraints)

        assert len(result) == 3
        assert all(callable(fn) for fn in result)

        # Test all work together
        config = {"model": "gpt-4", "temperature": 0.5, "max_tokens": 2000}
        assert all(fn(config) for fn in result)

    def test_invalid_constraint_type(self) -> None:
        """Test that invalid types raise TypeError."""
        from traigent.api.constraints import normalize_constraints

        with pytest.raises(TypeError, match="constraints\\[1\\]"):
            normalize_constraints([lambda cfg: True, "not a constraint"])  # type: ignore[list-item]

    def test_with_explicit_var_names(self) -> None:
        """Test normalizing with explicit var_names mapping."""
        from traigent.api.constraints import normalize_constraints

        model = Choices(["gpt-4", "gpt-3.5"])  # No name
        temp = Range(0.0, 2.0)  # No name

        constraints = [implies(model.equals("gpt-4"), temp.lte(0.7))]
        var_names = {id(model): "m", id(temp): "t"}

        result = normalize_constraints(constraints, var_names)

        assert len(result) == 1
        assert result[0]({"m": "gpt-4", "t": 0.5}) is True
        assert result[0]({"m": "gpt-4", "t": 1.0}) is False


class TestTVLExportFormat:
    """Tests for TVL 0.9 export format compliance."""

    def test_domain_has_kind_field(self) -> None:
        """Test that exported domains have explicit 'kind' field."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})
        spec = space.to_tvl_spec()

        for tvar in spec["tvars"]:
            assert "domain" in tvar
            assert "kind" in tvar["domain"]

    def test_range_domain_format(self) -> None:
        """Test range domain has 'range' array."""
        temp = Range(0.0, 2.0, name="temperature")
        space = ConfigSpace(tvars={"temperature": temp})
        spec = space.to_tvl_spec()

        domain = spec["tvars"][0]["domain"]
        assert domain["kind"] == "range"
        assert "range" in domain
        assert domain["range"] == [0.0, 2.0]

    def test_enum_domain_format(self) -> None:
        """Test enum domain has 'values' array."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        space = ConfigSpace(tvars={"model": model})
        spec = space.to_tvl_spec()

        domain = spec["tvars"][0]["domain"]
        assert domain["kind"] == "enum"
        assert "values" in domain
        assert domain["values"] == ["gpt-4", "gpt-3.5"]

    def test_constraint_params_prefix(self) -> None:
        """Test constraint expressions use params.<name> prefix."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )
        spec = space.to_tvl_spec()

        constraint = spec["constraints"]["structural"][0]
        assert "params.model" in constraint["when"]
        assert "params.temperature" in constraint["then"]

    def test_constraint_metadata_export(self) -> None:
        """Test constraint metadata (id, description, error_message) is exported."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"temperature": temp},
            constraints=[require(temp.gte(0.1), description="Min temp", id="c1")],
        )
        spec = space.to_tvl_spec()

        constraint = spec["constraints"]["structural"][0]
        assert constraint["id"] == "c1"
        assert constraint["description"] == "Min temp"
        assert constraint["error_message"] == "Min temp"
        assert "index" in constraint


class TestDecoratorIntegration:
    """Tests for decorator integration with Constraint and ConfigSpace objects."""

    def test_decorator_accepts_constraint_objects(self) -> None:
        """Test that @optimize accepts Constraint objects in constraints list."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        @optimize(
            configuration_space={"model": model, "temperature": temp},
            constraints=[
                implies(model.equals("gpt-4"), temp.lte(0.7)),
            ],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_decorator_accepts_mixed_constraints(self) -> None:
        """Test that @optimize accepts mixed Constraint and callable list."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        @optimize(
            configuration_space={"model": model, "temperature": temp},
            constraints=[
                implies(model.equals("gpt-4"), temp.lte(0.7)),  # Constraint
                lambda cfg: cfg["temperature"] >= 0.1,  # Callable
            ],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_decorator_accepts_config_space_object(self) -> None:
        """Test that @optimize accepts ConfigSpace object directly."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        @optimize(configuration_space=space)
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_decorator_config_space_with_no_constraints(self) -> None:
        """Test ConfigSpace without constraints can have external constraints."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})

        @optimize(
            configuration_space=space,
            constraints=[lambda cfg: cfg["temperature"] < 1.5],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_decorator_config_space_conflict_raises(self) -> None:
        """Test ConfigSpace with constraints AND explicit constraints raises error."""
        from traigent.api.decorators import optimize

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        with pytest.raises(TypeError, match="Cannot provide both"):

            @optimize(
                configuration_space=space,
                constraints=[lambda cfg: True],  # This should conflict
            )
            def test_func(text: str) -> str:
                return f"Response: {text}"

    def test_backward_compatibility_lambda_constraints(self) -> None:
        """Test that legacy lambda constraints still work."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        @optimize(
            configuration_space={
                "model": ["gpt-4", "gpt-3.5"],
                "temperature": (0.0, 2.0),
            },
            constraints=[
                lambda cfg: (
                    cfg["temperature"] < 1.0 if cfg["model"] == "gpt-4" else True
                ),
            ],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_backward_compatibility_simple_config_space(self) -> None:
        """Test that legacy simple config space syntax still works."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        @optimize(
            configuration_space={
                "model": ["gpt-4", "gpt-3.5"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
                "max_tokens": (100, 4096),
            }
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)

    def test_constraint_uses_config_key_not_tvar_name(self) -> None:
        """Test that constraints use config key from decorator, not tvar.name.

        This tests the fix for the var_names mapping issue where constraints
        would silently pass if tvar.name didn't match the config key.
        """
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        # Intentionally use different name vs config key
        temp = Range(0.0, 2.0, name="wrong_name")  # name doesn't match key
        model = Choices(["gpt-4", "gpt-3.5"], name="also_wrong")

        # The constraint uses the ParameterRange objects
        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))

        @optimize(
            configuration_space={"temperature": temp, "model": model},
            constraints=[constraint],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)
        # The key test: constraints should work even though tvar.name != config key
        # because var_names mapping is built from the config_space dict

    def test_unnamed_ranges_work_with_config_key(self) -> None:
        """Test that constraints with unnamed ranges work via config key mapping."""
        from traigent.api.decorators import optimize
        from traigent.core.optimized_function import OptimizedFunction

        # No name attribute set at all
        temp = Range(0.0, 2.0)
        model = Choices(["gpt-4", "gpt-3.5"])

        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))

        @optimize(
            configuration_space={"temperature": temp, "model": model},
            constraints=[constraint],
        )
        def test_func(text: str) -> str:
            return f"Response: {text}"

        assert isinstance(test_func, OptimizedFunction)


class TestConditionTvarProperty:
    """Tests for Condition._tvar field and tvar property."""

    def test_tvar_property_returns_parameter_range(self) -> None:
        """Test that tvar property returns the underlying ParameterRange."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lte(0.7)

        assert cond.tvar is temp
        assert cond.tvar.name == "temperature"

    def test_tvar_property_is_readonly(self) -> None:
        """Test that tvar property cannot be modified (frozen dataclass)."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = temp.lte(0.7)

        with pytest.raises(AttributeError):
            cond._tvar = Range(0.0, 1.0)  # type: ignore[misc]


class TestBoolExprImpliesMethod:
    """Tests for BoolExpr.implies() fluent method."""

    def test_implies_method_creates_constraint(self) -> None:
        """Test that implies() method creates a Constraint."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = model.equals("gpt-4").implies(temp.lte(0.7))

        assert isinstance(constraint, Constraint)
        assert constraint.is_implication is True

    def test_implies_method_works_with_and_condition(self) -> None:
        """Test implies() on AndCondition."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")
        tokens = IntRange(100, 4096, name="max_tokens")

        and_cond = model.equals("gpt-4") & temp.lte(0.7)
        constraint = and_cond.implies(tokens.gte(1000))

        assert isinstance(constraint, Constraint)
        assert constraint.when is and_cond


class TestCompoundConditionToExpression:
    """Tests for compound condition to_expression with string var_names."""

    def test_and_condition_to_expression_with_string(self) -> None:
        """Test AndCondition.to_expression with string parameter."""
        temp = Range(0.0, 2.0, name="temperature")
        and_cond = temp.lte(0.7) & temp.gte(0.1)

        # Note: with string var_names, it uses the same name for all conditions
        expr = and_cond.to_expression("params.temperature")
        assert "and" in expr
        assert "params.temperature" in expr

    def test_or_condition_to_expression_with_string(self) -> None:
        """Test OrCondition.to_expression with string parameter."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        or_cond = model.equals("gpt-4") | model.equals("gpt-3.5")

        expr = or_cond.to_expression("params.model")
        assert "or" in expr
        assert "params.model" in expr

    def test_not_condition_to_expression_with_string(self) -> None:
        """Test NotCondition.to_expression with string parameter."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        not_cond = ~model.equals("gpt-4")

        expr = not_cond.to_expression("params.model")
        assert "not" in expr
        assert "params.model" in expr


class TestNotConditionEvaluation:
    """Tests for NotCondition evaluation."""

    def test_not_condition_evaluate_true(self) -> None:
        """Test NotCondition returns True when inner condition is False."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        not_cond = ~model.equals("gpt-4")
        var_names = {id(model): "model"}

        # Inner condition is False (model is not gpt-4), so NOT returns True
        result = not_cond.evaluate_config({"model": "gpt-3.5"}, var_names)
        assert result is True

    def test_not_condition_evaluate_false(self) -> None:
        """Test NotCondition returns False when inner condition is True."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        not_cond = ~model.equals("gpt-4")
        var_names = {id(model): "model"}

        # Inner condition is True (model is gpt-4), so NOT returns False
        result = not_cond.evaluate_config({"model": "gpt-4"}, var_names)
        assert result is False

    def test_not_condition_explain(self) -> None:
        """Test NotCondition.explain() returns readable text."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        not_cond = ~model.equals("gpt-4")

        explanation = not_cond.explain()
        assert "NOT" in explanation
        assert "model" in explanation


class TestWhenBuilderFluent:
    """Tests for when().then() fluent builder syntax."""

    def test_when_then_creates_implication(self) -> None:
        """Test when().then() creates an implication constraint."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = when(model.equals("gpt-4")).then(temp.lte(0.7))

        assert isinstance(constraint, Constraint)
        assert constraint.is_implication is True

    def test_when_then_with_description(self) -> None:
        """Test when().then() with description and id."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = when(model.equals("gpt-4")).then(
            temp.lte(0.7),
            description="GPT-4 needs low temp",
            id="c1",
        )

        assert constraint.description == "GPT-4 needs low temp"
        assert constraint.id == "c1"


class TestConstraintExplain:
    """Tests for Constraint.explain() method."""

    def test_implication_explain(self) -> None:
        """Test explain() for implication constraint."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))
        explanation = constraint.explain()

        assert "IF" in explanation
        assert "THEN" in explanation
        assert "model" in explanation
        assert "temperature" in explanation

    def test_standalone_explain(self) -> None:
        """Test explain() for standalone constraint."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        explanation = constraint.explain()
        assert "REQUIRE" in explanation
        assert "temperature" in explanation


class TestCheckConstraintsConflict:
    """Tests for check_constraints_conflict() function."""

    def test_no_conflict_with_compatible_constraints(self) -> None:
        """Test that compatible constraints return no conflict."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.lte(1.5))
        c2 = require(temp.gte(0.1))

        # Sample configs that satisfy both
        result = check_constraints_conflict(
            [c1, c2],
            sample_configs=[{"temperature": 0.5}, {"temperature": 1.0}],
        )
        assert result is None

    def test_no_conflict_with_empty_constraints(self) -> None:
        """Test that empty constraints return no conflict."""
        result = check_constraints_conflict([])
        assert result is None

    def test_no_conflict_with_no_samples(self) -> None:
        """Test that missing samples return no conflict."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.lte(0.5))

        result = check_constraints_conflict([c1], sample_configs=None)
        assert result is None

    def test_conflict_with_exhaustive_samples(self) -> None:
        """Test that exhaustive samples report conflict for mutually exclusive constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.lte(0.5))
        c2 = require(temp.gte(0.8))

        # These constraints are mutually exclusive
        result = check_constraints_conflict(
            [c1, c2],
            sample_configs=[{"temperature": 0.3}, {"temperature": 0.9}],
            samples_exhaustive=True,
        )
        assert result is not None
        assert isinstance(result, ConstraintConflict)
        assert len(result.constraints) == 2

    def test_no_conflict_with_sparse_samples_default(self) -> None:
        """Test that sparse samples don't report conflict by default (issue #38)."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.lte(0.5))
        c2 = require(temp.gte(0.8))

        # Same mutually exclusive constraints, but samples_exhaustive=False (default)
        # should return None to avoid false positives
        result = check_constraints_conflict(
            [c1, c2],
            sample_configs=[{"temperature": 0.3}, {"temperature": 0.9}],
            # samples_exhaustive defaults to False
        )
        assert result is None

    def test_no_conflict_sparse_samples_explicit(self) -> None:
        """Test explicit samples_exhaustive=False returns None even with all failures."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.gte(0.3))
        c2 = require(temp.lte(0.7))

        # Samples outside valid range [0.3, 0.7]
        result = check_constraints_conflict(
            [c1, c2],
            sample_configs=[{"temperature": 0.2}, {"temperature": 0.8}],
            samples_exhaustive=False,
        )
        # Should return None since samples are not exhaustive
        # (valid configs like 0.5 exist but weren't sampled)
        assert result is None


class TestExplainConstraintViolation:
    """Tests for explain_constraint_violation() function."""

    def test_violation_returns_explanation(self) -> None:
        """Test that violated constraint returns explanation."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.lte(0.5))

        explanation = explain_constraint_violation(constraint, {"temperature": 0.9})

        assert explanation is not None
        assert "violated" in explanation.lower()
        assert "temperature" in explanation

    def test_satisfied_returns_none(self) -> None:
        """Test that satisfied constraint returns None."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.lte(0.5))

        result = explain_constraint_violation(constraint, {"temperature": 0.3})
        assert result is None

    def test_violation_with_var_names(self) -> None:
        """Test violation explanation with explicit var_names."""
        temp = Range(0.0, 2.0)  # No name
        constraint = require(temp.lte(0.5))
        var_names = {id(temp): "temp"}

        explanation = explain_constraint_violation(constraint, {"temp": 0.9}, var_names)

        assert explanation is not None
        assert "temp" in explanation


class TestBoolExprTvarProperty:
    """Tests for BoolExpr.tvar property on base and compound expressions."""

    def test_base_boolexpr_tvar_is_none(self) -> None:
        """Test that compound expressions return None for tvar."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        and_cond = temp.lte(0.7) & model.equals("gpt-4")
        assert and_cond.tvar is None

        or_cond = temp.lte(0.7) | model.equals("gpt-4")
        assert or_cond.tvar is None

        not_cond = ~temp.lte(0.7)
        assert not_cond.tvar is None


class TestConditionEvaluateOperators:
    """Tests for Condition.evaluate() with all operators."""

    def test_evaluate_greater_than(self) -> None:
        """Test > operator evaluation."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator=">", value=0.5)

        assert cond.evaluate(0.6) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.4) is False

    def test_evaluate_less_than(self) -> None:
        """Test < operator evaluation."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="<", value=0.5)

        assert cond.evaluate(0.4) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.6) is False

    def test_evaluate_in_range(self) -> None:
        """Test in_range operator evaluation."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="in_range", value=(0.3, 0.7))

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.3) is True
        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.2) is False
        assert cond.evaluate(0.8) is False

    def test_evaluate_unknown_operator_raises(self) -> None:
        """Test that unknown operator raises ValueError."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="??", value=0.5)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unknown operator"):
            cond.evaluate(0.5)


class TestConditionExplainOperators:
    """Tests for Condition.explain() with various operators."""

    def test_explain_in_operator(self) -> None:
        """Test explain for 'in' operator."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = model.is_in(["gpt-4"])

        explanation = cond.explain()
        assert "is one of" in explanation

    def test_explain_not_in_operator(self) -> None:
        """Test explain for 'not_in' operator."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = model.not_in(["gpt-4"])

        explanation = cond.explain()
        assert "is not one of" in explanation

    def test_explain_greater_than(self) -> None:
        """Test explain for '>' operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator=">", value=0.5)

        explanation = cond.explain()
        assert "greater than" in explanation

    def test_explain_less_than(self) -> None:
        """Test explain for '<' operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="<", value=0.5)

        explanation = cond.explain()
        assert "less than" in explanation

    def test_explain_in_range(self) -> None:
        """Test explain for 'in_range' operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="in_range", value=(0.3, 0.7))

        explanation = cond.explain()
        assert "between" in explanation


class TestConditionToExpressionFormats:
    """Tests for Condition.to_expression() with different operators."""

    def test_to_expression_in_range(self) -> None:
        """Test to_expression for 'in_range' operator."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(_tvar=temp, operator="in_range", value=(0.3, 0.7))
        var_names = {id(temp): "params.temperature"}

        expr = cond.to_expression(var_names)
        assert ">=" in expr
        assert "<=" in expr
        assert "0.3" in expr
        assert "0.7" in expr

    def test_to_expression_in_operator(self) -> None:
        """Test to_expression for 'in' operator."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = model.is_in(["gpt-4"])
        var_names = {id(model): "params.model"}

        expr = cond.to_expression(var_names)
        assert " in " in expr

    def test_to_expression_not_in_operator(self) -> None:
        """Test to_expression for 'not_in' operator."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = model.not_in(["gpt-4"])
        var_names = {id(model): "params.model"}

        expr = cond.to_expression(var_names)
        assert "not in" in expr


class TestConstraintToCallableMetadata:
    """Tests for Constraint.to_callable() metadata on returned function."""

    def test_callable_has_docstring(self) -> None:
        """Test that callable has __doc__ from constraint description."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1), description="Min temp required")

        fn = constraint.to_callable()
        assert fn.__doc__ == "Min temp required"

    def test_callable_has_name_with_id(self) -> None:
        """Test that callable has __name__ from constraint id."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1), id="min_temp")

        fn = constraint.to_callable()
        assert fn.__name__ == "constraint_min_temp"

    def test_callable_unnamed_constraint(self) -> None:
        """Test callable name for unnamed constraint."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        fn = constraint.to_callable()
        assert fn.__name__ == "constraint_unnamed"


class TestConstraintConflictStr:
    """Tests for ConstraintConflict.__str__() representation."""

    def test_conflict_str_format(self) -> None:
        """Test ConstraintConflict string representation."""
        temp = Range(0.0, 2.0, name="temperature")
        c1 = require(temp.lte(0.5))
        c2 = require(temp.gte(0.8))

        conflict = ConstraintConflict(
            constraints=[c1, c2],
            config={"temperature": 0.6},
            messages=["temp too high", "temp too low"],
        )

        str_repr = str(conflict)
        assert "conflict" in str_repr.lower()
        assert "temperature" in str_repr


# =============================================================================
# Constraint Scope Error Tests (Issue #82)
# =============================================================================


class TestConstraintScopeError:
    """Tests for ConstraintScopeError - detecting out-of-scope TVARs.

    Issue #82: Constraints referencing TVARs not in @traigent.optimize scope
    should raise an error, not be silently ignored.

    These tests verify that:
    1. Out-of-scope TVARs raise ConstraintScopeError at evaluation time
    2. Early validation catches scope errors in to_callable()
    3. Error messages include helpful hints for fixing the issue
    """

    def test_evaluate_config_fails_closed_on_out_of_scope_tvar(self) -> None:
        """Test that evaluate_config returns False for out-of-scope TVAR.

        When a TVAR is not in var_names and its name is also not in
        var_names values, the constraint fails closed (returns False).
        """
        # Define two TVARs
        budget = Range(1.0, 100.0, name="budget")
        model = Choices(["a", "b", "c"], name="model")

        # Create a condition using budget
        condition = budget.lte(10)

        # var_names only includes model, not budget
        var_names = {id(model): "model"}
        config = {"model": "a"}

        # Should return False (fail-closed) because budget is not in scope
        result = condition.evaluate_config(config, var_names)
        assert result is False

    def test_evaluate_config_fails_closed_on_missing_param_in_config(self) -> None:
        """Test that evaluate_config returns False when param missing from config.

        Fail-closed: missing parameter does not satisfy the constraint.
        """
        temp = Range(0.0, 2.0, name="temperature")
        condition = temp.lte(0.7)

        # temp is in var_names, but the config is missing "temperature"
        var_names = {id(temp): "temperature"}
        config = {"model": "gpt-4"}  # Missing "temperature"

        result = condition.evaluate_config(config, var_names)
        assert result is False

    def test_to_callable_validates_scope_early(self) -> None:
        """Test that to_callable() validates scope when var_names provided."""

        # Define TVARs
        budget = Range(1.0, 100.0, name="budget")
        model = Choices(["a", "b", "c"], name="model")

        # Create constraint using both TVARs
        constraint = when(budget.lte(10)).then(model.is_in(["a"]))

        # var_names only includes model, not budget
        var_names = {id(model): "model"}

        # to_callable should raise ConstraintScopeError during setup
        with pytest.raises(ConstraintScopeError) as exc_info:
            constraint.to_callable(var_names)

        error_msg = str(exc_info.value)
        assert "budget" in error_msg.lower()

    def test_to_callable_no_error_when_all_in_scope(self) -> None:
        """Test that to_callable() succeeds when all TVARs are in scope."""
        model = Choices(["a", "b", "c"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = when(model.equals("a")).then(temp.lte(0.7))

        # var_names includes both TVARs
        var_names = {id(model): "model", id(temp): "temperature"}

        # Should not raise - all TVARs in scope
        fn = constraint.to_callable(var_names)
        assert callable(fn)

        # And it should work correctly
        assert fn({"model": "a", "temperature": 0.5}) is True
        assert fn({"model": "a", "temperature": 1.0}) is False
        assert fn({"model": "b", "temperature": 1.0}) is True

    def test_out_of_scope_tvar_returns_false_via_evaluate_config(self) -> None:
        """Test that evaluate_config returns False for out-of-scope TVAR.

        evaluate_config uses fail-closed semantics: if the TVAR is not
        in var_names (by id or name), it returns False.
        """
        # Out of scope TVAR
        out_of_scope = Range(1.0, 100.0, name="out_of_scope_param")

        # In scope TVARs
        model = Choices(["a", "b"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        condition = out_of_scope.lte(10)
        var_names = {id(model): "model", id(temp): "temperature"}

        # Should return False (fail-closed) because out_of_scope_param
        # is not in var_names (neither by id nor by name)
        result = condition.evaluate_config(
            {"model": "a", "temperature": 0.5}, var_names
        )
        assert result is False

    def test_scope_error_with_compound_conditions(self) -> None:
        """Test scope validation with compound conditions (And, Or, Not)."""

        in_scope = Choices(["a", "b"], name="in_scope")
        out_of_scope = Range(0.0, 1.0, name="out_of_scope")

        # Compound condition with out-of-scope TVAR
        compound = in_scope.equals("a") & out_of_scope.lte(0.5)
        constraint = require(compound)

        var_names = {id(in_scope): "in_scope"}

        with pytest.raises(ConstraintScopeError):
            constraint.to_callable(var_names)

    def test_config_space_integration(self) -> None:
        """Test that ConfigSpace properly validates constraint scope."""

        # TVAR in config space
        model = Choices(["a", "b", "c"], name="model")

        # TVAR NOT in config space
        budget = Range(1.0, 100.0, name="budget")

        # Create ConfigSpace with only model
        space = ConfigSpace(
            tvars={"model": model},
            constraints=[],  # No constraints yet
        )

        # Constraint references budget which is not in space
        constraint = when(budget.lte(10)).then(model.is_in(["a"]))

        # Should raise when trying to convert to callable with space's var_names
        with pytest.raises(ConstraintScopeError) as exc_info:
            constraint.to_callable(space.var_names)

        error_msg = str(exc_info.value)
        assert "budget" in error_msg.lower()
        assert "model" in error_msg  # Available TVAR
