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
    OrCondition,
    constraints_to_callables,
    implies,
    require,
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
        cond = Condition(tvar=temp, operator="==", value=0.5)

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.7) is False

    def test_condition_not_equal(self) -> None:
        """Test not-equal operator condition."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        cond = Condition(tvar=model, operator="!=", value="gpt-4")

        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("gpt-4") is False

    def test_condition_greater_than(self) -> None:
        """Test greater-than operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(tvar=temp, operator=">", value=0.5)

        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.5) is False
        assert cond.evaluate(0.3) is False

    def test_condition_greater_equal(self) -> None:
        """Test greater-or-equal operator condition."""
        tokens = IntRange(100, 4096, name="max_tokens")
        cond = Condition(tvar=tokens, operator=">=", value=1000)

        assert cond.evaluate(1000) is True
        assert cond.evaluate(2000) is True
        assert cond.evaluate(500) is False

    def test_condition_less_than(self) -> None:
        """Test less-than operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(tvar=temp, operator="<", value=0.7)

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.7) is False
        assert cond.evaluate(0.9) is False

    def test_condition_less_equal(self) -> None:
        """Test less-or-equal operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(tvar=temp, operator="<=", value=0.7)

        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.9) is False

    def test_condition_in(self) -> None:
        """Test 'in' operator condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = Condition(tvar=model, operator="in", value=("gpt-4", "claude"))

        assert cond.evaluate("gpt-4") is True
        assert cond.evaluate("claude") is True
        assert cond.evaluate("gpt-3.5") is False

    def test_condition_not_in(self) -> None:
        """Test 'not_in' operator condition."""
        model = Choices(["gpt-4", "gpt-3.5", "claude"], name="model")
        cond = Condition(tvar=model, operator="not_in", value=("gpt-4",))

        assert cond.evaluate("gpt-3.5") is True
        assert cond.evaluate("claude") is True
        assert cond.evaluate("gpt-4") is False

    def test_condition_in_range(self) -> None:
        """Test 'in_range' operator condition."""
        temp = Range(0.0, 2.0, name="temperature")
        cond = Condition(tvar=temp, operator="in_range", value=(0.3, 0.7))

        assert cond.evaluate(0.5) is True
        assert cond.evaluate(0.3) is True
        assert cond.evaluate(0.7) is True
        assert cond.evaluate(0.2) is False
        assert cond.evaluate(0.8) is False

    def test_condition_to_expression(self) -> None:
        """Test expression generation."""
        temp = Range(0.0, 2.0, name="temperature")

        cond_eq = Condition(tvar=temp, operator="==", value=0.5)
        assert cond_eq.to_expression("temperature") == "temperature == 0.5"

        cond_lte = Condition(tvar=temp, operator="<=", value=0.7)
        assert cond_lte.to_expression("temp") == "temp <= 0.7"

        cond_range = Condition(tvar=temp, operator="in_range", value=(0.3, 0.7))
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

    def test_missing_var_name(self) -> None:
        """Test evaluation with missing variable returns True (doesn't apply)."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        # Missing variable - constraint doesn't apply
        # Use id() for identity-based lookup
        var_names = {id(temp): "temp"}
        result = constraint.evaluate({"other": 0.5}, var_names)
        assert result is True

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
        cond = Condition(tvar=temp, operator="<=", value=0.7)

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
