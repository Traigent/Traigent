"""Unit tests for the ConfigSpace module.

Tests cover:
- ConfigSpace creation and validation
- from_decorator_args factory method
- _sequence_to_range and _dict_to_range helpers
- TVL export methods
- var_names property and get_var_name method
- validate and check_satisfiability methods
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.constraints import implies, require
from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range
from traigent.api.validation_protocol import SatStatus


class TestConfigSpaceCreation:
    """Tests for ConfigSpace creation and validation."""

    def test_create_with_tvars(self) -> None:
        """Test creating ConfigSpace with tvars."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})

        assert len(space.tvars) == 2
        assert "temperature" in space.tvars
        assert "model" in space.tvars

    def test_create_with_constraints(self) -> None:
        """Test creating ConfigSpace with constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        assert len(space.constraints) == 1

    def test_create_with_description(self) -> None:
        """Test creating ConfigSpace with description."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"temperature": temp},
            description="Test config space",
        )

        assert space.description == "Test config space"

    def test_tvars_and_constraints_are_stored_immutably(self) -> None:
        """ConfigSpace should snapshot mutable constructor inputs."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        constraints = [implies(model.equals("gpt-4"), temp.lte(0.7))]
        tvars = {"temperature": temp}

        space = ConfigSpace(tvars=tvars, constraints=constraints)

        assert isinstance(space.tvars, MappingProxyType)
        assert isinstance(space.constraints, tuple)

        tvars["model"] = model
        constraints.append(require(temp.gte(0.1)))

        assert list(space.tvars.keys()) == ["temperature"]
        assert len(space.constraints) == 1

    def test_config_space_is_frozen(self) -> None:
        """Direct mutation should be rejected."""
        temp = Range(0.0, 2.0, name="temperature")
        space = ConfigSpace(tvars={"temperature": temp})

        with pytest.raises(TypeError):
            space.tvars["model"] = Choices(["gpt-4"], name="model")
        with pytest.raises(AttributeError):
            space.constraints.append(require(temp.gte(0.1)))  # type: ignore[attr-defined]
        with pytest.raises(FrozenInstanceError):
            space.description = "mutated"

    def test_repr(self) -> None:
        """Test ConfigSpace string representation."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[require(temp.gte(0.1))],
        )

        repr_str = repr(space)
        assert "ConfigSpace" in repr_str
        assert "constraints=1" in repr_str


class TestConfigSpaceFromDecoratorArgs:
    """Tests for ConfigSpace.from_decorator_args factory method."""

    def test_from_decorator_args_with_parameter_ranges(self) -> None:
        """Test creating from ParameterRange objects."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={
                "temperature": Range(0.0, 2.0),
                "model": Choices(["gpt-4", "gpt-3.5"]),
            }
        )

        assert len(space.tvars) == 2
        assert isinstance(space.tvars["temperature"], Range)
        assert isinstance(space.tvars["model"], Choices)

    def test_from_decorator_args_with_lists(self) -> None:
        """Test creating from list values (become Choices)."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"model": ["gpt-4", "gpt-3.5", "claude"]}
        )

        assert "model" in space.tvars
        assert isinstance(space.tvars["model"], Choices)

    def test_from_decorator_args_with_numeric_tuple(self) -> None:
        """Test creating from numeric tuple (becomes Range)."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"temperature": (0.0, 1.0)}
        )

        assert "temperature" in space.tvars
        assert isinstance(space.tvars["temperature"], Range)

    def test_from_decorator_args_with_int_tuple(self) -> None:
        """Test creating from int tuple (becomes IntRange)."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"max_tokens": (100, 4096)}
        )

        assert "max_tokens" in space.tvars
        assert isinstance(space.tvars["max_tokens"], IntRange)

    def test_from_decorator_args_with_inline_params(self) -> None:
        """Test creating from inline_params."""
        space = ConfigSpace.from_decorator_args(
            inline_params={
                "temperature": Range(0.0, 2.0),
                "model": ["gpt-4", "gpt-3.5"],
            }
        )

        assert len(space.tvars) == 2

    def test_from_decorator_args_inline_overrides_config_space(self) -> None:
        """Test that inline_params override configuration_space."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={"temperature": Range(0.0, 1.0)},
            inline_params={"temperature": Range(0.0, 2.0)},  # Override
        )

        assert space.tvars["temperature"].high == 2.0

    def test_from_decorator_args_with_constraints(self) -> None:
        """Test creating with constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        constraint = require(temp.gte(0.1))

        space = ConfigSpace.from_decorator_args(
            configuration_space={"temperature": temp},
            constraints=[constraint],
        )

        assert len(space.constraints) == 1

    def test_from_decorator_args_snapshots_mapping_and_constraints(self) -> None:
        """Factory inputs should be copied into immutable ConfigSpace storage."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        constraint = require(temp.gte(0.1))
        raw_configuration_space = {"temperature": temp}
        raw_inline_params = {"model": model}
        constraints = [constraint]

        space = ConfigSpace.from_decorator_args(
            configuration_space=MappingProxyType(raw_configuration_space),
            inline_params=MappingProxyType(raw_inline_params),
            constraints=constraints,
        )

        assert isinstance(space.tvars, MappingProxyType)
        assert isinstance(space.constraints, tuple)

        raw_configuration_space["max_tokens"] = IntRange(100, 4096, name="max_tokens")
        raw_inline_params["model"] = Choices(["gpt-4o-mini"], name="model")
        constraints.append(require(model.equals("gpt-4")))

        assert list(space.tvars.keys()) == ["temperature", "model"]
        assert tuple(space.tvars["model"].values) == ("gpt-4", "gpt-3.5")
        assert space.constraints == (constraint,)

    def test_from_decorator_args_skips_non_range_values(self) -> None:
        """Test that non-range values are skipped."""
        space = ConfigSpace.from_decorator_args(
            configuration_space={
                "temperature": Range(0.0, 2.0),
                "non_range": "just a string",  # Should be skipped
                "some_int": 42,  # Should be skipped
            }
        )

        assert len(space.tvars) == 1
        assert "temperature" in space.tvars


class TestSequenceToRange:
    """Tests for ConfigSpace._sequence_to_range."""

    def test_tuple_of_floats_becomes_range(self) -> None:
        """Test that (float, float) becomes Range."""
        result = ConfigSpace._sequence_to_range("temp", (0.1, 0.9))

        assert isinstance(result, Range)
        assert result.low == 0.1
        assert result.high == 0.9

    def test_tuple_of_ints_becomes_int_range(self) -> None:
        """Test that (int, int) becomes IntRange."""
        result = ConfigSpace._sequence_to_range("tokens", (100, 4096))

        assert isinstance(result, IntRange)
        assert result.low == 100
        assert result.high == 4096

    def test_tuple_of_mixed_numeric_becomes_range(self) -> None:
        """Test that (int, float) becomes Range."""
        result = ConfigSpace._sequence_to_range("temp", (0, 1.5))

        assert isinstance(result, Range)

    def test_list_becomes_choices(self) -> None:
        """Test that list becomes Choices."""
        result = ConfigSpace._sequence_to_range("model", ["gpt-4", "gpt-3.5"])

        assert isinstance(result, Choices)
        assert list(result.values) == ["gpt-4", "gpt-3.5"]

    def test_tuple_of_strings_becomes_choices(self) -> None:
        """Test that tuple of strings becomes Choices."""
        result = ConfigSpace._sequence_to_range("model", ("gpt-4", "gpt-3.5"))

        assert isinstance(result, Choices)

    def test_tuple_with_bool_becomes_choices(self) -> None:
        """Test that tuple with bools becomes Choices (not Range)."""
        result = ConfigSpace._sequence_to_range("flag", (True, False))

        assert isinstance(result, Choices)

    def test_long_tuple_becomes_choices(self) -> None:
        """Test that tuple with >2 elements becomes Choices."""
        result = ConfigSpace._sequence_to_range("model", ("a", "b", "c"))

        assert isinstance(result, Choices)

    def test_name_is_set(self) -> None:
        """Test that name attribute is set."""
        result = ConfigSpace._sequence_to_range("my_param", [1, 2, 3])

        assert result.name == "my_param"


class TestDictToRange:
    """Tests for ConfigSpace._dict_to_range."""

    def test_choices_format(self) -> None:
        """Test dict with 'choices' key becomes Choices."""
        result = ConfigSpace._dict_to_range("model", {"choices": ["gpt-4", "gpt-3.5"]})

        assert isinstance(result, Choices)
        assert list(result.values) == ["gpt-4", "gpt-3.5"]

    def test_categorical_type_format(self) -> None:
        """Test dict with type=categorical becomes Choices."""
        result = ConfigSpace._dict_to_range(
            "model", {"type": "categorical", "values": ["gpt-4", "gpt-3.5"]}
        )

        assert isinstance(result, Choices)

    def test_values_format(self) -> None:
        """Test dict with 'values' key becomes Choices."""
        result = ConfigSpace._dict_to_range("model", {"values": ["gpt-4", "gpt-3.5"]})

        assert isinstance(result, Choices)

    def test_range_format_floats(self) -> None:
        """Test dict with low/high floats becomes Range."""
        result = ConfigSpace._dict_to_range("temp", {"low": 0.0, "high": 2.0})

        assert isinstance(result, Range)
        assert result.low == 0.0
        assert result.high == 2.0

    def test_range_format_ints(self) -> None:
        """Test dict with low/high ints becomes IntRange."""
        result = ConfigSpace._dict_to_range("tokens", {"low": 100, "high": 4096})

        assert isinstance(result, IntRange)
        assert result.low == 100
        assert result.high == 4096

    def test_log_range_format(self) -> None:
        """Test dict with log=True becomes LogRange."""
        result = ConfigSpace._dict_to_range(
            "lr", {"low": 1e-5, "high": 1e-1, "log": True}
        )

        assert isinstance(result, LogRange)

    def test_range_with_step(self) -> None:
        """Test dict with step is preserved."""
        result = ConfigSpace._dict_to_range(
            "temp", {"low": 0.0, "high": 1.0, "step": 0.1}
        )

        assert isinstance(result, Range)
        assert result.step == 0.1

    def test_range_with_default(self) -> None:
        """Test dict with default is preserved."""
        result = ConfigSpace._dict_to_range(
            "temp", {"low": 0.0, "high": 1.0, "default": 0.5}
        )

        assert result.default == 0.5

    def test_range_with_unit(self) -> None:
        """Test dict with unit is preserved."""
        result = ConfigSpace._dict_to_range(
            "temp", {"low": 0.0, "high": 1.0, "unit": "ratio"}
        )

        assert result.unit == "ratio"

    def test_choices_with_default(self) -> None:
        """Test dict choices with default."""
        result = ConfigSpace._dict_to_range(
            "model", {"choices": ["gpt-4", "gpt-3.5"], "default": "gpt-4"}
        )

        assert result.default == "gpt-4"

    def test_invalid_dict_raises(self) -> None:
        """Test that invalid dict format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid range specification"):
            ConfigSpace._dict_to_range("bad", {"invalid": "format"})


class TestVarNames:
    """Tests for ConfigSpace.var_names property and get_var_name method."""

    def test_var_names_property(self) -> None:
        """Test var_names returns id-based mapping."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})
        var_names = space.var_names

        assert var_names[id(temp)] == "temperature"
        assert var_names[id(model)] == "model"

    def test_get_var_name(self) -> None:
        """Test get_var_name method."""
        temp = Range(0.0, 2.0, name="temperature")
        other = Range(0.0, 1.0)  # Not in space

        space = ConfigSpace(tvars={"temperature": temp})

        assert space.get_var_name(temp) == "temperature"
        assert space.get_var_name(other) is None

    def test_identical_ranges_dont_collide(self) -> None:
        """Test that identical Range objects have separate entries."""
        range1 = Range(0.0, 1.0, name="alpha")
        range2 = Range(0.0, 1.0, name="beta")  # Same values, different objects

        space = ConfigSpace(tvars={"alpha": range1, "beta": range2})

        assert len(space.var_names) == 2
        assert space.get_var_name(range1) == "alpha"
        assert space.get_var_name(range2) == "beta"


class TestValidation:
    """Tests for ConfigSpace.validate method."""

    def test_validate_valid_config(self) -> None:
        """Test validating a valid configuration."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        result = space.validate({"temperature": 0.5, "model": "gpt-4"})
        assert result.is_valid is True

    def test_validate_invalid_config(self) -> None:
        """Test validating an invalid configuration."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        result = space.validate({"temperature": 1.0, "model": "gpt-4"})
        assert result.is_valid is False
        assert len(result.violations) > 0

    def test_validate_no_constraints(self) -> None:
        """Test validating with no constraints."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(tvars={"temperature": temp})

        result = space.validate({"temperature": 0.5})
        assert result.is_valid is True


class TestCheckSatisfiability:
    """Tests for ConfigSpace.check_satisfiability method."""

    def test_satisfiability_no_constraints(self) -> None:
        """Test satisfiability with no constraints returns SAT."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(tvars={"temperature": temp})

        result = space.check_satisfiability()
        assert result.status == SatStatus.SAT

    def test_satisfiability_with_constraints(self) -> None:
        """Test satisfiability with constraints returns UNKNOWN (Python validator)."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"temperature": temp},
            constraints=[require(temp.gte(0.1))],
        )

        # Python validator returns UNKNOWN for complex constraints
        result = space.check_satisfiability()
        assert result.status == SatStatus.UNKNOWN


class TestTVLExport:
    """Tests for TVL export methods."""

    def test_to_tvl_tvars(self) -> None:
        """Test to_tvl_tvars method."""
        temp = Range(0.0, 2.0, name="temperature", unit="ratio")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})
        tvars = space.to_tvl_tvars()

        assert len(tvars) == 2
        tvar_names = [t.name for t in tvars]
        assert "temperature" in tvar_names
        assert "model" in tvar_names

    def test_to_tvl_constraints(self) -> None:
        """Test to_tvl_constraints method."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )
        constraints = space.to_tvl_constraints()

        assert len(constraints) == 1

    def test_to_tvl_spec_basic(self) -> None:
        """Test to_tvl_spec method returns valid format."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(tvars={"temperature": temp, "model": model})
        spec = space.to_tvl_spec()

        assert "tvars" in spec
        assert len(spec["tvars"]) == 2

    def test_to_tvl_spec_with_constraints(self) -> None:
        """Test to_tvl_spec includes constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )
        spec = space.to_tvl_spec()

        assert "constraints" in spec
        assert "structural" in spec["constraints"]

    def test_to_tvl_spec_with_description(self) -> None:
        """Test to_tvl_spec includes description."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"temperature": temp},
            description="Test space",
        )
        spec = space.to_tvl_spec()

        assert spec["description"] == "Test space"


class TestInferTvarType:
    """Tests for ConfigSpace._infer_tvar_type."""

    def test_infer_int_type(self) -> None:
        """Test inferring int type from IntRange."""
        tvar = IntRange(0, 100)
        assert ConfigSpace._infer_tvar_type(tvar) == "int"

    def test_infer_float_type_from_range(self) -> None:
        """Test inferring float type from Range."""
        tvar = Range(0.0, 1.0)
        assert ConfigSpace._infer_tvar_type(tvar) == "float"

    def test_infer_float_type_from_log_range(self) -> None:
        """Test inferring float type from LogRange."""
        tvar = LogRange(1e-5, 1e-1)
        assert ConfigSpace._infer_tvar_type(tvar) == "float"

    def test_infer_enum_type_from_string_choices(self) -> None:
        """Test inferring enum type from string Choices."""
        tvar = Choices(["a", "b", "c"])
        assert ConfigSpace._infer_tvar_type(tvar) == "enum"

    def test_infer_bool_type_from_bool_choices(self) -> None:
        """Test inferring bool type from bool Choices."""
        tvar = Choices([True, False])
        assert ConfigSpace._infer_tvar_type(tvar) == "bool"

    def test_infer_enum_type_from_mixed_choices(self) -> None:
        """Test inferring enum type from mixed Choices."""
        tvar = Choices([1, 2, 3])  # Numeric choices
        assert ConfigSpace._infer_tvar_type(tvar) == "enum"


class TestTvarToDomain:
    """Tests for ConfigSpace._tvar_to_domain."""

    def test_choices_domain(self) -> None:
        """Test Choices becomes enum domain."""
        tvar = Choices(["a", "b", "c"])
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["kind"] == "enum"
        assert domain["values"] == ["a", "b", "c"]

    def test_range_domain(self) -> None:
        """Test Range becomes range domain."""
        tvar = Range(0.0, 1.0)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["kind"] == "range"
        assert domain["range"] == [0.0, 1.0]

    def test_range_domain_with_step(self) -> None:
        """Test Range with step includes resolution."""
        tvar = Range(0.0, 1.0, step=0.1)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["resolution"] == 0.1

    def test_range_domain_with_log(self) -> None:
        """Test Range with log includes log flag."""
        tvar = Range(0.01, 1.0, log=True)  # log=True requires positive bounds
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["log"] is True

    def test_int_range_domain(self) -> None:
        """Test IntRange becomes range domain."""
        tvar = IntRange(0, 100)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["kind"] == "range"
        assert domain["range"] == [0, 100]

    def test_int_range_domain_with_step(self) -> None:
        """Test IntRange with step includes resolution."""
        tvar = IntRange(0, 100, step=10)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["resolution"] == 10

    def test_int_range_domain_with_log(self) -> None:
        """Test IntRange with log includes log flag."""
        tvar = IntRange(1, 100, log=True)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["log"] is True

    def test_log_range_domain(self) -> None:
        """Test LogRange becomes range domain with log=True."""
        tvar = LogRange(1e-5, 1e-1)
        domain = ConfigSpace._tvar_to_domain(tvar)

        assert domain["kind"] == "range"
        assert domain["log"] is True


class TestTVLSpecWithStandaloneConstraint:
    """Tests for to_tvl_spec with standalone (expr) constraints."""

    def test_standalone_constraint_export(self) -> None:
        """Test that standalone constraints are exported correctly."""
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"temperature": temp},
            constraints=[require(temp.gte(0.1))],
        )
        spec = space.to_tvl_spec()

        constraint = spec["constraints"]["structural"][0]
        assert "expr" in constraint
        assert "params.temperature" in constraint["expr"]

    def test_constraint_metadata_in_spec(self) -> None:
        """Test that constraint metadata is exported."""
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
