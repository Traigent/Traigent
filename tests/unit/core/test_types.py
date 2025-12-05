"""Comprehensive tests for traigent.core.types module."""

import pytest

from traigent.core.types import (
    ConfigurationSpace,
    Parameter,
    ParameterType,
)


class TestParameterType:
    """Test ParameterType enum."""

    def test_parameter_type_values(self):
        """Test ParameterType enum values."""
        assert ParameterType.FLOAT == "float"
        assert ParameterType.INTEGER == "integer"
        assert ParameterType.CATEGORICAL == "categorical"
        assert ParameterType.BOOLEAN == "boolean"

    def test_parameter_type_membership(self):
        """Test ParameterType membership."""
        assert "float" in [pt.value for pt in ParameterType]
        assert "integer" in [pt.value for pt in ParameterType]
        assert "categorical" in [pt.value for pt in ParameterType]
        assert "boolean" in [pt.value for pt in ParameterType]


class TestParameter:
    """Test Parameter dataclass."""

    def test_parameter_float_valid(self):
        """Test creating valid float parameter."""
        param = Parameter(
            name="temperature", type=ParameterType.FLOAT, bounds=(0.0, 1.0)
        )
        assert param.name == "temperature"
        assert param.type == ParameterType.FLOAT
        assert param.bounds == (0.0, 1.0)
        assert param.default is None
        assert param.description is None

    def test_parameter_float_with_default(self):
        """Test float parameter with default value."""
        param = Parameter(
            name="temperature",
            type=ParameterType.FLOAT,
            bounds=(0.0, 1.0),
            default=0.5,
            description="Model temperature",
        )
        assert param.default == 0.5
        assert param.description == "Model temperature"

    def test_parameter_float_invalid_bounds_not_tuple(self):
        """Test float parameter with non-tuple bounds raises ValueError."""
        with pytest.raises(ValueError, match="must have tuple bounds"):
            Parameter(name="temp", type=ParameterType.FLOAT, bounds=[0.0, 1.0])

    def test_parameter_float_invalid_bounds_wrong_length(self):
        """Test float parameter with wrong length bounds raises ValueError."""
        with pytest.raises(ValueError, match="must have tuple bounds"):
            Parameter(name="temp", type=ParameterType.FLOAT, bounds=(0.0,))

    def test_parameter_float_invalid_bounds_non_numeric(self):
        """Test float parameter with non-numeric bounds raises ValueError."""
        with pytest.raises(ValueError, match="bounds must be numeric"):
            Parameter(name="temp", type=ParameterType.FLOAT, bounds=("a", "b"))

    def test_parameter_integer_valid(self):
        """Test creating valid integer parameter."""
        param = Parameter(
            name="max_tokens", type=ParameterType.INTEGER, bounds=(100, 1000)
        )
        assert param.name == "max_tokens"
        assert param.type == ParameterType.INTEGER
        assert param.bounds == (100, 1000)

    def test_parameter_integer_invalid_bounds_not_tuple(self):
        """Test integer parameter with non-tuple bounds raises ValueError."""
        with pytest.raises(ValueError, match="must have tuple bounds"):
            Parameter(name="tokens", type=ParameterType.INTEGER, bounds=[100, 1000])

    def test_parameter_integer_invalid_bounds_non_integer(self):
        """Test integer parameter with non-integer bounds raises ValueError."""
        with pytest.raises(ValueError, match="bounds must be integers"):
            Parameter(name="tokens", type=ParameterType.INTEGER, bounds=(100.5, 1000.5))

    def test_parameter_categorical_valid(self):
        """Test creating valid categorical parameter."""
        param = Parameter(
            name="model",
            type=ParameterType.CATEGORICAL,
            bounds=["gpt-3.5-turbo", "gpt-4"],
        )
        assert param.name == "model"
        assert param.type == ParameterType.CATEGORICAL
        assert param.bounds == ["gpt-3.5-turbo", "gpt-4"]

    def test_parameter_categorical_empty_bounds(self):
        """Test categorical parameter with empty bounds raises ValueError."""
        with pytest.raises(ValueError, match="must have non-empty list"):
            Parameter(name="model", type=ParameterType.CATEGORICAL, bounds=[])

    def test_parameter_categorical_non_list_bounds(self):
        """Test categorical parameter with non-list bounds raises ValueError."""
        with pytest.raises(ValueError, match="must have non-empty list"):
            Parameter(name="model", type=ParameterType.CATEGORICAL, bounds="invalid")

    def test_parameter_boolean_valid(self):
        """Test creating valid boolean parameter."""
        param = Parameter(name="stream", type=ParameterType.BOOLEAN, bounds=None)
        assert param.name == "stream"
        assert param.type == ParameterType.BOOLEAN
        assert param.bounds == [True, False]

    def test_parameter_boolean_with_explicit_bounds(self):
        """Test boolean parameter with explicit bounds."""
        param = Parameter(
            name="stream", type=ParameterType.BOOLEAN, bounds=[True, False]
        )
        assert param.bounds == [True, False]

    def test_parameter_validate_float_valid(self):
        """Test validate_value for float parameter with valid values."""
        param = Parameter(name="temp", type=ParameterType.FLOAT, bounds=(0.0, 1.0))
        assert param.validate_value(0.0) is True
        assert param.validate_value(0.5) is True
        assert param.validate_value(1.0) is True

    def test_parameter_validate_float_invalid(self):
        """Test validate_value for float parameter with invalid values."""
        param = Parameter(name="temp", type=ParameterType.FLOAT, bounds=(0.0, 1.0))
        assert param.validate_value(-0.1) is False
        assert param.validate_value(1.1) is False
        assert param.validate_value("0.5") is False

    def test_parameter_validate_integer_valid(self):
        """Test validate_value for integer parameter with valid values."""
        param = Parameter(name="tokens", type=ParameterType.INTEGER, bounds=(100, 1000))
        assert param.validate_value(100) is True
        assert param.validate_value(500) is True
        assert param.validate_value(1000) is True

    def test_parameter_validate_integer_invalid(self):
        """Test validate_value for integer parameter with invalid values."""
        param = Parameter(name="tokens", type=ParameterType.INTEGER, bounds=(100, 1000))
        assert param.validate_value(99) is False
        assert param.validate_value(1001) is False
        assert param.validate_value(500.5) is False

    def test_parameter_validate_categorical_valid(self):
        """Test validate_value for categorical parameter with valid values."""
        param = Parameter(
            name="model", type=ParameterType.CATEGORICAL, bounds=["a", "b", "c"]
        )
        assert param.validate_value("a") is True
        assert param.validate_value("b") is True
        assert param.validate_value("c") is True

    def test_parameter_validate_categorical_invalid(self):
        """Test validate_value for categorical parameter with invalid values."""
        param = Parameter(
            name="model", type=ParameterType.CATEGORICAL, bounds=["a", "b", "c"]
        )
        assert param.validate_value("d") is False
        assert param.validate_value("") is False

    def test_parameter_validate_boolean_valid(self):
        """Test validate_value for boolean parameter with valid values."""
        param = Parameter(name="stream", type=ParameterType.BOOLEAN, bounds=None)
        assert param.validate_value(True) is True
        assert param.validate_value(False) is True

    def test_parameter_validate_boolean_invalid(self):
        """Test validate_value for boolean parameter with invalid values."""
        param = Parameter(name="stream", type=ParameterType.BOOLEAN, bounds=None)
        assert param.validate_value(1) is False
        assert param.validate_value(0) is False
        assert param.validate_value("true") is False


class TestConfigurationSpace:
    """Test ConfigurationSpace dataclass."""

    def test_configuration_space_empty(self):
        """Test creating empty configuration space."""
        space = ConfigurationSpace()
        assert space.parameters == []
        assert space.constraints is None
        assert space.name is None
        assert space.description is None

    def test_configuration_space_with_parameters(self):
        """Test creating configuration space with parameters."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
        ]
        space = ConfigurationSpace(
            parameters=params, name="test_space", description="Test"
        )
        assert len(space.parameters) == 2
        assert space.name == "test_space"
        assert space.description == "Test"

    def test_configuration_space_duplicate_names(self):
        """Test configuration space with duplicate parameter names raises ValueError."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("temp", ParameterType.FLOAT, (0.5, 1.5)),
        ]
        with pytest.raises(ValueError, match="Duplicate parameter names"):
            ConfigurationSpace(parameters=params)

    def test_configuration_space_items(self):
        """Test items() method returns parameter name-definition pairs."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("model", ParameterType.CATEGORICAL, ["a", "b"]),
            Parameter("stream", ParameterType.BOOLEAN, None),
        ]
        space = ConfigurationSpace(parameters=params)
        items_list = list(space.items())

        assert items_list[0] == ("temp", (0.0, 1.0))
        assert items_list[1] == ("model", ["a", "b"])
        assert items_list[2] == ("stream", [True, False])

    def test_configuration_space_keys(self):
        """Test keys() method returns parameter names."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
        ]
        space = ConfigurationSpace(parameters=params)
        keys = space.keys()
        assert keys == ["temp", "tokens"]

    def test_configuration_space_values(self):
        """Test values() method returns parameter definitions."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("model", ParameterType.CATEGORICAL, ["a", "b"]),
        ]
        space = ConfigurationSpace(parameters=params)
        values = space.values()
        assert values == [(0.0, 1.0), ["a", "b"]]

    def test_configuration_space_getitem(self):
        """Test __getitem__ method for dict-like access."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("model", ParameterType.CATEGORICAL, ["a", "b"]),
        ]
        space = ConfigurationSpace(parameters=params)

        assert space["temp"] == (0.0, 1.0)
        assert space["model"] == ["a", "b"]

    def test_configuration_space_getitem_not_found(self):
        """Test __getitem__ raises KeyError for non-existent parameter."""
        space = ConfigurationSpace()
        with pytest.raises(KeyError, match="not found"):
            _ = space["nonexistent"]

    def test_configuration_space_contains(self):
        """Test __contains__ method for membership testing."""
        params = [Parameter("temp", ParameterType.FLOAT, (0.0, 1.0))]
        space = ConfigurationSpace(parameters=params)

        assert "temp" in space
        assert "nonexistent" not in space

    def test_configuration_space_len(self):
        """Test __len__ method returns number of parameters."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
        ]
        space = ConfigurationSpace(parameters=params)
        assert len(space) == 2

    def test_configuration_space_add_parameter(self):
        """Test add_parameter method."""
        space = ConfigurationSpace()
        param = Parameter("temp", ParameterType.FLOAT, (0.0, 1.0))

        space.add_parameter(param)
        assert len(space.parameters) == 1
        assert space.parameters[0] == param

    def test_configuration_space_add_parameter_duplicate(self):
        """Test add_parameter raises ValueError for duplicate names."""
        space = ConfigurationSpace()
        param1 = Parameter("temp", ParameterType.FLOAT, (0.0, 1.0))
        param2 = Parameter("temp", ParameterType.FLOAT, (0.5, 1.5))

        space.add_parameter(param1)
        with pytest.raises(ValueError, match="already exists"):
            space.add_parameter(param2)

    def test_configuration_space_get_parameter_exists(self):
        """Test get_parameter returns parameter if it exists."""
        param = Parameter("temp", ParameterType.FLOAT, (0.0, 1.0))
        space = ConfigurationSpace(parameters=[param])

        result = space.get_parameter("temp")
        assert result == param

    def test_configuration_space_get_parameter_not_exists(self):
        """Test get_parameter returns None if parameter doesn't exist."""
        space = ConfigurationSpace()
        result = space.get_parameter("nonexistent")
        assert result is None

    def test_configuration_space_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
        ]
        space = ConfigurationSpace(parameters=params)

        config = {"temp": 0.5, "tokens": 500}
        assert space.validate_config(config) is True

    def test_configuration_space_validate_config_missing_parameter(self):
        """Test validate_config returns False for missing parameters."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
        ]
        space = ConfigurationSpace(parameters=params)

        config = {"temp": 0.5}  # Missing tokens
        assert space.validate_config(config) is False

    def test_configuration_space_validate_config_invalid_value(self):
        """Test validate_config returns False for invalid parameter values."""
        params = [Parameter("temp", ParameterType.FLOAT, (0.0, 1.0))]
        space = ConfigurationSpace(parameters=params)

        config = {"temp": 1.5}  # Out of bounds
        assert space.validate_config(config) is False

    def test_configuration_space_validate_config_with_constraints(self):
        """Test validate_config with constraints."""
        params = [
            Parameter("a", ParameterType.INTEGER, (0, 10)),
            Parameter("b", ParameterType.INTEGER, (0, 10)),
        ]
        constraints = [lambda config: config["a"] + config["b"] <= 10]
        space = ConfigurationSpace(parameters=params, constraints=constraints)

        assert space.validate_config({"a": 5, "b": 5}) is True
        assert space.validate_config({"a": 6, "b": 5}) is False

    def test_configuration_space_validate_config_constraint_exception(self):
        """Test validate_config handles constraint exceptions."""
        params = [Parameter("a", ParameterType.INTEGER, (0, 10))]
        constraints = [lambda config: config["nonexistent"] > 0]  # Will raise KeyError
        space = ConfigurationSpace(parameters=params, constraints=constraints)

        assert space.validate_config({"a": 5}) is False

    def test_configuration_space_sample_config(self):
        """Test sample_config generates valid random configuration."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
            Parameter("model", ParameterType.CATEGORICAL, ["a", "b", "c"]),
            Parameter("stream", ParameterType.BOOLEAN, None),
        ]
        space = ConfigurationSpace(parameters=params)

        config = space.sample_config()

        assert "temp" in config
        assert 0.0 <= config["temp"] <= 1.0
        assert "tokens" in config
        assert 100 <= config["tokens"] <= 1000
        assert "model" in config
        assert config["model"] in ["a", "b", "c"]
        assert "stream" in config
        assert config["stream"] in [True, False]

        # Validate the sampled config
        assert space.validate_config(config) is True

    def test_configuration_space_from_dict_float(self):
        """Test from_dict with float parameters."""
        space_dict = {"temperature": (0.0, 1.0)}
        space = ConfigurationSpace.from_dict(space_dict)

        assert len(space.parameters) == 1
        assert space.parameters[0].name == "temperature"
        assert space.parameters[0].type == ParameterType.FLOAT
        assert space.parameters[0].bounds == (0.0, 1.0)

    def test_configuration_space_from_dict_integer(self):
        """Test from_dict with integer parameters."""
        space_dict = {"max_tokens": (100, 1000)}
        space = ConfigurationSpace.from_dict(space_dict)

        assert len(space.parameters) == 1
        assert space.parameters[0].name == "max_tokens"
        assert space.parameters[0].type == ParameterType.INTEGER
        assert space.parameters[0].bounds == (100, 1000)

    def test_configuration_space_from_dict_categorical(self):
        """Test from_dict with categorical parameters."""
        space_dict = {"model": ["gpt-3.5-turbo", "gpt-4"]}
        space = ConfigurationSpace.from_dict(space_dict)

        assert len(space.parameters) == 1
        assert space.parameters[0].name == "model"
        assert space.parameters[0].type == ParameterType.CATEGORICAL
        assert space.parameters[0].bounds == ["gpt-3.5-turbo", "gpt-4"]

    def test_configuration_space_from_dict_boolean(self):
        """Test from_dict with boolean parameters."""
        space_dict = {"stream": "bool"}
        space = ConfigurationSpace.from_dict(space_dict)

        assert len(space.parameters) == 1
        assert space.parameters[0].name == "stream"
        assert space.parameters[0].type == ParameterType.BOOLEAN

    def test_configuration_space_from_dict_mixed_types(self):
        """Test from_dict with mixed parameter types."""
        space_dict = {
            "temperature": (0.0, 1.0),
            "max_tokens": (100, 1000),
            "model": ["a", "b"],
            "stream": "bool",
        }
        space = ConfigurationSpace.from_dict(space_dict)

        assert len(space.parameters) == 4
        assert space["temperature"] == (0.0, 1.0)
        assert space["max_tokens"] == (100, 1000)
        assert space["model"] == ["a", "b"]
        assert space["stream"] == [True, False]

    def test_configuration_space_from_dict_invalid_definition(self):
        """Test from_dict raises ValueError for unsupported definition."""
        space_dict = {"invalid": "unsupported"}
        with pytest.raises(ValueError, match="Unsupported parameter definition"):
            ConfigurationSpace.from_dict(space_dict)

    def test_configuration_space_to_dict(self):
        """Test to_dict converts configuration space to dictionary."""
        params = [
            Parameter("temp", ParameterType.FLOAT, (0.0, 1.0)),
            Parameter("tokens", ParameterType.INTEGER, (100, 1000)),
            Parameter("model", ParameterType.CATEGORICAL, ["a", "b"]),
            Parameter("stream", ParameterType.BOOLEAN, None),
        ]
        space = ConfigurationSpace(parameters=params)

        result = space.to_dict()

        assert result == {
            "temp": (0.0, 1.0),
            "tokens": (100, 1000),
            "model": ["a", "b"],
            "stream": "bool",
        }

    def test_configuration_space_round_trip(self):
        """Test round-trip conversion from_dict -> to_dict."""
        original_dict = {
            "temperature": (0.0, 1.0),
            "max_tokens": (100, 1000),
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "stream": "bool",
        }

        space = ConfigurationSpace.from_dict(original_dict)
        result_dict = space.to_dict()

        assert result_dict == original_dict
