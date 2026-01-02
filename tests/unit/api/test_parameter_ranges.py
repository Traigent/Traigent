"""Tests for SE-friendly parameter range classes.

Tests cover:
- Range, IntRange, LogRange, Choices construction and validation
- Error cases (invalid bounds, empty choices, etc.)
- to_config_value() output format
- get_default() behavior
- is_inline_param_definition() edge cases
- normalize_configuration_space() merging and precedence
"""

import pytest

from traigent.api.parameter_ranges import (
    Choices,
    IntRange,
    LogRange,
    ParameterRange,
    Range,
    is_inline_param_definition,
    is_parameter_range,
    normalize_config_value,
    normalize_configuration_space,
    normalize_parameter_value,
)


class TestRange:
    """Tests for Range class."""

    def test_basic_range(self):
        """Test basic float range construction."""
        r = Range(0.0, 1.0)
        assert r.low == 0.0
        assert r.high == 1.0
        assert r.step is None
        assert r.log is False
        assert r.default is None

    def test_range_to_config_value_simple(self):
        """Test simple range converts to tuple."""
        r = Range(0.0, 1.0)
        assert r.to_config_value() == (0.0, 1.0)

    def test_range_to_tuple(self):
        """Test to_tuple() method."""
        r = Range(0.5, 2.5)
        assert r.to_tuple() == (0.5, 2.5)

    def test_range_with_step(self):
        """Test range with step emits dict format."""
        r = Range(0.0, 1.0, step=0.1)
        config = r.to_config_value()
        assert isinstance(config, dict)
        assert config["type"] == "float"
        assert config["low"] == 0.0
        assert config["high"] == 1.0
        assert config["step"] == 0.1
        assert "log" not in config  # Not set when False

    def test_range_with_log(self):
        """Test range with log emits dict format."""
        r = Range(0.001, 1.0, log=True)
        config = r.to_config_value()
        assert isinstance(config, dict)
        assert config["type"] == "float"
        assert config["log"] is True

    def test_range_with_default(self):
        """Test range with default value."""
        r = Range(0.0, 1.0, default=0.5)
        assert r.default == 0.5
        assert r.get_default() == 0.5

    def test_range_validation_low_ge_high(self):
        """Test validation: low must be less than high."""
        with pytest.raises(ValueError, match="must be less than"):
            Range(1.0, 0.0)

    def test_range_validation_equal_bounds(self):
        """Test validation: equal bounds not allowed."""
        with pytest.raises(ValueError, match="must be less than"):
            Range(0.5, 0.5)

    def test_range_validation_negative_step(self):
        """Test validation: step must be positive."""
        with pytest.raises(ValueError, match="step must be positive"):
            Range(0.0, 1.0, step=-0.1)

    def test_range_validation_zero_step(self):
        """Test validation: step cannot be zero."""
        with pytest.raises(ValueError, match="step must be positive"):
            Range(0.0, 1.0, step=0.0)

    def test_range_log_requires_positive_low(self):
        """Test validation: log requires positive low bound."""
        with pytest.raises(ValueError, match="positive bounds"):
            Range(-1.0, 1.0, log=True)

    def test_range_log_requires_positive_zero(self):
        """Test validation: log requires low > 0, not just >= 0."""
        with pytest.raises(ValueError, match="positive bounds"):
            Range(0.0, 1.0, log=True)

    def test_range_log_and_step_forbidden(self):
        """Test validation: log and step cannot be combined (Optuna limitation)."""
        with pytest.raises(ValueError, match="Cannot use log=True with step"):
            Range(0.01, 1.0, step=0.1, log=True)

    def test_range_default_outside_bounds(self):
        """Test validation: default must be within bounds."""
        with pytest.raises(ValueError, match="outside range"):
            Range(0.0, 1.0, default=2.0)

    def test_range_default_at_bounds(self):
        """Test default at boundary values is valid."""
        r_low = Range(0.0, 1.0, default=0.0)
        assert r_low.default == 0.0
        r_high = Range(0.0, 1.0, default=1.0)
        assert r_high.default == 1.0

    def test_range_is_parameter_range(self):
        """Test Range is a ParameterRange."""
        r = Range(0.0, 1.0)
        assert isinstance(r, ParameterRange)
        assert is_parameter_range(r)

    def test_range_is_frozen(self):
        """Test Range is immutable."""
        r = Range(0.0, 1.0)
        with pytest.raises(AttributeError):
            r.low = 0.5  # type: ignore[misc]


class TestIntRange:
    """Tests for IntRange class."""

    def test_basic_int_range(self):
        """Test basic integer range construction."""
        r = IntRange(100, 4096)
        assert r.low == 100
        assert r.high == 4096
        assert r.step is None
        assert r.log is False

    def test_int_range_to_config_value_simple(self):
        """Test simple int range converts to tuple."""
        r = IntRange(1, 10)
        assert r.to_config_value() == (1, 10)

    def test_int_range_to_tuple(self):
        """Test to_tuple() method."""
        r = IntRange(100, 200)
        assert r.to_tuple() == (100, 200)

    def test_int_range_with_step(self):
        """Test int range with step emits dict format."""
        r = IntRange(0, 100, step=10)
        config = r.to_config_value()
        assert isinstance(config, dict)
        assert config["type"] == "int"
        assert config["step"] == 10

    def test_int_range_with_log(self):
        """Test int range with log emits dict format."""
        r = IntRange(1, 1000, log=True)
        config = r.to_config_value()
        assert isinstance(config, dict)
        assert config["log"] is True

    def test_int_range_type_validation_float_low(self):
        """Test validation: bounds must be integers."""
        with pytest.raises(TypeError, match="must be integers"):
            IntRange(1.5, 10)  # type: ignore[arg-type]

    def test_int_range_type_validation_float_high(self):
        """Test validation: high must be integer."""
        with pytest.raises(TypeError, match="must be integers"):
            IntRange(1, 10.5)  # type: ignore[arg-type]

    def test_int_range_validation_low_ge_high(self):
        """Test validation: low must be less than high."""
        with pytest.raises(ValueError, match="must be less than"):
            IntRange(10, 5)

    def test_int_range_log_and_step_forbidden(self):
        """Test validation: log and step cannot be combined."""
        with pytest.raises(ValueError, match="Cannot use log=True with step"):
            IntRange(1, 100, step=10, log=True)

    def test_int_range_step_must_be_positive(self):
        """Test validation: step must be positive."""
        with pytest.raises(ValueError, match="step must be positive"):
            IntRange(1, 100, step=0)
        with pytest.raises(ValueError, match="step must be positive"):
            IntRange(1, 100, step=-5)

    def test_int_range_log_requires_positive_low(self):
        """Test validation: log=True requires positive low bound."""
        with pytest.raises(ValueError, match="log=True requires positive bounds"):
            IntRange(0, 100, log=True)
        with pytest.raises(ValueError, match="log=True requires positive bounds"):
            IntRange(-5, 100, log=True)

    def test_int_range_with_default(self):
        """Test int range with default value."""
        r = IntRange(1, 100, default=50)
        assert r.default == 50
        assert r.get_default() == 50

    def test_int_range_default_outside_range(self):
        """Test validation: default must be within range."""
        with pytest.raises(ValueError, match="default.*outside range"):
            IntRange(1, 100, default=0)
        with pytest.raises(ValueError, match="default.*outside range"):
            IntRange(1, 100, default=101)


class TestLogRange:
    """Tests for LogRange class."""

    def test_basic_log_range(self):
        """Test basic log range construction."""
        r = LogRange(1e-5, 1e-1)
        assert r.low == 1e-5
        assert r.high == 1e-1
        assert r.default is None

    def test_log_range_to_config_value(self):
        """Test log range always emits dict with log=True."""
        r = LogRange(0.001, 10.0)
        config = r.to_config_value()
        assert isinstance(config, dict)
        assert config["type"] == "float"
        assert config["low"] == 0.001
        assert config["high"] == 10.0
        assert config["log"] is True

    def test_log_range_to_tuple(self):
        """Test to_tuple() method (loses log info)."""
        r = LogRange(0.01, 1.0)
        assert r.to_tuple() == (0.01, 1.0)

    def test_log_range_validation_zero_low(self):
        """Test validation: low must be positive."""
        with pytest.raises(ValueError, match="positive bounds"):
            LogRange(0.0, 1.0)

    def test_log_range_validation_negative_low(self):
        """Test validation: low must be positive."""
        with pytest.raises(ValueError, match="positive bounds"):
            LogRange(-0.1, 1.0)

    def test_log_range_validation_negative_high(self):
        """Test validation: high must be positive."""
        with pytest.raises(ValueError, match="positive bounds"):
            LogRange(0.1, -1.0)

    def test_log_range_validation_low_ge_high(self):
        """Test validation: low must be less than high."""
        with pytest.raises(ValueError, match="must be less than"):
            LogRange(1.0, 0.1)

    def test_log_range_with_default(self):
        """Test log range with default value."""
        r = LogRange(0.001, 1.0, default=0.01)
        assert r.default == 0.01
        assert r.get_default() == 0.01

    def test_log_range_default_outside_range(self):
        """Test validation: default must be within range."""
        with pytest.raises(ValueError, match="default.*outside range"):
            LogRange(0.001, 1.0, default=0.0001)
        with pytest.raises(ValueError, match="default.*outside range"):
            LogRange(0.001, 1.0, default=10.0)


class TestChoices:
    """Tests for Choices class."""

    def test_basic_choices(self):
        """Test basic choices construction."""
        c = Choices(["gpt-4", "gpt-3.5"])
        assert c.to_list() == ["gpt-4", "gpt-3.5"]
        assert len(c) == 2

    def test_choices_to_config_value(self):
        """Test choices converts to list."""
        c = Choices(["a", "b", "c"])
        assert c.to_config_value() == ["a", "b", "c"]

    def test_choices_to_list(self):
        """Test to_list() method."""
        c = Choices([1, 2, 3])
        assert c.to_list() == [1, 2, 3]

    def test_choices_contains(self):
        """Test __contains__ method."""
        c = Choices(["a", "b", "c"])
        assert "a" in c
        assert "d" not in c

    def test_choices_iteration(self):
        """Test __iter__ method."""
        c = Choices([1, 2, 3])
        assert list(c) == [1, 2, 3]

    def test_choices_with_default(self):
        """Test choices with default value."""
        c = Choices(["a", "b", "c"], default="b")
        assert c.default == "b"
        assert c.get_default() == "b"

    def test_choices_empty_raises(self):
        """Test validation: choices must have at least one value."""
        with pytest.raises(ValueError, match="at least one value"):
            Choices([])

    def test_choices_string_raises(self):
        """Test validation: string is not valid (common mistake)."""
        with pytest.raises(TypeError, match="must be a list or tuple, not str"):
            Choices("abc")  # type: ignore[arg-type]

    def test_choices_bytes_raises(self):
        """Test validation: bytes is not valid."""
        with pytest.raises(TypeError, match="must be a list or tuple, not str/bytes"):
            Choices(b"abc")  # type: ignore[arg-type]

    def test_choices_invalid_default(self):
        """Test validation: default must be in choices."""
        with pytest.raises(ValueError, match="not in choices"):
            Choices(["a", "b"], default="c")

    def test_choices_with_booleans(self):
        """Test choices with boolean values."""
        c = Choices([True, False], default=True)
        assert c.to_list() == [True, False]
        assert c.default is True

    def test_choices_with_numbers(self):
        """Test choices with numeric values."""
        c = Choices([0.1, 0.5, 0.9])
        assert c.to_list() == [0.1, 0.5, 0.9]

    def test_choices_tuple_input(self):
        """Test choices accepts tuple input."""
        c = Choices(("a", "b", "c"))
        assert c.to_list() == ["a", "b", "c"]

    def test_choices_values_immutable(self):
        """Test choices values are converted to tuple (immutable)."""
        c = Choices(["a", "b"])
        assert isinstance(c.values, tuple)


class TestIsParameterRange:
    """Tests for is_parameter_range function."""

    def test_range_is_parameter_range(self):
        assert is_parameter_range(Range(0.0, 1.0))

    def test_int_range_is_parameter_range(self):
        assert is_parameter_range(IntRange(1, 10))

    def test_log_range_is_parameter_range(self):
        assert is_parameter_range(LogRange(0.01, 1.0))

    def test_choices_is_parameter_range(self):
        assert is_parameter_range(Choices(["a", "b"]))

    def test_tuple_is_not_parameter_range(self):
        assert not is_parameter_range((0.0, 1.0))

    def test_list_is_not_parameter_range(self):
        assert not is_parameter_range(["a", "b"])

    def test_dict_is_not_parameter_range(self):
        assert not is_parameter_range({"type": "float", "low": 0, "high": 1})

    def test_none_is_not_parameter_range(self):
        assert not is_parameter_range(None)

    def test_string_is_not_parameter_range(self):
        assert not is_parameter_range("test")


class TestIsInlineParamDefinition:
    """Tests for is_inline_param_definition function."""

    def test_range_is_inline_param(self):
        assert is_inline_param_definition(Range(0.0, 1.0))

    def test_int_range_is_inline_param(self):
        assert is_inline_param_definition(IntRange(1, 10))

    def test_choices_is_inline_param(self):
        assert is_inline_param_definition(Choices(["a", "b"]))

    def test_tuple_two_floats_is_inline_param(self):
        assert is_inline_param_definition((0.0, 1.0))

    def test_tuple_two_ints_is_inline_param(self):
        assert is_inline_param_definition((1, 100))

    def test_tuple_mixed_numeric_is_inline_param(self):
        assert is_inline_param_definition((1, 10.5))

    def test_non_empty_list_is_not_inline_param(self):
        # Lists are NOT treated as inline params to catch typos like `objectivs=[...]`
        # Users should use Choices([...]) for categorical parameters
        assert not is_inline_param_definition(["a", "b"])

    def test_list_of_numbers_is_not_inline_param(self):
        # Lists are NOT treated as inline params to catch typos
        assert not is_inline_param_definition([1, 2, 3])

    def test_empty_list_is_not_inline_param(self):
        assert not is_inline_param_definition([])

    def test_tuple_wrong_length_is_not_inline_param(self):
        assert not is_inline_param_definition((1, 2, 3))
        assert not is_inline_param_definition((1,))

    def test_tuple_non_numeric_is_not_inline_param(self):
        assert not is_inline_param_definition(("a", "b"))

    def test_string_is_not_inline_param(self):
        assert not is_inline_param_definition("test")

    def test_int_is_not_inline_param(self):
        assert not is_inline_param_definition(42)

    def test_none_is_not_inline_param(self):
        assert not is_inline_param_definition(None)

    def test_dict_is_not_inline_param(self):
        # Dicts are NOT inline params - they could be typos for other args
        assert not is_inline_param_definition({"type": "float"})


class TestNormalizeConfigValue:
    """Tests for normalize_config_value function."""

    def test_normalize_range(self):
        result = normalize_config_value(Range(0.0, 1.0))
        assert result == (0.0, 1.0)

    def test_normalize_range_with_log(self):
        result = normalize_config_value(Range(0.01, 1.0, log=True))
        assert isinstance(result, dict)
        assert result["log"] is True

    def test_normalize_int_range(self):
        result = normalize_config_value(IntRange(1, 10))
        assert result == (1, 10)

    def test_normalize_log_range(self):
        result = normalize_config_value(LogRange(0.01, 1.0))
        assert isinstance(result, dict)
        assert result["log"] is True

    def test_normalize_choices(self):
        result = normalize_config_value(Choices(["a", "b"]))
        assert result == ["a", "b"]

    def test_normalize_legacy_tuple(self):
        result = normalize_config_value((0.0, 1.0))
        assert result == (0.0, 1.0)

    def test_normalize_legacy_list(self):
        result = normalize_config_value(["a", "b"])
        assert result == ["a", "b"]

    def test_normalize_parameter_value_alias(self):
        result = normalize_parameter_value(Range(0.0, 1.0))
        assert result == (0.0, 1.0)


class TestNormalizeConfigurationSpace:
    """Tests for normalize_configuration_space function."""

    def test_normalize_empty(self):
        result, defaults = normalize_configuration_space(None)
        assert result == {}
        assert defaults == {}

    def test_normalize_with_range(self):
        result, defaults = normalize_configuration_space(
            {
                "temperature": Range(0.0, 2.0),
            }
        )
        assert result["temperature"] == (0.0, 2.0)
        assert defaults == {}

    def test_normalize_with_choices(self):
        result, defaults = normalize_configuration_space(
            {
                "model": Choices(["gpt-4", "gpt-3.5"]),
            }
        )
        assert result["model"] == ["gpt-4", "gpt-3.5"]

    def test_normalize_with_defaults(self):
        result, defaults = normalize_configuration_space(
            {
                "temperature": Range(0.0, 2.0, default=0.7),
                "model": Choices(["gpt-4", "gpt-3.5"], default="gpt-4"),
            }
        )
        assert defaults["temperature"] == 0.7
        assert defaults["model"] == "gpt-4"

    def test_normalize_mixed_syntax(self):
        """Test mixed old and new syntax."""
        result, defaults = normalize_configuration_space(
            {
                "temperature": (0.0, 1.0),  # Old syntax
                "model": Choices(["gpt-4"]),  # New syntax
                "max_tokens": [100, 500, 1000],  # Old syntax
            }
        )
        assert result["temperature"] == (0.0, 1.0)
        assert result["model"] == ["gpt-4"]
        assert result["max_tokens"] == [100, 500, 1000]

    def test_inline_overrides_config_space(self):
        """Test inline params override config_space entries."""
        result, defaults = normalize_configuration_space(
            config_space={"temp": (0.0, 1.0)},
            inline_params={"temp": Range(0.0, 2.0)},
        )
        assert result["temp"] == (0.0, 2.0)

    def test_inline_merges_with_config_space(self):
        """Test inline params merge with config_space."""
        result, defaults = normalize_configuration_space(
            config_space={"model": ["gpt-4"]},
            inline_params={"temperature": Range(0.0, 1.0)},
        )
        assert "model" in result
        assert "temperature" in result

    def test_inline_only(self):
        """Test with only inline params."""
        result, defaults = normalize_configuration_space(
            config_space=None,
            inline_params={"temperature": Range(0.0, 1.0)},
        )
        assert result["temperature"] == (0.0, 1.0)

    def test_inline_defaults_override(self):
        """Test inline param defaults override config_space defaults."""
        result, defaults = normalize_configuration_space(
            config_space={"temp": Range(0.0, 1.0, default=0.5)},
            inline_params={"temp": Range(0.0, 2.0, default=1.0)},
        )
        assert defaults["temp"] == 1.0


class TestParameterRangeImmutability:
    """Tests to verify parameter range objects are immutable."""

    def test_range_frozen(self):
        r = Range(0.0, 1.0)
        with pytest.raises(AttributeError):
            r.low = 0.5  # type: ignore[misc]

    def test_int_range_frozen(self):
        r = IntRange(1, 10)
        with pytest.raises(AttributeError):
            r.high = 20  # type: ignore[misc]

    def test_log_range_frozen(self):
        r = LogRange(0.01, 1.0)
        with pytest.raises(AttributeError):
            r.default = 0.1  # type: ignore[misc]

    def test_choices_frozen(self):
        c = Choices(["a", "b"])
        with pytest.raises(AttributeError):
            c.default = "a"  # type: ignore[misc]
