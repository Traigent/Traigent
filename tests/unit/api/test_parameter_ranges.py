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

    # Type enforcement tests (D-2 fix)
    def test_choices_mixed_types_raises(self):
        """Test validation: mixed types raise TypeError by default."""
        with pytest.raises(TypeError, match="consistent types"):
            Choices(["string", 123])  # str and int mixed

    def test_choices_mixed_str_bool_raises(self):
        """Test validation: mixing str and bool raises."""
        with pytest.raises(TypeError, match="consistent types"):
            Choices(["yes", True, "no", False])

    def test_choices_mixed_int_bool_raises(self):
        """Test validation: mixing int and bool raises (bool is subclass of int)."""
        with pytest.raises(TypeError, match="consistent types"):
            Choices([0, 1, True, False])

    def test_choices_int_float_allowed(self):
        """Test validation: int and float can coexist (both numeric)."""
        c = Choices([1, 2.5, 3, 4.0])  # Should not raise
        assert len(c) == 4

    def test_choices_none_with_type_allowed(self):
        """Test validation: None can coexist with any type."""
        c = Choices([None, "default", "option"])  # Should not raise
        assert None in c
        assert "default" in c

    def test_choices_all_none_allowed(self):
        """Test validation: all None values allowed."""
        c = Choices([None, None])
        assert len(c) == 2

    def test_choices_enforce_type_false(self):
        """Test enforce_type=False allows mixed types."""
        c = Choices(["string", 123, True], enforce_type=False)
        assert len(c) == 3
        assert "string" in c
        assert 123 in c
        assert True in c

    def test_choices_single_value_no_validation(self):
        """Test single value doesn't trigger type validation."""
        c = Choices([42])  # Single value, no type comparison needed
        assert len(c) == 1


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


class TestAutoNamingFromKwargs:
    """Tests for D-1: Auto-naming parameters from decorator kwargs."""

    def test_range_name_auto_assigned(self):
        """Test Range without name gets name from kwarg key."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        r = Range(0.0, 1.0)  # No name set
        assert r.name is None

        # Process entry - simulates what happens in normalize_configuration_space
        returned_param = _process_param_entry("temperature", r, result, defaults)

        # The returned param should have the name auto-assigned
        assert returned_param is not None
        assert returned_param.name == "temperature"
        assert result["temperature"] == (0.0, 1.0)

    def test_int_range_name_auto_assigned(self):
        """Test IntRange without name gets name from kwarg key."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        r = IntRange(100, 4096)  # No name set
        assert r.name is None

        returned_param = _process_param_entry("max_tokens", r, result, defaults)

        assert returned_param is not None
        assert returned_param.name == "max_tokens"

    def test_choices_name_auto_assigned(self):
        """Test Choices without name gets name from kwarg key."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        c = Choices(["gpt-4", "gpt-3.5"])  # No name set
        assert c.name is None

        returned_param = _process_param_entry("model", c, result, defaults)

        assert returned_param is not None
        assert returned_param.name == "model"

    def test_log_range_name_auto_assigned(self):
        """Test LogRange without name gets name from kwarg key."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        r = LogRange(1e-5, 1e-1)  # No name set
        assert r.name is None

        returned_param = _process_param_entry("learning_rate", r, result, defaults)

        assert returned_param is not None
        assert returned_param.name == "learning_rate"

    def test_explicit_name_not_overwritten(self):
        """Test explicit name is preserved, not overwritten by kwarg key."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        r = Range(0.0, 1.0, name="temp")  # Explicit name set
        assert r.name == "temp"

        returned_param = _process_param_entry("temperature", r, result, defaults)

        # Explicit name should be preserved
        assert returned_param is not None
        assert returned_param.name == "temp"

    def test_auto_naming_preserves_other_attributes(self):
        """Test auto-naming doesn't affect other attributes."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}
        r = Range(0.0, 2.0, default=0.7, step=0.1, unit="ratio")

        returned_param = _process_param_entry("temp", r, result, defaults)

        assert returned_param is not None
        assert isinstance(returned_param, Range)
        assert returned_param.name == "temp"
        assert returned_param.low == 0.0
        assert returned_param.high == 2.0
        assert returned_param.default == 0.7
        assert returned_param.step == 0.1
        assert returned_param.unit == "ratio"

    def test_non_parameter_range_returns_none(self):
        """Test non-ParameterRange values return None."""
        from traigent.api.parameter_ranges import _process_param_entry

        result: dict = {}
        defaults: dict = {}

        # Tuple (legacy syntax)
        returned = _process_param_entry("temp", (0.0, 1.0), result, defaults)
        assert returned is None
        assert result["temp"] == (0.0, 1.0)

        # List (legacy syntax)
        result.clear()
        returned = _process_param_entry("model", ["a", "b"], result, defaults)
        assert returned is None
        assert result["model"] == ["a", "b"]


# =============================================================================
# Factory Method Tests - Range
# =============================================================================


class TestRangeFactoryMethods:
    """Tests for Range factory methods (domain presets)."""

    def test_temperature_default(self):
        """Test default temperature range."""
        temp = Range.temperature()
        assert temp.low == pytest.approx(0.0)
        assert temp.high == pytest.approx(1.0)
        assert temp.default == pytest.approx(0.7)
        assert temp.name == "temperature"

    def test_temperature_conservative(self):
        """Test conservative temperature range for factual tasks."""
        temp = Range.temperature(conservative=True)
        assert temp.low == pytest.approx(0.0)
        assert temp.high == pytest.approx(0.5)
        assert temp.default == pytest.approx(0.2)
        assert temp.name == "temperature"

    def test_temperature_creative(self):
        """Test creative temperature range."""
        temp = Range.temperature(creative=True)
        assert temp.low == pytest.approx(0.7)
        assert temp.high == pytest.approx(1.5)
        assert temp.default == pytest.approx(1.0)
        assert temp.name == "temperature"

    def test_temperature_both_flags_raises(self):
        """Test that both conservative and creative flags raise error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Range.temperature(conservative=True, creative=True)

    def test_top_p(self):
        """Test top_p factory method."""
        top_p = Range.top_p()
        assert top_p.low == pytest.approx(0.1)
        assert top_p.high == pytest.approx(1.0)
        assert top_p.default == pytest.approx(0.9)
        assert top_p.name == "top_p"

    def test_frequency_penalty(self):
        """Test frequency_penalty factory method."""
        fp = Range.frequency_penalty()
        assert fp.low == pytest.approx(0.0)
        assert fp.high == pytest.approx(2.0)
        assert fp.default == pytest.approx(0.0)
        assert fp.name == "frequency_penalty"

    def test_presence_penalty(self):
        """Test presence_penalty factory method."""
        pp = Range.presence_penalty()
        assert pp.low == pytest.approx(0.0)
        assert pp.high == pytest.approx(2.0)
        assert pp.default == pytest.approx(0.0)
        assert pp.name == "presence_penalty"

    def test_similarity_threshold(self):
        """Test similarity_threshold factory method."""
        st = Range.similarity_threshold()
        assert st.low == pytest.approx(0.0)
        assert st.high == pytest.approx(1.0)
        assert st.default == pytest.approx(0.5)
        assert st.name == "similarity_threshold"

    def test_mmr_lambda(self):
        """Test mmr_lambda factory method."""
        mmr = Range.mmr_lambda()
        assert mmr.low == pytest.approx(0.0)
        assert mmr.high == pytest.approx(1.0)
        assert mmr.default == pytest.approx(0.5)
        assert mmr.name == "mmr_lambda"

    def test_chunk_overlap_ratio(self):
        """Test chunk_overlap_ratio factory method."""
        cor = Range.chunk_overlap_ratio()
        assert cor.low == pytest.approx(0.0)
        assert cor.high == pytest.approx(0.5)
        assert cor.default == pytest.approx(0.1)
        assert cor.name == "chunk_overlap_ratio"


# =============================================================================
# Factory Method Tests - IntRange
# =============================================================================


class TestIntRangeFactoryMethods:
    """Tests for IntRange factory methods (domain presets)."""

    def test_max_tokens_default_medium(self):
        """Test default (medium) max_tokens range."""
        tokens = IntRange.max_tokens()
        assert tokens.low == 256
        assert tokens.high == 1024
        assert tokens.default == 512
        assert tokens.step == 64
        assert tokens.name == "max_tokens"

    def test_max_tokens_short(self):
        """Test short max_tokens range."""
        tokens = IntRange.max_tokens(task="short")
        assert tokens.low == 50
        assert tokens.high == 256
        assert tokens.default == 128

    def test_max_tokens_long(self):
        """Test long max_tokens range."""
        tokens = IntRange.max_tokens(task="long")
        assert tokens.low == 1024
        assert tokens.high == 4096
        assert tokens.default == 2048

    def test_max_tokens_invalid_task_raises(self):
        """Test invalid task type raises error."""
        with pytest.raises(ValueError, match="must be 'short', 'medium', or 'long'"):
            IntRange.max_tokens(task="invalid")  # type: ignore[arg-type]

    def test_k_retrieval_default(self):
        """Test default k_retrieval range."""
        k = IntRange.k_retrieval()
        assert k.low == 1
        assert k.high == 10
        assert k.default == 3
        assert k.name == "k"

    def test_k_retrieval_custom_max(self):
        """Test k_retrieval with custom max_k."""
        k = IntRange.k_retrieval(max_k=20)
        assert k.low == 1
        assert k.high == 20
        assert k.default == 3

    def test_chunk_size(self):
        """Test chunk_size factory method."""
        cs = IntRange.chunk_size()
        assert cs.low == 100
        assert cs.high == 1000
        assert cs.step == 100
        assert cs.default == 500
        assert cs.name == "chunk_size"

    def test_chunk_overlap(self):
        """Test chunk_overlap factory method."""
        co = IntRange.chunk_overlap()
        assert co.low == 0
        assert co.high == 200
        assert co.step == 25
        assert co.default == 50
        assert co.name == "chunk_overlap"

    def test_few_shot_count_default(self):
        """Test default few_shot_count range."""
        fsc = IntRange.few_shot_count()
        assert fsc.low == 0
        assert fsc.high == 10
        assert fsc.default == 3
        assert fsc.name == "few_shot_count"

    def test_few_shot_count_custom_max(self):
        """Test few_shot_count with custom max_examples."""
        fsc = IntRange.few_shot_count(max_examples=5)
        assert fsc.low == 0
        assert fsc.high == 5
        assert fsc.default == 3

    def test_batch_size_default(self):
        """Test default batch_size range."""
        bs = IntRange.batch_size()
        assert bs.low == 1
        assert bs.high == 64
        assert bs.default == 16
        assert bs.name == "batch_size"

    def test_batch_size_custom(self):
        """Test batch_size with custom parameters."""
        bs = IntRange.batch_size(min_size=4, max_size=128, default=32)
        assert bs.low == 4
        assert bs.high == 128
        assert bs.default == 32

    # Reasoning/Extended Thinking Factory Methods

    def test_reasoning_budget(self):
        """Test reasoning_budget factory method."""
        rb = IntRange.reasoning_budget()
        assert rb.low == 0
        assert rb.high == 128000
        assert rb.default == 8000
        assert rb.name == "reasoning_budget"
        assert rb.unit == "tokens"

    def test_reasoning_tokens(self):
        """Test reasoning_tokens factory method for OpenAI."""
        rt = IntRange.reasoning_tokens()
        assert rt.low == 1024
        assert rt.high == 128000
        assert rt.default == 32000
        assert rt.name == "max_completion_tokens"
        assert rt.unit == "tokens"

    def test_thinking_budget(self):
        """Test thinking_budget factory method for Anthropic."""
        tb = IntRange.thinking_budget()
        assert tb.low == 1024
        assert tb.high == 128000
        assert tb.default == 8000
        assert tb.name == "thinking_budget_tokens"
        assert tb.unit == "tokens"

    def test_gemini_thinking_budget(self):
        """Test gemini_thinking_budget factory method."""
        gtb = IntRange.gemini_thinking_budget()
        assert gtb.low == 0
        assert gtb.high == 32768
        assert gtb.default == 8192
        assert gtb.name == "thinking_budget"
        assert gtb.unit == "tokens"


# =============================================================================
# Factory Method Tests - Choices
# =============================================================================


class TestChoicesFactoryMethods:
    """Tests for Choices factory methods (domain presets)."""

    def test_model_default_balanced(self):
        """Test default model selection (balanced tier)."""
        model = Choices.model()
        assert model.name == "model"
        # Should include common models
        assert len(model) > 0

    def test_model_openai_fast(self):
        """Test OpenAI fast tier models."""
        model = Choices.model(provider="openai", tier="fast")
        assert model.name == "model"
        assert "gpt-4o-mini" in model

    def test_model_openai_balanced(self):
        """Test OpenAI balanced tier models."""
        model = Choices.model(provider="openai", tier="balanced")
        assert model.name == "model"
        assert "gpt-4o-mini" in model
        assert "gpt-4o" in model

    def test_model_openai_quality(self):
        """Test OpenAI quality tier models."""
        model = Choices.model(provider="openai", tier="quality")
        assert model.name == "model"
        assert "gpt-4o" in model
        assert "o1-preview" in model

    def test_model_anthropic_fast(self):
        """Test Anthropic fast tier models."""
        model = Choices.model(provider="anthropic", tier="fast")
        assert model.name == "model"
        assert "claude-3-haiku-20240307" in model

    def test_model_anthropic_balanced(self):
        """Test Anthropic balanced tier models."""
        model = Choices.model(provider="anthropic", tier="balanced")
        assert model.name == "model"
        assert "claude-3-5-sonnet-20241022" in model

    def test_model_anthropic_quality(self):
        """Test Anthropic quality tier models."""
        model = Choices.model(provider="anthropic", tier="quality")
        assert model.name == "model"
        assert "claude-3-opus-20240229" in model

    def test_model_from_env(self, monkeypatch):
        """Test model selection from environment variable."""
        monkeypatch.setenv("TRAIGENT_MODELS_OPENAI_FAST", "custom-model-1,custom-model-2")
        model = Choices.model(provider="openai", tier="fast")
        assert model.name == "model"
        assert "custom-model-1" in model
        assert "custom-model-2" in model

    def test_model_unknown_provider_tier(self):
        """Test model selection with unknown provider/tier falls back."""
        model = Choices.model(provider="unknown", tier="fast")
        assert model.name == "model"
        # Falls back to default
        assert len(model) > 0

    def test_prompting_strategy(self):
        """Test prompting_strategy factory method."""
        ps = Choices.prompting_strategy()
        assert ps.name == "prompting_strategy"
        assert ps.default == "direct"
        assert "direct" in ps
        assert "chain_of_thought" in ps
        assert "react" in ps
        assert "self_consistency" in ps

    def test_context_format(self):
        """Test context_format factory method."""
        cf = Choices.context_format()
        assert cf.name == "context_format"
        assert cf.default == "bullet"
        assert "bullet" in cf
        assert "numbered" in cf
        assert "xml" in cf
        assert "markdown" in cf
        assert "json" in cf

    def test_retriever_type(self):
        """Test retriever_type factory method."""
        rt = Choices.retriever_type()
        assert rt.name == "retriever"
        assert rt.default == "similarity"
        assert "similarity" in rt
        assert "mmr" in rt
        assert "bm25" in rt
        assert "hybrid" in rt

    def test_embedding_model_default(self):
        """Test default embedding_model selection."""
        em = Choices.embedding_model()
        assert em.name == "embedding_model"
        assert "text-embedding-3-small" in em
        assert "text-embedding-3-large" in em

    def test_embedding_model_openai(self):
        """Test OpenAI embedding_model selection."""
        em = Choices.embedding_model(provider="openai")
        assert em.name == "embedding_model"
        assert "text-embedding-3-small" in em
        assert "text-embedding-3-large" in em
        assert "text-embedding-ada-002" in em

    def test_reranker_model(self):
        """Test reranker_model factory method."""
        rm = Choices.reranker_model()
        assert rm.name == "reranker"
        assert rm.default == "none"
        assert "none" in rm
        assert "cohere-rerank-v3" in rm
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in rm
        assert "llm-rerank" in rm

    # Reasoning/Extended Thinking Factory Methods

    def test_reasoning_mode_default(self):
        """Test default reasoning_mode."""
        rm = Choices.reasoning_mode()
        assert rm.name == "reasoning_mode"
        assert rm.default == "standard"
        assert "none" in rm
        assert "standard" in rm
        assert "deep" in rm

    def test_reasoning_mode_custom_default(self):
        """Test reasoning_mode with custom default."""
        rm = Choices.reasoning_mode(default="deep")
        assert rm.default == "deep"

    def test_reasoning_effort_default(self):
        """Test default reasoning_effort for OpenAI."""
        re = Choices.reasoning_effort()
        assert re.name == "reasoning_effort"
        assert re.default == "medium"
        assert "minimal" in re
        assert "low" in re
        assert "medium" in re
        assert "high" in re
        assert "xhigh" in re

    def test_reasoning_effort_custom_default(self):
        """Test reasoning_effort with custom default."""
        re = Choices.reasoning_effort(default="high")
        assert re.default == "high"

    def test_extended_thinking_default(self):
        """Test default extended_thinking for Anthropic."""
        et = Choices.extended_thinking()
        assert et.name == "extended_thinking"
        assert et.default is False
        assert True in et
        assert False in et

    def test_extended_thinking_enabled(self):
        """Test extended_thinking enabled by default."""
        et = Choices.extended_thinking(default=True)
        assert et.default is True

    def test_thinking_level_default(self):
        """Test default thinking_level for Gemini 3."""
        tl = Choices.thinking_level()
        assert tl.name == "thinking_level"
        assert tl.default == "high"
        assert "MINIMAL" in tl
        assert "low" in tl
        assert "high" in tl

    def test_thinking_level_custom_default(self):
        """Test thinking_level with custom default."""
        tl = Choices.thinking_level(default="low")
        assert tl.default == "low"


# =============================================================================
# Edge Case Tests - Normalization
# =============================================================================


class TestNormalizeConfigurationSpaceEdgeCases:
    """Tests for edge cases in normalize_configuration_space."""

    def test_config_space_object_with_tvars(self):
        """Test that ConfigSpace-like objects extract tvars dict."""
        # Create a mock ConfigSpace-like object with tvars and constraints
        class MockConfigSpace:
            def __init__(self):
                self.tvars = {"temperature": Range(0.0, 1.0, default=0.5)}
                self.constraints = []

        mock_cs = MockConfigSpace()
        result, defaults = normalize_configuration_space(mock_cs)

        assert "temperature" in result
        assert result["temperature"] == (pytest.approx(0.0), pytest.approx(1.0))
        assert defaults["temperature"] == pytest.approx(0.5)

    def test_invalid_config_space_type_raises(self):
        """Test that invalid config_space types raise ValidationError."""
        from traigent.utils.exceptions import ValidationError

        # Pass an invalid type (not dict, not ConfigSpace-like)
        with pytest.raises(ValidationError, match="Expected dictionary"):
            normalize_configuration_space("invalid_string")  # type: ignore[arg-type]

        with pytest.raises(ValidationError, match="Expected dictionary"):
            normalize_configuration_space(12345)  # type: ignore[arg-type]

    def test_config_space_object_without_constraints(self):
        """Test that objects without constraints attr are treated as invalid."""
        from traigent.utils.exceptions import ValidationError

        # Object with tvars but no constraints - should fail validation
        class MockInvalidConfigSpace:
            def __init__(self):
                self.tvars = {"temp": Range(0.0, 1.0)}
                # No constraints attribute

        mock_cs = MockInvalidConfigSpace()
        with pytest.raises(ValidationError, match="Expected dictionary"):
            normalize_configuration_space(mock_cs)  # type: ignore[arg-type]


class TestProcessParamEntryEdgeCases:
    """Tests for edge cases in _process_param_entry."""

    def test_replace_fallback_on_non_dataclass(self):
        """Test fallback when replace() fails on non-dataclass ParameterRange.

        This tests the defensive fallback code path that handles cases where
        a custom ParameterRange subclass can't use dataclasses.replace().
        """
        from traigent.api.parameter_ranges import _process_param_entry

        # Create a custom ParameterRange that isn't a dataclass
        class CustomRange(ParameterRange):
            def __init__(self):
                self._name = None

            @property
            def name(self):
                return self._name

            def to_config_value(self):
                return (0.0, 1.0)

            def get_default(self):
                return None

        result: dict = {}
        defaults: dict = {}
        custom = CustomRange()

        # This should trigger the fallback since CustomRange isn't a dataclass
        _process_param_entry("my_param", custom, result, defaults)

        # Should still work, but name may not be auto-assigned
        assert "my_param" in result
        assert result["my_param"] == (pytest.approx(0.0), pytest.approx(1.0))
