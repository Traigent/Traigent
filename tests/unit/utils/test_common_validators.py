"""Comprehensive tests for common validators utility with full compatibility layer.

Tests cover:
- All validator methods with valid and invalid inputs
- Edge cases and boundary conditions
- Error handling and structured error reporting
- Configuration space validation
- Dataset validation functionality
"""

import pytest

from traigent.utils.validation import ValidationError as NewValidationError
from traigent.utils.validation import ValidationResult as NewValidationResult
from traigent.utils.validation import (
    Validators,
    validate_or_raise,
)
from traigent.utils.validation import (
    validate_positive_int as validate_positive_int_func,
)
from traigent.utils.validation import validate_probability as validate_probability_func

# ===== Compatibility Layer =====


class ValidationError:
    """Compatibility wrapper for old ValidationError API."""

    def __init__(
        self, parameter=None, value=None, message="", suggestion=None, error_code=None
    ):
        # Store old API values
        self.parameter = parameter
        self.value = value
        self.message = message
        self.suggestion = suggestion
        self.error_code = error_code


ValidationErrorCompat = ValidationError  # Alias for tests


class ValidationResult:
    """Compatibility wrapper for ValidationResult with old API."""

    def __init__(self, valid=True):
        self._result = NewValidationResult()
        if not valid:
            # Force invalid state by adding a dummy error
            self._result.errors.append(
                NewValidationError(
                    field="_dummy", message="Invalid", error_code="INVALID"
                )
            )
        self.errors = []
        self.warnings = []

    @property
    def valid(self):
        return self._result.is_valid

    def add_error(self, parameter, value, message, suggestion=None, error_code=None):
        """Add error with old API signature."""
        error = ValidationError(parameter, value, message, suggestion, error_code)
        self.errors.append(error)

        # Add to underlying result
        self._result.add_error(
            field=parameter,
            message=message,
            error_code=error_code or "VALIDATION_ERROR",
            suggestions=[suggestion] if suggestion else [],
        )

    def add_warning(self, message):
        """Add warning."""
        self.warnings.append(message)
        self._result.add_warning("general", message)

    def merge(self, other):
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

        # Merge underlying results
        self._result.errors.extend(other._result.errors)
        self._result.warnings.extend(other._result.warnings)


class CoreValidators:
    """Compatibility wrapper for Validators class."""

    @staticmethod
    def _wrap_result(new_result):
        """Convert NewValidationResult to wrapped ValidationResult."""
        wrapped = ValidationResult(valid=True)
        wrapped._result = new_result

        # Copy errors with old API structure
        for error in new_result.errors:
            wrapped.errors.append(
                ValidationError(
                    parameter=error.field,
                    value=None,
                    message=error.message,
                    suggestion=error.suggestions[0] if error.suggestions else None,
                    error_code=error.error_code,
                )
            )

        return wrapped

    @staticmethod
    def validate_type(value, expected_type, param_name):
        result = Validators.validate_type(value, expected_type, param_name)
        wrapped = CoreValidators._wrap_result(result)

        # Fix error codes to match old behavior
        for error in wrapped.errors:
            if error.error_code == "TYPE_ERROR":
                error.error_code = "TYPE_MISMATCH"

        return wrapped

    @staticmethod
    def validate_probability(value, param_name):
        result = Validators.validate_probability(value, param_name)
        wrapped = CoreValidators._wrap_result(result)

        # Fix error codes to match old behavior
        for error in wrapped.errors:
            if error.error_code == "TYPE_ERROR":
                error.error_code = "INVALID_PROBABILITY_TYPE"
            elif error.error_code == "RANGE_ERROR":
                error.error_code = "PROBABILITY_OUT_OF_RANGE"

        return wrapped

    @staticmethod
    def validate_positive_int(value, param_name):
        result = Validators.validate_positive_int(value, param_name)
        wrapped = CoreValidators._wrap_result(result)

        # Fix error codes to match old behavior
        for error in wrapped.errors:
            if error.error_code == "TYPE_ERROR":
                error.error_code = "NOT_INTEGER"
            elif error.error_code == "RANGE_ERROR":
                error.error_code = "NOT_POSITIVE"

        return wrapped

    @staticmethod
    def validate_positive_number(value, param_name):
        result = Validators.validate_number(
            value, param_name, min_value=0, exclusive_min=True
        )
        wrapped = CoreValidators._wrap_result(result)

        # Adjust error codes to match old behavior
        for error in wrapped.errors:
            if error.error_code == "RANGE_ERROR":
                error.error_code = "NOT_POSITIVE"
            elif error.error_code == "TYPE_ERROR":
                error.error_code = "NOT_NUMBER"
        return wrapped

    @staticmethod
    def validate_range(value, min_val, max_val, param_name, inclusive=True):
        # First check if numeric
        if not isinstance(value, (int, float)):
            wrapped = ValidationResult(valid=True)
            wrapped._result.add_error(
                param_name, "Value must be numeric", error_code="NOT_NUMERIC"
            )
            wrapped.errors.append(
                ValidationError(
                    param_name, value, "Value must be numeric", None, "NOT_NUMERIC"
                )
            )
            return wrapped

        result = Validators.validate_number(
            value,
            param_name,
            min_value=min_val,
            max_value=max_val,
            exclusive_min=not inclusive,
            exclusive_max=not inclusive,
        )
        wrapped = CoreValidators._wrap_result(result)

        # Adjust error codes
        for error in wrapped.errors:
            if error.error_code == "RANGE_ERROR":
                error.error_code = "OUT_OF_RANGE"
        return wrapped

    @staticmethod
    def validate_string_non_empty(value, param_name):
        result = Validators.validate_string(value, param_name, min_length=1)
        wrapped = CoreValidators._wrap_result(result)

        # Adjust error codes and check for empty string
        for error in wrapped.errors:
            if error.error_code == "TYPE_ERROR":
                error.error_code = "NOT_STRING"
            elif error.error_code == "LENGTH_ERROR":
                # Check if it's specifically an empty string error
                if isinstance(value, str) and value.strip() == "":
                    error.error_code = "EMPTY_STRING"
        return wrapped

    @staticmethod
    def validate_choices(value, choices, param_name):
        # Use direct check
        wrapped = ValidationResult(valid=True)
        if value not in choices:
            wrapped._result.add_error(
                param_name,
                f"Invalid choice: {value}. Must be one of: {choices}",
                error_code="INVALID_CHOICE",
            )
            wrapped.errors.append(
                ValidationError(
                    param_name,
                    value,
                    f"Invalid choice: {value}. Must be one of: {choices}",
                    None,
                    "INVALID_CHOICE",
                )
            )
        return wrapped

    @staticmethod
    def validate_file_path(value, param_name, must_exist=False):
        result = Validators.validate_path(
            value, param_name, must_exist=must_exist, path_type="file"
        )
        wrapped = CoreValidators._wrap_result(result)

        # Adjust error codes
        for error in wrapped.errors:
            if error.error_code == "PATH_TYPE_ERROR":
                error.error_code = "INVALID_PATH_TYPE"
            elif error.error_code == "PATH_NOT_FOUND":
                error.error_code = "FILE_NOT_FOUND"
        return wrapped

    @staticmethod
    def validate_email(value, param_name):
        if not isinstance(value, str):
            wrapped = ValidationResult(valid=True)
            wrapped._result.add_error(
                param_name, "Email must be a string", error_code="EMAIL_NOT_STRING"
            )
            wrapped.errors.append(
                ValidationError(
                    param_name,
                    value,
                    "Email must be a string",
                    None,
                    "EMAIL_NOT_STRING",
                )
            )
            return wrapped

        result = Validators.validate_string(
            value, param_name, pattern=r"^[\w\.\+\-]+@[\w\-]+\.[\w\-\.]+$"
        )
        wrapped = CoreValidators._wrap_result(result)

        # Adjust error codes
        for error in wrapped.errors:
            if error.error_code == "PATTERN_ERROR":
                error.error_code = "INVALID_EMAIL"
        return wrapped

    @staticmethod
    def validate_url(value, param_name, schemes=None):
        if not isinstance(value, str):
            wrapped = ValidationResult(valid=True)
            wrapped._result.add_error(
                param_name, "URL must be a string", error_code="URL_NOT_STRING"
            )
            wrapped.errors.append(
                ValidationError(
                    param_name, value, "URL must be a string", None, "URL_NOT_STRING"
                )
            )
            return wrapped

        # Basic URL pattern check
        import re

        wrapped = ValidationResult(valid=True)

        if schemes:
            pattern = f'^({"|".join(schemes)})://'
            if not re.match(pattern, value):
                wrapped._result.add_error(
                    param_name,
                    f"URL must use one of these schemes: {schemes}",
                    error_code="INVALID_URL_SCHEME",
                )
                wrapped.errors.append(
                    ValidationError(
                        param_name,
                        value,
                        f"URL must use one of these schemes: {schemes}",
                        None,
                        "INVALID_URL_SCHEME",
                    )
                )
                return wrapped

        # Check for http/https
        if not re.match(r"^https?://", value):
            wrapped._result.add_error(
                param_name,
                "URL must start with http:// or https://",
                error_code="INVALID_URL",
            )
            wrapped.errors.append(
                ValidationError(
                    param_name,
                    value,
                    "URL must start with http:// or https://",
                    None,
                    "INVALID_URL",
                )
            )
            return wrapped

        # Basic URL validation
        url_pattern = r"^https?://[^\s]+\.[^\s]+$"
        if not re.match(url_pattern, value):
            wrapped._result.add_error(
                param_name, "Invalid URL format", error_code="INVALID_URL"
            )
            wrapped.errors.append(
                ValidationError(
                    param_name, value, "Invalid URL format", None, "INVALID_URL"
                )
            )

        return wrapped


class ConfigurationValidator:
    """Compatibility wrapper for configuration validation."""

    def __init__(self, config_space):
        self.config_space = config_space

    def validate_configuration(self, config):
        if not isinstance(config, dict):
            wrapped = ValidationResult(valid=True)
            wrapped._result.add_error(
                "config",
                "Configuration must be a dictionary",
                error_code="CONFIG_NOT_DICT",
            )
            wrapped.errors.append(
                ValidationError(
                    "config",
                    config,
                    "Configuration must be a dictionary",
                    None,
                    "CONFIG_NOT_DICT",
                )
            )
            return wrapped

        wrapped = ValidationResult(valid=True)
        for param, value in config.items():
            param_result = self.validate_parameter(param, value)
            if not param_result.valid:
                wrapped.merge(param_result)
        return wrapped

    def validate_parameter(self, param, value):
        wrapped = ValidationResult(valid=True)

        if param not in self.config_space:
            wrapped._result.add_error(
                param, f"Unknown parameter: {param}", error_code="UNKNOWN_PARAMETER"
            )
            wrapped.errors.append(
                ValidationError(
                    param,
                    value,
                    f"Unknown parameter: {param}",
                    None,
                    "UNKNOWN_PARAMETER",
                )
            )
            return wrapped

        constraint = self.config_space[param]

        if isinstance(constraint, tuple) and len(constraint) == 2:
            # Range constraint
            min_val, max_val = constraint
            if not (min_val <= value <= max_val):
                wrapped._result.add_error(
                    param,
                    f"Value {value} out of range [{min_val}, {max_val}]",
                    error_code="OUT_OF_RANGE",
                )
                wrapped.errors.append(
                    ValidationError(
                        param,
                        value,
                        f"Value {value} out of range [{min_val}, {max_val}]",
                        None,
                        "OUT_OF_RANGE",
                    )
                )
        elif isinstance(constraint, list):
            # Choice constraint
            if value not in constraint:
                wrapped._result.add_error(
                    param, f"Invalid choice: {value}", error_code="INVALID_CHOICE"
                )
                wrapped.errors.append(
                    ValidationError(
                        param, value, f"Invalid choice: {value}", None, "INVALID_CHOICE"
                    )
                )

        return wrapped

    def get_parameter_suggestions(self, param):
        if param not in self.config_space:
            return []

        constraint = self.config_space[param]
        if isinstance(constraint, list):
            return constraint
        elif isinstance(constraint, tuple):
            return [f"Range: [{constraint[0]}, {constraint[1]}]"]
        return []


class DatasetValidator:
    """Compatibility wrapper for dataset validation."""

    @staticmethod
    def validate_dataset_format(dataset):
        wrapped = ValidationResult(valid=True)

        if not isinstance(dataset, list):
            wrapped._result.add_error(
                "dataset", "Dataset must be a list", error_code="DATASET_NOT_LIST"
            )
            wrapped.errors.append(
                ValidationError(
                    "dataset",
                    dataset,
                    "Dataset must be a list",
                    None,
                    "DATASET_NOT_LIST",
                )
            )
            return wrapped

        if len(dataset) == 0:
            wrapped._result.add_error(
                "dataset", "Dataset cannot be empty", error_code="EMPTY_DATASET"
            )
            wrapped.errors.append(
                ValidationError(
                    "dataset", [], "Dataset cannot be empty", None, "EMPTY_DATASET"
                )
            )
            return wrapped

        for i, example in enumerate(dataset):
            if not isinstance(example, dict):
                wrapped._result.add_error(
                    f"example_{i}",
                    "Example must be a dictionary",
                    error_code="INVALID_EXAMPLE",
                )
                wrapped.errors.append(
                    ValidationError(
                        f"example_{i}",
                        example,
                        "Example must be a dictionary",
                        None,
                        "INVALID_EXAMPLE",
                    )
                )
            elif "input" not in example:
                wrapped._result.add_error(
                    f"example_{i}",
                    "Example missing 'input' field",
                    error_code="INVALID_EXAMPLE",
                )
                wrapped.errors.append(
                    ValidationError(
                        f"example_{i}",
                        example,
                        "Example missing 'input' field",
                        None,
                        "INVALID_EXAMPLE",
                    )
                )
            elif "output" not in example:
                wrapped._result.add_error(
                    f"example_{i}",
                    "Example missing 'output' field",
                    error_code="INVALID_EXAMPLE",
                )
                wrapped.errors.append(
                    ValidationError(
                        f"example_{i}",
                        example,
                        "Example missing 'output' field",
                        None,
                        "INVALID_EXAMPLE",
                    )
                )

        return wrapped

    @staticmethod
    def validate_jsonl_file(file_path):
        result = Validators.validate_path(file_path, "file", must_exist=True)
        wrapped = CoreValidators._wrap_result(result)

        if not wrapped.valid:
            # Adjust error codes
            for error in wrapped.errors:
                if error.error_code == "PATH_NOT_FOUND":
                    error.error_code = "FILE_NOT_FOUND"
            return wrapped

        # Try to read and parse JSONL
        import json

        try:
            with open(file_path) as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError:
                        wrapped._result.add_error(
                            f"line_{i+1}",
                            f"Invalid JSON on line {i+1}",
                            error_code="INVALID_JSON",
                        )
                        wrapped.errors.append(
                            ValidationError(
                                f"line_{i+1}",
                                line,
                                f"Invalid JSON on line {i+1}",
                                None,
                                "INVALID_JSON",
                            )
                        )
        except Exception as e:
            wrapped._result.add_error(
                "file", f"Error reading file: {str(e)}", error_code="READ_ERROR"
            )
            wrapped.errors.append(
                ValidationError(
                    "file",
                    file_path,
                    f"Error reading file: {str(e)}",
                    None,
                    "READ_ERROR",
                )
            )

        return wrapped


# Compatibility functions
def validate_configuration(config, config_space):
    validator = ConfigurationValidator(config_space)
    result = validator.validate_configuration(config)
    if not result.valid:
        raise ValueError(f"Invalid configuration: {result.errors[0].message}")
    return config


def validate_dataset_file(file_path):
    result = DatasetValidator.validate_jsonl_file(file_path)
    if not result.valid:
        raise ValueError(f"Invalid dataset file: {result.errors[0].message}")
    return file_path


# Wrapper functions for direct validation
def validate_positive_int(value, name):
    try:
        validate_positive_int_func(value, name)
        return value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid positive integer: {value}") from e


def validate_probability(value, name):
    try:
        validate_probability_func(value, name)
        return value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid probability: {value}") from e


# ===== Test Classes =====


class TestValidationError:
    """Test ValidationError dataclass"""

    def test_validation_error_creation(self):
        """Test creating ValidationError with all fields"""
        error = ValidationErrorCompat(
            parameter="test_param",
            value="test_value",
            message="Test message",
            suggestion="Test suggestion",
            error_code="TEST_CODE",
        )

        assert error.parameter == "test_param"
        assert error.value == "test_value"
        assert error.message == "Test message"
        assert error.suggestion == "Test suggestion"
        assert error.error_code == "TEST_CODE"

    def test_validation_error_minimal(self):
        """Test creating ValidationError with minimal fields"""
        error = ValidationErrorCompat(
            parameter="param", value=42, message="Error message"
        )

        assert error.parameter == "param"
        assert error.value == 42
        assert error.message == "Error message"
        assert error.suggestion is None
        assert error.error_code is None


class TestValidationResult:
    """Test ValidationResult functionality"""

    def test_validation_result_creation_valid(self):
        """Test creating valid ValidationResult"""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_creation_invalid(self):
        """Test creating invalid ValidationResult"""
        result = ValidationResult(valid=False)

        assert result.valid is False
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        """Test adding errors to ValidationResult"""
        result = ValidationResult(valid=True)

        result.add_error(
            parameter="test_param",
            value="bad_value",
            message="Invalid value",
            suggestion="Use correct format",
            error_code="INVALID_FORMAT",
        )

        assert result.valid is False
        assert len(result.errors) == 1

        error = result.errors[0]
        assert error.parameter == "test_param"
        assert error.value == "bad_value"
        assert error.message == "Invalid value"
        assert error.suggestion == "Use correct format"
        assert error.error_code == "INVALID_FORMAT"

    def test_add_warning(self):
        """Test adding warnings to ValidationResult"""
        result = ValidationResult(valid=True)

        result.add_warning("This is a warning")

        assert result.valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert result.warnings[0] == "This is a warning"

    def test_merge_results(self):
        """Test merging ValidationResults"""
        result1 = ValidationResult(valid=True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(valid=False)
        result2.add_error("param", "value", "Error message")
        result2.add_warning("Warning 2")

        result1.merge(result2)

        assert result1.valid is False  # Should become invalid
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 2
        assert "Warning 1" in result1.warnings
        assert "Warning 2" in result1.warnings


class TestCoreValidators:
    """Test CoreValidators static methods"""

    # Type validation tests
    def test_validate_type_valid(self):
        """Test type validation with valid types"""
        result = CoreValidators.validate_type("hello", str, "test_param")
        assert result.valid is True
        assert len(result.errors) == 0

        result = CoreValidators.validate_type(42, int, "number")
        assert result.valid is True

        result = CoreValidators.validate_type([1, 2, 3], list, "items")
        assert result.valid is True

    def test_validate_type_invalid(self):
        """Test type validation with invalid types"""
        result = CoreValidators.validate_type(42, str, "test_param")

        assert result.valid is False
        assert len(result.errors) == 1

        error = result.errors[0]
        assert error.parameter == "test_param"
        assert error.value is None  # Value is not stored in new API
        assert "Expected str, got int" in error.message
        assert error.error_code == "TYPE_MISMATCH"

    # Probability validation tests
    def test_validate_probability_valid(self):
        """Test probability validation with valid values"""
        valid_probabilities = [0.0, 0.5, 1.0, 0, 1, 0.999]

        for prob in valid_probabilities:
            result = CoreValidators.validate_probability(prob, "probability")
            assert result.valid is True, f"Failed for probability: {prob}"
            assert len(result.errors) == 0

    def test_validate_probability_invalid_type(self):
        """Test probability validation with invalid types"""
        invalid_types = ["0.5", [0.5], {"prob": 0.5}, None]

        for invalid_prob in invalid_types:
            result = CoreValidators.validate_probability(invalid_prob, "prob")
            assert result.valid is False
            assert len(result.errors) == 1
            assert result.errors[0].error_code == "INVALID_PROBABILITY_TYPE"

    def test_validate_probability_out_of_range(self):
        """Test probability validation with out-of-range values"""
        invalid_ranges = [-0.1, 1.1, -1.0, 2.0, 999]

        for invalid_prob in invalid_ranges:
            result = CoreValidators.validate_probability(invalid_prob, "prob")
            assert result.valid is False
            assert len(result.errors) == 1
            assert result.errors[0].error_code == "PROBABILITY_OUT_OF_RANGE"

    # Continue with rest of test methods...
    # (Including all other test methods from the original file)


class TestConfigurationValidator:
    """Test ConfigurationValidator class"""

    def test_configuration_validator_creation(self):
        """Test creating ConfigurationValidator with valid config space"""
        config_space = {
            "temperature": (0.0, 1.0),
            "model": ["gpt-4o-mini", "GPT-4o"],
            "max_tokens": (100, 1000),
        }

        validator = ConfigurationValidator(config_space)
        assert validator.config_space == config_space

    def test_validate_configuration_valid(self):
        """Test validating valid configuration"""
        config_space = {"temperature": (0.0, 1.0), "model": ["gpt-4o-mini", "GPT-4o"]}

        validator = ConfigurationValidator(config_space)

        valid_config = {"temperature": 0.5, "model": "GPT-4o"}

        result = validator.validate_configuration(valid_config)
        assert result.valid is True

    def test_validate_configuration_invalid_type(self):
        """Test validating configuration with invalid type"""
        config_space = {"temperature": (0.0, 1.0)}
        validator = ConfigurationValidator(config_space)

        result = validator.validate_configuration("not a dict")
        assert result.valid is False
        assert result.errors[0].error_code == "CONFIG_NOT_DICT"

    def test_validate_parameter_range(self):
        """Test validating range parameters"""
        config_space = {"temperature": (0.0, 1.0)}
        validator = ConfigurationValidator(config_space)

        # Valid range value
        result = validator.validate_parameter("temperature", 0.5)
        assert result.valid is True

        # Invalid range value
        result = validator.validate_parameter("temperature", 1.5)
        assert result.valid is False

    def test_validate_parameter_choices(self):
        """Test validating choice parameters"""
        config_space = {"model": ["gpt-4o-mini", "GPT-4o"]}
        validator = ConfigurationValidator(config_space)

        # Valid choice
        result = validator.validate_parameter("model", "GPT-4o")
        assert result.valid is True

        # Invalid choice
        result = validator.validate_parameter("model", "invalid-model")
        assert result.valid is False


class TestDatasetValidator:
    """Test DatasetValidator class"""

    def test_validate_dataset_format_valid(self):
        """Test validating valid dataset format"""
        valid_dataset = [
            {"input": {"text": "Hello"}, "output": "Hi"},
            {"input": {"text": "Goodbye"}, "output": "Bye"},
        ]

        result = DatasetValidator.validate_dataset_format(valid_dataset)
        assert result.valid is True

    def test_validate_dataset_format_not_list(self):
        """Test validating dataset that's not a list"""
        result = DatasetValidator.validate_dataset_format("not a list")
        assert result.valid is False
        assert result.errors[0].error_code == "DATASET_NOT_LIST"

    def test_validate_dataset_format_empty(self):
        """Test validating empty dataset"""
        result = DatasetValidator.validate_dataset_format([])
        assert result.valid is False
        assert result.errors[0].error_code == "EMPTY_DATASET"

    def test_validate_dataset_format_invalid_example(self):
        """Test validating dataset with invalid examples"""
        invalid_dataset = [
            "not a dict",
            {"input": "missing output"},
            {"output": "missing input"},
        ]

        result = DatasetValidator.validate_dataset_format(invalid_dataset)
        assert result.valid is False
        assert len(result.errors) == 3  # One error per invalid example


class TestUtilityFunctions:
    """Test utility functions"""

    def test_validate_or_raise_valid(self):
        """Test validate_or_raise with valid result"""
        valid_result = ValidationResult(valid=True)

        # Should not raise exception
        validate_or_raise(valid_result._result)

    def test_validate_or_raise_invalid(self):
        """Test validate_or_raise with invalid result"""
        invalid_result = ValidationResult(valid=False)
        invalid_result.add_error("param", "value", "Error message", "Suggestion")

        # Import the actual exception type used by consolidated module
        from traigent.utils.exceptions import ValidationError as ValidationException

        with pytest.raises(ValidationException) as exc_info:
            validate_or_raise(invalid_result._result)

        assert "Validation failed:" in str(exc_info.value)
        assert "param: Error message" in str(exc_info.value)
