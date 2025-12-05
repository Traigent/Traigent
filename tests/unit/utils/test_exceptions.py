"""Comprehensive tests for traigent.utils.exceptions module.

Tests cover all custom exception classes with focus on initialization,
inheritance, attributes, and error message handling.
"""

from __future__ import annotations

import pytest

from traigent.utils.exceptions import (
    AgentExecutionError,
    AuthenticationError,
    ClientError,
    ConfigurationError,
    ConfigurationSpaceError,
    DatasetValidationError,
    EvaluationError,
    InvocationError,
    JWTValidationError,
    ObjectiveValidationError,
    OptimizationError,
    PlatformCapabilityError,
    PluginError,
    QuotaExceededError,
    RetryError,
    ServiceError,
    ServiceUnavailableError,
    SessionError,
    StandardizedClientError,
    StorageError,
    TraigentConnectionError,
    TraigentError,
    TraigentStorageError,
    TraigentTimeoutError,
    TraigentValidationError,
    TrialPrunedError,
    ValidationError,
)


class TestTraigentError:
    """Test TraigentError base exception class."""

    def test_create_basic_error(self):
        """Test creating basic TraigentError."""
        error = TraigentError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_create_error_with_details(self):
        """Test TraigentError with details dictionary."""
        details = {"key": "value", "count": 42}
        error = TraigentError("Error with details", details=details)

        assert error.message == "Error with details"
        assert error.details == details
        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_error_is_exception(self):
        """Test that TraigentError inherits from Exception."""
        error = TraigentError("Test")
        assert isinstance(error, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching TraigentError."""
        with pytest.raises(TraigentError) as exc_info:
            raise TraigentError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_empty_details(self):
        """Test TraigentError with None details."""
        error = TraigentError("Test", details=None)
        assert error.details == {}


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_inheritance(self):
        """Test ConfigurationError inherits from TraigentError."""
        error = ConfigurationError("Config error")

        assert isinstance(error, TraigentError)
        assert isinstance(error, Exception)

    def test_with_details(self):
        """Test ConfigurationError with details."""
        details = {"invalid_param": "temperature", "value": -1}
        error = ConfigurationError("Invalid configuration", details=details)

        assert error.message == "Invalid configuration"
        assert error.details["invalid_param"] == "temperature"


class TestValidationError:
    """Test ValidationError."""

    def test_inheritance(self):
        """Test ValidationError inherits from TraigentError."""
        error = ValidationError("Validation failed")

        assert isinstance(error, TraigentError)
        assert isinstance(error, Exception)

    def test_validation_with_field_info(self):
        """Test ValidationError with field information."""
        details = {"field": "dataset", "reason": "empty"}
        error = ValidationError("Dataset is empty", details=details)

        assert error.details["field"] == "dataset"
        assert error.details["reason"] == "empty"


class TestTraigentValidationError:
    """Test TraigentValidationError."""

    def test_inheritance(self):
        """Test TraigentValidationError inherits from ValidationError."""
        error = TraigentValidationError("Enhanced validation error")

        assert isinstance(error, ValidationError)
        assert isinstance(error, TraigentError)


class TestClientError:
    """Test ClientError."""

    def test_create_basic_client_error(self):
        """Test creating basic ClientError."""
        error = ClientError("Client communication failed")

        assert error.message == "Client communication failed"
        assert error.status_code is None
        assert error.details == {}

    def test_create_with_status_code(self):
        """Test ClientError with HTTP status code."""
        error = ClientError("Not found", status_code=404)

        assert error.status_code == 404
        assert error.message == "Not found"

    def test_create_with_all_fields(self):
        """Test ClientError with status code and details."""
        details = {"url": "https://api.example.com", "method": "POST"}
        error = ClientError("Request failed", status_code=500, details=details)

        assert error.status_code == 500
        assert error.details["url"] == "https://api.example.com"

    def test_inheritance(self):
        """Test ClientError inherits from TraigentError."""
        error = ClientError("Test")
        assert isinstance(error, TraigentError)


class TestStandardizedClientError:
    """Test StandardizedClientError."""

    def test_inheritance(self):
        """Test StandardizedClientError inherits from ClientError."""
        error = StandardizedClientError("Standardized error")

        assert isinstance(error, ClientError)
        assert isinstance(error, TraigentError)


class TestAuthenticationError:
    """Test AuthenticationError."""

    def test_inheritance(self):
        """Test AuthenticationError inherits from TraigentError."""
        error = AuthenticationError("Authentication failed")

        assert isinstance(error, TraigentError)
        assert error.message == "Authentication failed"

    def test_with_auth_details(self):
        """Test AuthenticationError with authentication details."""
        details = {"username": "test_user", "reason": "invalid_token"}
        error = AuthenticationError("Token expired", details=details)

        assert error.details["reason"] == "invalid_token"


class TestJWTValidationError:
    """Test JWTValidationError."""

    def test_inheritance(self):
        """Test JWTValidationError inherits from ValidationError."""
        error = JWTValidationError("JWT token invalid")

        assert isinstance(error, ValidationError)
        assert isinstance(error, TraigentError)


class TestConfigurationSpaceError:
    """Test ConfigurationSpaceError."""

    def test_inheritance(self):
        """Test ConfigurationSpaceError inherits from ConfigurationError."""
        error = ConfigurationSpaceError("Invalid configuration space")

        assert isinstance(error, ConfigurationError)
        assert isinstance(error, TraigentError)


class TestDatasetValidationError:
    """Test DatasetValidationError."""

    def test_inheritance(self):
        """Test DatasetValidationError inherits from ValidationError."""
        error = DatasetValidationError("Invalid dataset")

        assert isinstance(error, ValidationError)


class TestObjectiveValidationError:
    """Test ObjectiveValidationError."""

    def test_inheritance(self):
        """Test ObjectiveValidationError inherits from ValidationError."""
        error = ObjectiveValidationError("Invalid objective")

        assert isinstance(error, ValidationError)


class TestInvocationError:
    """Test InvocationError."""

    def test_create_basic_invocation_error(self):
        """Test creating basic InvocationError."""
        error = InvocationError("Function invocation failed")

        assert error.message == "Function invocation failed"
        assert error.config is None
        assert error.input_data is None
        assert error.original_error is None

    def test_create_with_config(self):
        """Test InvocationError with configuration."""
        config = {"model": "gpt-4", "temperature": 0.7}
        error = InvocationError("Invocation failed", config=config)

        assert error.config == config
        assert error.config["model"] == "gpt-4"

    def test_create_with_input_data(self):
        """Test InvocationError with input data."""
        input_data = {"prompt": "Test prompt", "max_tokens": 100}
        error = InvocationError("Failed", input_data=input_data)

        assert error.input_data == input_data

    def test_create_with_original_error(self):
        """Test InvocationError with original exception."""
        original = ValueError("Original error")
        error = InvocationError("Wrapped error", original_error=original)

        assert error.original_error is original
        assert isinstance(error.original_error, ValueError)

    def test_create_with_all_fields(self):
        """Test InvocationError with all fields."""
        config = {"model": "gpt-4"}
        input_data = {"prompt": "test"}
        original = RuntimeError("Original")
        details = {"attempt": 3}

        error = InvocationError(
            "Complete error",
            config=config,
            input_data=input_data,
            original_error=original,
            details=details,
        )

        assert error.config == config
        assert error.input_data == input_data
        assert error.original_error is original
        assert error.details["attempt"] == 3


class TestEvaluationError:
    """Test EvaluationError."""

    def test_create_basic_evaluation_error(self):
        """Test creating basic EvaluationError."""
        error = EvaluationError("Evaluation failed")

        assert error.message == "Evaluation failed"
        assert error.config is None
        assert error.original_error is None

    def test_create_with_config(self):
        """Test EvaluationError with configuration."""
        config = {"metric": "accuracy", "threshold": 0.8}
        error = EvaluationError("Metric failed", config=config)

        assert error.config == config
        assert error.config["metric"] == "accuracy"

    def test_create_with_original_error(self):
        """Test EvaluationError with original exception."""
        original = KeyError("Missing metric")
        error = EvaluationError("Evaluation error", original_error=original)

        assert error.original_error is original

    def test_create_with_all_fields(self):
        """Test EvaluationError with all fields."""
        config = {"metric": "f1"}
        original = ValueError("Invalid")
        details = {"trial": 5}

        error = EvaluationError(
            "Complete evaluation error",
            config=config,
            original_error=original,
            details=details,
        )

        assert error.config == config
        assert error.original_error is original
        assert error.details["trial"] == 5


class TestOptimizationError:
    """Test OptimizationError."""

    def test_inheritance(self):
        """Test OptimizationError inherits from TraigentError."""
        error = OptimizationError("Optimization failed")

        assert isinstance(error, TraigentError)
        assert error.message == "Optimization failed"


class TestPluginError:
    """Test PluginError."""

    def test_inheritance(self):
        """Test PluginError inherits from TraigentError."""
        error = PluginError("Plugin loading failed")

        assert isinstance(error, TraigentError)


class TestStorageError:
    """Test StorageError."""

    def test_inheritance(self):
        """Test StorageError inherits from TraigentError."""
        error = StorageError("Storage operation failed")

        assert isinstance(error, TraigentError)


class TestTraigentStorageError:
    """Test TraigentStorageError."""

    def test_inheritance(self):
        """Test TraigentStorageError inherits from StorageError."""
        error = TraigentStorageError("Edge Analytics storage failed")

        assert isinstance(error, StorageError)
        assert isinstance(error, TraigentError)


class TestServiceError:
    """Test ServiceError."""

    def test_create_basic_service_error(self):
        """Test creating basic ServiceError."""
        error = ServiceError("Service failed")

        assert error.message == "Service failed"
        assert error.service_name is None
        assert error.endpoint is None
        assert error.status_code is None

    def test_create_with_service_name(self):
        """Test ServiceError with service name."""
        error = ServiceError("Failed", service_name="OptiGen Backend")

        assert error.service_name == "OptiGen Backend"

    def test_create_with_endpoint(self):
        """Test ServiceError with endpoint."""
        error = ServiceError("Failed", endpoint="/api/v1/experiments")

        assert error.endpoint == "/api/v1/experiments"

    def test_create_with_status_code(self):
        """Test ServiceError with HTTP status code."""
        error = ServiceError("Service unavailable", status_code=503)

        assert error.status_code == 503

    def test_create_with_all_fields(self):
        """Test ServiceError with all fields."""
        details = {"retry_after": 60}
        error = ServiceError(
            "Complete service error",
            service_name="API Gateway",
            endpoint="/health",
            status_code=503,
            details=details,
        )

        assert error.service_name == "API Gateway"
        assert error.endpoint == "/health"
        assert error.status_code == 503
        assert error.details["retry_after"] == 60


class TestTraigentConnectionError:
    """Test TraigentConnectionError."""

    def test_inheritance(self):
        """Test TraigentConnectionError inherits from ServiceError."""
        error = TraigentConnectionError("Connection failed")

        assert isinstance(error, ServiceError)
        assert isinstance(error, TraigentError)


class TestServiceUnavailableError:
    """Test ServiceUnavailableError."""

    def test_inheritance(self):
        """Test ServiceUnavailableError inherits from ServiceError."""
        error = ServiceUnavailableError("Service temporarily unavailable")

        assert isinstance(error, ServiceError)


class TestQuotaExceededError:
    """Test QuotaExceededError."""

    def test_inheritance(self):
        """Test QuotaExceededError inherits from ServiceError."""
        error = QuotaExceededError("Rate limit exceeded")

        assert isinstance(error, ServiceError)

    def test_with_quota_details(self):
        """Test QuotaExceededError with quota information."""
        details = {"limit": 1000, "used": 1001, "reset_time": "2024-01-01T00:00:00Z"}
        error = QuotaExceededError("Quota exceeded", details=details)

        assert error.details["limit"] == 1000
        assert error.details["used"] == 1001


class TestSessionError:
    """Test SessionError."""

    def test_inheritance(self):
        """Test SessionError inherits from TraigentError."""
        error = SessionError("Session expired")

        assert isinstance(error, TraigentError)


class TestAgentExecutionError:
    """Test AgentExecutionError."""

    def test_inheritance(self):
        """Test AgentExecutionError inherits from TraigentError."""
        error = AgentExecutionError("Agent execution failed")

        assert isinstance(error, TraigentError)


class TestPlatformCapabilityError:
    """Test PlatformCapabilityError."""

    def test_inheritance(self):
        """Test PlatformCapabilityError inherits from TraigentError."""
        error = PlatformCapabilityError("Platform doesn't support feature")

        assert isinstance(error, TraigentError)


class TestRetryError:
    """Test RetryError."""

    def test_inheritance(self):
        """Test RetryError inherits from TraigentError."""
        error = RetryError("Retry failed")

        assert isinstance(error, TraigentError)


class TestTraigentTimeoutError:
    """Test TraigentTimeoutError."""

    def test_inheritance(self):
        """Test TraigentTimeoutError inherits from TraigentError."""
        error = TraigentTimeoutError("Operation timed out")

        assert isinstance(error, TraigentError)

    def test_with_timeout_details(self):
        """Test TraigentTimeoutError with timeout information."""
        details = {"timeout_seconds": 30, "elapsed": 35}
        error = TraigentTimeoutError("Timeout", details=details)

        assert error.details["timeout_seconds"] == 30


class TestTrialPrunedError:
    """Test TrialPrunedError."""

    def test_create_basic_pruned_error(self):
        """Test creating basic TrialPrunedError."""
        error = TrialPrunedError()

        assert error.message == "Trial pruned"
        assert error.step is None

    def test_create_with_custom_message(self):
        """Test TrialPrunedError with custom message."""
        error = TrialPrunedError("Early stopping triggered")

        assert error.message == "Early stopping triggered"

    def test_create_with_step(self):
        """Test TrialPrunedError with step number."""
        error = TrialPrunedError("Pruned at step 50", step=50)

        assert error.message == "Pruned at step 50"
        assert error.step == 50

    def test_inheritance(self):
        """Test TrialPrunedError inherits from TraigentError."""
        error = TrialPrunedError()

        assert isinstance(error, TraigentError)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_traigent_error(self):
        """Test that all custom exceptions inherit from TraigentError."""
        exceptions = [
            ConfigurationError("test"),
            ValidationError("test"),
            ClientError("test"),
            AuthenticationError("test"),
            InvocationError("test"),
            EvaluationError("test"),
            OptimizationError("test"),
            PluginError("test"),
            StorageError("test"),
            ServiceError("test"),
            SessionError("test"),
            AgentExecutionError("test"),
            PlatformCapabilityError("test"),
            RetryError("test"),
            TraigentTimeoutError("test"),
            TrialPrunedError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, TraigentError)
            assert isinstance(exc, Exception)

    def test_validation_hierarchy(self):
        """Test validation error hierarchy."""
        errors = [
            TraigentValidationError("test"),
            JWTValidationError("test"),
            DatasetValidationError("test"),
            ObjectiveValidationError("test"),
        ]

        for error in errors:
            assert isinstance(error, ValidationError)
            assert isinstance(error, TraigentError)

    def test_configuration_hierarchy(self):
        """Test configuration error hierarchy."""
        error = ConfigurationSpaceError("test")

        assert isinstance(error, ConfigurationError)
        assert isinstance(error, TraigentError)

    def test_client_hierarchy(self):
        """Test client error hierarchy."""
        error = StandardizedClientError("test")

        assert isinstance(error, ClientError)
        assert isinstance(error, TraigentError)

    def test_service_hierarchy(self):
        """Test service error hierarchy."""
        errors = [
            TraigentConnectionError("test"),
            ServiceUnavailableError("test"),
            QuotaExceededError("test"),
        ]

        for error in errors:
            assert isinstance(error, ServiceError)
            assert isinstance(error, TraigentError)

    def test_storage_hierarchy(self):
        """Test storage error hierarchy."""
        error = TraigentStorageError("test")

        assert isinstance(error, StorageError)
        assert isinstance(error, TraigentError)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_message(self):
        """Test exception with empty message."""
        error = TraigentError("")
        assert error.message == ""

    def test_very_long_message(self):
        """Test exception with very long message."""
        long_message = "Error " * 1000
        error = TraigentError(long_message)
        assert len(error.message) > 5000

    def test_unicode_in_message(self):
        """Test exception with Unicode characters."""
        error = TraigentError("Error: 失败 🚨 spëcial")
        assert "失败" in error.message
        assert "🚨" in error.message

    def test_nested_details_dict(self):
        """Test exception with nested details dictionary."""
        details = {"level1": {"level2": {"level3": "deep_value"}}}
        error = TraigentError("Nested", details=details)
        assert error.details["level1"]["level2"]["level3"] == "deep_value"

    def test_none_values_in_details(self):
        """Test exception with None values in details."""
        details = {"key1": None, "key2": "value"}
        error = TraigentError("Test", details=details)
        assert error.details["key1"] is None
        assert error.details["key2"] == "value"

    def test_exception_chaining(self):
        """Test exception chaining with original error."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise InvocationError("Wrapped", original_error=e) from e
        except InvocationError as exc:
            assert isinstance(exc.original_error, ValueError)
            assert "Original error" in str(exc.original_error)
