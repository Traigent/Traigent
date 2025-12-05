"""
Comprehensive tests for error handling patterns across Traigent SDK.
"""

from unittest.mock import patch

import pytest

from traigent.utils.exceptions import (
    AuthenticationError,
    ConfigurationError,
    OptimizationError,
    TraigentError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test custom exception hierarchy."""

    def test_base_exception(self):
        """Test base TraiGent exception."""
        error = TraigentError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test validation error inheritance."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, TraigentError)

    def test_configuration_error(self):
        """Test configuration error inheritance."""
        error = ConfigurationError("Config invalid")
        assert str(error) == "Config invalid"
        assert isinstance(error, TraigentError)

    def test_optimization_error(self):
        """Test optimization error inheritance."""
        error = OptimizationError("Optimization failed")
        assert str(error) == "Optimization failed"
        assert isinstance(error, TraigentError)

    def test_authentication_error(self):
        """Test authentication error inheritance."""
        error = AuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, TraigentError)


class TestExceptionChaining:
    """Test proper exception chaining patterns."""

    def test_parameter_validation_chaining(self):
        """Test exception chaining in parameter validation."""
        from traigent.api.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        # Test that internal ValueErrors are properly chained
        with pytest.raises(ValidationError) as exc_info:
            validator._validate_execution_mode("invalid_mode")

        # Should be a ValidationError with proper message
        assert "Invalid execution_mode" in str(exc_info.value)

    def test_configuration_error_chaining(self):
        """Test exception chaining in configuration building."""
        from traigent.api.config_builder import ConfigurationBuilder
        from traigent.config.types import InjectionMode

        builder = ConfigurationBuilder()

        # Test configuration error with proper chaining
        with pytest.raises(ConfigurationError) as exc_info:
            builder._resolve_injection_mode(InjectionMode.PARAMETER, None)

        assert "config_param must be specified" in str(exc_info.value)

    @patch("traigent.agents.config_mapper.logger")
    def test_config_mapper_error_chaining(self, mock_logger):
        """Test error chaining in config mapper module."""
        # Import here to avoid circular imports in test setup
        from traigent.agents.config_mapper import ConfigurationError as MapperError

        # Create a mock validation function that raises ValueError
        def failing_validation(value):
            raise ValueError("Validation failed for test")

        # Test that the error is properly chained
        try:
            raise MapperError("Test error") from ValueError("Original error")
        except MapperError as e:
            assert str(e) == "Test error"
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"

    def test_platform_error_chaining(self):
        """Test error chaining in platform modules."""
        # Test that platform errors properly chain OpenAI exceptions
        from traigent.utils.exceptions import AgentExecutionError

        # Simulate the pattern used in the actual code
        try:
            original_error = Exception("OpenAI API error")
            raise AgentExecutionError("Agent execution failed") from original_error
        except AgentExecutionError as e:
            assert "Agent execution failed" in str(e)
            assert isinstance(e.__cause__, Exception)
            assert str(e.__cause__) == "OpenAI API error"


class TestErrorRecovery:
    """Test error recovery and fallback patterns."""

    def test_graceful_degradation(self):
        """Test graceful degradation when optional components fail."""
        # This tests the pattern where we continue operation even if
        # non-critical components fail

        def optional_operation():
            raise Exception("Optional component failed")

        def main_operation_with_fallback():
            result = {"status": "success", "optional_data": None}
            try:
                result["optional_data"] = optional_operation()
            except Exception:
                # Graceful degradation - continue without optional data
                pass
            return result

        result = main_operation_with_fallback()
        assert result["status"] == "success"
        assert result["optional_data"] is None

    def test_retry_pattern(self):
        """Test retry patterns with exponential backoff."""
        import time

        class RetryableOperation:
            def __init__(self, max_attempts=3):
                self.attempts = 0
                self.max_attempts = max_attempts

            def execute(self):
                self.attempts += 1
                if self.attempts < self.max_attempts:
                    raise Exception(f"Attempt {self.attempts} failed")
                return f"Success on attempt {self.attempts}"

        def retry_with_backoff(operation, max_retries=3, base_delay=0.1):
            for attempt in range(max_retries):
                try:
                    return operation.execute()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e from None
                    delay = base_delay * (2**attempt)
                    time.sleep(delay)

        operation = RetryableOperation(max_attempts=2)
        result = retry_with_backoff(operation, max_retries=3, base_delay=0.01)
        assert result == "Success on attempt 2"

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for external service calls."""
        import time

        class CircuitBreaker:
            def __init__(self, failure_threshold=3, timeout=60):
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            def call(self, func, *args, **kwargs):
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time < self.timeout:
                        raise Exception("Circuit breaker is OPEN")
                    else:
                        self.state = "HALF_OPEN"

                try:
                    result = func(*args, **kwargs)
                    self.reset()
                    return result
                except Exception as e:
                    self.record_failure()
                    raise e

            def record_failure(self):
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

            def reset(self):
                self.failure_count = 0
                self.state = "CLOSED"

        def failing_service():
            raise Exception("Service unavailable")

        def working_service():
            return "Service response"

        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)

        # First failure
        with pytest.raises(Exception):  # noqa: B017
            breaker.call(failing_service)
        assert breaker.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(Exception):  # noqa: B017
            breaker.call(failing_service)
        assert breaker.state == "OPEN"

        # Circuit is open - should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(working_service)


class TestContextualErrorInfo:
    """Test that errors provide sufficient context for debugging."""

    def test_validation_error_context(self):
        """Test that validation errors include parameter context."""
        from traigent.api.parameter_validator import ParameterValidator

        validator = ParameterValidator()

        with pytest.raises(ValidationError) as exc_info:
            validator._validate_execution_mode("invalid_mode")

        error_msg = str(exc_info.value)
        assert "invalid_mode" in error_msg
        assert "execution_mode" in error_msg
        assert (
            "edge_analytics" in error_msg or "cloud" in error_msg
        )  # Valid options mentioned

    def test_configuration_error_context(self):
        """Test that configuration errors include parameter context."""
        from traigent.api.config_builder import ConfigurationBuilder
        from traigent.config.types import InjectionMode

        builder = ConfigurationBuilder()

        with pytest.raises(ConfigurationError) as exc_info:
            builder._resolve_injection_mode(InjectionMode.PARAMETER, None)

        error_msg = str(exc_info.value)
        assert "config_param" in error_msg
        assert "parameter" in error_msg


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling across the system."""

    def test_end_to_end_error_propagation(self):
        """Test error propagation from validation through configuration."""
        from traigent.api.parameter_validator import validate_optimize_parameters

        # This should trigger validation error that propagates properly
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(execution_mode="definitely_invalid_mode")

        # Error should have proper context
        error_msg = str(exc_info.value)
        assert "execution_mode" in error_msg
        assert "definitely_invalid_mode" in error_msg

    def test_removed_parameters_raise_validation_error(self):
        """Removed decorator parameters should raise a validation error."""
        from traigent.api.parameter_validator import validate_optimize_parameters

        with pytest.raises(ValidationError, match="auto_optimize"):
            validate_optimize_parameters(auto_optimize=True)
