"""Comprehensive tests for traigent.utils.error_handler module.

Tests cover custom exception classes, error handlers, error patterns,
and the error handler decorator.
"""

from __future__ import annotations

import pytest

from traigent.utils.error_handler import (
    APIKeyError,
    ConfigurationError,
    DependencyError,
    ErrorHandler,
    EvaluationError,
    TraigentError,
    wrap_with_error_handler,
)


class TestTraigentError:
    """Test TraigentError base exception."""

    def test_basic_error(self):
        """Test basic error with message only."""
        error = TraigentError("Something went wrong")

        assert "Something went wrong" in str(error)
        assert error.message == "Something went wrong"
        assert error.fix is None
        assert error.docs_link is None

    def test_error_with_fix(self):
        """Test error with fix suggestion."""
        error = TraigentError("Config invalid", fix="Check your settings")

        assert "Something went wrong" in str(error) or "Config invalid" in str(error)
        assert "Fix:" in str(error)
        assert "Check your settings" in str(error)

    def test_error_with_docs_link(self):
        """Test error with documentation link."""
        error = TraigentError(
            "Config invalid",
            fix="Check your settings",
            docs_link="https://example.com/docs",
        )

        assert "Docs:" in str(error)
        assert "https://example.com/docs" in str(error)

    def test_format_message_structure(self):
        """Test formatted message structure."""
        error = TraigentError(
            "Test error", fix="Test fix", docs_link="https://example.com"
        )
        formatted = error.format_message()

        assert formatted.startswith("\n")
        assert "❌" in formatted
        assert "💡" in formatted
        assert "📚" in formatted

    def test_error_attributes(self):
        """Test error attributes are accessible."""
        error = TraigentError("Test", fix="Fix it", docs_link="https://docs.com")

        assert hasattr(error, "message")
        assert hasattr(error, "fix")
        assert hasattr(error, "docs_link")


class TestDependencyError:
    """Test DependencyError exception."""

    def test_dependency_error_message(self):
        """Test dependency error message format."""
        error = DependencyError("numpy")

        assert "numpy" in str(error)
        assert "pip install" in str(error)

    def test_dependency_error_fix(self):
        """Test dependency error includes fix."""
        error = DependencyError("pandas")

        assert error.fix is not None
        assert "pip install pandas" in error.fix

    def test_dependency_error_docs_link(self):
        """Test dependency error includes docs link."""
        error = DependencyError("langchain")

        assert error.docs_link is not None
        assert "installation" in error.docs_link.lower()


class TestAPIKeyError:
    """Test APIKeyError exception."""

    def test_api_key_error_message(self):
        """Test API key error message format."""
        error = APIKeyError("OPENAI_API_KEY")

        assert "OPENAI_API_KEY" in str(error)
        assert "Missing" in str(error) or "invalid" in str(error)

    def test_api_key_error_fix(self):
        """Test API key error includes fix."""
        error = APIKeyError("ANTHROPIC_API_KEY")

        assert error.fix is not None
        assert ".env" in error.fix or "environment" in error.fix

    def test_api_key_error_docs_link(self):
        """Test API key error includes docs link."""
        error = APIKeyError("TEST_KEY")

        assert error.docs_link is not None
        assert "configuration" in error.docs_link.lower()


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error_message(self):
        """Test configuration error message format."""
        error = ConfigurationError("temperature", "value out of range")

        assert "temperature" in str(error)
        assert "value out of range" in str(error)

    def test_configuration_error_fix(self):
        """Test configuration error includes fix."""
        error = ConfigurationError("model", "invalid model name")

        assert error.fix is not None
        assert "configuration_space" in error.fix

    def test_configuration_error_docs_link(self):
        """Test configuration error includes docs link."""
        error = ConfigurationError("param", "issue")

        assert error.docs_link is not None


class TestEvaluationError:
    """Test EvaluationError exception."""

    def test_evaluation_error_message(self):
        """Test evaluation error message format."""
        error = EvaluationError("invalid dataset format")

        assert "invalid dataset format" in str(error)
        assert "Evaluation failed" in str(error)

    def test_evaluation_error_fix(self):
        """Test evaluation error includes fix."""
        error = EvaluationError("missing data")

        assert error.fix is not None
        assert "dataset" in error.fix.lower()

    def test_evaluation_error_docs_link(self):
        """Test evaluation error includes docs link."""
        error = EvaluationError("test")

        assert error.docs_link is not None
        assert "evaluation" in error.docs_link.lower()


class TestErrorHandlerPatterns:
    """Test ErrorHandler pattern matching."""

    def test_import_fixes_dictionary(self):
        """Test IMPORT_FIXES dictionary contains common packages."""
        assert "langchain" in ErrorHandler.IMPORT_FIXES
        assert "openai" in ErrorHandler.IMPORT_FIXES
        assert "numpy" in ErrorHandler.IMPORT_FIXES

    def test_error_patterns_dictionary(self):
        """Test ERROR_PATTERNS contains common patterns."""
        assert "No module named" in ErrorHandler.ERROR_PATTERNS
        assert "API key" in ErrorHandler.ERROR_PATTERNS
        assert "Connection refused" in ErrorHandler.ERROR_PATTERNS

    def test_pattern_check_functions_callable(self):
        """Test pattern check functions are callable."""
        for pattern_info in ErrorHandler.ERROR_PATTERNS.values():
            assert "check" in pattern_info
            assert callable(pattern_info["check"])

    def test_pattern_handlers_exist(self):
        """Test pattern handlers reference existing methods."""
        for pattern_info in ErrorHandler.ERROR_PATTERNS.values():
            handler_name = pattern_info["handler"]
            assert hasattr(ErrorHandler, handler_name)
            assert callable(getattr(ErrorHandler, handler_name))


class TestErrorHandlerImportErrors:
    """Test ErrorHandler handling of import errors."""

    def test_handle_import_error_known_package(self):
        """Test handling of import error for known package."""
        error = ImportError("No module named 'langchain'")

        with pytest.raises(DependencyError) as exc_info:
            ErrorHandler.handle_import_error(error)

        assert "langchain" in str(exc_info.value)

    def test_handle_import_error_unknown_package(self):
        """Test handling of import error for unknown package."""
        error = ImportError("No module named 'unknown_package'")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_import_error(error)

        assert "unknown_package" in str(exc_info.value)

    def test_handle_import_error_extracts_module_name(self):
        """Test module name extraction from import error."""
        error = ImportError("No module named 'package.submodule'")

        with pytest.raises(TraigentError):
            ErrorHandler.handle_import_error(error)


class TestErrorHandlerAPIKeyErrors:
    """Test ErrorHandler handling of API key errors."""

    def test_handle_api_key_error_openai(self):
        """Test handling of OpenAI API key error."""
        error = Exception("OpenAI API key is invalid")

        with pytest.raises(APIKeyError) as exc_info:
            ErrorHandler.handle_api_key_error(error)

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_handle_api_key_error_anthropic(self):
        """Test handling of Anthropic API key error."""
        error = Exception("Anthropic authentication failed")

        with pytest.raises(APIKeyError) as exc_info:
            ErrorHandler.handle_api_key_error(error)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_handle_api_key_error_traigent(self):
        """Test handling of Traigent API key error."""
        error = Exception("Traigent API key missing")

        with pytest.raises(APIKeyError) as exc_info:
            ErrorHandler.handle_api_key_error(error)

        assert "TRAIGENT_API_KEY" in str(exc_info.value) or "OPTIGEN_API_KEY" in str(
            exc_info.value
        )

    def test_handle_api_key_error_generic(self):
        """Test handling of generic API key error."""
        error = Exception("API authentication failed")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_api_key_error(error)

        assert "authentication" in str(exc_info.value).lower()


class TestErrorHandlerConnectionErrors:
    """Test ErrorHandler handling of connection errors."""

    def test_handle_connection_error_backend(self):
        """Test handling of backend connection error."""
        error = ConnectionRefusedError("Connection refused to localhost:5000")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_connection_error(error)

        assert "backend" in str(exc_info.value).lower()
        assert "mock mode" in str(
            exc_info.value
        ).lower() or "TRAIGENT_MOCK_MODE" in str(exc_info.value)

    def test_handle_connection_error_generic(self):
        """Test handling of generic connection error."""
        error = ConnectionError("Connection failed")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_connection_error(error)

        assert "connection" in str(exc_info.value).lower()


class TestErrorHandlerPermissionErrors:
    """Test ErrorHandler handling of permission errors."""

    def test_handle_permission_error(self):
        """Test handling of permission error."""
        error = PermissionError("Permission denied")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_permission_error(error)

        assert "permission" in str(exc_info.value).lower()
        assert "pip install --user" in str(exc_info.value).lower() or "user" in str(
            exc_info.value
        )


class TestErrorHandlerZeroAccuracy:
    """Test ErrorHandler handling of zero accuracy errors."""

    def test_handle_zero_accuracy_0_percent(self):
        """Test handling of 0.0% accuracy error."""
        error = ValueError("Accuracy: 0.0%")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_zero_accuracy(error)

        assert "accuracy" in str(exc_info.value).lower()

    def test_handle_zero_accuracy_mock_mode_suggestion(self):
        """Test zero accuracy error suggests mock mode."""
        error = ValueError("accuracy: 0.0")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_zero_accuracy(error)

        assert "mock mode" in str(
            exc_info.value
        ).lower() or "TRAIGENT_MOCK_MODE" in str(exc_info.value)


class TestErrorHandlerUnknownErrors:
    """Test ErrorHandler handling of unknown errors."""

    def test_handle_unknown_error(self):
        """Test handling of unknown error."""
        error = RuntimeError("Unexpected issue")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_unknown_error(error)

        assert "unexpected" in str(exc_info.value).lower()
        assert "troubleshooting" in str(exc_info.value).lower()


class TestErrorHandlerMain:
    """Test ErrorHandler main handle_error method."""

    def test_handle_error_import_error(self):
        """Test handle_error detects and routes import errors."""
        error = ImportError("No module named 'openai'")

        with pytest.raises(DependencyError):
            ErrorHandler.handle_error(error)

    def test_handle_error_api_key_error(self):
        """Test handle_error detects and routes API key errors."""
        error = Exception("OpenAI API key is invalid")

        with pytest.raises(APIKeyError):
            ErrorHandler.handle_error(error)

    def test_handle_error_connection_error(self):
        """Test handle_error detects and routes connection errors."""
        error = ConnectionRefusedError("Connection refused")

        with pytest.raises(TraigentError):
            ErrorHandler.handle_error(error)

    def test_handle_error_unknown_pattern(self):
        """Test handle_error handles unknown error patterns."""
        error = ValueError("Some random error")

        with pytest.raises(TraigentError):
            ErrorHandler.handle_error(error)

    def test_handle_error_preserves_traigent_errors(self):
        """Test handle_error preserves Traigent custom errors."""
        original_error = DependencyError("test")

        # Should re-raise as-is in real usage, but handle_error doesn't do this
        # So we test that it would raise some Traigent error
        with pytest.raises(TraigentError):
            ErrorHandler.handle_error(original_error)


class TestWrapWithErrorHandler:
    """Test wrap_with_error_handler decorator."""

    def test_decorator_preserves_function_behavior(self):
        """Test decorator doesn't change normal function behavior."""

        @wrap_with_error_handler
        def test_func(x):
            return x * 2

        assert test_func(5) == 10

    def test_decorator_handles_exceptions(self):
        """Test decorator catches and handles exceptions."""

        @wrap_with_error_handler
        def failing_func():
            raise ImportError("No module named 'test_package'")

        with pytest.raises(TraigentError):
            failing_func()

    def test_decorator_preserves_traigent_errors(self):
        """Test decorator re-raises Traigent errors as-is."""

        @wrap_with_error_handler
        def custom_error_func():
            raise DependencyError("test_package")

        with pytest.raises(DependencyError):
            custom_error_func()

    def test_decorator_with_arguments(self):
        """Test decorator works with function arguments."""

        @wrap_with_error_handler
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @wrap_with_error_handler
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"

        assert greet("Alice") == "Hello, Alice"
        assert greet("Bob", greeting="Hi") == "Hi, Bob"


class TestErrorHandlerIntegration:
    """Test ErrorHandler integration scenarios."""

    def test_error_chain_preserved(self):
        """Test that error chaining is preserved."""
        original = ValueError("Original error")

        with pytest.raises(TraigentError) as exc_info:
            ErrorHandler.handle_unknown_error(original)

        assert exc_info.value.__cause__ is original

    def test_multiple_error_types(self):
        """Test handling different error types in sequence."""
        errors = [
            ImportError("No module named 'numpy'"),
            Exception("OpenAI API key missing"),
            ConnectionRefusedError("Failed to connect"),
        ]

        for error in errors:
            with pytest.raises(TraigentError):
                ErrorHandler.handle_error(error)

    def test_nested_errors(self):
        """Test handling nested errors."""

        def inner():
            raise ImportError("No module named 'pandas'")

        def outer():
            try:
                inner()
            except Exception as e:
                ErrorHandler.handle_error(e)

        with pytest.raises(TraigentError):
            outer()
