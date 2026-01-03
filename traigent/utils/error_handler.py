"""
Intelligent Error Handler for Traigent SDK

Provides helpful error messages with actionable fixes and documentation links.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-ORCH-003 REQ-CLOUD-009 REQ-SEC-010 SYNC-OptimizationFlow SYNC-CloudHybrid

from collections.abc import Callable
from typing import Any


class TraigentError(Exception):
    """Base exception for Traigent errors with helpful messages."""

    def __init__(
        self, message: str, fix: str | None = None, docs_link: str | None = None
    ) -> None:
        self.message = message
        self.fix = fix
        self.docs_link = docs_link
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with fix and docs link."""
        parts = [f"\n❌ {self.message}"]

        if self.fix:
            parts.append(f"\n💡 Fix: {self.fix}")

        if self.docs_link:
            parts.append(f"\n📚 Docs: {self.docs_link}")

        return "\n".join(parts)


class DependencyError(TraigentError):
    """Error for missing dependencies."""

    def __init__(self, package: str) -> None:
        super().__init__(
            message=f"Missing required package: {package}",
            fix=f"Run: pip install {package}",
            docs_link="https://github.com/Traigent/Traigent#installation",
        )


class APIKeyError(TraigentError):
    """Error for missing or invalid API keys."""

    def __init__(self, key_name: str) -> None:
        super().__init__(
            message=f"Missing or invalid API key: {key_name}",
            fix=f"Add {key_name} to your .env file or set as environment variable",
            docs_link="https://github.com/Traigent/Traigent#configuration",
        )


class ConfigurationError(TraigentError):
    """Error for invalid configuration."""

    def __init__(self, param: str, issue: str) -> None:
        super().__init__(
            message=f"Invalid configuration for '{param}': {issue}",
            fix="Check the configuration_space parameter in @traigent.optimize()",
            docs_link="https://github.com/Traigent/Traigent#configuration-space",
        )


class EvaluationError(TraigentError):
    """Error during evaluation."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Evaluation failed: {reason}",
            fix="Check your evaluation dataset format and custom evaluator (if used)",
            docs_link="https://github.com/Traigent/Traigent#evaluation",
        )


class ErrorHandler:
    """Intelligent error handler that provides helpful fixes."""

    # Common import errors and their fixes
    IMPORT_FIXES = {
        "langchain": "pip install langchain",
        "langchain_openai": "pip install langchain-openai",
        "langchain_chroma": "pip install langchain-chroma",
        "openai": "pip install openai",
        "dotenv": "pip install python-dotenv",
        "numpy": "pip install numpy",
        "pandas": "pip install pandas",
        "traigent": "pip install -e . (from project root)",
    }

    # Common error patterns and their solutions
    ERROR_PATTERNS = {
        "No module named": {
            "check": lambda e: "No module named" in str(e),
            "handler": "handle_import_error",
        },
        "API key": {
            "check": lambda e: any(
                k in str(e).lower() for k in ["api key", "api_key", "authentication"]
            ),
            "handler": "handle_api_key_error",
        },
        "Connection refused": {
            "check": lambda e: "Connection refused" in str(e)
            or "Failed to connect" in str(e),
            "handler": "handle_connection_error",
        },
        "Permission denied": {
            "check": lambda e: "Permission denied" in str(e),
            "handler": "handle_permission_error",
        },
        "accuracy = 0.0": {
            "check": lambda e: "0.0%" in str(e) or "accuracy: 0.0" in str(e).lower(),
            "handler": "handle_zero_accuracy",
        },
    }

    @classmethod
    def handle_error(cls, error: Exception) -> None:
        """Handle an error with helpful suggestions."""
        str(error)

        # Check for known error patterns
        for _pattern_name, pattern_info in cls.ERROR_PATTERNS.items():
            check_func = pattern_info["check"]
            if callable(check_func) and check_func(error):
                handler_name = str(pattern_info["handler"])
                handler: Callable[[Exception], None] | None = getattr(
                    cls, handler_name, None
                )
                if handler is not None and callable(handler):
                    handler(error)
                    return

        # Default error handling
        cls.handle_unknown_error(error)

    @classmethod
    def handle_import_error(cls, error: Exception) -> None:
        """Handle import errors with installation suggestions."""
        error_str = str(error)

        # Extract module name from error
        if "No module named" in error_str:
            parts = error_str.split("'")
            if len(parts) >= 2:
                module_name = parts[1].split(".")[0]

                if module_name in cls.IMPORT_FIXES:
                    raise DependencyError(module_name) from error
                else:
                    raise TraigentError(
                        message=f"Cannot import {module_name}",
                        fix=f"Try: pip install {module_name}",
                        docs_link="https://github.com/Traigent/Traigent#dependencies",
                    ) from error

    @classmethod
    def handle_api_key_error(cls, error: Exception) -> None:
        """Handle API key errors."""
        error_str = str(error).lower()

        # Detect which API key is missing
        if "openai" in error_str:
            raise APIKeyError("OPENAI_API_KEY") from error
        elif "anthropic" in error_str:
            raise APIKeyError("ANTHROPIC_API_KEY") from error
        elif "traigent" in error_str:
            raise APIKeyError("TRAIGENT_API_KEY or legacy OPTIGEN_API_KEY") from error
        else:
            raise TraigentError(
                message="API authentication failed",
                fix="Check your API keys in .env file",
                docs_link="https://github.com/Traigent/Traigent#api-keys",
            ) from error

    @classmethod
    def handle_connection_error(cls, error: Exception) -> None:
        """Handle connection errors."""
        error_str = str(error)

        if "localhost:5000" in error_str or "backend" in error_str.lower():
            raise TraigentError(
                message="Cannot connect to Traigent backend",
                fix="1. Start backend: cd backend && python app.py\n"
                "   2. Or use offline mode: TRAIGENT_OFFLINE_MODE=true",
                docs_link="https://github.com/Traigent/Traigent#backend-setup",
            ) from error
        else:
            raise TraigentError(
                message="Connection failed",
                fix="Check your internet connection and firewall settings",
                docs_link="https://github.com/Traigent/Traigent#troubleshooting",
            ) from error

    @classmethod
    def handle_permission_error(cls, error: Exception) -> None:
        """Handle permission errors."""
        raise TraigentError(
            message="Permission denied",
            fix="1. Check file permissions\n"
            "   2. Run in virtual environment\n"
            "   3. Use --user flag: pip install --user",
            docs_link="https://github.com/Traigent/Traigent#permissions",
        ) from error

    @classmethod
    def handle_zero_accuracy(cls, error: Exception) -> None:
        """Handle zero accuracy issues."""
        raise TraigentError(
            message="Optimization showing 0.0% accuracy",
            fix="1. Enable mock mode: TRAIGENT_MOCK_LLM=true\n"
            "   2. Check evaluation dataset format\n"
            "   3. Verify API keys for embedding evaluation\n"
            "   4. Use custom evaluator for specific metrics",
            docs_link="https://github.com/Traigent/Traigent#evaluation",
        ) from error

    @classmethod
    def handle_unknown_error(cls, error: Exception) -> None:
        """Handle unknown errors with general troubleshooting."""
        raise TraigentError(
            message=f"Unexpected error: {str(error)}",
            fix="1. Check the stack trace above\n"
            "   2. Verify all dependencies: python scripts/verify_installation.py\n"
            "   3. Try mock mode: TRAIGENT_MOCK_LLM=true\n"
            "   4. Report issue: https://github.com/Traigent/Traigent/issues",
            docs_link="https://github.com/Traigent/Traigent#troubleshooting",
        ) from error


def wrap_with_error_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to wrap functions with intelligent error handling."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except TraigentError:
            # Re-raise our custom errors as-is
            raise
        except Exception as e:
            # Handle other errors with our intelligent handler
            ErrorHandler.handle_error(e)

    return wrapper
