"""
Intelligent Error Handler for Traigent SDK

Provides helpful error messages with actionable fixes and documentation links.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-ORCH-LIFECYCLE FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-ORCH-003 REQ-CLOUD-009 REQ-SEC-010 SYNC-OptimizationFlow SYNC-CloudHybrid

import re
from collections.abc import Callable
from typing import Any

_PERMISSION_DENIED = "Permission denied"

_NO_MODULE_NAMED = "No module named"


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


_SAFE_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _classify_credential_error(error: BaseException) -> str:
    """Map a credential failure to a closed category without echoing its text."""
    text = str(error).lower()
    if any(t in text for t in ("permission", "forbidden", "not authorized", "403")):
        return "permission"
    if any(
        t in text
        for t in ("missing", "not set", "no api key", "must be set", "required")
    ):
        return "missing-key"
    if any(
        t in text
        for t in ("invalid", "incorrect", "authentication", "unauthorized", "401")
    ):
        return "auth"
    return "unknown"


class APIKeyError(TraigentError):
    """Error for missing or invalid API keys.

    Pass ``key_name`` for the classic "add KEY to your .env" form, or pass
    ``message`` (and optionally ``fix``) for a fully formed explanation. A
    message is never treated as a key name (Traigent#2417).
    """

    def __init__(
        self,
        key_name: str | None = None,
        *,
        message: str | None = None,
        fix: str | None = None,
    ) -> None:
        if message is None:
            name = key_name or "API key"
            message = f"Missing or invalid API key: {name}"
            if fix is None:
                fix = f"Add {name} to your .env file or set as environment variable"
        super().__init__(
            message=message,
            fix=fix,
            docs_link="https://github.com/Traigent/Traigent#troubleshooting",
        )

    @classmethod
    def from_provider_error(cls, error: BaseException) -> "APIKeyError":
        """Build the error for a credential failure raised by user code."""
        from traigent.utils.env_config import is_mock_llm

        # Never interpolate the exception text: it can carry a rejected key,
        # prompt text or a file path. Only the exception class name and a
        # closed category derived from the text are reported; the original
        # exception stays chained (``raise ... from error``) for local debugging.
        category = _classify_credential_error(error)
        class_name = type(error).__name__
        if not _SAFE_CLASS_NAME.fullmatch(class_name):
            class_name = "Exception"
        message = (
            "API key error detected "
            f"(category: {category}, error type: {class_name}). "
            "Details are withheld; see the chained exception for the original error."
        )
        if is_mock_llm():
            fix = (
                "Mock mode (TRAIGENT_MOCK_LLM) is already on, but this client checks "
                "for a key when it is constructed, before mock interception can "
                "run. Set a placeholder key (for example OPENAI_API_KEY=mock-key), "
                "route the call through LiteLLM, or stub the client."
            )
        else:
            fix = (
                "Set the required API key environment variable, or use "
                "TRAIGENT_MOCK_LLM=true for testing."
            )
        return cls(message=message, fix=fix)


class ConfigurationError(TraigentError):
    """Error for invalid configuration."""

    def __init__(self, param: str, issue: str) -> None:
        super().__init__(
            message=f"Invalid configuration for '{param}': {issue}",
            fix="Check the configuration_space parameter in @traigent.optimize()",
            docs_link="https://github.com/Traigent/Traigent#using-it-in-your-own-code",
        )


class EvaluationError(TraigentError):
    """Error during evaluation."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Evaluation failed: {reason}",
            fix="Check your evaluation dataset format and custom evaluator (if used)",
            docs_link="https://github.com/Traigent/Traigent#evaluation",
        )


class OfflineModeError(TraigentError):
    """Raised when a backend request is attempted while offline mode is enabled.

    ``TRAIGENT_OFFLINE_MODE=true`` is an explicit "no backend egress" switch, so
    standalone backend clients fail closed (raise) rather than silently sending
    data off-box.
    """

    def __init__(self, operation: str = "This backend request") -> None:
        super().__init__(
            message=(
                f"{operation} was blocked because TRAIGENT_OFFLINE_MODE is enabled "
                "— no data is sent to the Traigent backend."
            ),
            fix="Unset TRAIGENT_OFFLINE_MODE (or set it to 'false') to allow backend communication.",
            docs_link="https://github.com/Traigent/Traigent#optimization-routing",
        )


_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB = (
    "https://github.com/Traigent/Traigent#troubleshooting"
)


class ErrorHandler:
    """Intelligent error handler that provides helpful fixes."""

    # Common import errors and their fixes
    IMPORT_FIXES = {
        "langchain": "pip install langchain",
        "langchain_openai": "pip install langchain-openai",
        "openai": "pip install openai",
        "dotenv": "pip install python-dotenv",
        "numpy": "pip install numpy",
        "pandas": "pip install pandas",
        "traigent": "pip install -e . (from project root)",
    }

    # Common error patterns and their solutions
    ERROR_PATTERNS = {
        _NO_MODULE_NAMED: {
            "check": lambda e: _NO_MODULE_NAMED in str(e),
            "handler": "handle_import_error",
        },
        "API key": {
            "check": lambda e: any(
                k in str(e).lower() for k in ["api key", "api_key", "authentication"]
            ),
            "handler": "handle_api_key_error",
        },
        "Connection refused": {
            "check": lambda e: (
                "Connection refused" in str(e) or "Failed to connect" in str(e)
            ),
            "handler": "handle_connection_error",
        },
        _PERMISSION_DENIED: {
            "check": lambda e: _PERMISSION_DENIED in str(e),
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
        if _NO_MODULE_NAMED in error_str:
            parts = error_str.split("'")
            if len(parts) >= 2:
                module_name = parts[1].split(".")[0]

                if module_name in cls.IMPORT_FIXES:
                    raise DependencyError(module_name) from error
                else:
                    raise TraigentError(
                        message=f"Cannot import {module_name}",
                        fix=f"Try: pip install {module_name}",
                        docs_link="https://github.com/Traigent/Traigent#installation",
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
            raise APIKeyError("TRAIGENT_API_KEY") from error
        else:
            raise TraigentError(
                message="API authentication failed",
                fix="Check your API keys in .env file",
                docs_link=_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB,
            ) from error

    @classmethod
    def handle_connection_error(cls, error: Exception) -> None:
        """Handle connection errors."""
        error_str = str(error)

        if (
            re.search(r"\b(?:localhost|127\.0\.0\.1)(?::\d+)?\b", error_str)
            or "backend" in error_str.lower()
        ):
            raise TraigentError(
                message="Cannot connect to Traigent backend",
                fix="1. Start your Traigent backend service\n"
                "   2. Or use offline mode: TRAIGENT_OFFLINE_MODE=true",
                docs_link=_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB,
            ) from error
        else:
            raise TraigentError(
                message="Connection failed",
                fix="Check your internet connection and firewall settings",
                docs_link=_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB,
            ) from error

    @classmethod
    def handle_permission_error(cls, error: Exception) -> None:
        """Handle permission errors."""
        raise TraigentError(
            message=_PERMISSION_DENIED,
            fix="1. Check file permissions\n"
            "   2. Run in virtual environment\n"
            "   3. Use --user flag: pip install --user",
            docs_link=_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB,
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
            docs_link=_HTTPS_GITHUB_COM_TRAIGENT_TRAIGENT_TROUB,
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
