"""Vendor error classification and interactive pause prompts.

Provides error classification for vendor/provider errors (rate limits, quota
exhaustion, service unavailability) and an injectable prompt adapter for
interactive pause-and-resume during optimization runs.

The TerminalPausePrompt is the default implementation, using ``input()`` for
interactive terminals and auto-stopping in non-interactive environments (CI,
pipes, scripts).
"""

from __future__ import annotations

import re
import sqlite3
import sys
from enum import Enum
from typing import Protocol

from traigent.utils.console import _safe_print
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class VendorErrorCategory(Enum):
    """Categories of vendor errors that can trigger interactive pause."""

    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    AUTHENTICATION = "authentication"
    SERVICE_UNAVAILABLE = "service_unavailable"


class PausePromptAdapter(Protocol):
    """Protocol for interactive pause prompts. Injectable for testing."""

    def prompt_vendor_pause(
        self, error: Exception, category: VendorErrorCategory
    ) -> str:
        """Prompt user after vendor error. Returns 'resume' or 'stop'."""
        ...

    def prompt_budget_pause(self, accumulated: float, limit: float) -> str:
        """Prompt user after budget limit. Returns 'raise:<amount>' or 'stop'."""
        ...


_VENDOR_ERROR_PATTERNS: list[tuple[str, VendorErrorCategory]] = [
    ("rate limit", VendorErrorCategory.RATE_LIMIT),
    ("rate_limit", VendorErrorCategory.RATE_LIMIT),
    ("ratelimit", VendorErrorCategory.RATE_LIMIT),
    ("status 429", VendorErrorCategory.RATE_LIMIT),
    ("code: 429", VendorErrorCategory.RATE_LIMIT),
    ("code 429", VendorErrorCategory.RATE_LIMIT),
    ("http 429", VendorErrorCategory.RATE_LIMIT),
    ("error 429", VendorErrorCategory.RATE_LIMIT),
    ("429 too many", VendorErrorCategory.RATE_LIMIT),
    ("too many requests", VendorErrorCategory.RATE_LIMIT),
    ("status 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("code: 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("code 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("http 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("error 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("402 payment required", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient_funds", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient funds", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient credits", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("billing hard limit", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("exceeded your current billing", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("payment required", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("quota", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("quota_exceeded", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("insufficient_quota", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("status 401", VendorErrorCategory.AUTHENTICATION),
    ("code: 401", VendorErrorCategory.AUTHENTICATION),
    ("code 401", VendorErrorCategory.AUTHENTICATION),
    ("http 401", VendorErrorCategory.AUTHENTICATION),
    ("error 401", VendorErrorCategory.AUTHENTICATION),
    ("401 unauthorized", VendorErrorCategory.AUTHENTICATION),
    ("status 403", VendorErrorCategory.AUTHENTICATION),
    ("code: 403", VendorErrorCategory.AUTHENTICATION),
    ("code 403", VendorErrorCategory.AUTHENTICATION),
    ("http 403", VendorErrorCategory.AUTHENTICATION),
    ("error 403", VendorErrorCategory.AUTHENTICATION),
    ("403 forbidden", VendorErrorCategory.AUTHENTICATION),
    ("unauthorized", VendorErrorCategory.AUTHENTICATION),
    ("unauthorised", VendorErrorCategory.AUTHENTICATION),
    ("forbidden", VendorErrorCategory.AUTHENTICATION),
    ("invalid api key", VendorErrorCategory.AUTHENTICATION),
    ("invalid_api_key", VendorErrorCategory.AUTHENTICATION),
    ("api key invalid", VendorErrorCategory.AUTHENTICATION),
    ("authentication", VendorErrorCategory.AUTHENTICATION),
    ("auth error", VendorErrorCategory.AUTHENTICATION),
    ("permission denied", VendorErrorCategory.AUTHENTICATION),
    ("service unavailable", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("status 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("code: 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("code 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("http 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("error 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("503 service", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("overloaded", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("server_error", VendorErrorCategory.SERVICE_UNAVAILABLE),
]


# Generic single words that also appear in ordinary identifiers and data
# ("quota_used", "forbidden_words", "authentication" table). They only count
# as vendor wording when not glued to identifier characters (#2843).
_BARE_WORD_PATTERNS = frozenset(
    {
        "quota",
        "unauthorized",
        "unauthorised",
        "forbidden",
        "authentication",
        "overloaded",
    }
)
_BARE_WORD_REGEX = {
    word: re.compile(rf"(?<![a-z0-9_]){word}(?![a-z0-9_])")
    for word in _BARE_WORD_PATTERNS
}

#: Exceptions raised by the user's own code or the standard library. Their
#: message is domain text, not a provider verdict, so it is never pattern
#: matched for AUTH/QUOTA wording. Provider SDK errors (openai, anthropic, ...)
#: and httpx/requests status errors are not in this set.
_LOCAL_ERROR_TYPE_NAMES = (
    "OSError|IOError|PermissionError|FileNotFoundError|FileExistsError|"
    "IsADirectoryError|NotADirectoryError|KeyError|IndexError|LookupError|"
    "AttributeError|NameError|TypeError|ValueError|AssertionError|ImportError|"
    "ModuleNotFoundError|ZeroDivisionError|UnicodeError|UnicodeDecodeError|"
    "UnicodeEncodeError|OperationalError|ProgrammingError|IntegrityError|"
    "DatabaseError|InterfaceError|JSONDecodeError|StopIteration|RecursionError|"
    "NotImplementedError"
)
_LOCAL_ERROR_TEXT_RE = re.compile(
    rf"(?:^|[\s:])(?:{_LOCAL_ERROR_TYPE_NAMES}):|\[errno \d+\]", re.IGNORECASE
)


_HTTP_LIBRARY_ROOTS = frozenset({"requests", "urllib3", "httpx", "aiohttp", "httpcore"})


def _is_local_exception(exc: BaseException) -> bool:
    """True for filesystem/lookup/SQL/arithmetic errors (not provider SDK errors).

    ``requests`` exceptions subclass ``OSError`` but are HTTP failures, so HTTP
    client libraries are excluded.
    """
    if type(exc).__module__.split(".")[0] in _HTTP_LIBRARY_ROOTS:
        return False
    return isinstance(exc, (OSError, LookupError, ArithmeticError, sqlite3.Error))


def http_status_of(exc: BaseException) -> int | None:
    """Structured HTTP status carried by ``exc`` or its ``response``, if any."""
    response = getattr(exc, "response", None)
    for source in (exc, response):
        if source is None:
            continue
        for attr in ("status_code", "status", "http_status", "code"):
            value = getattr(source, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                if 100 <= value <= 599:
                    return value
    return None


_STATUS_CATEGORIES: dict[int, VendorErrorCategory] = {
    401: VendorErrorCategory.AUTHENTICATION,
    403: VendorErrorCategory.AUTHENTICATION,
    402: VendorErrorCategory.INSUFFICIENT_FUNDS,
    429: VendorErrorCategory.RATE_LIMIT,
    503: VendorErrorCategory.SERVICE_UNAVAILABLE,
}


def _matches_pattern(pattern: str, msg: str) -> bool:
    regex = _BARE_WORD_REGEX.get(pattern)
    if regex is not None:
        return regex.search(msg) is not None
    return pattern in msg


def classify_vendor_error(exc: Exception) -> VendorErrorCategory | None:
    """Classify an exception as a vendor-pausable error category.

    Order: known Traigent exception types, then a structured HTTP status, then
    (for non-local errors only) message pattern matching. Builtin/stdlib/user
    exceptions and text that carries a local-error marker (``PermissionError:``,
    ``[Errno 13]``, ``OperationalError:``) are never pattern matched, so a
    missing file or a SQL column named ``quota_used`` is not a provider outage.

    Args:
        exc: The exception to classify.

    Returns:
        VendorErrorCategory if the error is vendor-pausable, None otherwise.
    """
    from traigent.utils.exceptions import (
        InsufficientFundsError,
        QuotaExceededError,
        RateLimitError,
        ServiceUnavailableError,
    )

    if isinstance(exc, RateLimitError):
        return VendorErrorCategory.RATE_LIMIT
    if isinstance(exc, InsufficientFundsError):
        return VendorErrorCategory.INSUFFICIENT_FUNDS
    if isinstance(exc, QuotaExceededError):
        return VendorErrorCategory.QUOTA_EXHAUSTED
    if isinstance(exc, ServiceUnavailableError):
        return VendorErrorCategory.SERVICE_UNAVAILABLE

    status = http_status_of(exc)
    if (
        status is not None
        and status in _STATUS_CATEGORIES
        # 401/403 from a non-provider library are the user's own service, not a
        # verdict on the provider key; 402/429/503 are transport-level facts.
        and (status not in (401, 403) or _is_provider_sdk_exception(exc))
    ):
        return _STATUS_CATEGORIES[status]

    if _is_local_exception(exc):
        return None

    # Fallback: pattern-match the error message
    msg = str(exc).lower()
    if _LOCAL_ERROR_TEXT_RE.search(msg):
        return None
    for pattern, category in _VENDOR_ERROR_PATTERNS:
        if _matches_pattern(pattern, msg):
            return category

    return None


#: Top-level packages whose exceptions are provider/vendor verdicts. Message
#: text is only trusted as an auth signal for exceptions raised by these.
_PROVIDER_SDK_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "cohere",
        "mistralai",
        "litellm",
        "groq",
        "ollama",
        "together",
        "boto3",
        "botocore",
        "azure",
        "huggingface_hub",
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_community",
        "traigent",
    }
)
_AUTH_EXCEPTION_NAMES = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "InvalidAPIKeyError",
        "UnauthorizedError",
        "APIKeyError",
        "CredentialError",
    }
)
_PROVIDER_KEY_VARS = (
    r"(?:OPENAI|ANTHROPIC|GOOGLE|GEMINI|COHERE|MISTRAL|GROQ|OPENROUTER|TOGETHER|"
    r"AZURE_OPENAI|HUGGINGFACE|HF|DEEPSEEK|XAI|PERPLEXITY|FIREWORKS)_API_KEY"
)
_MISSING_WORDS = (
    r"(?:not set|must be set|is missing|missing|not found|not configured|unset|"
    r"is required|required|not provided)"
)
#: A provider key env var reported missing, in either word order
#: ("OPENAI_API_KEY is not set" / "missing OPENAI_API_KEY"). Restricted to
#: known provider variables so a domain error quoting some other ``*_API_KEY``
#: token stays a per-example failure.
_MISSING_KEY_RE = re.compile(
    rf"\b{_PROVIDER_KEY_VARS}\b.{{0,60}}?{_MISSING_WORDS}"
    rf"|{_MISSING_WORDS}.{{0,60}}?\b{_PROVIDER_KEY_VARS}\b",
    re.IGNORECASE,
)
_API_KEY_TEXT_TOKENS = ("api key", "api_key", "authentication", "openai_api_key")


def _is_provider_sdk_exception(exc: BaseException) -> bool:
    return type(exc).__module__.split(".")[0] in _PROVIDER_SDK_ROOTS


def is_provider_api_key_failure(exc: BaseException) -> bool:
    """True when ``exc`` is a provider verdict on the API key, not user-code text.

    Decided by structure first: an HTTP 401/403, or an auth exception class
    (``openai.AuthenticationError`` ...). Message wording counts only for
    exceptions raised by a known provider SDK. A SQL error mentioning an
    ``authentication`` table or a ``KeyError('api_key')`` from the user's agent
    is an ordinary per-example failure and must not abort the run (#2842).
    """
    provider_sdk = _is_provider_sdk_exception(exc)
    # A 401/403 from the user's own service (requests/httpx to an internal API,
    # a custom exception with ``.code = 403``) is not a verdict on the provider
    # key: the status is only trusted from provider SDK exceptions.
    if provider_sdk and http_status_of(exc) in (401, 403):
        return True
    if any(base.__name__ in _AUTH_EXCEPTION_NAMES for base in type(exc).__mro__):
        return True
    if _is_local_exception(exc):
        return False
    text = str(exc).lower()
    if provider_sdk:
        return any(token in text for token in _API_KEY_TEXT_TOKENS)
    # A missing provider key surfaces from wrappers/agents as a plain
    # ValueError/RuntimeError naming the env var ("OPENAI_API_KEY is not set").
    # That is a run-level setup error, not per-example domain text, so abort.
    return _MISSING_KEY_RE.search(str(exc)) is not None


def classify_systematic_provider_failure(
    error: Exception | str,
) -> VendorErrorCategory | None:
    """Classify non-transient provider failures that should abort when systematic."""

    exc = error if isinstance(error, Exception) else RuntimeError(str(error))
    category = classify_vendor_error(exc)
    if category in _SYSTEMATIC_FATAL_CATEGORIES:
        return category
    return None


def provider_failure_action_hint(category: VendorErrorCategory | str | None) -> str:
    """Return a user-actionable hint for fatal provider call failures."""

    if isinstance(category, str):
        try:
            category = VendorErrorCategory(category)
        except ValueError:
            category = None
    if category in _SYSTEMATIC_FATAL_CATEGORIES:
        return _CATEGORY_DESCRIPTIONS[category]
    return (
        "Provider calls are failing before evaluation can score outputs. "
        "Check provider credentials, quota, billing, and model access."
    )


_CATEGORY_DESCRIPTIONS: dict[VendorErrorCategory, str] = {
    VendorErrorCategory.RATE_LIMIT: (
        "The LLM provider is rate-limiting requests. "
        "This usually resolves after a short wait."
    ),
    VendorErrorCategory.QUOTA_EXHAUSTED: (
        "The LLM provider quota has been exhausted. "
        "Check your billing dashboard or wait for the quota to reset."
    ),
    VendorErrorCategory.INSUFFICIENT_FUNDS: (
        "Your LLM provider API key has insufficient credits. "
        "Please top up your account at your provider's billing page and retry."
    ),
    VendorErrorCategory.AUTHENTICATION: (
        "The LLM provider rejected the API key or permissions. "
        "Check the configured provider key, auth scope, project, and model access."
    ),
    VendorErrorCategory.SERVICE_UNAVAILABLE: (
        "The LLM provider service is temporarily unavailable. "
        "This is usually a transient issue."
    ),
}

_SYSTEMATIC_FATAL_CATEGORIES = {
    VendorErrorCategory.AUTHENTICATION,
    VendorErrorCategory.INSUFFICIENT_FUNDS,
    VendorErrorCategory.QUOTA_EXHAUSTED,
}


class TerminalPausePrompt:
    """Default terminal-based prompt implementation.

    Uses ``input()`` for interactive prompts. In non-interactive mode
    (``sys.stdin.isatty()`` is False), automatically returns ``"stop"``.
    """

    def prompt_vendor_pause(
        self, error: Exception, category: VendorErrorCategory
    ) -> str:
        """Prompt user to resume or stop after a vendor error.

        Args:
            error: The original vendor exception.
            category: Classified error category.

        Returns:
            ``"resume"`` or ``"stop"``.
        """
        if not sys.stdin.isatty():
            logger.info(
                "Non-interactive mode: auto-stopping after vendor error (%s)",
                category.value,
            )
            return "stop"

        description = _CATEGORY_DESCRIPTIONS.get(category, str(error))
        _safe_print(f"\n⚠️  Vendor error: {category.value}")
        _safe_print(f"   {description}")
        _safe_print(f"   Error: {error}")

        # Insufficient funds is non-recoverable — stop immediately
        if category == VendorErrorCategory.INSUFFICIENT_FUNDS:
            _safe_print()
            _safe_print(
                "Stopping optimization (insufficient funds cannot be resolved by retrying)."
            )
            return "stop"

        _safe_print()
        _safe_print("Options:")
        _safe_print("  [r] Resume optimization")
        _safe_print("  [s] Stop and return partial results")

        try:
            choice = input("\nYour choice (r/s): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _safe_print()
            return "stop"

        if choice in ("r", "resume"):
            return "resume"
        return "stop"

    def prompt_budget_pause(self, accumulated: float, limit: float) -> str:
        """Prompt user to raise budget or stop after budget limit reached.

        Args:
            accumulated: Total cost accumulated so far (USD).
            limit: Current cost limit (USD).

        Returns:
            ``"raise:<new_limit>"`` or ``"stop"``.
        """
        if not sys.stdin.isatty():
            logger.info(
                "Non-interactive mode: auto-stopping at budget limit ($%.2f / $%.2f)",
                accumulated,
                limit,
            )
            return "stop"

        _safe_print(f"\n💰 Budget limit reached: ${accumulated:.2f} / ${limit:.2f}")
        _safe_print()
        _safe_print("Options:")
        _safe_print("  Enter a new limit (e.g. 5.0) to raise and continue")
        _safe_print("  [s] Stop and return partial results")

        try:
            choice = input("\nNew limit or 's' to stop: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _safe_print()
            return "stop"

        if choice in ("s", "stop", ""):
            return "stop"

        try:
            new_limit = float(choice)
            if new_limit <= accumulated:
                _safe_print(
                    f"  New limit must be greater than current spend (${accumulated:.2f})"
                )
                return "stop"
            if not (0 < new_limit < float("inf")):
                _safe_print("  Invalid amount. Stopping.")
                return "stop"
            return f"raise:{new_limit}"
        except ValueError:
            _safe_print("  Invalid input. Stopping.")
            return "stop"
