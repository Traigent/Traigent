"""Provider validation for Traigent SDK.

This module provides provider key validation to ensure API keys are valid
before starting potentially long and expensive optimization runs.

Validation behavior:
- **Known providers** (OpenAI, Anthropic, Google, Mistral, Cohere):
  - Auth errors (401, InvalidAPIKey) → fail-fast with ProviderValidationError
  - Transient errors (rate limit, timeout, network) → warn, allow to proceed
    (key may be valid; service temporarily unavailable)
- **Unknown models**: warn only, do not block (cannot validate without SDK)

Transient errors are intentionally allowed because a momentary rate limit or
network blip during validation doesn't indicate an invalid key. The actual
LLM calls have their own retry logic.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Provider detection patterns (model name -> provider)
# Matches: gpt-*, o1-*, o3-* -> openai
#          claude-*, haiku-*, sonnet-*, opus-* -> anthropic
#          gemini-* -> google
#          mistral-*, codestral-*, pixtral-* -> mistral
#          command-* -> cohere
_PROVIDER_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai": re.compile(r"^(gpt-|o[13]-|text-davinci|text-embedding|whisper-)"),
    "anthropic": re.compile(r"^(claude-|haiku-|sonnet-|opus-)"),
    "google": re.compile(r"^(gemini-|models/gemini-)"),
    "mistral": re.compile(r"^(mistral-|codestral-|pixtral-|open-mistral-)"),
    "cohere": re.compile(r"^command-"),
}

# Also accept provider/model and provider:model formats (LiteLLM compatibility)
_LITELLM_PREFIX_PATTERN = re.compile(r"^([a-z]+)[:/](.+)$")

# Default validation timeout in seconds
_DEFAULT_VALIDATION_TIMEOUT = 5.0

# Authentication error types that indicate invalid keys (vs transient errors)
# These are checked by exception class name to avoid importing all SDKs
_AUTH_ERROR_TYPES = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "InvalidAPIKeyError",
        "UnauthorizedError",
        "APIKeyError",
        "CredentialError",
    }
)

# Transient error types that don't indicate invalid keys
_TRANSIENT_ERROR_TYPES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "TimeoutError",
        "ConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
    }
)


def _is_auth_error(exc: Exception) -> bool:
    """Check if exception indicates an authentication/authorization failure.

    Auth errors mean the key is definitely invalid.
    """
    exc_type = type(exc).__name__
    if exc_type in _AUTH_ERROR_TYPES:
        return True
    # Also check parent classes for inheritance
    for parent in type(exc).__mro__:
        if parent.__name__ in _AUTH_ERROR_TYPES:
            return True
    # Check exception message for common auth patterns
    msg = str(exc).lower()
    auth_keywords = (
        "invalid api key",
        "unauthorized",
        "authentication",
        "invalid_api_key",
    )
    return any(kw in msg for kw in auth_keywords)


def _is_transient_error(exc: Exception) -> bool:
    """Check if exception indicates a transient error (network, rate limit, timeout).

    Transient errors don't mean the key is invalid - the service might be temporarily unavailable.
    """
    exc_type = type(exc).__name__
    if exc_type in _TRANSIENT_ERROR_TYPES:
        return True
    # Also check parent classes
    for parent in type(exc).__mro__:
        if parent.__name__ in _TRANSIENT_ERROR_TYPES:
            return True
    return False


def _run_with_timeout(func: Callable[[], T], timeout: float, provider: str) -> T:
    """Run a function with a timeout using ThreadPoolExecutor.

    Used for SDKs that don't support native timeout parameters (e.g., google-genai).

    Args:
        func: Zero-argument callable to execute.
        timeout: Timeout in seconds.
        provider: Provider name for error messages.

    Returns:
        Result of the function.

    Raises:
        TimeoutError: If the function doesn't complete within the timeout.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"{provider} validation timed out after {timeout}s"
            ) from None


@dataclass
class ProviderStatus:
    """Status of a provider validation check.

    Attributes:
        provider: Provider name (e.g., "openai", "anthropic", "google").
        valid: Whether the provider is available and key is valid.
        message: Human-readable status message.
        models: List of models associated with this provider (optional).
        error_type: Type of error if validation failed (optional).
    """

    provider: str
    valid: bool
    message: str
    models: list[str] | None = None
    error_type: str | None = None


@dataclass
class ProviderValidator:
    """Validates provider API keys before optimization runs.

    This validator performs lightweight API calls to verify that provider
    keys are valid before starting potentially expensive optimization runs.

    Validation behavior:
    - Known providers (OpenAI, Anthropic, Google, Mistral, Cohere):
      - Auth errors → valid=False, raise ProviderValidationError
      - Transient errors (rate limit, timeout) → valid=True with warning
        (key may be valid; actual call will retry as needed)
    - Unknown models: Warn only, do not block (cannot validate without SDK).

    Caching:
    - Success-only caching, session-scoped (instance-level, thread-safe).
    - Cache key is provider + key fingerprint (first 8 chars of SHA256).
    - Failures and transient errors are not cached.

    Timeout:
    - Default 5.0 seconds, configurable via TRAIGENT_VALIDATION_TIMEOUT.
    - OpenAI/Anthropic/Mistral/Cohere: native SDK timeout parameter.
    - Google: ThreadPoolExecutor wrapper (SDK lacks timeout param).

    Attributes:
        timeout: Validation timeout in seconds (default: 5.0).
    """

    timeout: float = _DEFAULT_VALIDATION_TIMEOUT
    _validated_cache: dict[str, str] = field(default_factory=dict)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock)

    def _get_key_fingerprint(self, key: str | None) -> str:
        """Generate a fingerprint for a key (first 8 chars of SHA256)."""
        if not key:
            return "none"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def _is_cached(self, provider: str, key: str | None) -> bool:
        """Check if a provider validation result is cached (thread-safe)."""
        cache_key = f"{provider}:{self._get_key_fingerprint(key)}"
        with self._cache_lock:
            return cache_key in self._validated_cache

    def _cache_success(self, provider: str, key: str | None) -> None:
        """Cache a successful validation result (thread-safe)."""
        cache_key = f"{provider}:{self._get_key_fingerprint(key)}"
        with self._cache_lock:
            self._validated_cache[cache_key] = "valid"

    def validate_all(self, models: Sequence[str]) -> dict[str, ProviderStatus]:
        """Validate all providers needed for the given models.

        Args:
            models: List of model names to validate providers for.

        Returns:
            Dict mapping provider name to ProviderStatus.
        """
        # Extract unique providers from model list
        providers_to_validate: dict[str, list[str]] = {}
        unknown_models: list[str] = []

        for model in models:
            provider = get_provider_for_model(model)
            if provider:
                if provider not in providers_to_validate:
                    providers_to_validate[provider] = []
                providers_to_validate[provider].append(model)
            else:
                unknown_models.append(model)

        # Warn about unknown models
        if unknown_models:
            for model in unknown_models:
                logger.warning(
                    "Unknown model '%s' - provider not recognized. "
                    "Skipping provider validation for this model.",
                    model,
                )

        # Validate each provider
        results: dict[str, ProviderStatus] = {}
        for provider, provider_models in providers_to_validate.items():
            status = self._validate_provider(provider)
            status.models = provider_models
            results[provider] = status

        return results

    def _validate_provider(self, provider: str) -> ProviderStatus:
        """Validate a single provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            ProviderStatus with validation result.
        """
        validate_method = getattr(self, f"_validate_{provider}", None)
        if validate_method is None:
            return ProviderStatus(
                provider=provider,
                valid=False,
                message=f"No validator for provider '{provider}'",
                error_type="UnsupportedProvider",
            )

        result: ProviderStatus = validate_method()
        return result

    def _validate_openai(self) -> ProviderStatus:
        """Validate OpenAI API key."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return ProviderStatus(
                provider="openai",
                valid=False,
                message="Set OPENAI_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("openai", key):
            return ProviderStatus(
                provider="openai",
                valid=True,
                message="Available (cached)",
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, timeout=self.timeout)
            client.models.list()
            self._cache_success("openai", key)
            return ProviderStatus(
                provider="openai",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="openai",
                valid=False,
                message="SDK not installed (pip install openai)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="openai",
                    valid=False,
                    message=f"Invalid key ({error_type})",
                    error_type=error_type,
                )
            if _is_transient_error(exc):
                # Transient error - key might be valid, warn but don't block
                logger.warning(
                    "OpenAI validation got transient error (%s), "
                    "assuming key is valid. Actual call may fail.",
                    error_type,
                )
                return ProviderStatus(
                    provider="openai",
                    valid=True,
                    message=f"Available (unverified: {error_type})",
                    error_type=error_type,
                )
            # Unknown error - treat as auth failure (conservative)
            return ProviderStatus(
                provider="openai",
                valid=False,
                message=f"Validation failed ({error_type})",
                error_type=error_type,
            )

    def _validate_anthropic(self) -> ProviderStatus:
        """Validate Anthropic API key."""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return ProviderStatus(
                provider="anthropic",
                valid=False,
                message="Set ANTHROPIC_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("anthropic", key):
            return ProviderStatus(
                provider="anthropic",
                valid=True,
                message="Available (cached)",
            )

        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=key, timeout=self.timeout)
            # Use count_tokens - zero cost, validates auth
            client.messages.count_tokens(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
            )
            self._cache_success("anthropic", key)
            return ProviderStatus(
                provider="anthropic",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="anthropic",
                valid=False,
                message="SDK not installed (pip install anthropic)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="anthropic",
                    valid=False,
                    message=f"Invalid key ({error_type})",
                    error_type=error_type,
                )
            if _is_transient_error(exc):
                logger.warning(
                    "Anthropic validation got transient error (%s), "
                    "assuming key is valid. Actual call may fail.",
                    error_type,
                )
                return ProviderStatus(
                    provider="anthropic",
                    valid=True,
                    message=f"Available (unverified: {error_type})",
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="anthropic",
                valid=False,
                message=f"Validation failed ({error_type})",
                error_type=error_type,
            )

    def _validate_google(self) -> ProviderStatus:
        """Validate Google API key."""
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            return ProviderStatus(
                provider="google",
                valid=False,
                message="Set GOOGLE_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("google", key):
            return ProviderStatus(
                provider="google",
                valid=True,
                message="Available (cached)",
            )

        try:
            from google import genai

            client = genai.Client(api_key=key)

            # google-genai doesn't support timeout in constructor, so we
            # wrap the API call with our timeout helper
            def _list_models() -> None:
                list(client.models.list())

            _run_with_timeout(_list_models, self.timeout, "Google")
            self._cache_success("google", key)
            return ProviderStatus(
                provider="google",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="google",
                valid=False,
                message="SDK not installed (pip install google-genai)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="google",
                    valid=False,
                    message=f"Invalid key ({error_type})",
                    error_type=error_type,
                )
            if _is_transient_error(exc):
                logger.warning(
                    "Google validation got transient error (%s), "
                    "assuming key is valid. Actual call may fail.",
                    error_type,
                )
                return ProviderStatus(
                    provider="google",
                    valid=True,
                    message=f"Available (unverified: {error_type})",
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="google",
                valid=False,
                message=f"Validation failed ({error_type})",
                error_type=error_type,
            )

    def _validate_mistral(self) -> ProviderStatus:
        """Validate Mistral API key."""
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            return ProviderStatus(
                provider="mistral",
                valid=False,
                message="Set MISTRAL_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("mistral", key):
            return ProviderStatus(
                provider="mistral",
                valid=True,
                message="Available (cached)",
            )

        try:
            from mistralai import Mistral

            # Mistral SDK uses timeout in seconds (default 120)
            client = Mistral(api_key=key, timeout=int(self.timeout))
            client.models.list()
            self._cache_success("mistral", key)
            return ProviderStatus(
                provider="mistral",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="mistral",
                valid=False,
                message="SDK not installed (pip install mistralai)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="mistral",
                    valid=False,
                    message=f"Invalid key ({error_type})",
                    error_type=error_type,
                )
            if _is_transient_error(exc):
                logger.warning(
                    "Mistral validation got transient error (%s), "
                    "assuming key is valid. Actual call may fail.",
                    error_type,
                )
                return ProviderStatus(
                    provider="mistral",
                    valid=True,
                    message=f"Available (unverified: {error_type})",
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="mistral",
                valid=False,
                message=f"Validation failed ({error_type})",
                error_type=error_type,
            )

    def _validate_cohere(self) -> ProviderStatus:
        """Validate Cohere API key."""
        key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not key:
            return ProviderStatus(
                provider="cohere",
                valid=False,
                message="Set COHERE_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("cohere", key):
            return ProviderStatus(
                provider="cohere",
                valid=True,
                message="Available (cached)",
            )

        try:
            import cohere

            # Cohere SDK supports timeout parameter
            client = cohere.Client(api_key=key, timeout=self.timeout)
            client.models.list()
            self._cache_success("cohere", key)
            return ProviderStatus(
                provider="cohere",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="cohere",
                valid=False,
                message="SDK not installed (pip install cohere)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="cohere",
                    valid=False,
                    message=f"Invalid key ({error_type})",
                    error_type=error_type,
                )
            if _is_transient_error(exc):
                logger.warning(
                    "Cohere validation got transient error (%s), "
                    "assuming key is valid. Actual call may fail.",
                    error_type,
                )
                return ProviderStatus(
                    provider="cohere",
                    valid=True,
                    message=f"Available (unverified: {error_type})",
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="cohere",
                valid=False,
                message=f"Validation failed ({error_type})",
                error_type=error_type,
            )


def get_provider_for_model(model_name: str) -> str | None:
    """Detect provider from model name.

    Args:
        model_name: Model name (e.g., "gpt-4o-mini", "claude-3-haiku-20240307").

    Returns:
        Provider name if detected, None otherwise.

    Examples:
        >>> get_provider_for_model("gpt-4o-mini")
        'openai'
        >>> get_provider_for_model("claude-3-haiku-20240307")
        'anthropic'
        >>> get_provider_for_model("gemini-1.5-flash")
        'google'
        >>> get_provider_for_model("openai/gpt-4")  # LiteLLM format
        'openai'
        >>> get_provider_for_model("my-custom-model")  # Unknown
        None
    """
    # Check for LiteLLM provider prefix format (provider/model or provider:model)
    prefix_match = _LITELLM_PREFIX_PATTERN.match(model_name)
    if prefix_match:
        prefix = prefix_match.group(1).lower()
        # Map common LiteLLM prefixes to our provider names
        prefix_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "gemini": "google",
            "vertex_ai": "google",
            "mistral": "mistral",
            "cohere": "cohere",
        }
        if prefix in prefix_map:
            return prefix_map[prefix]

    # Check against known patterns
    model_lower = model_name.lower()
    for provider, pattern in _PROVIDER_PATTERNS.items():
        if pattern.match(model_lower):
            return provider

    return None


def validate_providers(
    models: Sequence[str],
    *,
    timeout: float = _DEFAULT_VALIDATION_TIMEOUT,
) -> dict[str, ProviderStatus]:
    """Validate providers for the given models.

    This is a convenience function that creates a ProviderValidator and
    validates all providers needed for the given models.

    Args:
        models: List of model names to validate providers for.
        timeout: Validation timeout in seconds (default: 5.0).

    Returns:
        Dict mapping provider name to ProviderStatus.
    """
    validator = ProviderValidator(timeout=timeout)
    return validator.validate_all(models)


def print_provider_status(results: dict[str, ProviderStatus]) -> None:
    """Print provider validation results in a standardized format.

    Args:
        results: Dict mapping provider name to ProviderStatus.

    Output format:
        Provider Status:
          [OK] OpenAI: Available
               Models: gpt-4o-mini, gpt-4o
          [--] Anthropic: Set ANTHROPIC_API_KEY
          [--] Google: Invalid key (AuthenticationError)
    """
    print("\nProvider Status:")
    for provider, status in sorted(results.items()):
        symbol = "[OK]" if status.valid else "[--]"
        print(f"  {symbol} {provider.capitalize()}: {status.message}")
        if status.valid and status.models:
            models_str = ", ".join(status.models)
            print(f"       Models: {models_str}")


def get_failed_providers(
    results: dict[str, ProviderStatus],
) -> list[tuple[str, str]]:
    """Get list of failed providers with their error types.

    Args:
        results: Dict mapping provider name to ProviderStatus.

    Returns:
        List of (provider, error_type) tuples for failed providers.
    """
    return [
        (provider, status.error_type or "Unknown")
        for provider, status in results.items()
        if not status.valid
    ]
