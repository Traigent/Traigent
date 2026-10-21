"""Provider validation for Traigent SDK.

This module provides an opt-in, standalone provider key validation helper
(:func:`validate_providers`) callers can use to check API keys are valid
before starting potentially long and expensive optimization runs. It is NOT
wired into ``@traigent.optimize`` automatically -- there is no
``validate_providers=`` decorator parameter, and nothing in this module
raises :class:`~traigent.utils.exceptions.ProviderValidationError` on its
own. Call :func:`validate_providers`, inspect the returned
``dict[str, ProviderStatus]`` (or use :func:`get_failed_providers`), and
raise :class:`~traigent.utils.exceptions.ProviderValidationError` yourself
if you want a hard failure.

Validation behavior (as reported in the returned ``ProviderStatus``, not by
raising):
- **Known providers** (OpenAI, Anthropic, Google, Mistral, Cohere):
  - Auth errors (401, InvalidAPIKey) → ``valid=False``
  - Transient errors (rate limit, timeout, network) → ``valid=True`` with a
    warning (key may be valid; service temporarily unavailable)
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

# Common validation message templates (avoid S1192 duplicate strings)
_MSG_INVALID_KEY = "Invalid key ({error_type})"
_MSG_TRANSIENT_WARNING = (
    "%s validation got transient error (%s), "
    "assuming key is valid. Actual call may fail."
)
_MSG_AVAILABLE_UNVERIFIED = "Available (unverified: {error_type})"
_MSG_VALIDATION_FAILED = "Validation failed ({error_type})"
_MSG_AVAILABLE_CACHED = "Available (cached)"
_MSG_INSUFFICIENT_FUNDS = (
    "Insufficient funds — top up your {provider} account and retry"
)

# Provider detection patterns (model name -> provider)
# Matches: gpt-*, o1-*, o3-* -> openai
#          claude-*, haiku-*, sonnet-*, opus-* -> anthropic
#          gemini-* -> google
#          mistral-*, codestral-*, pixtral-* -> mistral
#          command-* -> cohere
#
# The SET of providers here (plus _KNOWN_MODELS and the _validate_<provider>
# methods below) is pinned to the canonical provider-support table
# (traigent/config/provider_support.py) by the drift regression test
# (tests/unit/config/test_provider_support_drift.py): every provider with
# detection="prefix" must appear here. HuggingFace is deliberately absent — it
# is the last-resort bare org/model match handled in get_provider_for_model().
_PROVIDER_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai": re.compile(r"^(gpt-|o[13]-|text-davinci|text-embedding|whisper-)"),
    "anthropic": re.compile(r"^(claude-|haiku-|sonnet-|opus-)"),
    "google": re.compile(r"^(gemini-|models/gemini-)"),
    "mistral": re.compile(r"^(mistral-|codestral-|pixtral-|open-mistral-)"),
    "cohere": re.compile(r"^command-"),
    # NOTE: HuggingFace (org/model format) is intentionally NOT in this dict.
    # It is applied as a true last resort in get_provider_for_model(), after all
    # known-provider prefix checks have been exhausted.
}

# HuggingFace bare org/model pattern (e.g. "meta-llama/Meta-Llama-3-8B-Instruct").
# Applied as a LAST RESORT only when the leading segment is not a known provider prefix.
_HF_ORG_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$")

# Also accept provider/model and provider:model formats (LiteLLM compatibility).
# Supports underscores and digits in the prefix (e.g. vertex_ai, together_ai).
_LITELLM_PREFIX_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)[:/](.+)$")

# Known valid models per provider (as of Jan 2026)
# Used to warn users about potentially invalid model names before API calls
# Note: This list may not be exhaustive; unknown models get a warning, not an error
_KNOWN_MODELS: dict[str, frozenset[str]] = {
    "openai": frozenset(
        {
            # Ids come from the bundled litellm price table; ids whose
            # deprecation_date has passed are dropped (#2651).
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
            "gpt-5.6",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "o3",
            # Retired o1-preview / o1-mini pruned (#1936/#1937).
            "o3-mini",
        }
    ),
    "anthropic": frozenset(
        {
            # Current SDK default fallback IDs (#1936) — keep in sync with
            # parameter_ranges.Choices.model / providers._FALLBACK_MODELS so the
            # SDK never warns "unknown" on its own defaults.
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            # Ids from the bundled litellm price table with no past
            # deprecation_date (#2651). Retired claude-sonnet-4-20250514,
            # claude-opus-4-20250514 (2026-06-15), claude-opus-4-1 (2026-08-05),
            # claude-3-7-sonnet, claude-3-opus, claude-3-haiku are dropped.
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-5",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-opus-4-6-20260205",
            "claude-opus-4-7",
            "claude-opus-4-7-20260416",
            "claude-opus-5",
            "claude-fable-5",
            "claude-fable-5-1",
            "claude-mythos-5",
            "claude-mythos-5-1",
            # Claude 5.5 family: current Anthropic model ids that are not yet
            # in the bundled litellm price table, so they are listed
            # explicitly (review r2). claude-fable-5-1 is listed above.
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            # Retired claude-3* ids (3.5 Sonnet/Haiku, 3 Sonnet) were removed
            # from this list per Anthropic's deprecations page (#2646).
        }
    ),
    "google": frozenset(
        {
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.0-flash",
            # Retired gemini-1.5 family, gemini-2.0-flash-exp preview, and
            # gemini-1.0-pro / gemini-pro pruned (#1936/#1937).
        }
    ),
    "mistral": frozenset(
        {
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
            "codestral-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
        }
    ),
    "cohere": frozenset(
        {
            "command",
            "command-light",
            "command-r",
            "command-r-plus",
        }
    ),
}

# Default validation timeout in seconds
_DEFAULT_VALIDATION_TIMEOUT = 5.0

# OpenRouter's key-info endpoint: 200 for a valid key, 401 otherwise. Free.
_OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

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


_INSUFFICIENT_FUNDS_KEYWORDS = (
    "insufficient_funds",
    "insufficient funds",
    "insufficient credits",
    "billing hard limit",
    "exceeded your current billing",
    "payment required",
    "status 402",
    "http 402",
    "code: 402",
    "error 402",
    "402 payment",
)


def _is_insufficient_funds_error(exc: Exception) -> bool:
    """Check if exception indicates insufficient funds / billing failure."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _INSUFFICIENT_FUNDS_KEYWORDS)


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


def validate_model_names(
    models: list[str], provider: str
) -> tuple[list[str], list[str]]:
    """Validate model names against known models list.

    Args:
        models: List of model names to validate.
        provider: Provider name to check against.

    Returns:
        Tuple of (valid_models, unknown_models).
        Unknown models are not necessarily invalid - just not in our known list.
    """
    known = _KNOWN_MODELS.get(provider, frozenset())
    if provider == "huggingface":
        logger.debug(
            "HuggingFace is an open model namespace - skipping model-name sanity check"
        )
        return models, []
    if not known:
        # No known models list for this provider - assume all valid
        return models, []

    valid = []
    unknown = []
    for model in models:
        if model in known:
            valid.append(model)
        else:
            unknown.append(model)
    return valid, unknown


# What a status says about the account's funds. "not_checked": the provider has
# no balance API, so nothing was verified. "unknown": a check ran but could not
# tell. "available": a reported limit has headroom. "depleted": a reported limit
# is exhausted. Only "available" is a claim that funds exist, and then only for
# the key's own spending limit, never the account balance.
_FUNDS_NOT_CHECKED = "not_checked"
_FUNDS_UNKNOWN = "unknown"
_FUNDS_AVAILABLE = "available"
_FUNDS_DEPLETED = "depleted"


def _openrouter_funds(response: object) -> str:
    """Funds state from an OpenRouter ``/api/v1/key`` response, never raising."""
    try:
        body = response.json()  # type: ignore[attr-defined]
        data = body.get("data") if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001 - unreadable body is an unknown, not a pass
        return _FUNDS_UNKNOWN
    if not isinstance(data, dict):
        return _FUNDS_UNKNOWN

    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if value == value else None  # NaN is not a number

    limit = _number(data.get("limit"))
    if limit is None:
        return _FUNDS_UNKNOWN  # no per-key limit: balance is not visible here
    remaining = _number(data.get("limit_remaining"))
    if remaining is None:
        usage = _number(data.get("usage"))
        remaining = None if usage is None else limit - usage
    if remaining is None:
        return _FUNDS_UNKNOWN
    return _FUNDS_DEPLETED if remaining <= 0 else _FUNDS_AVAILABLE


@dataclass
class ProviderStatus:
    """Status of a provider validation check.

    Attributes:
        provider: Provider name (e.g., "openai", "anthropic", "google").
        valid: Whether the provider is available and key is valid.
        message: Human-readable status message.
        models: List of valid models associated with this provider (optional).
        unknown_models: List of models not in known list (may still work).
        error_type: Type of error if validation failed (optional).
        skipped: True when the provider was deliberately not checked (for
            example Bedrock, which has no key/balance probe). A skipped
            provider is never a pass (``valid`` is False) and never counts as
            a failure in :func:`get_failed_providers`.
        funds: ``"not_checked"`` (default; the provider has no balance API),
            ``"unknown"``, ``"available"`` or ``"depleted"``. Never claims
            funds were verified unless a balance/limit was actually read.
    """

    provider: str
    valid: bool
    message: str
    models: list[str] | None = None
    unknown_models: list[str] | None = None
    error_type: str | None = None
    skipped: bool = False
    funds: str = _FUNDS_NOT_CHECKED


@dataclass
class ProviderValidator:
    """Validates provider API keys, on demand, before optimization runs.

    This validator performs lightweight API calls to verify that provider
    keys are valid before starting potentially expensive optimization runs.
    It can be called standalone; the optimizer also runs it automatically
    before cost approval for paid runs (a cost limit is set or cost is
    approved), honouring ``TRAIGENT_SKIP_PROVIDER_VALIDATION``. It never raises
    ``ProviderValidationError`` itself; it only reports status (see
    :func:`validate_providers` and :func:`get_failed_providers`).

    Validation behavior:
    - Known providers (OpenAI, Anthropic, Google, Mistral, Cohere):
      - Auth errors → valid=False (caller may raise ProviderValidationError)
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

    def _group_models_by_provider(
        self, models: Sequence[str]
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Group models by provider and identify unknown providers."""
        providers: dict[str, list[str]] = {}
        unknown: list[str] = []

        for model in models:
            provider = get_run_provider_for_model(model)
            if provider:
                providers.setdefault(provider, []).append(model)
            else:
                unknown.append(model)

        return providers, unknown

    def _warn_unknown_models(self, unknown_models: list[str], provider: str) -> None:
        """Log warnings for models not in known list."""
        for model in unknown_models:
            logger.warning(
                "Model '%s' not in known %s models list. "
                "It may be deprecated, a typo, or a new model. "
                "API calls may fail with 404.",
                model,
                provider,
            )

    def validate_all(self, models: Sequence[str]) -> dict[str, ProviderStatus]:
        """Validate all providers needed for the given models.

        Args:
            models: List of model names to validate providers for.

        Returns:
            Dict mapping provider name to ProviderStatus.
        """
        providers_to_validate, unknown_provider_models = self._group_models_by_provider(
            models
        )

        # Warn about models with unknown providers
        for model in unknown_provider_models:
            logger.warning(
                "Unknown model '%s' - provider not recognized. "
                "Skipping provider validation for this model.",
                model,
            )

        # Validate each provider and check model names
        results: dict[str, ProviderStatus] = {}
        for provider, provider_models in providers_to_validate.items():
            status = self._validate_provider(provider)

            # Validate model names against known list
            valid_models, unknown_models = validate_model_names(
                provider_models, provider
            )
            status.models = valid_models or provider_models
            status.unknown_models = unknown_models or None

            if unknown_models:
                self._warn_unknown_models(unknown_models, provider)

            results[provider] = status

        return results

    def _validate_provider(self, provider: str) -> ProviderStatus:
        """Validate a single provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").

        Returns:
            ProviderStatus with validation result.
        """
        if provider == "bedrock":
            return ProviderStatus(
                provider="bedrock",
                valid=False,
                message=(
                    "Skipped: Bedrock authenticates with AWS credentials and "
                    "has no key or balance check; its key is not "
                    "preflight-checked."
                ),
                error_type="NotValidated",
                skipped=True,
            )
        if provider == "vertex_ai":
            return ProviderStatus(
                provider="vertex_ai",
                valid=False,
                message=(
                    "Skipped: Vertex AI authenticates with application default "
                    "credentials, not an API key; credentials not checked."
                ),
                error_type="NotValidated",
                skipped=True,
            )
        if provider == "openrouter":
            return self._check_openrouter()
        validate_method = getattr(self, f"_validate_{provider}", None)
        if validate_method is None:
            # Distinguish a provider that is *recognized* by the SDK but
            # intentionally not validated (e.g. azure_openai / bedrock — present
            # in models.yaml / discovery but with no _validate_<provider>
            # method) from a truly unknown provider. The former gets a clearly
            # labeled "mapping-only / not validated" status instead of the
            # generic UnsupportedProvider error (#1568).
            from traigent.config.provider_support import get_provider_spec

            spec = get_provider_spec(provider)
            if spec is not None and not spec.validated:
                return ProviderStatus(
                    provider=provider,
                    valid=False,
                    message=(
                        f"Provider '{provider}' is recognized but not "
                        "validated by the SDK (mapping-only). Its key is not "
                        "preflight-checked; the actual LLM call will surface "
                        "any auth error."
                    ),
                    error_type="NotValidated",
                )
            return ProviderStatus(
                provider=provider,
                valid=False,
                message=f"No validator for provider '{provider}'",
                error_type="UnsupportedProvider",
            )

        result: ProviderStatus = validate_method()
        return result

    def _check_openrouter(self) -> ProviderStatus:
        """Validate an OpenRouter API key via its key-info endpoint.

        Not a ``_validate_<provider>`` method on purpose: OpenRouter is routed
        through the OpenAI-compatible client and is not in the provider-support
        table, so it must not look like a first-class validated provider.
        """
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return ProviderStatus(
                provider="openrouter",
                valid=False,
                message="Set OPENROUTER_API_KEY",
                error_type="MissingKey",
            )

        if self._is_cached("openrouter", key):
            return ProviderStatus(
                provider="openrouter",
                valid=True,
                message=_MSG_AVAILABLE_CACHED,
                funds=_FUNDS_UNKNOWN,
            )

        try:
            import requests

            response = requests.get(
                _OPENROUTER_KEY_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=self.timeout,
            )
            code = response.status_code
        except Exception as exc:
            # Never include the exception text: it can echo the request.
            error_type = type(exc).__name__
            logger.warning(_MSG_TRANSIENT_WARNING, "OpenRouter", error_type)
            return ProviderStatus(
                provider="openrouter",
                valid=True,
                message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                error_type=error_type,
            )

        if 200 <= code < 300:
            funds = _openrouter_funds(response)
            if funds == _FUNDS_DEPLETED:
                return ProviderStatus(
                    provider="openrouter",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="OpenRouter"),
                    error_type="InsufficientFunds",
                    funds=_FUNDS_DEPLETED,
                )
            self._cache_success("openrouter", key)
            return ProviderStatus(
                provider="openrouter", valid=True, message="Available", funds=funds
            )
        if code in (401, 403):
            return ProviderStatus(
                provider="openrouter",
                valid=False,
                message=_MSG_INVALID_KEY.format(error_type=f"HTTP{code}"),
                error_type="AuthenticationError",
            )
        if code == 402:
            return ProviderStatus(
                provider="openrouter",
                valid=False,
                message=_MSG_INSUFFICIENT_FUNDS.format(provider="OpenRouter"),
                error_type="InsufficientFunds",
            )
        if code == 429 or code >= 500:
            error_type = f"HTTP{code}"
            logger.warning(_MSG_TRANSIENT_WARNING, "OpenRouter", error_type)
            return ProviderStatus(
                provider="openrouter",
                valid=True,
                message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                error_type=error_type,
            )
        return ProviderStatus(
            provider="openrouter",
            valid=False,
            message=_MSG_VALIDATION_FAILED.format(error_type=f"HTTP{code}"),
            error_type=f"HTTP{code}",
        )

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
                message=_MSG_AVAILABLE_CACHED,
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
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="openai",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="OpenAI"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                # Transient error - key might be valid, warn but don't block
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "OpenAI",
                    error_type,
                )
                return ProviderStatus(
                    provider="openai",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            # Unknown error - treat as auth failure (conservative)
            return ProviderStatus(
                provider="openai",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
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
                message=_MSG_AVAILABLE_CACHED,
            )

        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=key, timeout=self.timeout)
            # Use count_tokens - zero cost, validates auth
            client.messages.count_tokens(
                model="claude-haiku-4-5-20251001",
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
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="anthropic",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="Anthropic"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "Anthropic",
                    error_type,
                )
                return ProviderStatus(
                    provider="anthropic",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="anthropic",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
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
                message=_MSG_AVAILABLE_CACHED,
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
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="google",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="Google"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "Google",
                    error_type,
                )
                return ProviderStatus(
                    provider="google",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="google",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
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
                message=_MSG_AVAILABLE_CACHED,
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
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="mistral",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="Mistral"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "Mistral",
                    error_type,
                )
                return ProviderStatus(
                    provider="mistral",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="mistral",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
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
                message=_MSG_AVAILABLE_CACHED,
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
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="cohere",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="Cohere"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "Cohere",
                    error_type,
                )
                return ProviderStatus(
                    provider="cohere",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="cohere",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
                error_type=error_type,
            )

    def _validate_huggingface(self) -> ProviderStatus:
        """Validate HuggingFace API token."""
        key = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_API_KEY")
        )
        if not key:
            return ProviderStatus(
                provider="huggingface",
                valid=False,
                message="Set HF_TOKEN",
                error_type="MissingKey",
            )

        if self._is_cached("huggingface", key):
            return ProviderStatus(
                provider="huggingface",
                valid=True,
                message=_MSG_AVAILABLE_CACHED,
            )

        try:
            from huggingface_hub import HfApi

            HfApi().whoami(token=key)
            self._cache_success("huggingface", key)
            return ProviderStatus(
                provider="huggingface",
                valid=True,
                message="Available",
            )
        except ImportError:
            return ProviderStatus(
                provider="huggingface",
                valid=False,
                message="SDK not installed (pip install huggingface_hub)",
                error_type="ModuleNotFoundError",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            if _is_auth_error(exc):
                return ProviderStatus(
                    provider="huggingface",
                    valid=False,
                    message=_MSG_INVALID_KEY.format(error_type=error_type),
                    error_type=error_type,
                )
            if _is_insufficient_funds_error(exc):
                return ProviderStatus(
                    provider="huggingface",
                    valid=False,
                    message=_MSG_INSUFFICIENT_FUNDS.format(provider="HuggingFace"),
                    error_type="InsufficientFunds",
                )
            if _is_transient_error(exc):
                logger.warning(
                    _MSG_TRANSIENT_WARNING,
                    "HuggingFace",
                    error_type,
                )
                return ProviderStatus(
                    provider="huggingface",
                    valid=True,
                    message=_MSG_AVAILABLE_UNVERIFIED.format(error_type=error_type),
                    error_type=error_type,
                )
            return ProviderStatus(
                provider="huggingface",
                valid=False,
                message=_MSG_VALIDATION_FAILED.format(error_type=error_type),
                error_type=error_type,
            )


def get_provider_for_model(model_name: str) -> str | None:
    """Detect provider from model name.

    Args:
        model_name: Model name (e.g., "gpt-4o-mini", "claude-haiku-4-5-20251001").

    Returns:
        Provider name if detected, None otherwise.

    Examples:
        >>> get_provider_for_model("gpt-4o-mini")
        'openai'
        >>> get_provider_for_model("claude-haiku-4-5-20251001")
        'anthropic'
        >>> get_provider_for_model("gemini-1.5-flash")
        'google'
        >>> get_provider_for_model("openai/gpt-4")  # LiteLLM format
        'openai'
        >>> get_provider_for_model("my-custom-model")  # Unknown
        None
    """
    # Map of LiteLLM provider prefixes (including those with underscores/digits) to
    # our canonical provider names.  None means "known prefix but not a Traigent-managed
    # provider" — these are still routed away from HuggingFace.
    _LITELLM_PREFIX_MAP: dict[str, str | None] = {
        # Core providers we validate
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "gemini": "google",
        "vertex_ai": "google",
        "mistral": "mistral",
        "cohere": "cohere",
        "huggingface": "huggingface",
        "hf": "huggingface",
        "huggingface_hub": "huggingface",
        # Additional known LiteLLM prefixes (not HuggingFace, not our validators)
        "azure": None,
        "groq": None,
        "bedrock": None,
        "together_ai": None,
        "replicate": None,
        "ollama": None,
        "openrouter": None,
        "perplexity": None,
        "fireworks_ai": None,
        "anyscale": None,
        "deepinfra": None,
        "voyage": None,
        "palm": None,
        "ai21": None,
        "nlp_cloud": None,
        "aleph_alpha": None,
        "petals": None,
        "oobabooga": None,
        "databricks": None,
    }

    # Step 1: Check for explicit provider/model or provider:model prefix (LiteLLM format).
    # This handles underscores/digits in prefixes (e.g. vertex_ai, together_ai).
    prefix_match = _LITELLM_PREFIX_PATTERN.match(model_name)
    if prefix_match:
        prefix = prefix_match.group(1).lower()
        if prefix in _LITELLM_PREFIX_MAP:
            return _LITELLM_PREFIX_MAP[prefix]

    # Step 2: Check against known provider name-prefix patterns (no slash in model id).
    model_lower = model_name.lower()
    for provider, pattern in _PROVIDER_PATTERNS.items():
        if pattern.match(model_lower):
            return provider

    # Step 3: Last resort — bare "org/model" HuggingFace format.
    # Only attribute to huggingface when the segment before the slash is NOT a known
    # provider prefix (which was already handled in step 1).
    if _HF_ORG_MODEL_PATTERN.match(model_name):
        org_segment = model_name.split("/", 1)[0].lower()
        if org_segment not in _LITELLM_PREFIX_MAP:
            return "huggingface"

    return None


def get_run_provider_for_model(model_name: str) -> str | None:
    """Like :func:`get_provider_for_model`, but also names OpenRouter/Bedrock/Vertex AI.

    Those two have no first-class validator mapping in
    :func:`get_provider_for_model` (it returns ``None`` for them), yet a run
    that uses them still spends money, so the pre-run check needs to see them.
    """
    prefix_match = _LITELLM_PREFIX_PATTERN.match(model_name)
    if prefix_match:
        prefix = prefix_match.group(1).lower()
        if prefix in ("openrouter", "bedrock", "vertex_ai"):
            return prefix
    return get_provider_for_model(model_name)


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
          [!!] Unknown models (may fail): claude-3-5-sonnet-20241022
    """
    print("\nProvider Status:")
    unknown_all: list[str] = []

    for provider, status in sorted(results.items()):
        symbol = "[OK]" if status.valid else ("[SK]" if status.skipped else "[--]")
        funds_note = ""
        if status.valid and status.funds in (_FUNDS_NOT_CHECKED, _FUNDS_UNKNOWN):
            funds_note = " (funds not checked)"
        print(f"  {symbol} {provider.capitalize()}: {status.message}{funds_note}")
        if status.valid and status.models:
            models_str = ", ".join(status.models)
            print(f"       Models: {models_str}")
        if status.unknown_models:
            unknown_all.extend(status.unknown_models)

    # Show warning for unknown models at the end
    if unknown_all:
        print(f"  [!!] Unknown models (may fail): {', '.join(unknown_all)}")
        print("       These models are not in known list - check for typos")


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
        if not status.valid and not status.skipped
    ]


# Error types that mean "could not check", not "the key is bad": the run's own
# call will surface a real problem, so the pre-run gate must not block on them.
_NON_BLOCKING_ERROR_TYPES = frozenset(
    {"NotValidated", "UnsupportedProvider", "ModuleNotFoundError", "Skipped"}
)


def preflight_run_providers(
    models: Sequence[str],
    *,
    timeout: float | None = None,
) -> dict[str, ProviderStatus]:
    """Check the keys of the providers a paid run will use, failing closed.

    Called once per run by the optimizer, before cost approval, for paid runs.
    Raises :class:`~traigent.utils.exceptions.ProviderValidationError` when a
    provider's key is missing, invalid or out of funds. The message names the
    provider and the error type only -- never key material or server text.
    Providers that cannot be checked (Bedrock, mapping-only providers, an SDK
    that is not installed) are reported in the result but never block.

    Returns:
        Dict mapping provider name to ProviderStatus (skipped ones included).
    """
    from traigent.utils.env_config import get_validation_timeout
    from traigent.utils.exceptions import ProviderValidationError

    results = validate_providers(
        models,
        timeout=get_validation_timeout() if timeout is None else timeout,
    )
    failed = [
        (provider, status.error_type or "Unknown")
        for provider, status in results.items()
        if not status.valid
        and not status.skipped
        and (status.error_type or "Unknown") not in _NON_BLOCKING_ERROR_TYPES
    ]
    for provider, status in results.items():
        if status.skipped:
            logger.info(
                "Provider '%s' not preflight-checked: %s", provider, status.message
            )
    if failed:
        listing = ", ".join(f"{provider} ({error})" for provider, error in failed)
        raise ProviderValidationError(
            "Provider key check failed before the paid run started: "
            f"{listing}. Fix the key (or set TRAIGENT_SKIP_PROVIDER_VALIDATION=true "
            "to skip this check).",
            failed_providers=failed,
        )
    return results
