"""Environment configuration utility for Traigent.

This module provides secure access to environment variables and configuration.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-INVOKERS FUNC-SECURITY REQ-INJ-002 REQ-SEC-010 SYNC-OptimizationFlow

import os
import secrets
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from dotenv import load_dotenv

from .logging import get_logger

# ============================================================================
# MIGRATION GUARD: Reject deprecated TRAIGENT_MOCK_MODE at import time
# ============================================================================
# This must be checked BEFORE any other code runs to ensure a clean break.
if os.environ.get("TRAIGENT_MOCK_MODE"):
    raise OSError(
        "TRAIGENT_MOCK_MODE is deprecated and no longer supported.\n"
        "Please migrate to:\n"
        "  - traigent.testing.enable_mock_mode_for_quickstart() (in code) "
        "to mock LLM API calls\n"
        "  - TRAIGENT_OFFLINE_MODE=true (to skip backend communication)"
    )


def _is_production_env_name(name: str | None) -> bool:
    """Internal helper used by the prod-guard before :func:`is_production`
    is defined. Mirrors :func:`is_production`'s semantics so the guard
    and the runtime check stay in lockstep."""
    if name is None:
        return False
    return name.strip().lower() in {"prod", "production"}


def _is_truthy_env_value(value: str | None) -> bool:
    """Internal version of :func:`is_truthy` (defined later in the file).

    Defined here so the prod-guard can run before :func:`is_truthy` is
    in scope. Same accepted truthy strings.
    """
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _check_mock_llm_prod_guard() -> None:
    """Reject ``TRAIGENT_MOCK_LLM=true`` in production; warn elsewhere.

    Reads ``os.environ`` live so the result reflects the current process
    state — important because this is called both before AND after
    :func:`load_dotenv`, and a ``.env`` file may add the variables on
    the second pass. Tightness is intentional:

    * ``TRAIGENT_MOCK_LLM=false`` (or unset) — no-op, no warn.
    * ``TRAIGENT_MOCK_LLM=true`` and ``ENVIRONMENT=production`` —
      ``OSError`` (hard-block; the surviving guard against the original
      prod incident).
    * ``TRAIGENT_MOCK_LLM=true`` outside production — emit a single
      ``DeprecationWarning`` so the migration path is visible in
      pytest's warnings summary and interactive Python sessions.
    """
    if not _is_truthy_env_value(os.environ.get("TRAIGENT_MOCK_LLM")):
        return
    if _is_production_env_name(os.environ.get("ENVIRONMENT")):
        raise OSError(
            "TRAIGENT_MOCK_LLM=true is set in a production environment "
            "(ENVIRONMENT=production). Mock mode is hard-blocked in "
            "production to prevent silent substitution of real LLM "
            "calls. Either unset TRAIGENT_MOCK_LLM, or run with "
            "ENVIRONMENT!=production. For programmatic mock activation "
            "use traigent.testing.enable_mock_mode_for_quickstart()."
        )
    # Suppress the user-facing deprecation warning when the env var was
    # written by the quickstart bootstrap in ``traigent/__init__.py``
    # (the user IS using the recommended in-code path; the env var is
    # internal handoff across the import boundary). The prod hard-block
    # above still applies — bootstrap cannot bypass it.
    if os.environ.get("_TRAIGENT_QUICKSTART_BOOTSTRAP") == "1":
        return
    warnings.warn(
        "TRAIGENT_MOCK_LLM is deprecated. Call "
        "traigent.testing.enable_mock_mode_for_quickstart() in code "
        "instead. The env var still works in non-production environments "
        "for backward compatibility but will be removed in a future "
        "release.",
        DeprecationWarning,
        stacklevel=2,
    )


# ============================================================================
# DEPRECATION: TRAIGENT_MOCK_LLM still works in dev/test, blocked in prod
# ============================================================================
# Run the guard before dotenv to catch the explicit-env-var case fast,
# THEN run it again after dotenv so a `.env` setting `ENVIRONMENT=production
# TRAIGENT_MOCK_LLM=true` is also caught (even though those vars only land
# in os.environ after load_dotenv). is_mock_llm() recomputes the prod
# check live as a defence in depth — no module-level cache is read after
# import, so late env mutation cannot bypass the guard.
_check_mock_llm_prod_guard()

# Load environment variables from .env file if it exists, unless the
# caller has explicitly opted out via TRAIGENT_SKIP_DOTENV. Tests and
# hermetic subprocess smokes set this so the repo's ``.env`` (which
# typically contains TRAIGENT_BACKEND_URL=localhost:5000 etc.) does
# NOT leak into a "clean env" run.
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists() and os.environ.get("TRAIGENT_SKIP_DOTENV", "").lower() not in (
    "1",
    "true",
    "yes",
    "on",
):
    load_dotenv(env_file)
    _check_mock_llm_prod_guard()

_MIN_JWT_SECRET_LENGTH = 32
_PRODUCTION_ENV_NAMES = {"prod", "production"}

logger = get_logger(__name__)

# Cache generated development secrets so repeated calls stay consistent per process.
_GENERATED_DEV_JWT_SECRET: str | None = None

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig


def _normalize_str(value: str | None) -> str | None:
    """Return a stripped string or None when the value is blank."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _get_environment_name() -> str:
    """Return normalized environment name with development as default."""
    value = os.getenv("ENVIRONMENT", "development")
    return value.strip().lower()


def is_production() -> bool:
    """Check if running in production mode."""
    return _get_environment_name() in _PRODUCTION_ENV_NAMES


@overload
def get_env_var(
    key: str,
    default: str,
    required: bool = False,
    mask_in_logs: bool = False,
) -> str: ...


@overload
def get_env_var(
    key: str,
    default: str | None = None,
    required: Literal[True] = True,
    mask_in_logs: bool = False,
) -> str: ...


@overload
def get_env_var(
    key: str,
    default: None = None,
    required: bool = False,
    mask_in_logs: bool = False,
) -> str | None: ...


def get_env_var(
    key: str,
    default: str | None = None,
    required: bool = False,
    mask_in_logs: bool = False,
) -> str | None:
    """Get environment variable with optional validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether this variable is required
        mask_in_logs: Whether to mask value in logs

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(key, default)

    if required and not value:
        raise ValueError(
            f"Required environment variable {key} not found. "
            f"Please set it in your .env file or environment."
        )

    if value and mask_in_logs:
        # Never log actual sensitive values
        masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
        logger.info("Loaded environment variable %s (masked): %s", key, masked_value)

    return value


# Convenience functions for common environment variables
def get_api_key(provider: str) -> str | None:
    """Get API key for a specific provider from environment."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "huggingface": "HF_API_KEY",
        "traigent": "TRAIGENT_API_KEY",
    }

    env_var = env_map.get(provider.lower())
    if not env_var:
        return None

    return get_env_var(env_var, mask_in_logs=True)


def get_database_url() -> str:
    """Get database URL from environment."""
    if is_production():
        return get_env_var("DATABASE_URL", required=True)
    return get_env_var("DATABASE_URL", default="postgresql://localhost:5432/traigent")


def get_redis_url() -> str:
    """Get Redis URL from environment."""
    if is_production():
        return get_env_var("REDIS_URL", required=True)
    return get_env_var("REDIS_URL", default="redis://localhost:6379")


def get_jwt_secret() -> str:
    """Get JWT secret key from environment."""
    secret = _normalize_str(get_env_var("JWT_SECRET_KEY"))

    if secret:
        if len(secret) < _MIN_JWT_SECRET_LENGTH:
            warnings.warn(
                "JWT_SECRET_KEY is shorter than the recommended 32 characters."
                " Please rotate to a stronger value for production deployments.",
                UserWarning,
                stacklevel=2,
            )
        return secret

    if is_mock_llm() or is_development():
        # Allow overriding the fallback for integration tests while keeping it stable.
        override = _normalize_str(os.getenv("TRAIGENT_DEV_JWT_SECRET"))
        if override:
            return override
        global _GENERATED_DEV_JWT_SECRET
        if _GENERATED_DEV_JWT_SECRET is None:
            _GENERATED_DEV_JWT_SECRET = secrets.token_urlsafe(48)
            warnings.warn(
                "JWT_SECRET_KEY not set. Generated a temporary development secret. "
                "Set TRAIGENT_DEV_JWT_SECRET or JWT_SECRET_KEY for stable behavior.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Generated ephemeral JWT secret for development use; configure "
                "TRAIGENT_DEV_JWT_SECRET for stability."
            )
        return _GENERATED_DEV_JWT_SECRET

    raise ValueError(
        "JWT_SECRET_KEY must be set when ENVIRONMENT is production. "
        "Set a strong secret in the environment or .env file."
    )


def is_truthy(value: str | None) -> bool:
    """Normalize an environment-variable string to a boolean.

    Accepts ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive) as
    true. Anything else — including ``None`` and empty string — is false.

    Use this at user-facing entry points (CLI wrappers, example scripts,
    ``__main__`` modules) so the set of accepted truthy values is consistent
    across the codebase instead of drifting per call site.
    """
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_mock_llm() -> bool:
    """Check if LLM API calls should be mocked with simulated responses.

    Mock mode is on when EITHER:

    * the in-code flag set via
      :func:`traigent.testing.enable_mock_mode_for_quickstart` is true
      (the recommended path), OR
    * ``TRAIGENT_MOCK_LLM=true`` is set AND ``ENVIRONMENT`` is not
      ``production`` (legacy path kept for backward compatibility with
      existing test fixtures and examples; hard-blocked in prod by the
      import-time guard at the top of this module).

    This is separate from backend communication - use
    :func:`is_backend_offline` to control whether Traigent backend calls
    are skipped.

    Returns:
        True if LLM calls should be mocked, False for real LLM API calls.

    See also:
        - :func:`is_backend_offline`: Check if Traigent backend calls should be skipped
    """
    # Production hard-block runs FIRST — even if the in-code flag was
    # flipped on in a previous shell-script step or a misconfigured
    # deployment, ``is_mock_llm()`` must NEVER report True in production.
    # ``is_production()`` is recomputed live from ``os.environ`` so late
    # mutation cannot bypass the guard.
    if is_production():
        return False
    from traigent.testing import is_mock_mode_enabled

    if is_mock_mode_enabled():
        return True
    # Match the broader truthy set used by the import-time prod guard
    # (1/true/yes/on) so a value like ``TRAIGENT_MOCK_LLM=1`` doesn't get
    # blocked at import but then quietly return False here — the two
    # paths must agree on what counts as "set".
    return is_truthy(os.environ.get("TRAIGENT_MOCK_LLM"))


def is_strict_cost_accounting() -> bool:
    """Check whether runtime cost accounting should fail fast.

    When TRAIGENT_STRICT_COST_ACCOUNTING=true, post-call cost paths should:
    - Require priced models (no silent 0.0 for unknown models)
    - Raise on unknown/missing trial cost instead of fallback mode
    """
    return get_env_var("TRAIGENT_STRICT_COST_ACCOUNTING", "false").lower() == "true"


def should_show_cloud_notice(traigent_config: "TraigentConfig") -> bool:
    """Return True when the cloud API key notice should be shown."""
    if is_mock_llm() or is_backend_offline():
        return False
    if traigent_config.is_edge_analytics_mode():
        return False

    from traigent.config.backend_config import BackendConfig

    return not BackendConfig.get_api_key()


def is_backend_offline() -> bool:
    """Check if Traigent backend calls should be skipped.

    When TRAIGENT_OFFLINE_MODE=true, all communication with the Traigent backend
    is skipped. This is useful for:
    - Air-gapped environments without network access
    - Local development without backend setup
    - Testing scenarios where backend is not needed

    This is independent of LLM mocking - you can:
    - Mock LLMs but still send to backend (MOCK_LLM=true, OFFLINE=false)
    - Use real LLMs but skip backend (MOCK_LLM=false, OFFLINE=true)
    - Both mock and offline (MOCK_LLM=true, OFFLINE=true) for local testing

    Returns:
        True if backend calls should be skipped, False for normal backend communication.

    See also:
        - is_mock_llm(): Check if LLM API calls should be mocked
    """
    return get_env_var("TRAIGENT_OFFLINE_MODE", "false").lower() == "true"


def is_development() -> bool:
    """Check if running in development mode."""
    return _get_environment_name() in {"dev", "development"}


def skip_provider_validation() -> bool:
    """Check if provider validation should be skipped.

    Provider validation is skipped when:
    - TRAIGENT_SKIP_PROVIDER_VALIDATION=true is set
    - Mock mode is on via :func:`traigent.testing.enable_mock_mode_for_quickstart`
      (no real API calls)

    This allows users to bypass validation for:
    - Custom/internal models not recognized by Traigent
    - Testing scenarios where validation is not needed
    - Environments where validation requests are blocked

    Returns:
        True if provider validation should be skipped, False otherwise.

    See also:
        - :func:`is_mock_llm`: Check if LLM API calls should be mocked
        - ``validate_providers`` param in ``@traigent.optimize`` decorator
    """
    # Skip if mock mode is enabled (no real API calls anyway)
    if is_mock_llm():
        return True

    # Skip if explicitly disabled
    skip_env = get_env_var("TRAIGENT_SKIP_PROVIDER_VALIDATION", "false")
    return skip_env.lower() in ("true", "1", "yes")


def get_validation_timeout() -> float:
    """Get provider validation timeout in seconds.

    Reads from TRAIGENT_VALIDATION_TIMEOUT environment variable.
    Defaults to 5.0 seconds if not set.

    Returns:
        Validation timeout in seconds.
    """
    timeout_str = get_env_var("TRAIGENT_VALIDATION_TIMEOUT", "5.0")
    try:
        return float(timeout_str)
    except ValueError:
        logger.warning(
            "Invalid TRAIGENT_VALIDATION_TIMEOUT '%s', using default 5.0s",
            timeout_str,
        )
        return 5.0
