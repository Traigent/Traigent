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
        "  - TRAIGENT_MOCK_LLM=true (to mock LLM API calls)\n"
        "  - TRAIGENT_OFFLINE_MODE=true (to skip backend communication)\n"
        "For local testing, set both: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true"
    )

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

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
    return get_env_var("DATABASE_URL", default="postgresql://localhost:5432/traigent")


def get_redis_url() -> str:
    """Get Redis URL from environment."""
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


def is_mock_llm() -> bool:
    """Check if LLM API calls should be mocked with simulated responses.

    When TRAIGENT_MOCK_LLM=true, LLM provider calls (OpenAI, Anthropic, etc.)
    are intercepted and return simulated responses instead of making real API calls.

    This is separate from backend communication - use is_backend_offline() to
    control whether Traigent backend calls are skipped.

    Returns:
        True if LLM calls should be mocked, False for real LLM API calls.

    See also:
        - is_backend_offline(): Check if Traigent backend calls should be skipped
    """
    return get_env_var("TRAIGENT_MOCK_LLM", "false").lower() == "true"


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
