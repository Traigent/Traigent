"""Environment configuration utility for TraiGent.

This module provides secure access to environment variables and configuration.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-INVOKERS FUNC-SECURITY REQ-INJ-002 REQ-SEC-010 SYNC-OptimizationFlow

import os
import secrets
import warnings
from pathlib import Path
from typing import Literal, overload

from dotenv import load_dotenv

from .logging import get_logger

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try to load from configs/env-templates/.env.local.template as fallback (development only)
    template_file = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "env-templates"
        / ".env.local.template"
    )
    if template_file.exists():
        # Only warn in non-mock mode to reduce noise for demo/testing
        mock_mode = os.getenv("TRAIGENT_MOCK_MODE", "false").lower() == "true"
        if not mock_mode:
            warnings.warn(
                "Using configs/env-templates/.env.local.template file. Create a .env file with actual values for production.",
                UserWarning,
                stacklevel=2,
            )

_MIN_JWT_SECRET_LENGTH = 32
_PRODUCTION_ENV_NAMES = {"prod", "production"}

logger = get_logger(__name__)

# Cache generated development secrets so repeated calls stay consistent per process.
_GENERATED_DEV_JWT_SECRET: str | None = None


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

    if is_mock_mode() or is_development():
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


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return get_env_var("TRAIGENT_MOCK_MODE", "false").lower() == "true"


def is_development() -> bool:
    """Check if running in development mode."""
    return _get_environment_name() in {"dev", "development"}
