"""CI/CD approval checks for optimization runs.

Functions to detect CI environments, validate approval tokens (legacy and
HMAC-signed), and gate optimization runs behind approvals.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from traigent.config.types import TraigentConfig
from traigent.utils.env_config import is_mock_llm, is_production
from traigent.utils.exceptions import ConfigurationError, OptimizationError
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import safe_open, validate_path

logger = get_logger(__name__)

# ISO 8601 timezone suffix for UTC (used in token validation)
_UTC_TIMEZONE_SUFFIX = "+00:00"


def _is_ci_environment() -> bool:
    """Detect if running in a CI environment (robust detection - 10 providers)."""
    return (
        os.getenv("CI") in ("true", "1")
        or os.getenv("GITHUB_ACTIONS") in ("true", "1")
        or os.getenv("JENKINS_URL") is not None
        or os.getenv("GITLAB_CI") in ("true", "1")
        or os.getenv("CIRCLECI") in ("true", "1")
        or os.getenv("TRAVIS") in ("true", "1")
        or os.getenv("BUILDKITE") in ("true", "1")
        or os.getenv("TEAMCITY_VERSION") is not None
        or os.getenv("AZURE_HTTP_USER_AGENT") is not None
        or os.getenv("BITBUCKET_BUILD_NUMBER") is not None
    )


def _check_env_var_approval() -> bool:
    """Check if CI run is approved via environment variable."""
    if os.getenv("TRAIGENT_RUN_APPROVED") == "1":
        approved_by = os.getenv("TRAIGENT_APPROVED_BY", "environment_variable")
        logger.info(f"CI optimization approved by: {approved_by}")
        return True
    return False


def _sanitize_for_log(value: str, max_len: int = 50) -> str:
    """Sanitize user-controlled data for safe logging.

    Removes any characters that could be used for log injection attacks,
    keeping only alphanumeric characters and common safe symbols.
    """
    return "".join(c for c in str(value)[:max_len] if c.isalnum() or c in "-_@.")


def _validate_legacy_token(token_data: dict[str, Any]) -> bool:
    """Validate a legacy format approval token."""
    if "approved_by" not in token_data or "expires_at" not in token_data:
        return False
    try:
        expires_str = token_data["expires_at"]
        expires_at = datetime.fromisoformat(
            expires_str.replace("Z", _UTC_TIMEZONE_SUFFIX)
        )
        now = datetime.now()
        if expires_at.tzinfo:
            now = now.replace(tzinfo=UTC)
        if expires_at > now:
            safe_approver = _sanitize_for_log(token_data.get("approved_by", "unknown"))
            logger.info("CI optimization approved by legacy token: %s", safe_approver)
            return True
    except (ValueError, KeyError):
        pass
    return False


def _validate_hmac_token(token_data: dict[str, Any]) -> bool:
    """Validate an HMAC-signed approval token."""
    required_fields = ["approver", "expires_iso", "nonce", "signature"]
    if not all(field in token_data for field in required_fields):
        return False

    secret = os.getenv("TRAIGENT_APPROVAL_SECRET", "").encode()
    if not secret:
        logger.warning(
            "TRAIGENT_APPROVAL_SECRET not set, cannot validate token signature"
        )
        try:
            expires_at = datetime.fromisoformat(
                token_data["expires_iso"].replace("Z", _UTC_TIMEZONE_SUFFIX)
            )
            now = datetime.now(UTC) if expires_at.tzinfo else datetime.now()
            if expires_at > now:
                logger.warning(
                    "Token approved by %s (signature not verified)",
                    _sanitize_for_log(token_data.get("approver", "unknown")),
                )
                return True
        except (ValueError, KeyError):
            pass
        return False

    # Compute expected signature
    payload = f"{token_data['approver']}|{token_data['expires_iso']}|{token_data['nonce']}".encode()
    expected_sig = base64.b64encode(
        hmac.new(secret, payload, hashlib.sha256).digest()
    ).decode()

    # Constant-time comparison
    if not hmac.compare_digest(token_data.get("signature", ""), expected_sig):
        logger.warning("Token signature validation failed")
        return False

    # Check expiration
    try:
        expires_at = datetime.fromisoformat(
            token_data["expires_iso"].replace("Z", _UTC_TIMEZONE_SUFFIX)
        )
        now = datetime.now(UTC) if expires_at.tzinfo else datetime.now()

        # Enforce max TTL of 24 hours from creation
        if expires_at - now > timedelta(hours=24):
            logger.warning("Token TTL exceeds 24 hours, rejecting")
            return False
        if expires_at > now:
            logger.info(
                "CI optimization approved by HMAC token: %s",
                _sanitize_for_log(token_data.get("approver", "unknown")),
            )
            return True
        logger.debug("Token expired")
    except (ValueError, KeyError):
        pass
    return False


def _check_token_file_approval(token_file: Path, base_dir: Path) -> bool:
    """Check if CI run is approved via token file."""
    if not token_file.exists():
        return False

    try:
        with safe_open(token_file, base_dir, mode="r", encoding="utf-8") as f:
            token_data = json.load(f)

        if _validate_hmac_token(token_data):
            return True
        if _validate_legacy_token(token_data):
            return True
    except (ValueError, KeyError) as e:  # JSONDecodeError is subclass of ValueError
        logger.debug(f"Invalid approval token: {e}")

    return False


def check_ci_approval(traigent_config: TraigentConfig) -> None:
    """Check if approval is required and granted for CI runs.

    Args:
        traigent_config: Traigent configuration

    Raises:
        OptimizationError: If running in CI without proper approval
    """
    if not traigent_config.is_edge_analytics_mode():
        return

    if is_mock_llm():
        msg = "Skipping CI approval in mock LLM mode"
        if is_production():
            logger.warning(f"{msg} while ENVIRONMENT=production.")
        else:
            logger.info(f"{msg}.")
        return

    if not _is_ci_environment():
        return

    if _check_env_var_approval():
        return

    storage_path = traigent_config.get_local_storage_path()
    if storage_path is None:
        raise ConfigurationError("Storage path not configured")
    storage_root = Path(storage_path).expanduser().resolve()
    token_file = validate_path(
        storage_root / "approval.token", storage_root, must_exist=False
    )
    if _check_token_file_approval(token_file, storage_root):
        return

    raise OptimizationError(
        """
CI/CD Approval Required

This optimization was triggered in a CI environment and requires approval.

To approve, use one of these methods:

1. Environment variable (recommended for CI):
   export TRAIGENT_RUN_APPROVED=1
   export TRAIGENT_APPROVED_BY="your_name"

2. Approval token file:
   echo '{"approved_by": "your_name", "expires_at": "2024-12-31T23:59:59"}' > ~/.traigent/approval.token

3. GitHub Actions with environment protection:
   Use 'environment: production' with required reviewers
        """
    )
