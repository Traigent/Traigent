"""Security profile and feature flag configuration for the Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn


class SecurityProfile(Enum):
    """High-level security posture for the SDK runtime."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"


@dataclass(frozen=True)
class SecurityFlags:
    """Resolved security feature flags with sane defaults per profile."""

    profile: SecurityProfile
    allow_weak_credentials: bool
    auto_discovery: bool
    strict_sql_validation: bool
    use_decimal_rate_limiter: bool
    emit_security_telemetry: bool


def _str_to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_security_profile() -> SecurityProfile:
    """Resolve the current security profile using environment configuration."""

    explicit = os.getenv("TRAIGENT_SECURITY_PROFILE")
    if explicit:
        try:
            return SecurityProfile(explicit.strip().lower())
        except ValueError:
            pass

    env = (os.getenv("TRAIGENT_ENV") or "production").strip().lower()
    if os.getenv("PYTEST_CURRENT_TEST") and explicit is None:
        return SecurityProfile.DEVELOPMENT
    if env in {"dev", "development", "local"}:
        return SecurityProfile.DEVELOPMENT
    if env in {"stage", "staging"}:
        return SecurityProfile.STAGING
    return SecurityProfile.PRODUCTION


def get_security_flags() -> SecurityFlags:
    """Return resolved security flags honouring profile defaults and overrides."""

    profile = get_security_profile()

    allow_weak_default = profile != SecurityProfile.PRODUCTION
    auto_discovery_default = profile != SecurityProfile.PRODUCTION
    strict_sql_default = profile != SecurityProfile.DEVELOPMENT

    allow_env = os.getenv("ALLOW_WEAK_CREDS")
    if allow_env is not None:
        allow_weak_credentials = _str_to_bool(allow_env)
    else:
        allow_weak_credentials = allow_weak_default

    discovery_env = os.getenv("AUTO_DISCOVERY")
    if discovery_env is not None:
        auto_discovery = _str_to_bool(discovery_env)
    else:
        auto_discovery = auto_discovery_default

    strict_sql_env = os.getenv("STRICT_SQL")
    if strict_sql_env is not None:
        strict_sql_validation = _str_to_bool(strict_sql_env)
    else:
        strict_sql_validation = strict_sql_default

    decimal_env = os.getenv("TRAIGENT_USE_DECIMAL_RATE_LIMIT")
    use_decimal_rate_limiter = (
        _str_to_bool(decimal_env) if decimal_env is not None else False
    )

    emit_security_telemetry = not _str_to_bool(os.getenv("DISABLE_SECURITY_TELEMETRY"))

    return SecurityFlags(
        profile=profile,
        allow_weak_credentials=allow_weak_credentials,
        auto_discovery=auto_discovery,
        strict_sql_validation=strict_sql_validation,
        use_decimal_rate_limiter=use_decimal_rate_limiter,
        emit_security_telemetry=emit_security_telemetry,
    )


def reset_security_cache() -> NoReturn:
    """Compatibility shim. Previously a silent `return None`, which the
    validation spine flagged as a `public_stub_runtime` (Batch 1 of the
    public-surface gate). Either a real cache-reset implementation lands
    here, or the symbol gets removed from `__all__`. Until then, fail loud
    so callers don't silently get a no-op."""
    raise NotImplementedError("reset_security_cache is not implemented")
