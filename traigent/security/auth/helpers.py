"""Shared sanitization helpers and constants for authentication providers.

This module provides common utilities used by multiple auth providers to sanitize
user input, ensuring consistent behavior across SAML, OIDC, TOTP, and SMS auth.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import re
from typing import Any

# Regex pattern for control characters to remove during sanitization
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Regex pattern for validating email addresses (RFC 5322 simplified)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Regex pattern for validating role names (lowercase alphanumeric with dash/underscore)
ROLE_PATTERN = re.compile(r"^[a-z0-9_-]+$")

# Security constant: Delay to prevent timing attacks on authentication.
# This intentionally blocks to ensure constant-time failure responses.
# Note: This module is sync-only; do not call from async contexts without wrapping.
TIMING_ATTACK_DELAY_SECONDS = 0.1


def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize a string value by removing control characters and limiting length.

    Args:
        value: The string to sanitize
        max_length: Maximum allowed length (default 255)

    Returns:
        Sanitized string, empty string if input is invalid
    """
    if not value or not isinstance(value, str):
        return ""
    sanitized = _CONTROL_CHARS_PATTERN.sub("", value)
    return sanitized[:max_length].strip()


def sanitize_email(email: str, default_domain: str = "unknown.local") -> str:
    """Sanitize and validate email address.

    Args:
        email: The email address to sanitize
        default_domain: Domain to use for invalid emails (default "unknown.local")

    Returns:
        Sanitized email or default fallback email
    """
    if not email or not isinstance(email, str):
        return f"unknown@{default_domain}"
    email = sanitize_string(email.lower(), max_length=255)
    if not EMAIL_PATTERN.match(email):
        return f"unknown@{default_domain}"
    return email


def sanitize_roles(roles: Any) -> list[str]:
    """Sanitize role list.

    Args:
        roles: List of roles or single role value

    Returns:
        List of sanitized role strings, defaults to ["user"] if empty
    """
    if not roles:
        return ["user"]
    if not isinstance(roles, list):
        roles = [roles]
    sanitized = []
    for role in roles:
        if isinstance(role, str):
            clean_role = sanitize_string(role, max_length=50).lower()
            if clean_role and ROLE_PATTERN.match(clean_role):
                sanitized.append(clean_role)
    return sanitized if sanitized else ["user"]
