"""Authentication providers for TraiGent.

This package provides complete implementations of SAML, OIDC, TOTP, and SMS
authentication methods with enterprise security features.

Example usage:
    from traigent.security.auth import User, SAMLAuthProvider, OIDCAuthProvider
    from traigent.security.auth import TOTPAuthProvider, SMSAuthProvider
    from traigent.security.auth import MultiFactorAuth
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from .helpers import (
    EMAIL_PATTERN,
    ROLE_PATTERN,
    TIMING_ATTACK_DELAY_SECONDS,
    sanitize_email,
    sanitize_roles,
    sanitize_string,
)
from .mfa import MultiFactorAuth
from .models import User
from .oidc import JWT_AVAILABLE, OIDCAuthProvider
from .saml import SAML_AVAILABLE, SAMLAuthProvider
from .sms import TWILIO_AVAILABLE, SMSAuthProvider
from .totp import PYOTP_AVAILABLE, TOTPAuthProvider

__all__ = [
    # Models
    "User",
    # Providers
    "SAMLAuthProvider",
    "OIDCAuthProvider",
    "TOTPAuthProvider",
    "SMSAuthProvider",
    "MultiFactorAuth",
    # Helpers (for backwards compatibility)
    "sanitize_string",
    "sanitize_email",
    "sanitize_roles",
    # Constants
    "EMAIL_PATTERN",
    "ROLE_PATTERN",
    "TIMING_ATTACK_DELAY_SECONDS",
    # Availability flags (for backwards compatibility)
    "SAML_AVAILABLE",
    "JWT_AVAILABLE",
    "PYOTP_AVAILABLE",
    "TWILIO_AVAILABLE",
]
