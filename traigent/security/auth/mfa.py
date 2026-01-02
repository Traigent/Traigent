"""Unified Multi-Factor Authentication manager."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import re
from typing import Any

from traigent.utils.logging import get_logger

from .sms import SMSAuthProvider
from .totp import TOTPAuthProvider

logger = get_logger(__name__)


class MultiFactorAuth:
    """Unified Multi-Factor Authentication manager."""

    def __init__(self) -> None:
        """Initialize MFA manager with all providers."""
        self.totp_provider: TOTPAuthProvider | None = None
        self.sms_provider: SMSAuthProvider | None = None

    def enable_totp(self, issuer_name: str = "Traigent") -> None:
        """Enable TOTP authentication."""
        self.totp_provider = TOTPAuthProvider(issuer_name)

    def enable_sms(self, twilio_settings: dict[str, Any]) -> None:
        """Enable SMS authentication."""
        self.sms_provider = SMSAuthProvider(twilio_settings)

    def verify_mfa(
        self, user_id: str, method: str, code: str, secret: str | None = None
    ) -> bool:
        """Verify MFA code using specified method.

        Args:
            user_id: User identifier
            method: MFA method ("totp" or "sms")
            code: Verification code
            secret: TOTP secret (required for TOTP method)

        Returns:
            True if verification successful, False otherwise
        """
        if method not in ["totp", "sms"]:
            logger.warning(f"Unknown MFA method attempted: {method}")
            raise ValueError(f"Unknown MFA method: {method}")

        if not user_id or not isinstance(user_id, str) or len(user_id) > 255:
            logger.warning("Invalid user_id for MFA verification")
            return False

        if not code or not re.match(r"^\d{6,8}$", code):
            logger.warning(f"Invalid code format for MFA method {method}")
            return False

        try:
            if method == "totp":
                if not self.totp_provider:
                    raise ValueError("TOTP provider not enabled")
                if not secret:
                    raise ValueError("TOTP secret required")
                return self.totp_provider.verify_totp_code(code, secret, user_id)

            elif method == "sms":
                if not self.sms_provider:
                    raise ValueError("SMS provider not enabled")
                return self.sms_provider.verify_sms_code(user_id, code)

        except Exception as e:
            logger.error(f"MFA verification error for {method}: {e}")
            return False

        return False  # Unreachable but satisfies type checker
