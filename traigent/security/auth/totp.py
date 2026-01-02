"""Time-based One-Time Password (TOTP) authentication provider implementation."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import base64
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import cast

from traigent.utils.logging import get_logger

from .helpers import sanitize_string

logger = get_logger(__name__)

# Optional pyotp dependency
try:
    import pyotp

    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False
    pyotp = None


class TOTPAuthProvider:
    """Time-based One-Time Password (TOTP) authentication provider."""

    def __init__(self, issuer_name: str = "Traigent", secret_length: int = 32) -> None:
        """Initialize TOTP provider.

        Args:
            issuer_name: Name to display in authenticator apps
            secret_length: Length of generated secrets in bytes (default 32 = 256 bits)
        """
        if not PYOTP_AVAILABLE:
            raise ImportError(
                "pyotp is required for TOTP authentication. "
                "Install it with: pip install pyotp"
            )

        self.issuer_name = sanitize_string(issuer_name, max_length=50)
        self.secret_length = min(max(secret_length, 16), 64)
        self._used_codes: dict[str, set[str]] = {}
        self._code_expiry: dict[str, datetime] = {}

    def generate_totp_secret(self) -> str:
        """Generate a new TOTP secret for a user."""
        random_bytes = secrets.token_bytes(self.secret_length)
        secret = base64.b32encode(random_bytes).decode("utf-8")
        logger.info(f"Generated new TOTP secret ({self.secret_length * 8} bits)")
        return secret

    def generate_provisioning_uri(self, username: str, secret: str) -> str:
        """Generate provisioning URI for QR code."""
        totp = pyotp.TOTP(secret)
        return cast(
            str, totp.provisioning_uri(name=username, issuer_name=self.issuer_name)
        )

    def verify_totp_code(
        self, code: str, secret: str, user_id: str, window: int = 1
    ) -> bool:
        """Verify a TOTP code.

        Args:
            code: 6-digit TOTP code from user
            secret: User's TOTP secret
            user_id: User identifier for replay prevention
            window: Number of time windows to check (for clock skew)

        Returns:
            True if code is valid, False otherwise
        """
        try:
            if not code or not re.match(r"^\d{6,8}$", code):
                logger.warning("Invalid TOTP code format")
                return False

            if not secret or not re.match(r"^[A-Z2-7]+=*$", secret):
                logger.warning("Invalid TOTP secret format")
                return False

            if user_id in self._used_codes:
                if code in self._used_codes[user_id]:
                    logger.warning(f"TOTP code replay attempt for user {user_id}")
                    return False

                if (
                    user_id in self._code_expiry
                    and datetime.now(UTC) > self._code_expiry[user_id]
                ):
                    self._used_codes[user_id].clear()

            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(code, valid_window=min(window, 2))

            if is_valid:
                if user_id not in self._used_codes:
                    self._used_codes[user_id] = set()
                self._used_codes[user_id].add(code)
                self._code_expiry[user_id] = datetime.now(UTC) + timedelta(seconds=90)

                if len(self._used_codes[user_id]) > 10:
                    self._used_codes[user_id] = set(
                        list(self._used_codes[user_id])[-5:]
                    )

                logger.info(f"TOTP verification successful for user {user_id}")
            else:
                logger.warning(f"TOTP verification failed for user {user_id}")

            return cast(bool, is_valid)

        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False

    def generate_backup_codes(self, count: int = 10, code_length: int = 8) -> list[str]:
        """Generate backup codes for account recovery."""
        count = min(max(count, 1), 20)
        code_length = min(max(code_length, 8), 16)

        codes: set[str] = set()
        attempts = 0
        max_attempts = count * 10

        while len(codes) < count and attempts < max_attempts:
            bytes_needed = (code_length * 3) // 4 + 1
            raw_code = base64.urlsafe_b64encode(secrets.token_bytes(bytes_needed))[
                :code_length
            ]
            code_str = (
                raw_code.decode("ascii").upper().replace("-", "").replace("_", "")
            )

            if len(code_str) >= code_length:
                codes.add(code_str[:code_length])
            attempts += 1

        codes_list = list(codes)
        logger.info(f"Generated {len(codes_list)} unique backup codes")
        return codes_list
