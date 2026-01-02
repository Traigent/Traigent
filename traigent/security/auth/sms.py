"""SMS-based authentication provider using Twilio."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from traigent.utils.exceptions import AuthenticationError
from traigent.utils.logging import get_logger

from .helpers import TIMING_ATTACK_DELAY_SECONDS

logger = get_logger(__name__)

# Optional Twilio dependency
try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioClient = None


class SMSAuthProvider:
    """SMS-based authentication provider using Twilio."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialize SMS provider with Twilio settings."""
        if not TWILIO_AVAILABLE:
            raise ImportError(
                "twilio is required for SMS authentication. "
                "Install it with: pip install twilio"
            )

        account_sid = settings.get("account_sid")
        auth_token = settings.get("auth_token")
        from_number = settings.get("from_number")

        if not all([account_sid, auth_token, from_number]):
            raise ValueError("SMS settings missing required Twilio configuration")

        self.account_sid = str(account_sid)
        self.auth_token = str(auth_token)
        self.from_number = str(from_number)

        self.max_attempts = int(settings.get("max_attempts", 3))
        self.code_expiry_minutes = int(settings.get("code_expiry_minutes", 5))
        self.rate_limit_minutes = int(settings.get("rate_limit_minutes", 1))
        self.max_codes_per_hour = int(settings.get("max_codes_per_hour", 5))

        if not re.match(r"^\+[1-9]\d{1,14}$", self.from_number):
            raise ValueError("Invalid from_number format. Must be E.164 format")

        self.client = TwilioClient(self.account_sid, self.auth_token)
        self._verification_codes: dict[str, dict[str, Any]] = {}
        self._rate_limit: dict[str, list[datetime]] = {}

    def send_verification_code(self, phone_number: str, user_id: str) -> str:
        """Send SMS verification code to user."""
        try:
            if not re.match(r"^\+[1-9]\d{1,14}$", phone_number):
                raise ValueError("Invalid phone number format. Must be E.164 format")

            if not user_id or not isinstance(user_id, str) or len(user_id) > 255:
                raise ValueError("Invalid user_id")

            now = datetime.now(UTC)
            if user_id in self._rate_limit:
                self._rate_limit[user_id] = [
                    t for t in self._rate_limit[user_id] if now - t < timedelta(hours=1)
                ]

                if len(self._rate_limit[user_id]) >= self.max_codes_per_hour:
                    logger.warning(f"Rate limit exceeded for user {user_id}")
                    raise AuthenticationError(
                        "Too many verification codes requested. Please try again later."
                    )

                if (
                    self._rate_limit[user_id]
                    and (now - self._rate_limit[user_id][-1]).seconds
                    < self.rate_limit_minutes * 60
                ):
                    logger.warning(f"SMS requested too soon for user {user_id}")
                    raise AuthenticationError(
                        f"Please wait {self.rate_limit_minutes} minute(s) before requesting a new code."
                    )

            code = str(secrets.randbelow(900000) + 100000)

            code_hash = hmac.new(
                self.auth_token.encode(), f"{user_id}:{code}".encode(), hashlib.sha256
            ).hexdigest()

            self._verification_codes[user_id] = {
                "code_hash": code_hash,
                "phone_number": phone_number,
                "expires_at": now + timedelta(minutes=self.code_expiry_minutes),
                "attempts": 0,
                "created_at": now,
            }

            if user_id not in self._rate_limit:
                self._rate_limit[user_id] = []
            self._rate_limit[user_id].append(now)

            message = self.client.messages.create(
                body=f"Your Traigent verification code is: {code}\n\nThis code expires in {self.code_expiry_minutes} minutes.",
                from_=self.from_number,
                to=phone_number,
            )

            logger.info(
                f"SMS verification code sent to user {user_id} (SID: {message.sid})"
            )
            return cast(str, message.sid)

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            raise AuthenticationError(
                f"SMS sending failed: {str(e).replace(self.auth_token, '***')}"
            ) from e

    def verify_sms_code(self, user_id: str, code: str) -> bool:
        """Verify an SMS verification code."""
        if not user_id or not isinstance(user_id, str):
            logger.warning("Invalid user_id for SMS verification")
            return False

        if not code or not re.match(r"^\d{6}$", code):
            logger.warning("Invalid code format for SMS verification")
            return False

        if user_id not in self._verification_codes:
            logger.warning(f"No verification code found for user {user_id}")
            time.sleep(TIMING_ATTACK_DELAY_SECONDS)
            return False

        stored = self._verification_codes[user_id]

        if datetime.now(UTC) > stored["expires_at"]:
            logger.warning(f"Verification code expired for user {user_id}")
            del self._verification_codes[user_id]
            time.sleep(TIMING_ATTACK_DELAY_SECONDS)
            return False

        stored["attempts"] += 1

        if stored["attempts"] > self.max_attempts:
            logger.warning(f"Too many verification attempts for user {user_id}")
            del self._verification_codes[user_id]
            time.sleep(TIMING_ATTACK_DELAY_SECONDS)
            return False

        code_hash = hmac.new(
            self.auth_token.encode(), f"{user_id}:{code}".encode(), hashlib.sha256
        ).hexdigest()

        is_valid = hmac.compare_digest(stored["code_hash"], code_hash)

        if is_valid:
            logger.info(f"SMS verification successful for user {user_id}")
            del self._verification_codes[user_id]
            if user_id in self._rate_limit:
                del self._rate_limit[user_id]
            return True
        else:
            logger.warning(
                f"Invalid SMS code attempt {stored['attempts']} for user {user_id}"
            )
            time.sleep(TIMING_ATTACK_DELAY_SECONDS * stored["attempts"])
            return False

    def send_notification(self, phone_number: str, message: str) -> str:
        """Send a notification SMS."""
        try:
            msg = self.client.messages.create(
                body=message, from_=self.from_number, to=phone_number
            )
            logger.info(f"Notification SMS sent to {phone_number}")
            return cast(str, msg.sid)
        except Exception as e:
            logger.error(f"Failed to send notification SMS: {e}")
            raise AuthenticationError(f"SMS notification failed: {e}") from None
