"""Comprehensive input validation for TraiGent SDK.

Provides secure input validation, sanitization, and protection against
injection attacks including SQL injection, XSS, and command injection.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import html
import re
from typing import Any, TypeVar
from urllib.parse import urlparse

from traigent.security.config import get_security_flags

try:
    import bleach

    BLEACH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BLEACH_AVAILABLE = False

from traigent.utils.exceptions import ValidationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class InputValidator:
    """Comprehensive input validation and sanitization."""

    # Regex patterns for validation
    PATTERNS = {
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "username": re.compile(r"^[a-zA-Z0-9_-]{3,32}$"),
        "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
        "numeric": re.compile(r"^[0-9]+$"),
        "uuid": re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        ),
        "url": re.compile(r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$"),
        "phone": re.compile(r"^\+?[1-9]\d{1,14}$"),  # E.164 format
        "safe_filename": re.compile(r"^[a-zA-Z0-9._-]+$"),
    }

    # SQL injection patterns (common attack vectors)
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        r"(--|#|/\*|\*/)",  # SQL comments
        r"(\bOR\b.*=.*)",  # OR conditions
        r"(\bAND\b.*=.*)",  # AND conditions
        r"(;|\\x00|\\n|\\r|\\x1a)",  # Special characters (quotes handled separately)
        r"(\bSLEEP\b|\bBENCHMARK\b|\bWAITFOR\b)",  # Time-based attacks
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"</iframe>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",  # Shell metacharacters
        r"\$\{.*\}",  # Variable expansion
        r"\\x[0-9a-fA-F]{2}",  # Hex encoding
    ]

    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate and sanitize email address.

        Args:
            email: Email address to validate

        Returns:
            Sanitized email address

        Raises:
            ValidationError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValidationError("Email must be a non-empty string")

        email = email.strip().lower()

        if len(email) > 255:
            raise ValidationError("Email address too long")

        if not cls.PATTERNS["email"].match(email):
            raise ValidationError("Invalid email format")

        return email

    @classmethod
    def validate_username(cls, username: str) -> str:
        """Validate and sanitize username.

        Args:
            username: Username to validate

        Returns:
            Sanitized username

        Raises:
            ValidationError: If username is invalid
        """
        if not username or not isinstance(username, str):
            raise ValidationError("Username must be a non-empty string")

        username = username.strip()

        if not cls.PATTERNS["username"].match(username):
            raise ValidationError(
                "Username must be 3-32 characters, alphanumeric, underscore, or dash"
            )

        if username.isdigit():
            raise ValidationError("Username cannot be numeric only")

        return username

    @classmethod
    def validate_password(cls, password: str, min_length: int = 8) -> None:
        """Validate password strength.

        Args:
            password: Password to validate
            min_length: Minimum password length

        Raises:
            ValidationError: If password is too weak
        """
        if not password or not isinstance(password, str):
            raise ValidationError("Password must be a non-empty string")

        password = password.strip()

        # Check for common weak patterns
        if password.lower() in ["password", "12345678", "qwerty", "admin"]:
            raise ValidationError("Password is too common")

        if len(password) < min_length:
            raise ValidationError(f"Password must be at least {min_length} characters")

        if len(password) > 256:
            raise ValidationError("Password too long (max 256 characters)")

        # Require complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        complexity = sum([has_upper, has_lower, has_digit, has_special])
        if complexity < 3:
            raise ValidationError(
                "Password must contain at least 3 of: uppercase, lowercase, digit, special character"
            )

    @classmethod
    def validate_url(cls, url: str, allowed_schemes: list[str] | None = None) -> str:
        """Validate and sanitize URL.

        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes (default: ['http', 'https'])

        Returns:
            Sanitized URL

        Raises:
            ValidationError: If URL is invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        raw_url = url.strip()
        allowed = [scheme.lower() for scheme in (allowed_schemes or ["http", "https"])]

        parsed = urlparse(raw_url)
        scheme = parsed.scheme.lower()

        if scheme in {"javascript", "data", "vbscript"}:
            raise ValidationError("Potentially malicious URL")

        if scheme not in allowed:
            raise ValidationError(f"URL must use one of: {allowed}")

        if not parsed.netloc:
            raise ValidationError("Invalid URL format")

        if not re.fullmatch(r"[a-zA-Z0-9.-]+", parsed.netloc):
            raise ValidationError("Invalid URL host")

        if any(char in raw_url for char in ['"', "'", "<", ">"]):
            raise ValidationError("Invalid URL format")

        return raw_url

    @classmethod
    def sanitize_html(cls, html_content: str) -> str:
        """Sanitize HTML content to prevent XSS.

        Args:
            html_content: HTML content to sanitize

        Returns:
            Sanitized HTML content
        """
        if not html_content:
            return ""

        flags = get_security_flags()

        if BLEACH_AVAILABLE:
            sanitized = str(
                bleach.clean(
                    html_content,
                    tags=[],
                    attributes={},
                    strip=True,
                )
            )
        else:
            sanitized = html_content
            for pattern in cls.XSS_PATTERNS:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
            sanitized = html.escape(sanitized)

        if flags.emit_security_telemetry and sanitized != html_content:
            logger.info("HTML content sanitized under profile %s", flags.profile.value)

        return sanitized

    @classmethod
    def validate_sql_input(cls, value: str) -> str:
        """Validate input for SQL queries (use parameterized queries instead!).

        Args:
            value: Value to validate

        Returns:
            Sanitized value

        Raises:
            ValidationError: If SQL injection attempt detected
        """
        if not value:
            return ""

        value_upper = value.upper()

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper):
                logger.warning(f"SQL injection attempt detected: {pattern}")
                raise ValidationError("Invalid input detected")

        flags = get_security_flags()
        if flags.strict_sql_validation:
            suspicious_quotes = value.count("'") >= 2 or value.count('"') >= 2
            if suspicious_quotes and re.search(
                r"(['\"][^'\"]*[;]|['\"]\s*(OR|AND)\s)", value_upper
            ):
                logger.warning("Strict SQL validation rejected quoted pattern")
                raise ValidationError("Potentially unsafe quoted SQL input detected")

            if flags.emit_security_telemetry and suspicious_quotes:
                logger.debug(
                    "SQL input contained balanced quotes under profile %s",
                    flags.profile.value,
                )

        # Escape single quotes (minimal protection - USE PARAMETERIZED QUERIES!)
        return value.replace("'", "''")

    @classmethod
    def validate_command_input(cls, value: str) -> str:
        """Validate input for shell commands (avoid shell=True!).

        Args:
            value: Value to validate

        Returns:
            Sanitized value

        Raises:
            ValidationError: If command injection attempt detected
        """
        if not value:
            return ""

        # Check for command injection patterns
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                logger.warning(f"Command injection attempt detected: {pattern}")
                raise ValidationError("Invalid input detected")

        # Remove dangerous characters
        sanitized = re.sub(r"[;&|`$()<>]", "", value)
        return sanitized

    @classmethod
    def validate_numeric(
        cls,
        value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        """Validate numeric input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated numeric value

        Raises:
            ValidationError: If value is invalid
        """
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Value must be numeric") from None

        if min_value is not None and num_value < min_value:
            raise ValidationError(f"Value must be >= {min_value}")

        if max_value is not None and num_value > max_value:
            raise ValidationError(f"Value must be <= {max_value}")

        return num_value

    @classmethod
    def validate_integer(
        cls,
        value: Any,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        """Validate integer input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer value

        Raises:
            ValidationError: If value is invalid
        """
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValidationError("Value must be an integer") from None

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"Value must be >= {min_value}")

        if max_value is not None and int_value > max_value:
            raise ValidationError(f"Value must be <= {max_value}")

        return int_value

    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate and sanitize filename.

        Args:
            filename: Filename to validate

        Returns:
            Sanitized filename

        Raises:
            ValidationError: If filename is invalid
        """
        if not filename or not isinstance(filename, str):
            raise ValidationError("Filename must be a non-empty string")

        # Remove path components
        filename = filename.replace("/", "").replace("\\", "")

        # Check for directory traversal
        if ".." in filename:
            raise ValidationError("Invalid filename")

        # Allow only safe characters
        if not cls.PATTERNS["safe_filename"].match(filename):
            raise ValidationError(
                "Filename must contain only alphanumeric, dot, underscore, or dash"
            )

        # Limit length
        if len(filename) > 255:
            raise ValidationError("Filename too long")

        return filename

    @classmethod
    def validate_json_input(
        cls, data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate JSON input against a schema.

        Args:
            data: JSON data to validate
            schema: Validation schema

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails
        """
        validated = {}

        for field, rules in schema.items():
            value = data.get(field)

            # Check required
            if rules.get("required", False) and value is None:
                raise ValidationError(f"Field '{field}' is required")

            if value is not None:
                # Type validation
                expected_type = rules.get("type")
                if expected_type:
                    if expected_type == "string" and not isinstance(value, str):
                        raise ValidationError(f"Field '{field}' must be a string")
                    elif expected_type == "integer" and not isinstance(value, int):
                        raise ValidationError(f"Field '{field}' must be an integer")
                    elif expected_type == "number" and not isinstance(
                        value, (int, float)
                    ):
                        raise ValidationError(f"Field '{field}' must be a number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        raise ValidationError(f"Field '{field}' must be a boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        raise ValidationError(f"Field '{field}' must be an array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        raise ValidationError(f"Field '{field}' must be an object")

                # Additional validation
                validator = rules.get("validator")
                if validator and callable(validator):
                    value = validator(value)

                validated[field] = value

        return validated


class SanitizationHelper:
    """Helper class for data sanitization."""

    @staticmethod
    def remove_null_bytes(value: str) -> str:
        """Remove null bytes from string."""
        return value.replace("\x00", "")

    @staticmethod
    def normalize_whitespace(value: str) -> str:
        """Normalize whitespace in string."""
        return " ".join(value.split())

    @staticmethod
    def strip_control_chars(value: str) -> str:
        """Remove control characters from string."""
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

    @staticmethod
    def truncate(value: str, max_length: int) -> str:
        """Truncate string to maximum length."""
        return value[:max_length] if len(value) > max_length else value
