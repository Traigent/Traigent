"""SAML authentication provider implementation."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from traigent.utils.exceptions import AuthenticationError
from traigent.utils.logging import get_logger

from .helpers import sanitize_email, sanitize_roles, sanitize_string
from .models import User

logger = get_logger(__name__)

# Optional SAML dependency
try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth

    SAML_AVAILABLE = True
except ImportError:
    SAML_AVAILABLE = False
    OneLogin_Saml2_Auth = None


class SAMLAuthProvider:
    """SAML authentication provider with full implementation."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialize SAML provider with settings.

        Args:
            settings: SAML configuration including SP and IdP settings
        """
        if not SAML_AVAILABLE:
            raise ImportError(
                "python3-saml is required for SAML authentication. "
                "Install it with: pip install python3-saml"
            )

        self.settings = settings
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate SAML settings are properly configured."""
        required_sp = ["entityId", "assertionConsumerService"]
        required_idp = ["entityId", "singleSignOnService", "x509cert"]

        if "sp" not in self.settings:
            raise ValueError("SAML settings missing 'sp' configuration")
        if "idp" not in self.settings:
            raise ValueError("SAML settings missing 'idp' configuration")

        for key in required_sp:
            if key not in self.settings["sp"]:
                raise ValueError(f"SAML SP settings missing required key: {key}")

        for key in required_idp:
            if key not in self.settings["idp"]:
                raise ValueError(f"SAML IdP settings missing required key: {key}")

        x509cert = self.settings["idp"].get("x509cert", "")
        if not x509cert or not x509cert.strip():
            raise ValueError("Invalid or empty X.509 certificate")

        sso_url = self.settings["idp"].get("singleSignOnService", {}).get("url", "")
        if not sso_url or not sso_url.startswith("https://"):
            raise ValueError("Single Sign-On Service URL must use HTTPS")

        acs_url = self.settings["sp"].get("assertionConsumerService", {}).get("url", "")
        if not acs_url or not acs_url.startswith("https://"):
            raise ValueError("Assertion Consumer Service URL must use HTTPS")

    def authenticate_saml(
        self, request: dict[str, Any], saml_response: str
    ) -> User | None:
        """Authenticate user via SAML response.

        Args:
            request: HTTP request data (URL, method, etc.)
            saml_response: Base64-encoded SAML response from IdP

        Returns:
            Authenticated User object or None if authentication fails
        """
        try:
            request.setdefault("post_data", {})["SAMLResponse"] = saml_response
            auth = OneLogin_Saml2_Auth(request, self.settings)
            auth.process_response()

            if not auth.is_authenticated():
                errors = auth.get_errors()
                logger.warning(f"SAML authentication failed: {errors}")
                return None

            attributes = auth.get_attributes()
            nameid = auth.get_nameid()
            nameid_format = auth.get_nameid_format()

            if not nameid or not isinstance(nameid, str):
                logger.warning("Invalid or missing SAML NameID")
                return None

            username = sanitize_string(
                attributes.get("uid", [nameid])[0], max_length=32
            )
            email = sanitize_email(
                attributes.get("email", [f"{nameid}@saml"])[0],
                default_domain="saml.local",
            )
            try:
                roles = sanitize_roles(attributes.get("roles", ["user"]), strict=True)
            except ValueError as exc:
                logger.warning("SAML response contains invalid role claims: %s", exc)
                return None

            user = User(
                user_id=sanitize_string(nameid, max_length=255),
                username=username,
                email=email,
                roles=roles,
                metadata={
                    "saml_attributes": self._sanitize_attributes(attributes),
                    "nameid_format": nameid_format,
                    "session_index": auth.get_session_index(),
                    "authenticated_at": datetime.now(UTC).isoformat(),
                    "auth_method": "saml",
                },
            )

            logger.info(f"SAML authentication successful for user: {user.username}")
            return user

        except Exception as e:
            logger.error(f"SAML authentication error: {e}")
            raise AuthenticationError(f"SAML authentication failed: {e}") from None

    def create_login_request(self, request: dict[str, Any]) -> str:
        """Create SAML login request URL."""
        auth = OneLogin_Saml2_Auth(request, self.settings)
        return cast(str, auth.login())

    def create_logout_request(
        self, request: dict[str, Any], name_id: str, session_index: str
    ) -> str:
        """Create SAML logout request URL."""
        auth = OneLogin_Saml2_Auth(request, self.settings)
        return cast(str, auth.logout(name_id=name_id, session_index=session_index))

    def _sanitize_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Sanitize SAML attributes dictionary."""
        if not isinstance(attributes, dict):
            return {}
        sanitized: dict[str, Any] = {}
        for key, value in attributes.items():
            if isinstance(key, str) and len(key) < 100:
                clean_key = sanitize_string(key, max_length=100)
                if isinstance(value, (str, int, float, bool)):
                    sanitized[clean_key] = value
                elif isinstance(value, list) and len(value) < 100:
                    sanitized[clean_key] = [
                        sanitize_string(str(v), max_length=255) for v in value[:100]
                    ]
        return sanitized
