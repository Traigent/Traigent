"""Security headers middleware for TraiGent API endpoints.

Provides comprehensive security headers following OWASP best practices
to protect against common web vulnerabilities.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import urlparse

from traigent.config.backend_config import BackendConfig
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware:
    """Middleware to add security headers to HTTP responses."""

    def __init__(
        self,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_cors: bool = False,
        allowed_origins: list[str] | None = None,
        csp_directives: dict[str, str] | None = None,
    ) -> None:
        """Initialize security headers middleware.

        Args:
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            enable_cors: Enable CORS headers (for APIs)
            allowed_origins: List of allowed origins for CORS
            csp_directives: Custom CSP directives
        """
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.enable_cors = enable_cors
        self.allowed_origins = allowed_origins or []

        # Default CSP directives (restrictive)
        self.csp_directives = csp_directives or {
            "default-src": "'self'",
            "script-src": "'self'",
            "style-src": "'self'",
            "img-src": "'self' data: https:",
            "font-src": "'self' data:",
            "connect-src": "'self'",
            "media-src": "'self'",
            "object-src": "'none'",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "upgrade-insecure-requests": "",
        }

    def apply_headers(self, response: Any, request: Any | None = None) -> Any:
        """Apply security headers to response.

        Args:
            response: HTTP response object
            request: HTTP request object (for CORS)

        Returns:
            Response with security headers applied
        """
        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), "
            "accelerometer=(), gyroscope=()"
        )

        # HTTP Strict Transport Security
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        if self.enable_csp:
            csp_header = self._build_csp_header()
            response.headers["Content-Security-Policy"] = csp_header
            # Report-only mode for testing
            # response.headers["Content-Security-Policy-Report-Only"] = csp_header

        # CORS headers (for APIs)
        if self.enable_cors and request:
            origin = request.headers.get("Origin")
            if origin and self._is_allowed_origin(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS"
                )
                response.headers["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization, X-Requested-With"
                )
                response.headers["Access-Control-Max-Age"] = "86400"
            elif self.allowed_origins == ["*"]:
                response.headers["Access-Control-Allow-Origin"] = "*"

        # Remove server identification headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)

        # Add cache control for sensitive endpoints
        if self._is_sensitive_endpoint(request):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

    def _build_csp_header(self) -> str:
        """Build Content Security Policy header value.

        Returns:
            CSP header string
        """
        directives = []
        for directive, value in self.csp_directives.items():
            if value:
                directives.append(f"{directive} {value}")
            else:
                directives.append(directive)

        return "; ".join(directives)

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed for CORS.

        Args:
            origin: Origin header value

        Returns:
            True if origin is allowed
        """
        if "*" in self.allowed_origins:
            return True

        # Exact match
        if origin in self.allowed_origins:
            return True

        # Check for wildcard subdomains
        for allowed in self.allowed_origins:
            if allowed.startswith("*."):
                domain = allowed[2:]
                if origin.endswith(domain):
                    return True

        return False

    def _is_sensitive_endpoint(self, request: Any | None) -> bool:
        """Check if endpoint handles sensitive data.

        Args:
            request: HTTP request object

        Returns:
            True if endpoint is sensitive
        """
        if not request:
            return False

        sensitive_paths = [
            "/api/auth",
            "/api/login",
            "/api/logout",
            "/api/session",
            "/api/user",
            "/api/admin",
            "/api/config",
            "/api/keys",
        ]

        path = getattr(request, "path", "")
        return any(path.startswith(p) for p in sensitive_paths)


class NonceGenerator:
    """Generate nonces for CSP inline scripts/styles."""

    @staticmethod
    def generate_nonce() -> str:
        """Generate a cryptographic nonce.

        Returns:
            Base64-encoded nonce string
        """
        return secrets.token_urlsafe(16)

    @staticmethod
    def generate_hash(content: str, algorithm: str = "sha256") -> str:
        """Generate hash for inline content.

        Args:
            content: Inline script/style content
            algorithm: Hash algorithm (sha256, sha384, sha512)

        Returns:
            Base64-encoded hash with algorithm prefix
        """
        if algorithm not in ["sha256", "sha384", "sha512"]:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode("utf-8"))
        hash_b64 = base64.b64encode(hash_obj.digest()).decode("ascii")

        return f"'{algorithm}-{hash_b64}'"


def create_flask_security_headers(app: Any) -> None:
    """Apply security headers to Flask application.

    Args:
        app: Flask application instance
    """
    middleware = SecurityHeadersMiddleware()

    @app.after_request
    def add_security_headers(response: Any) -> Any:
        """Add security headers to every response."""
        return middleware.apply_headers(response, request=None)


def create_fastapi_security_headers(app: Any) -> None:
    """Apply security headers to FastAPI application.

    Args:
        app: FastAPI application instance
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class SecurityHeadersMiddlewareFastAPI(BaseHTTPMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(app)
            self.headers_middleware = SecurityHeadersMiddleware()

        async def dispatch(
            self, request: Request, call_next: Callable[..., Any]
        ) -> Response:
            response = await call_next(request)
            self.headers_middleware.apply_headers(response, request)
            return cast(Response, response)

    app.add_middleware(SecurityHeadersMiddlewareFastAPI)


def _default_backend_origin() -> str:
    """Return the origin for the configured Traigent backend."""

    backend_url = BackendConfig.get_backend_url().rstrip("/")
    parsed = urlparse(backend_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return backend_url or BackendConfig.DEFAULT_PROD_URL


# Example configurations for common scenarios

STRICT_CSP = {
    "default-src": "'none'",
    "script-src": "'self'",
    "style-src": "'self'",
    "img-src": "'self'",
    "font-src": "'self'",
    "connect-src": "'self'",
    "media-src": "'none'",
    "object-src": "'none'",
    "frame-ancestors": "'none'",
    "base-uri": "'self'",
    "form-action": "'self'",
    "upgrade-insecure-requests": "",
}

API_CSP = {
    "default-src": "'none'",
    "frame-ancestors": "'none'",
    "base-uri": "'self'",
}

RELAXED_CSP = {
    "default-src": "'self'",
    "script-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "img-src": "'self' data: https:",
    "font-src": "'self' data: https:",
    "connect-src": f"'self' {_default_backend_origin()}",
    "media-src": "'self'",
    "object-src": "'none'",
    "frame-ancestors": "'self'",
    "base-uri": "'self'",
    "form-action": "'self'",
}
