"""Unit tests for traigent/security/headers.py.

Tests for security headers middleware, CORS validation, CSP header construction,
and framework middleware integration.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.security.headers import (
    API_CSP,
    RELAXED_CSP,
    STRICT_CSP,
    NonceGenerator,
    SecurityHeadersMiddleware,
    _default_backend_origin,
    create_fastapi_security_headers,
    create_flask_security_headers,
)


class TestSecurityHeadersMiddleware:
    """Tests for SecurityHeadersMiddleware class."""

    @pytest.fixture
    def middleware(self) -> SecurityHeadersMiddleware:
        """Create middleware instance with default configuration."""
        return SecurityHeadersMiddleware()

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create mock HTTP response object."""
        response = MagicMock()
        response.headers = {}
        return response

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock HTTP request object."""
        request = MagicMock()
        request.headers = {}
        request.path = "/api/test"
        return request

    # Initialization tests
    def test_init_default_values(self) -> None:
        """Test middleware initialization with default values."""
        middleware = SecurityHeadersMiddleware()
        assert middleware.enable_hsts is True
        assert middleware.enable_csp is True
        assert middleware.enable_cors is False
        assert middleware.allowed_origins == []
        assert "default-src" in middleware.csp_directives
        assert middleware.csp_directives["default-src"] == "'self'"

    def test_init_custom_values(self) -> None:
        """Test middleware initialization with custom values."""
        custom_csp = {"default-src": "'none'", "script-src": "'self'"}
        middleware = SecurityHeadersMiddleware(
            enable_hsts=False,
            enable_csp=False,
            enable_cors=True,
            allowed_origins=["https://example.com"],
            csp_directives=custom_csp,
        )
        assert middleware.enable_hsts is False
        assert middleware.enable_csp is False
        assert middleware.enable_cors is True
        assert middleware.allowed_origins == ["https://example.com"]
        assert middleware.csp_directives == custom_csp

    def test_init_empty_allowed_origins(self) -> None:
        """Test middleware initialization with None allowed_origins."""
        middleware = SecurityHeadersMiddleware(allowed_origins=None)
        assert middleware.allowed_origins == []

    # Core security headers tests
    def test_apply_headers_core_security(
        self, middleware: SecurityHeadersMiddleware, mock_response: MagicMock
    ) -> None:
        """Test that core security headers are applied."""
        result = middleware.apply_headers(mock_response)

        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-Frame-Options"] == "DENY"
        assert result.headers["X-XSS-Protection"] == "1; mode=block"
        assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "geolocation=()" in result.headers["Permissions-Policy"]
        assert "microphone=()" in result.headers["Permissions-Policy"]
        assert "camera=()" in result.headers["Permissions-Policy"]

    def test_apply_headers_hsts_enabled(self, mock_response: MagicMock) -> None:
        """Test HSTS header when enabled."""
        middleware = SecurityHeadersMiddleware(enable_hsts=True)
        result = middleware.apply_headers(mock_response)

        assert "Strict-Transport-Security" in result.headers
        assert "max-age=31536000" in result.headers["Strict-Transport-Security"]
        assert "includeSubDomains" in result.headers["Strict-Transport-Security"]
        assert "preload" in result.headers["Strict-Transport-Security"]

    def test_apply_headers_hsts_disabled(self, mock_response: MagicMock) -> None:
        """Test HSTS header when disabled."""
        middleware = SecurityHeadersMiddleware(enable_hsts=False)
        result = middleware.apply_headers(mock_response)

        assert "Strict-Transport-Security" not in result.headers

    def test_apply_headers_csp_enabled(self, mock_response: MagicMock) -> None:
        """Test CSP header when enabled."""
        middleware = SecurityHeadersMiddleware(enable_csp=True)
        result = middleware.apply_headers(mock_response)

        assert "Content-Security-Policy" in result.headers
        csp = result.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "object-src 'none'" in csp

    def test_apply_headers_csp_disabled(self, mock_response: MagicMock) -> None:
        """Test CSP header when disabled."""
        middleware = SecurityHeadersMiddleware(enable_csp=False)
        result = middleware.apply_headers(mock_response)

        assert "Content-Security-Policy" not in result.headers

    # CORS tests
    def test_apply_headers_cors_disabled(
        self,
        middleware: SecurityHeadersMiddleware,
        mock_response: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """Test no CORS headers when CORS is disabled."""
        # CORS disabled by default
        result = middleware.apply_headers(mock_response, mock_request)

        assert "Access-Control-Allow-Origin" not in result.headers
        assert "Access-Control-Allow-Credentials" not in result.headers

    def test_apply_headers_cors_allowed_origin(
        self, mock_response: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test CORS headers with allowed origin."""
        middleware = SecurityHeadersMiddleware(
            enable_cors=True, allowed_origins=["https://example.com"]
        )
        mock_request.headers = {"Origin": "https://example.com"}

        result = middleware.apply_headers(mock_response, mock_request)

        assert result.headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert result.headers["Access-Control-Allow-Credentials"] == "true"
        assert (
            "GET, POST, PUT, DELETE, OPTIONS"
            in result.headers["Access-Control-Allow-Methods"]
        )
        assert "Content-Type" in result.headers["Access-Control-Allow-Headers"]
        assert "Authorization" in result.headers["Access-Control-Allow-Headers"]
        assert result.headers["Access-Control-Max-Age"] == "86400"

    def test_apply_headers_cors_disallowed_origin(
        self, mock_response: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test CORS headers with disallowed origin."""
        middleware = SecurityHeadersMiddleware(
            enable_cors=True, allowed_origins=["https://example.com"]
        )
        mock_request.headers = {"Origin": "https://malicious.com"}

        result = middleware.apply_headers(mock_response, mock_request)

        assert "Access-Control-Allow-Origin" not in result.headers

    def test_apply_headers_cors_wildcard_with_origin(
        self, mock_response: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test CORS headers with wildcard when origin is present."""
        middleware = SecurityHeadersMiddleware(enable_cors=True, allowed_origins=["*"])
        mock_request.headers = {"Origin": "https://any-origin.com"}

        result = middleware.apply_headers(mock_response, mock_request)

        # When origin header is present, it's echoed back
        assert result.headers["Access-Control-Allow-Origin"] == (
            "https://any-origin.com"
        )

    def test_apply_headers_cors_wildcard_no_origin(
        self, mock_response: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test CORS headers with wildcard when no origin header."""
        middleware = SecurityHeadersMiddleware(enable_cors=True, allowed_origins=["*"])
        mock_request.headers = {}

        result = middleware.apply_headers(mock_response, mock_request)

        # When no origin header, wildcard is used
        assert result.headers["Access-Control-Allow-Origin"] == "*"

    def test_apply_headers_cors_no_request(self, mock_response: MagicMock) -> None:
        """Test CORS headers when no request provided."""
        middleware = SecurityHeadersMiddleware(enable_cors=True)
        result = middleware.apply_headers(mock_response, request=None)

        assert "Access-Control-Allow-Origin" not in result.headers

    def test_apply_headers_cors_no_origin_header(
        self, mock_response: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test CORS headers when request has no Origin header."""
        middleware = SecurityHeadersMiddleware(
            enable_cors=True, allowed_origins=["https://example.com"]
        )
        mock_request.headers = {}

        result = middleware.apply_headers(mock_response, mock_request)

        assert "Access-Control-Allow-Origin" not in result.headers

    # Server header removal tests
    def test_apply_headers_removes_server_headers(
        self, middleware: SecurityHeadersMiddleware, mock_response: MagicMock
    ) -> None:
        """Test removal of server identification headers."""
        mock_response.headers = {"Server": "Apache/2.4.1", "X-Powered-By": "PHP/7.4"}

        result = middleware.apply_headers(mock_response)

        assert "Server" not in result.headers
        assert "X-Powered-By" not in result.headers

    # Sensitive endpoint tests
    def test_apply_headers_sensitive_endpoint_auth(
        self,
        middleware: SecurityHeadersMiddleware,
        mock_response: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """Test cache control headers for auth endpoints."""
        mock_request.path = "/api/auth/login"

        result = middleware.apply_headers(mock_response, mock_request)

        assert "no-store" in result.headers["Cache-Control"]
        assert "no-cache" in result.headers["Cache-Control"]
        assert "must-revalidate" in result.headers["Cache-Control"]
        assert "private" in result.headers["Cache-Control"]
        assert result.headers["Pragma"] == "no-cache"
        assert result.headers["Expires"] == "0"

    def test_apply_headers_sensitive_endpoint_user(
        self,
        middleware: SecurityHeadersMiddleware,
        mock_response: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """Test cache control headers for user endpoints."""
        mock_request.path = "/api/user/profile"

        result = middleware.apply_headers(mock_response, mock_request)

        assert "Cache-Control" in result.headers
        assert "no-store" in result.headers["Cache-Control"]

    def test_apply_headers_non_sensitive_endpoint(
        self,
        middleware: SecurityHeadersMiddleware,
        mock_response: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """Test no cache control headers for non-sensitive endpoints."""
        mock_request.path = "/api/public/data"

        result = middleware.apply_headers(mock_response, mock_request)

        # Cache-Control should not be set for non-sensitive endpoints
        assert (
            "Cache-Control" not in result.headers
            or "no-store" not in result.headers.get("Cache-Control", "")
        )

    # CSP header building tests
    def test_build_csp_header_default(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test building CSP header with default directives."""
        csp = middleware._build_csp_header()

        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "object-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'self'" in csp

    def test_build_csp_header_custom(self) -> None:
        """Test building CSP header with custom directives."""
        custom_csp = {
            "default-src": "'none'",
            "script-src": "'self' 'unsafe-inline'",
            "style-src": "'self'",
        }
        middleware = SecurityHeadersMiddleware(csp_directives=custom_csp)

        csp = middleware._build_csp_header()

        assert "default-src 'none'" in csp
        assert "script-src 'self' 'unsafe-inline'" in csp
        assert "style-src 'self'" in csp

    def test_build_csp_header_empty_value(self) -> None:
        """Test building CSP header with empty directive values."""
        custom_csp = {
            "upgrade-insecure-requests": "",
            "block-all-mixed-content": "",
        }
        middleware = SecurityHeadersMiddleware(csp_directives=custom_csp)

        csp = middleware._build_csp_header()

        assert "upgrade-insecure-requests" in csp
        assert "block-all-mixed-content" in csp
        # Empty values should not have trailing space
        assert "upgrade-insecure-requests;" in csp or csp.endswith(
            "upgrade-insecure-requests"
        )

    # Origin validation tests
    def test_is_allowed_origin_wildcard(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test origin validation with wildcard."""
        middleware.allowed_origins = ["*"]

        assert middleware._is_allowed_origin("https://example.com")
        assert middleware._is_allowed_origin("http://localhost:3000")
        assert middleware._is_allowed_origin("https://any-domain.com")

    def test_is_allowed_origin_exact_match(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test origin validation with exact match."""
        middleware.allowed_origins = ["https://example.com", "https://app.example.com"]

        assert middleware._is_allowed_origin("https://example.com")
        assert middleware._is_allowed_origin("https://app.example.com")
        assert not middleware._is_allowed_origin("https://other.com")

    def test_is_allowed_origin_subdomain_wildcard(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test origin validation with subdomain wildcard."""
        middleware.allowed_origins = ["*.example.com"]

        assert middleware._is_allowed_origin("https://app.example.com")
        assert middleware._is_allowed_origin("http://api.example.com")
        assert middleware._is_allowed_origin("https://subdomain.example.com")
        # endswith() matches the domain itself too
        assert middleware._is_allowed_origin("https://example.com")
        assert not middleware._is_allowed_origin("https://other.com")

    def test_is_allowed_origin_subdomain_wildcard_with_protocol(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test origin validation with subdomain wildcard including protocol."""
        middleware.allowed_origins = ["*.example.com"]

        # Should match any subdomain regardless of protocol
        assert middleware._is_allowed_origin("https://api.example.com")
        assert middleware._is_allowed_origin("wss://websocket.example.com")

    def test_is_allowed_origin_empty_list(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test origin validation with empty allowed origins."""
        middleware.allowed_origins = []

        assert not middleware._is_allowed_origin("https://example.com")

    # Sensitive endpoint detection tests
    def test_is_sensitive_endpoint_no_request(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test sensitive endpoint detection with no request."""
        assert middleware._is_sensitive_endpoint(None) is False

    def test_is_sensitive_endpoint_auth_paths(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test sensitive endpoint detection for auth paths."""
        mock_request = MagicMock()

        sensitive_paths = [
            "/api/auth/login",
            "/api/auth/register",
            "/api/login",
            "/api/logout",
        ]

        for path in sensitive_paths:
            mock_request.path = path
            assert middleware._is_sensitive_endpoint(mock_request) is True

    def test_is_sensitive_endpoint_user_paths(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test sensitive endpoint detection for user paths."""
        mock_request = MagicMock()

        sensitive_paths = [
            "/api/user/profile",
            "/api/user/settings",
            "/api/session/active",
            "/api/admin/users",
            "/api/config/settings",
            "/api/keys/list",
        ]

        for path in sensitive_paths:
            mock_request.path = path
            assert middleware._is_sensitive_endpoint(mock_request) is True

    def test_is_sensitive_endpoint_public_paths(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test sensitive endpoint detection for public paths."""
        mock_request = MagicMock()

        public_paths = [
            "/api/public/data",
            "/api/health",
            "/api/status",
            "/docs",
            "/",
        ]

        for path in public_paths:
            mock_request.path = path
            assert middleware._is_sensitive_endpoint(mock_request) is False

    def test_is_sensitive_endpoint_no_path_attribute(
        self, middleware: SecurityHeadersMiddleware
    ) -> None:
        """Test sensitive endpoint detection when request has no path."""
        mock_request = MagicMock(spec=[])

        # Should handle missing path attribute gracefully
        assert middleware._is_sensitive_endpoint(mock_request) is False


class TestNonceGenerator:
    """Tests for NonceGenerator class."""

    def test_generate_nonce_returns_string(self) -> None:
        """Test that generate_nonce returns a string."""
        nonce = NonceGenerator.generate_nonce()
        assert isinstance(nonce, str)

    def test_generate_nonce_not_empty(self) -> None:
        """Test that generated nonce is not empty."""
        nonce = NonceGenerator.generate_nonce()
        assert len(nonce) > 0

    def test_generate_nonce_is_unique(self) -> None:
        """Test that generated nonces are unique."""
        nonces = [NonceGenerator.generate_nonce() for _ in range(100)]
        assert len(set(nonces)) == 100

    def test_generate_nonce_is_url_safe(self) -> None:
        """Test that generated nonce is URL-safe."""
        nonce = NonceGenerator.generate_nonce()
        # URL-safe base64 only contains alphanumeric, -, and _
        assert all(c.isalnum() or c in ["-", "_"] for c in nonce)

    def test_generate_hash_sha256(self) -> None:
        """Test hash generation with SHA-256."""
        content = "console.log('test');"
        hash_value = NonceGenerator.generate_hash(content, "sha256")

        assert hash_value.startswith("'sha256-")
        assert hash_value.endswith("'")
        assert len(hash_value) > 10

    def test_generate_hash_sha384(self) -> None:
        """Test hash generation with SHA-384."""
        content = "console.log('test');"
        hash_value = NonceGenerator.generate_hash(content, "sha384")

        assert hash_value.startswith("'sha384-")
        assert hash_value.endswith("'")

    def test_generate_hash_sha512(self) -> None:
        """Test hash generation with SHA-512."""
        content = "console.log('test');"
        hash_value = NonceGenerator.generate_hash(content, "sha512")

        assert hash_value.startswith("'sha512-")
        assert hash_value.endswith("'")

    def test_generate_hash_deterministic(self) -> None:
        """Test that hash generation is deterministic."""
        content = "console.log('test');"
        hash1 = NonceGenerator.generate_hash(content)
        hash2 = NonceGenerator.generate_hash(content)

        assert hash1 == hash2

    def test_generate_hash_different_content(self) -> None:
        """Test that different content produces different hashes."""
        hash1 = NonceGenerator.generate_hash("content1")
        hash2 = NonceGenerator.generate_hash("content2")

        assert hash1 != hash2

    def test_generate_hash_unsupported_algorithm(self) -> None:
        """Test that unsupported algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            NonceGenerator.generate_hash("content", "md5")

    def test_generate_hash_empty_content(self) -> None:
        """Test hash generation with empty content."""
        hash_value = NonceGenerator.generate_hash("")

        assert hash_value.startswith("'sha256-")
        assert hash_value.endswith("'")

    def test_generate_hash_unicode_content(self) -> None:
        """Test hash generation with unicode content."""
        content = "console.log('Hello, 世界!');"
        hash_value = NonceGenerator.generate_hash(content)

        assert hash_value.startswith("'sha256-")
        assert hash_value.endswith("'")


class TestFlaskIntegration:
    """Tests for Flask integration."""

    @pytest.fixture
    def mock_flask_app(self) -> MagicMock:
        """Create mock Flask application."""
        app = MagicMock()
        app.after_request = MagicMock()
        return app

    def test_create_flask_security_headers(self, mock_flask_app: MagicMock) -> None:
        """Test Flask security headers integration."""
        create_flask_security_headers(mock_flask_app)

        # Verify after_request decorator was called
        mock_flask_app.after_request.assert_called_once()

    def test_flask_decorator_applies_headers(self, mock_flask_app: MagicMock) -> None:
        """Test that Flask decorator applies security headers."""
        create_flask_security_headers(mock_flask_app)

        # Get the registered callback
        callback = mock_flask_app.after_request.call_args[0][0]

        # Create mock response
        mock_response = MagicMock()
        mock_response.headers = {}

        # Call the callback
        result = callback(mock_response)

        # Verify headers were applied
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers


class TestFastAPIIntegration:
    """Tests for FastAPI integration."""

    @pytest.fixture
    def mock_fastapi_app(self) -> MagicMock:
        """Create mock FastAPI application."""
        app = MagicMock()
        app.add_middleware = MagicMock()
        return app

    def test_create_fastapi_security_headers(self, mock_fastapi_app: MagicMock) -> None:
        """Test FastAPI security headers integration."""
        create_fastapi_security_headers(mock_fastapi_app)

        # Verify middleware was added
        mock_fastapi_app.add_middleware.assert_called_once()

    @pytest.mark.asyncio
    async def test_fastapi_middleware_applies_headers(self) -> None:
        """Test that FastAPI middleware applies security headers."""
        # Import here to avoid dependency issues
        try:
            import starlette.middleware.base  # noqa: F401
        except ImportError:
            pytest.skip("Starlette not installed")

        # Create mock app and middleware
        mock_app = MagicMock()

        # Create middleware instance
        create_fastapi_security_headers(mock_app)

        # Get the middleware class
        middleware_class = mock_app.add_middleware.call_args[0][0]

        # Verify it's the right middleware
        assert middleware_class.__name__ == "SecurityHeadersMiddlewareFastAPI"

    def test_fastapi_middleware_class_exists(self) -> None:
        """Test that FastAPI middleware class is created properly."""
        mock_app = MagicMock()
        create_fastapi_security_headers(mock_app)

        # Verify middleware was added
        assert mock_app.add_middleware.called

        # Get middleware class
        middleware_class = mock_app.add_middleware.call_args[0][0]
        assert hasattr(middleware_class, "__init__")
        assert hasattr(middleware_class, "dispatch")


class TestDefaultBackendOrigin:
    """Tests for _default_backend_origin function."""

    @patch("traigent.security.headers.BackendConfig.get_backend_url")
    def test_default_backend_origin_with_scheme(self, mock_get_url: MagicMock) -> None:
        """Test default backend origin with full URL."""
        mock_get_url.return_value = "https://api.traigent.ai/api/v1"

        origin = _default_backend_origin()

        assert origin == "https://api.traigent.ai"

    @patch("traigent.security.headers.BackendConfig.get_backend_url")
    def test_default_backend_origin_with_port(self, mock_get_url: MagicMock) -> None:
        """Test default backend origin with port."""
        mock_get_url.return_value = "http://localhost:5000/api/v1"

        origin = _default_backend_origin()

        assert origin == "http://localhost:5000"

    @patch("traigent.security.headers.BackendConfig.get_backend_url")
    def test_default_backend_origin_strips_trailing_slash(
        self, mock_get_url: MagicMock
    ) -> None:
        """Test that trailing slashes are stripped."""
        mock_get_url.return_value = "https://api.traigent.ai/"

        origin = _default_backend_origin()

        assert origin == "https://api.traigent.ai"

    @patch("traigent.security.headers.BackendConfig.get_backend_url")
    @patch(
        "traigent.security.headers.BackendConfig.DEFAULT_PROD_URL",
        "https://api.traigent.ai",
    )
    def test_default_backend_origin_fallback(self, mock_get_url: MagicMock) -> None:
        """Test fallback to DEFAULT_PROD_URL when URL is empty."""
        mock_get_url.return_value = ""

        origin = _default_backend_origin()

        assert origin == "https://api.traigent.ai"

    @patch("traigent.security.headers.BackendConfig.get_backend_url")
    def test_default_backend_origin_no_scheme(self, mock_get_url: MagicMock) -> None:
        """Test backend origin when URL has no scheme."""
        mock_get_url.return_value = "localhost:5000"

        origin = _default_backend_origin()

        # Should return as-is if no scheme
        assert origin == "localhost:5000"


class TestCSPPresets:
    """Tests for CSP preset configurations."""

    def test_strict_csp_preset(self) -> None:
        """Test STRICT_CSP preset configuration."""
        assert STRICT_CSP["default-src"] == "'none'"
        assert STRICT_CSP["script-src"] == "'self'"
        assert STRICT_CSP["object-src"] == "'none'"
        assert STRICT_CSP["frame-ancestors"] == "'none'"
        assert STRICT_CSP["media-src"] == "'none'"

    def test_api_csp_preset(self) -> None:
        """Test API_CSP preset configuration."""
        assert API_CSP["default-src"] == "'none'"
        assert API_CSP["frame-ancestors"] == "'none'"
        assert API_CSP["base-uri"] == "'self'"
        # API CSP should be minimal
        assert len(API_CSP) == 3

    def test_relaxed_csp_preset(self) -> None:
        """Test RELAXED_CSP preset configuration."""
        assert RELAXED_CSP["default-src"] == "'self'"
        assert "'unsafe-inline'" in RELAXED_CSP["script-src"]
        assert "cdn.jsdelivr.net" in RELAXED_CSP["script-src"]
        assert "'unsafe-inline'" in RELAXED_CSP["style-src"]
        assert "connect-src" in RELAXED_CSP

    def test_relaxed_csp_includes_backend_origin(self) -> None:
        """Test that RELAXED_CSP includes backend origin in connect-src."""
        # Should include the backend origin
        assert "connect-src" in RELAXED_CSP
        # Should start with 'self'
        assert "'self'" in RELAXED_CSP["connect-src"]

    def test_csp_presets_have_upgrade_insecure_requests(self) -> None:
        """Test that CSP presets include upgrade-insecure-requests where appropriate."""
        assert "upgrade-insecure-requests" in STRICT_CSP
        # API CSP might not need it since it's minimal
        # RELAXED_CSP doesn't have it in the source

    def test_strict_csp_with_middleware(self) -> None:
        """Test using STRICT_CSP with middleware."""
        middleware = SecurityHeadersMiddleware(csp_directives=STRICT_CSP)
        csp = middleware._build_csp_header()

        assert "default-src 'none'" in csp
        assert "script-src 'self'" in csp

    def test_api_csp_with_middleware(self) -> None:
        """Test using API_CSP with middleware."""
        middleware = SecurityHeadersMiddleware(csp_directives=API_CSP)
        csp = middleware._build_csp_header()

        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_relaxed_csp_with_middleware(self) -> None:
        """Test using RELAXED_CSP with middleware."""
        middleware = SecurityHeadersMiddleware(csp_directives=RELAXED_CSP)
        csp = middleware._build_csp_header()

        assert "default-src 'self'" in csp
        assert "'unsafe-inline'" in csp


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_apply_headers_with_existing_headers(self) -> None:
        """Test applying headers when response already has some headers."""
        middleware = SecurityHeadersMiddleware()
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "application/json",
            "X-Custom-Header": "value",
        }

        result = middleware.apply_headers(mock_response)

        # Custom headers should be preserved
        assert result.headers["Content-Type"] == "application/json"
        assert result.headers["X-Custom-Header"] == "value"
        # Security headers should be added
        assert result.headers["X-Content-Type-Options"] == "nosniff"

    def test_apply_headers_overwrites_server_headers(self) -> None:
        """Test that server headers are always removed even if set."""
        middleware = SecurityHeadersMiddleware()
        mock_response = MagicMock()
        mock_response.headers = {
            "Server": "Custom/1.0",
            "X-Powered-By": "TraiGent/1.0",
        }

        result = middleware.apply_headers(mock_response)

        assert "Server" not in result.headers
        assert "X-Powered-By" not in result.headers

    def test_is_allowed_origin_case_sensitivity(self) -> None:
        """Test origin validation is case-sensitive."""
        middleware = SecurityHeadersMiddleware()
        middleware.allowed_origins = ["https://Example.com"]

        # Domain names should be case-insensitive in practice, but test exact matching
        assert middleware._is_allowed_origin("https://Example.com")
        # This will fail because we do exact match
        assert not middleware._is_allowed_origin("https://example.com")

    def test_build_csp_header_order_consistency(self) -> None:
        """Test that CSP header building maintains consistent order."""
        # Note: dict order is preserved in Python 3.7+
        middleware = SecurityHeadersMiddleware()
        csp1 = middleware._build_csp_header()
        csp2 = middleware._build_csp_header()

        assert csp1 == csp2

    def test_multiple_middleware_instances(self) -> None:
        """Test that multiple middleware instances don't interfere."""
        middleware1 = SecurityHeadersMiddleware(enable_hsts=True)
        middleware2 = SecurityHeadersMiddleware(enable_hsts=False)

        assert middleware1.enable_hsts is True
        assert middleware2.enable_hsts is False

    def test_cors_with_multiple_allowed_origins(self) -> None:
        """Test CORS validation with multiple allowed origins."""
        middleware = SecurityHeadersMiddleware(
            enable_cors=True,
            allowed_origins=["https://app1.example.com", "https://app2.example.com"],
        )

        assert middleware._is_allowed_origin("https://app1.example.com")
        assert middleware._is_allowed_origin("https://app2.example.com")
        assert not middleware._is_allowed_origin("https://app3.example.com")
