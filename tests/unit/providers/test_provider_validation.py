"""Comprehensive unit tests for provider validation module.

Tests cover:
- Provider detection from model names (regex patterns + LiteLLM prefix)
- Model name validation against known models lists
- Error classification (auth vs transient)
- Timeout handling
- Provider-specific validators (OpenAI, Anthropic, Google, Mistral, Cohere)
- Caching behavior
- Status printing and filtering
"""

from __future__ import annotations

import hashlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from traigent.providers.validation import (
    _AUTH_ERROR_TYPES,
    _KNOWN_MODELS,
    _PROVIDER_PATTERNS,
    _TRANSIENT_ERROR_TYPES,
    ProviderStatus,
    ProviderValidator,
    _is_auth_error,
    _is_transient_error,
    _run_with_timeout,
    get_failed_providers,
    get_provider_for_model,
    print_provider_status,
    validate_model_names,
    validate_providers,
)


@pytest.mark.unit
class TestGetProviderForModel:
    """Tests for get_provider_for_model function."""

    def test_openai_gpt_models(self):
        """Test OpenAI GPT model detection."""
        assert get_provider_for_model("gpt-4o-mini") == "openai"
        assert get_provider_for_model("gpt-4") == "openai"
        assert get_provider_for_model("gpt-3.5-turbo") == "openai"

    def test_openai_o_series_models(self):
        """Test OpenAI o1/o3 series detection."""
        assert get_provider_for_model("o1-mini") == "openai"
        assert get_provider_for_model("o1-preview") == "openai"
        assert get_provider_for_model("o3-mini") == "openai"

    def test_anthropic_claude_models(self):
        """Test Anthropic Claude model detection."""
        assert get_provider_for_model("claude-3-haiku-20240307") == "anthropic"
        assert get_provider_for_model("claude-sonnet-4-20250514") == "anthropic"
        assert get_provider_for_model("claude-opus-4-20250514") == "anthropic"

    def test_anthropic_family_names(self):
        """Test Anthropic family name detection (haiku, sonnet, opus)."""
        assert get_provider_for_model("haiku-test") == "anthropic"
        assert get_provider_for_model("sonnet-test") == "anthropic"
        assert get_provider_for_model("opus-test") == "anthropic"

    def test_google_gemini_models(self):
        """Test Google Gemini model detection."""
        assert get_provider_for_model("gemini-1.5-flash") == "google"
        assert get_provider_for_model("gemini-2.0-flash-exp") == "google"
        assert get_provider_for_model("models/gemini-pro") == "google"

    def test_mistral_models(self):
        """Test Mistral model detection."""
        assert get_provider_for_model("mistral-small-latest") == "mistral"
        assert get_provider_for_model("codestral-latest") == "mistral"
        assert get_provider_for_model("pixtral-12b") == "mistral"
        assert get_provider_for_model("open-mistral-7b") == "mistral"

    def test_cohere_models(self):
        """Test Cohere model detection (pattern requires 'command-' prefix)."""
        assert get_provider_for_model("command-r-plus") == "cohere"
        assert get_provider_for_model("command-light") == "cohere"
        assert get_provider_for_model("command-r") == "cohere"

    def test_litellm_slash_format(self):
        """Test LiteLLM provider/model format detection."""
        assert get_provider_for_model("openai/gpt-4") == "openai"
        assert get_provider_for_model("anthropic/claude-3-haiku") == "anthropic"
        assert get_provider_for_model("google/gemini-pro") == "google"
        assert get_provider_for_model("mistral/mistral-small") == "mistral"
        assert get_provider_for_model("cohere/command") == "cohere"

    def test_litellm_colon_format(self):
        """Test LiteLLM provider:model format detection."""
        assert get_provider_for_model("openai:gpt-4") == "openai"
        assert get_provider_for_model("anthropic:claude-3-haiku") == "anthropic"

    def test_litellm_gemini_alias(self):
        """Test LiteLLM gemini alias maps to google."""
        assert get_provider_for_model("gemini/gemini-pro") == "google"

    def test_litellm_vertex_ai_prefix_not_matched(self):
        """Test vertex_ai prefix with underscore is not matched.

        The LiteLLM prefix pattern only matches [a-z]+ (lowercase letters),
        so vertex_ai with underscore doesn't match. The full string
        "vertex_ai/gemini-pro" also doesn't match any provider patterns.
        """
        result = get_provider_for_model("vertex_ai/gemini-pro")
        # vertex_ai prefix has underscore, doesn't match [a-z]+ pattern
        # and full string doesn't match any provider pattern
        assert result is None

    def test_unknown_model(self):
        """Test unknown model returns None."""
        assert get_provider_for_model("my-custom-model") is None
        assert get_provider_for_model("unknown-provider/model") is None
        assert get_provider_for_model("") is None

    def test_case_insensitive(self):
        """Test detection is case-insensitive."""
        assert get_provider_for_model("GPT-4") == "openai"
        assert get_provider_for_model("CLAUDE-3-HAIKU") == "anthropic"
        assert get_provider_for_model("GEMINI-PRO") == "google"


@pytest.mark.unit
class TestValidateModelNames:
    """Tests for validate_model_names function."""

    def test_all_valid_openai_models(self):
        """Test validation with all known OpenAI models."""
        models = ["gpt-4o", "gpt-4o-mini", "o1-mini"]
        valid, unknown = validate_model_names(models, "openai")
        assert valid == models
        assert unknown == []

    def test_all_valid_anthropic_models(self):
        """Test validation with all known Anthropic models."""
        models = ["claude-3-haiku-20240307", "claude-sonnet-4-20250514"]
        valid, unknown = validate_model_names(models, "anthropic")
        assert valid == models
        assert unknown == []

    def test_mixed_valid_and_unknown(self):
        """Test validation with mix of valid and unknown models."""
        models = ["gpt-4o", "gpt-6-ultra", "gpt-3.5-turbo"]
        valid, unknown = validate_model_names(models, "openai")
        assert "gpt-4o" in valid
        assert "gpt-3.5-turbo" in valid
        assert unknown == ["gpt-6-ultra"]

    def test_all_unknown_models(self):
        """Test validation with all unknown models."""
        models = ["fake-model-1", "fake-model-2"]
        valid, unknown = validate_model_names(models, "openai")
        assert valid == []
        assert unknown == models

    def test_unknown_provider(self):
        """Test validation with provider not in known list."""
        models = ["custom-model-1", "custom-model-2"]
        valid, unknown = validate_model_names(models, "custom-provider")
        assert valid == models
        assert unknown == []

    def test_empty_models_list(self):
        """Test validation with empty models list."""
        valid, unknown = validate_model_names([], "openai")
        assert valid == []
        assert unknown == []


@pytest.mark.unit
class TestIsAuthError:
    """Tests for _is_auth_error function."""

    def test_known_auth_error_types(self):
        """Test detection of known auth error class names."""
        for error_type in _AUTH_ERROR_TYPES:
            # Create mock exception with matching class name
            mock_exc = Mock()
            mock_exc.__class__.__name__ = error_type
            assert _is_auth_error(mock_exc)

    def test_auth_error_by_message_content(self):
        """Test detection of auth error by message keywords."""
        assert _is_auth_error(Exception("Invalid API key provided"))
        assert _is_auth_error(Exception("Unauthorized access"))
        assert _is_auth_error(Exception("Authentication failed"))
        assert _is_auth_error(Exception("invalid_api_key error"))

    def test_non_auth_error(self):
        """Test non-auth errors return False."""
        assert not _is_auth_error(ValueError("Invalid value"))
        assert not _is_auth_error(RuntimeError("Runtime error"))
        assert not _is_auth_error(Exception("Some other error"))


@pytest.mark.unit
class TestIsTransientError:
    """Tests for _is_transient_error function."""

    def test_known_transient_error_types(self):
        """Test detection of known transient error class names."""
        for error_type in _TRANSIENT_ERROR_TYPES:
            mock_exc = Mock()
            mock_exc.__class__.__name__ = error_type
            assert _is_transient_error(mock_exc)

    def test_non_transient_error(self):
        """Test non-transient errors return False."""
        assert not _is_transient_error(ValueError("Invalid value"))
        assert not _is_transient_error(Exception("Authentication failed"))


@pytest.mark.unit
class TestRunWithTimeout:
    """Tests for _run_with_timeout function."""

    def test_function_completes_within_timeout(self):
        """Test function that completes before timeout."""

        def quick_func():
            return "success"

        result = _run_with_timeout(quick_func, timeout=1.0, provider="test")
        assert result == "success"

    def test_function_times_out(self):
        """Test function that exceeds timeout."""
        import time

        def slow_func():
            time.sleep(2.0)
            return "never"

        with pytest.raises(TimeoutError, match="test validation timed out after 0.5s"):
            _run_with_timeout(slow_func, timeout=0.5, provider="test")

    def test_function_raises_exception(self):
        """Test function that raises exception."""

        def error_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_with_timeout(error_func, timeout=1.0, provider="test")


@pytest.mark.unit
class TestProviderStatus:
    """Tests for ProviderStatus dataclass."""

    def test_create_valid_status(self):
        """Test creating valid provider status."""
        status = ProviderStatus(
            provider="openai", valid=True, message="Available", models=["gpt-4o"]
        )
        assert status.provider == "openai"
        assert status.valid is True
        assert status.message == "Available"
        assert status.models == ["gpt-4o"]
        assert status.unknown_models is None
        assert status.error_type is None

    def test_create_invalid_status_with_error(self):
        """Test creating invalid provider status with error."""
        status = ProviderStatus(
            provider="anthropic",
            valid=False,
            message="Invalid key",
            error_type="AuthenticationError",
        )
        assert status.provider == "anthropic"
        assert status.valid is False
        assert status.error_type == "AuthenticationError"


@pytest.mark.unit
class TestProviderValidatorCaching:
    """Tests for ProviderValidator caching behavior."""

    def test_get_key_fingerprint(self):
        """Test key fingerprint generation."""
        validator = ProviderValidator()
        key = "test-api-key-12345"  # pragma: allowlist secret
        fingerprint = validator._get_key_fingerprint(key)
        expected = hashlib.sha256(key.encode()).hexdigest()[:8]
        assert fingerprint == expected
        assert len(fingerprint) == 8

    def test_get_key_fingerprint_none(self):
        """Test key fingerprint for None key."""
        validator = ProviderValidator()
        assert validator._get_key_fingerprint(None) == "none"

    def test_cache_success_and_is_cached(self):
        """Test caching successful validation."""
        validator = ProviderValidator()
        key = "test-key-123"  # pragma: allowlist secret
        assert not validator._is_cached("openai", key)
        validator._cache_success("openai", key)
        assert validator._is_cached("openai", key)

    def test_cache_thread_safety(self):
        """Test cache operations are thread-safe."""
        validator = ProviderValidator()
        key = "test-key-456"  # pragma: allowlist secret
        results = []

        def cache_and_check():
            validator._cache_success("openai", key)
            results.append(validator._is_cached("openai", key))

        threads = [threading.Thread(target=cache_and_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see cached result
        assert all(results)
        assert len(results) == 10


@pytest.mark.unit
class TestProviderValidatorGrouping:
    """Tests for ProviderValidator model grouping."""

    def test_group_models_by_provider(self):
        """Test grouping models by provider."""
        validator = ProviderValidator()
        models = [
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-3-haiku-20240307",
            "gemini-1.5-flash",
        ]
        providers, unknown = validator._group_models_by_provider(models)
        assert "openai" in providers
        assert len(providers["openai"]) == 2
        assert "anthropic" in providers
        assert len(providers["anthropic"]) == 1
        assert "google" in providers
        assert len(providers["google"]) == 1
        assert unknown == []

    def test_group_models_with_unknown_provider(self):
        """Test grouping with unknown provider models."""
        validator = ProviderValidator()
        models = ["gpt-4o", "my-custom-model", "another-unknown"]
        providers, unknown = validator._group_models_by_provider(models)
        assert "openai" in providers
        assert unknown == ["my-custom-model", "another-unknown"]


@pytest.mark.unit
class TestProviderValidatorOpenAI:
    """Tests for ProviderValidator._validate_openai."""

    def test_missing_api_key(self, monkeypatch):
        """Test OpenAI validation with missing API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_openai()
        assert status.provider == "openai"
        assert status.valid is False
        assert status.message == "Set OPENAI_API_KEY"
        assert status.error_type == "MissingKey"

    def test_cached_validation(self, monkeypatch):
        """Test OpenAI validation with cached result."""
        key = "sk-test123"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", key)
        validator = ProviderValidator()
        validator._cache_success("openai", key)
        status = validator._validate_openai()
        assert status.provider == "openai"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_import_error(self, monkeypatch):
        """Test OpenAI validation with SDK not installed."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"openai": None}):
            # Force import to fail
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_openai()

        assert status.provider == "openai"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert status.error_type == "ModuleNotFoundError"

    def test_auth_error(self, monkeypatch):
        """Test OpenAI validation with authentication error."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid")  # pragma: allowlist secret
        validator = ProviderValidator()

        class AuthenticationError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = AuthenticationError("Invalid API key")
        mock_client = Mock()
        mock_client.models = mock_models

        # Mock OpenAI module
        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            status = validator._validate_openai()

        assert status.provider == "openai"
        assert status.valid is False
        assert "Invalid key" in status.message
        assert status.error_type == "AuthenticationError"

    def test_transient_error(self, monkeypatch):
        """Test OpenAI validation with transient error."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")  # pragma: allowlist secret
        validator = ProviderValidator()

        class RateLimitError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = RateLimitError("Rate limit exceeded")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            status = validator._validate_openai()

        assert status.provider == "openai"
        assert status.valid is True
        assert "Available (unverified" in status.message
        assert status.error_type == "RateLimitError"

    def test_success(self, monkeypatch):
        """Test OpenAI validation success."""
        key = "sk-test123"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", key)
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            status = validator._validate_openai()

        assert status.provider == "openai"
        assert status.valid is True
        assert status.message == "Available"
        assert validator._is_cached("openai", key)

    def test_unknown_error(self, monkeypatch):
        """Test OpenAI validation with unknown error."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.side_effect = RuntimeError("Unknown error")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            status = validator._validate_openai()

        assert status.provider == "openai"
        assert status.valid is False
        assert "Validation failed" in status.message
        assert status.error_type == "RuntimeError"


@pytest.mark.unit
class TestProviderValidatorAnthropic:
    """Tests for ProviderValidator._validate_anthropic."""

    def test_missing_api_key(self, monkeypatch):
        """Test Anthropic validation with missing API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_anthropic()
        assert status.provider == "anthropic"
        assert status.valid is False
        assert status.message == "Set ANTHROPIC_API_KEY"
        assert status.error_type == "MissingKey"

    def test_cached_validation(self, monkeypatch):
        """Test Anthropic validation with cached result."""
        key = "sk-ant-test123"  # pragma: allowlist secret
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
        validator = ProviderValidator()
        validator._cache_success("anthropic", key)
        status = validator._validate_anthropic()
        assert status.provider == "anthropic"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_import_error(self, monkeypatch):
        """Test Anthropic validation with SDK not installed."""
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-test123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_anthropic()

        assert status.provider == "anthropic"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert status.error_type == "ModuleNotFoundError"

    def test_auth_error(self, monkeypatch):
        """Test Anthropic validation with authentication error."""
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-invalid"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class PermissionDeniedError(Exception):
            pass

        mock_messages = Mock()
        mock_messages.count_tokens.side_effect = PermissionDeniedError(
            "Invalid API key"
        )
        mock_client = Mock()
        mock_client.messages = mock_messages

        mock_anthropic = SimpleNamespace(Anthropic=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            status = validator._validate_anthropic()

        assert status.provider == "anthropic"
        assert status.valid is False
        assert "Invalid key" in status.message
        assert status.error_type == "PermissionDeniedError"

    def test_transient_error(self, monkeypatch):
        """Test Anthropic validation with transient error."""
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-test123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class APITimeoutError(Exception):
            pass

        mock_messages = Mock()
        mock_messages.count_tokens.side_effect = APITimeoutError("Timeout")
        mock_client = Mock()
        mock_client.messages = mock_messages

        mock_anthropic = SimpleNamespace(Anthropic=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            status = validator._validate_anthropic()

        assert status.provider == "anthropic"
        assert status.valid is True
        assert "Available (unverified" in status.message
        assert status.error_type == "APITimeoutError"

    def test_success(self, monkeypatch):
        """Test Anthropic validation success."""
        key = "sk-ant-test123"  # pragma: allowlist secret
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
        validator = ProviderValidator()

        mock_messages = Mock()
        mock_messages.count_tokens.return_value = {"input_tokens": 1}
        mock_client = Mock()
        mock_client.messages = mock_messages

        mock_anthropic = SimpleNamespace(Anthropic=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            status = validator._validate_anthropic()

        assert status.provider == "anthropic"
        assert status.valid is True
        assert status.message == "Available"
        assert validator._is_cached("anthropic", key)

    def test_unknown_error(self, monkeypatch):
        """Test Anthropic validation with unknown error."""
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-test123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_messages = Mock()
        mock_messages.count_tokens.side_effect = RuntimeError("Unknown error")
        mock_client = Mock()
        mock_client.messages = mock_messages

        mock_anthropic = SimpleNamespace(Anthropic=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            status = validator._validate_anthropic()

        assert status.provider == "anthropic"
        assert status.valid is False
        assert "Validation failed" in status.message
        assert status.error_type == "RuntimeError"


@pytest.mark.unit
class TestProviderValidatorGoogle:
    """Tests for ProviderValidator._validate_google."""

    def test_missing_api_key(self, monkeypatch):
        """Test Google validation with missing API keys."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_google()
        assert status.provider == "google"
        assert status.valid is False
        assert status.message == "Set GOOGLE_API_KEY"
        assert status.error_type == "MissingKey"

    def test_gemini_api_key_fallback(self, monkeypatch):
        """Test Google validation falls back to GEMINI_API_KEY."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        key = "gemini-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("GEMINI_API_KEY", key)
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_genai = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"google": SimpleNamespace(genai=mock_genai)}):
            with patch(
                "traigent.providers.validation._run_with_timeout"
            ) as mock_timeout:
                mock_timeout.return_value = None
                status = validator._validate_google()

        assert status.provider == "google"
        assert status.valid is True
        assert validator._is_cached("google", key)

    def test_cached_validation(self, monkeypatch):
        """Test Google validation with cached result."""
        key = "google-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("GOOGLE_API_KEY", key)
        validator = ProviderValidator()
        validator._cache_success("google", key)
        status = validator._validate_google()
        assert status.provider == "google"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_import_error(self, monkeypatch):
        """Test Google validation with SDK not installed."""
        monkeypatch.setenv(
            "GOOGLE_API_KEY", "google-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"google": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_google()

        assert status.provider == "google"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert status.error_type == "ModuleNotFoundError"


@pytest.mark.unit
class TestProviderValidatorMistral:
    """Tests for ProviderValidator._validate_mistral."""

    def test_missing_api_key(self, monkeypatch):
        """Test Mistral validation with missing API key."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_mistral()
        assert status.provider == "mistral"
        assert status.valid is False
        assert status.message == "Set MISTRAL_API_KEY"
        assert status.error_type == "MissingKey"

    def test_cached_validation(self, monkeypatch):
        """Test Mistral validation with cached result."""
        key = "mistral-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("MISTRAL_API_KEY", key)
        validator = ProviderValidator()
        validator._cache_success("mistral", key)
        status = validator._validate_mistral()
        assert status.provider == "mistral"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_import_error(self, monkeypatch):
        """Test Mistral validation with SDK not installed."""
        monkeypatch.setenv(
            "MISTRAL_API_KEY", "mistral-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"mistralai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_mistral()

        assert status.provider == "mistral"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert status.error_type == "ModuleNotFoundError"

    def test_auth_error(self, monkeypatch):
        """Test Mistral validation with authentication error."""
        monkeypatch.setenv(
            "MISTRAL_API_KEY", "mistral-invalid"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class UnauthorizedError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = UnauthorizedError("Unauthorized")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_mistralai = SimpleNamespace(Mistral=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"mistralai": mock_mistralai}):
            status = validator._validate_mistral()

        assert status.provider == "mistral"
        assert status.valid is False
        assert "Invalid key" in status.message
        assert status.error_type == "UnauthorizedError"

    def test_transient_error(self, monkeypatch):
        """Test Mistral validation with transient error."""
        monkeypatch.setenv(
            "MISTRAL_API_KEY", "mistral-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class ConnectionError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = ConnectionError("Connection failed")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_mistralai = SimpleNamespace(Mistral=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"mistralai": mock_mistralai}):
            status = validator._validate_mistral()

        assert status.provider == "mistral"
        assert status.valid is True
        assert "Available (unverified" in status.message
        assert status.error_type == "ConnectionError"

    def test_success(self, monkeypatch):
        """Test Mistral validation success."""
        key = "mistral-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("MISTRAL_API_KEY", key)
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_mistralai = SimpleNamespace(Mistral=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"mistralai": mock_mistralai}):
            status = validator._validate_mistral()

        assert status.provider == "mistral"
        assert status.valid is True
        assert status.message == "Available"
        assert validator._is_cached("mistral", key)

    def test_unknown_error(self, monkeypatch):
        """Test Mistral validation with unknown error."""
        monkeypatch.setenv(
            "MISTRAL_API_KEY", "mistral-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.side_effect = RuntimeError("Unknown error")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_mistralai = SimpleNamespace(Mistral=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"mistralai": mock_mistralai}):
            status = validator._validate_mistral()

        assert status.provider == "mistral"
        assert status.valid is False
        assert "Validation failed" in status.message
        assert status.error_type == "RuntimeError"


@pytest.mark.unit
class TestProviderValidatorCohere:
    """Tests for ProviderValidator._validate_cohere."""

    def test_missing_api_key(self, monkeypatch):
        """Test Cohere validation with missing API keys."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_cohere()
        assert status.provider == "cohere"
        assert status.valid is False
        assert status.message == "Set COHERE_API_KEY"
        assert status.error_type == "MissingKey"

    def test_co_api_key_fallback(self, monkeypatch):
        """Test Cohere validation falls back to CO_API_KEY."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        key = "co-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("CO_API_KEY", key)
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_cohere = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is True
        assert validator._is_cached("cohere", key)

    def test_cached_validation(self, monkeypatch):
        """Test Cohere validation with cached result."""
        key = "cohere-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("COHERE_API_KEY", key)
        validator = ProviderValidator()
        validator._cache_success("cohere", key)
        status = validator._validate_cohere()
        assert status.provider == "cohere"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_import_error(self, monkeypatch):
        """Test Cohere validation with SDK not installed."""
        monkeypatch.setenv(
            "COHERE_API_KEY", "cohere-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"cohere": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert status.error_type == "ModuleNotFoundError"

    def test_auth_error(self, monkeypatch):
        """Test Cohere validation with authentication error."""
        monkeypatch.setenv(
            "COHERE_API_KEY", "cohere-invalid"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class APIKeyError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = APIKeyError("Invalid API key")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_cohere = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is False
        assert "Invalid key" in status.message
        assert status.error_type == "APIKeyError"

    def test_transient_error(self, monkeypatch):
        """Test Cohere validation with transient error."""
        monkeypatch.setenv(
            "COHERE_API_KEY", "cohere-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        class InternalServerError(Exception):
            pass

        mock_models = Mock()
        mock_models.list.side_effect = InternalServerError("Internal server error")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_cohere = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is True
        assert "Available (unverified" in status.message
        assert status.error_type == "InternalServerError"

    def test_success(self, monkeypatch):
        """Test Cohere validation success."""
        key = "cohere-key-123"  # pragma: allowlist secret
        monkeypatch.setenv("COHERE_API_KEY", key)
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_cohere = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is True
        assert status.message == "Available"
        assert validator._is_cached("cohere", key)

    def test_unknown_error(self, monkeypatch):
        """Test Cohere validation with unknown error."""
        monkeypatch.setenv(
            "COHERE_API_KEY", "cohere-key-123"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.side_effect = RuntimeError("Unknown error")
        mock_client = Mock()
        mock_client.models = mock_models

        mock_cohere = SimpleNamespace(Client=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"cohere": mock_cohere}):
            status = validator._validate_cohere()

        assert status.provider == "cohere"
        assert status.valid is False
        assert "Validation failed" in status.message
        assert status.error_type == "RuntimeError"


@pytest.mark.unit
class TestProviderValidatorValidateProvider:
    """Tests for ProviderValidator._validate_provider dispatch."""

    def test_unsupported_provider(self):
        """Test validation of unsupported provider."""
        validator = ProviderValidator()
        status = validator._validate_provider("unknown-provider")
        assert status.provider == "unknown-provider"
        assert status.valid is False
        assert "No validator for provider" in status.message
        assert status.error_type == "UnsupportedProvider"

    def test_supported_provider_dispatch(self, monkeypatch):
        """Test validation dispatches to correct provider method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models

        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            status = validator._validate_provider("openai")

        assert status.provider == "openai"
        assert status.valid is True


@pytest.mark.unit
class TestProviderValidatorValidateAll:
    """Tests for ProviderValidator.validate_all."""

    def test_validate_all_mixed_providers(self, monkeypatch):
        """Test validate_all with multiple providers."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY", "sk-ant-test"
        )  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_openai_models = Mock()
        mock_openai_models.list.return_value = []
        mock_openai_client = Mock()
        mock_openai_client.models = mock_openai_models
        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_openai_client))

        mock_anthropic_messages = Mock()
        mock_anthropic_messages.count_tokens.return_value = {"input_tokens": 1}
        mock_anthropic_client = Mock()
        mock_anthropic_client.messages = mock_anthropic_messages
        mock_anthropic = SimpleNamespace(
            Anthropic=Mock(return_value=mock_anthropic_client)
        )

        models = ["gpt-4o", "claude-3-haiku-20240307"]

        with patch.dict(
            sys.modules, {"openai": mock_openai, "anthropic": mock_anthropic}
        ):
            results = validator.validate_all(models)

        assert "openai" in results
        assert results["openai"].valid is True
        assert "anthropic" in results
        assert results["anthropic"].valid is True

    def test_validate_all_with_unknown_models(self, monkeypatch):
        """Test validate_all warns about unknown models."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models
        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        models = ["gpt-4o", "gpt-unknown-model"]

        with patch.dict(sys.modules, {"openai": mock_openai}):
            results = validator.validate_all(models)

        assert "openai" in results
        assert results["openai"].unknown_models == ["gpt-unknown-model"]

    def test_validate_all_with_unknown_provider_models(self):
        """Test validate_all returns empty dict for models with unknown providers."""
        validator = ProviderValidator()
        models = ["my-custom-model"]

        results = validator.validate_all(models)

        assert results == {}


@pytest.mark.unit
class TestValidateProvidersConvenience:
    """Tests for validate_providers convenience function."""

    def test_validate_providers_creates_validator(self, monkeypatch):
        """Test validate_providers creates validator with custom timeout."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret

        mock_models = Mock()
        mock_models.list.return_value = []
        mock_client = Mock()
        mock_client.models = mock_models
        mock_openai = SimpleNamespace(OpenAI=Mock(return_value=mock_client))

        with patch.dict(sys.modules, {"openai": mock_openai}):
            results = validate_providers(["gpt-4o"], timeout=10.0)

        assert "openai" in results
        assert results["openai"].valid is True


@pytest.mark.unit
class TestPrintProviderStatus:
    """Tests for print_provider_status function."""

    def test_print_valid_providers(self, capsys):
        """Test printing valid provider status."""
        results = {
            "openai": ProviderStatus(
                provider="openai",
                valid=True,
                message="Available",
                models=["gpt-4o", "gpt-4o-mini"],
            ),
            "anthropic": ProviderStatus(
                provider="anthropic", valid=True, message="Available (cached)"
            ),
        }

        print_provider_status(results)

        captured = capsys.readouterr()
        assert "Provider Status:" in captured.out
        assert "[OK] Openai: Available" in captured.out
        assert "Models: gpt-4o, gpt-4o-mini" in captured.out
        assert "[OK] Anthropic: Available (cached)" in captured.out

    def test_print_invalid_providers(self, capsys):
        """Test printing invalid provider status."""
        results = {
            "openai": ProviderStatus(
                provider="openai",
                valid=False,
                message="Set OPENAI_API_KEY",
                error_type="MissingKey",
            )
        }

        print_provider_status(results)

        captured = capsys.readouterr()
        assert "[--] Openai: Set OPENAI_API_KEY" in captured.out

    def test_print_with_unknown_models(self, capsys):
        """Test printing status with unknown models warning."""
        results = {
            "openai": ProviderStatus(
                provider="openai",
                valid=True,
                message="Available",
                models=["gpt-4o"],
                unknown_models=["gpt-unknown"],
            )
        }

        print_provider_status(results)

        captured = capsys.readouterr()
        assert "[!!] Unknown models (may fail): gpt-unknown" in captured.out
        assert "check for typos" in captured.out


@pytest.mark.unit
class TestGetFailedProviders:
    """Tests for get_failed_providers function."""

    def test_get_failed_providers_filters_invalid(self):
        """Test get_failed_providers returns only invalid providers."""
        results = {
            "openai": ProviderStatus(
                provider="openai", valid=True, message="Available"
            ),
            "anthropic": ProviderStatus(
                provider="anthropic",
                valid=False,
                message="Invalid key",
                error_type="AuthenticationError",
            ),
            "google": ProviderStatus(
                provider="google",
                valid=False,
                message="Set GOOGLE_API_KEY",
                error_type="MissingKey",
            ),
        }

        failed = get_failed_providers(results)

        assert len(failed) == 2
        assert ("anthropic", "AuthenticationError") in failed
        assert ("google", "MissingKey") in failed

    def test_get_failed_providers_empty_when_all_valid(self):
        """Test get_failed_providers returns empty list when all valid."""
        results = {
            "openai": ProviderStatus(
                provider="openai", valid=True, message="Available"
            ),
            "anthropic": ProviderStatus(
                provider="anthropic", valid=True, message="Available"
            ),
        }

        failed = get_failed_providers(results)

        assert failed == []

    def test_get_failed_providers_handles_missing_error_type(self):
        """Test get_failed_providers handles None error_type."""
        results = {
            "custom": ProviderStatus(
                provider="custom", valid=False, message="Failed", error_type=None
            )
        }

        failed = get_failed_providers(results)

        assert len(failed) == 1
        assert failed[0] == ("custom", "Unknown")


@pytest.mark.unit
class TestModuleLevelConstants:
    """Tests for module-level constants."""

    def test_provider_patterns_exist(self):
        """Test _PROVIDER_PATTERNS contains expected providers."""
        assert "openai" in _PROVIDER_PATTERNS
        assert "anthropic" in _PROVIDER_PATTERNS
        assert "google" in _PROVIDER_PATTERNS
        assert "mistral" in _PROVIDER_PATTERNS
        assert "cohere" in _PROVIDER_PATTERNS

    def test_known_models_exist(self):
        """Test _KNOWN_MODELS contains expected providers."""
        assert "openai" in _KNOWN_MODELS
        assert "anthropic" in _KNOWN_MODELS
        assert "google" in _KNOWN_MODELS
        assert "mistral" in _KNOWN_MODELS
        assert "cohere" in _KNOWN_MODELS
        # Verify some known models
        assert "gpt-4o" in _KNOWN_MODELS["openai"]
        assert "claude-3-haiku-20240307" in _KNOWN_MODELS["anthropic"]
        assert "gemini-1.5-flash" in _KNOWN_MODELS["google"]

    def test_auth_error_types_exist(self):
        """Test _AUTH_ERROR_TYPES contains expected error types."""
        assert "AuthenticationError" in _AUTH_ERROR_TYPES
        assert "PermissionDeniedError" in _AUTH_ERROR_TYPES
        assert "InvalidAPIKeyError" in _AUTH_ERROR_TYPES
        assert "UnauthorizedError" in _AUTH_ERROR_TYPES

    def test_transient_error_types_exist(self):
        """Test _TRANSIENT_ERROR_TYPES contains expected error types."""
        assert "RateLimitError" in _TRANSIENT_ERROR_TYPES
        assert "APIConnectionError" in _TRANSIENT_ERROR_TYPES
        assert "APITimeoutError" in _TRANSIENT_ERROR_TYPES
        assert "TimeoutError" in _TRANSIENT_ERROR_TYPES
        assert "ConnectionError" in _TRANSIENT_ERROR_TYPES


@pytest.mark.unit
class TestProvidersInitImports:
    """Tests for traigent.providers.__init__.py imports."""

    def test_providers_init_imports_all_exports(self):
        """Test importing all exported symbols from traigent.providers."""
        from traigent.providers import (
            ProviderStatus,
            ProviderValidator,
            get_failed_providers,
            get_provider_for_model,
            print_provider_status,
            validate_model_names,
            validate_providers,
        )

        # Verify all imports are not None
        assert ProviderStatus is not None
        assert ProviderValidator is not None
        assert get_failed_providers is not None
        assert get_provider_for_model is not None
        assert print_provider_status is not None
        assert validate_model_names is not None
        assert validate_providers is not None

    def test_providers_init_exports_correct_types(self):
        """Test imported symbols have correct types."""
        from traigent.providers import (
            ProviderStatus,
            ProviderValidator,
            get_failed_providers,
            get_provider_for_model,
            print_provider_status,
            validate_model_names,
            validate_providers,
        )

        # Check types
        assert isinstance(ProviderStatus, type)
        assert isinstance(ProviderValidator, type)
        assert callable(get_failed_providers)
        assert callable(get_provider_for_model)
        assert callable(print_provider_status)
        assert callable(validate_model_names)
        assert callable(validate_providers)
