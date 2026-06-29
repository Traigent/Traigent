"""Focused tests for HuggingFace provider detection and validation.

Covers issues #1566 (_validate_huggingface) and #1567 (HF auto-detection).
All tests are fully mocked — no real HF API calls are made.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from traigent.providers.validation import (
    ProviderValidator,
    get_provider_for_model,
    validate_model_names,
)


@pytest.mark.unit
class TestGetProviderForModelHuggingFace:
    """Issue #1567 — HF models resolve to 'huggingface', not None."""

    def test_bare_hf_repo_id_detected(self):
        """Bare org/model format resolves to huggingface."""
        assert get_provider_for_model("meta-llama/Meta-Llama-3-8B-Instruct") == "huggingface"

    def test_bare_hf_repo_id_other_orgs(self):
        """Other orgs in org/model format also resolve to huggingface.

        Note: orgs whose name matches a known LiteLLM prefix (e.g. 'google')
        are caught by the prefix map first and resolve to that provider instead.
        """
        assert get_provider_for_model("mistralai/Mistral-7B-Instruct-v0.2") == "huggingface"
        assert get_provider_for_model("HuggingFaceH4/zephyr-7b-beta") == "huggingface"
        # 'google' is a known LiteLLM prefix, so it routes to Google, not HF
        assert get_provider_for_model("google/flan-t5-base") == "google"

    def test_litellm_huggingface_prefix(self):
        """'huggingface/' prefix resolves to huggingface."""
        assert get_provider_for_model("huggingface/meta-llama/Llama-2-7b") == "huggingface"

    def test_litellm_hf_prefix(self):
        """'hf/' prefix resolves to huggingface."""
        assert get_provider_for_model("hf/some-model") == "huggingface"

    def test_hf_model_with_dots_and_dashes(self):
        """Model names with dots and dashes match the HF pattern."""
        assert get_provider_for_model("org/model-v1.0-alpha") == "huggingface"
        assert get_provider_for_model("org/model_name.v2") == "huggingface"

    def test_no_slash_still_returns_none(self):
        """A model name without '/' does not match HF pattern."""
        assert get_provider_for_model("my-custom-model") is None

    def test_known_providers_not_affected(self):
        """Existing provider detection is unchanged after adding HF."""
        assert get_provider_for_model("gpt-4o") == "openai"
        assert get_provider_for_model("claude-3-haiku-20240307") == "anthropic"
        assert get_provider_for_model("gemini-1.5-flash") == "google"
        assert get_provider_for_model("mistral-small-latest") == "mistral"
        assert get_provider_for_model("command-r-plus") == "cohere"

    def test_litellm_provider_prefix_not_overridden(self):
        """LiteLLM-prefixed known providers still resolve correctly."""
        assert get_provider_for_model("openai/gpt-4") == "openai"
        assert get_provider_for_model("anthropic/claude-3-haiku") == "anthropic"
        assert get_provider_for_model("google/gemini-pro") == "google"

    def test_known_provider_prefixes_never_resolve_to_huggingface(self):
        """Known provider/platform prefixes must NOT leak into HuggingFace (last-resort guard).

        This is the critical last-resort boundary: any model id whose leading
        segment (before the first '/') is a known LiteLLM provider prefix must
        be routed away from HuggingFace, regardless of whether we have a
        dedicated validator for that provider.
        """
        # These have dedicated Traigent validators
        assert get_provider_for_model("vertex_ai/gemini-pro") != "huggingface"
        assert get_provider_for_model("vertex_ai/gemini-pro") == "google"
        # These are known LiteLLM prefixes without a Traigent validator
        assert get_provider_for_model("azure/my-deployment") != "huggingface"
        assert get_provider_for_model("groq/llama-3.1-8b-instant") != "huggingface"
        assert get_provider_for_model("bedrock/anthropic.claude-3") != "huggingface"
        assert get_provider_for_model("together_ai/togethercomputer/llama-2-70b") != "huggingface"
        assert get_provider_for_model("ollama/llama3") != "huggingface"
        assert get_provider_for_model("openrouter/meta-llama/llama-3.1-8b") != "huggingface"


@pytest.mark.unit
class TestValidateModelNamesHuggingFace:
    """Issue #1567 — HF open-namespace skip message instead of 'provider not recognized'."""

    def test_hf_models_all_valid_open_namespace(self):
        """HuggingFace skips model-name check; all models returned as valid."""
        models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
        valid, unknown = validate_model_names(models, "huggingface")
        assert valid == models
        assert unknown == []

    def test_hf_validate_logs_skip_message(self, caplog):
        """validate_model_names logs the open-namespace skip message for HF."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="traigent.providers.validation"):
            validate_model_names(["org/model"], "huggingface")

        assert any(
            "HuggingFace is an open model namespace" in record.message
            for record in caplog.records
        )

    def test_empty_hf_model_list(self):
        """Empty model list for HF returns ([], [])."""
        valid, unknown = validate_model_names([], "huggingface")
        assert valid == []
        assert unknown == []


@pytest.mark.unit
class TestValidateHuggingFace:
    """Issue #1566 — _validate_huggingface covers the full validator contract."""

    def test_missing_token_all_env_vars(self, monkeypatch):
        """Returns MissingKey when no HF token env vars are set."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_API_KEY", raising=False)
        validator = ProviderValidator()
        status = validator._validate_huggingface()
        assert status.provider == "huggingface"
        assert status.valid is False
        assert status.message == "Set HF_TOKEN"
        assert status.error_type == "MissingKey"

    def test_hf_token_env_var_primary(self, monkeypatch):
        """HF_TOKEN is the primary env var."""
        key = "hf-test-token"  # pragma: allowlist secret
        monkeypatch.setenv("HF_TOKEN", key)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_API_KEY", raising=False)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.valid is True
        assert status.message == "Available"
        mock_api.whoami.assert_called_once_with(token=key)

    def test_hugging_face_hub_token_fallback(self, monkeypatch):
        """HUGGING_FACE_HUB_TOKEN is the second fallback."""
        key = "hf-hub-token"  # pragma: allowlist secret
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", key)
        monkeypatch.delenv("HF_API_KEY", raising=False)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.valid is True
        mock_api.whoami.assert_called_once_with(token=key)

    def test_hf_api_key_third_fallback(self, monkeypatch):
        """HF_API_KEY is the third fallback."""
        key = "hf-api-key"  # pragma: allowlist secret
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv("HF_API_KEY", key)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.valid is True
        mock_api.whoami.assert_called_once_with(token=key)

    def test_success_caches_result(self, monkeypatch):
        """Successful validation is cached for subsequent calls."""
        key = "hf-test-token"  # pragma: allowlist secret
        monkeypatch.setenv("HF_TOKEN", key)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            validator._validate_huggingface()

        assert validator._is_cached("huggingface", key)

    def test_cached_returns_without_api_call(self, monkeypatch):
        """Cached result is returned without calling HfApi."""
        key = "hf-test-token"  # pragma: allowlist secret
        monkeypatch.setenv("HF_TOKEN", key)
        validator = ProviderValidator()
        validator._cache_success("huggingface", key)

        status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is True
        assert status.message == "Available (cached)"

    def test_sdk_not_installed(self, monkeypatch):
        """Returns ModuleNotFoundError status when huggingface_hub is absent."""
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")  # pragma: allowlist secret
        validator = ProviderValidator()

        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is False
        assert "SDK not installed" in status.message
        assert "huggingface_hub" in status.message
        assert status.error_type == "ModuleNotFoundError"

    def test_auth_error_fails_fast(self, monkeypatch):
        """Auth errors produce valid=False (fail-fast)."""
        monkeypatch.setenv("HF_TOKEN", "hf-invalid-token")  # pragma: allowlist secret
        validator = ProviderValidator()

        class RepositoryNotFoundError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        mock_api = Mock()
        mock_api.whoami.side_effect = AuthenticationError("Invalid token")
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is False
        assert "Invalid key" in status.message
        assert status.error_type == "AuthenticationError"

    def test_auth_error_by_message_keyword(self, monkeypatch):
        """Auth errors detected via message keyword also fail fast."""
        monkeypatch.setenv("HF_TOKEN", "hf-invalid-token")  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.side_effect = Exception("invalid api key")
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is False
        assert "Invalid key" in status.message

    def test_transient_error_warns_and_allows(self, monkeypatch):
        """Transient errors produce valid=True with a warning."""
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")  # pragma: allowlist secret
        validator = ProviderValidator()

        class ConnectionError(Exception):
            pass

        mock_api = Mock()
        mock_api.whoami.side_effect = ConnectionError("Connection refused")
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is True
        assert "Available (unverified" in status.message
        assert status.error_type == "ConnectionError"

    def test_unknown_error_fails(self, monkeypatch):
        """Unknown errors produce valid=False (conservative)."""
        monkeypatch.setenv("HF_TOKEN", "hf-test-token")  # pragma: allowlist secret
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.side_effect = RuntimeError("Unexpected error")
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_huggingface()

        assert status.provider == "huggingface"
        assert status.valid is False
        assert "Validation failed" in status.message
        assert status.error_type == "RuntimeError"

    def test_validate_provider_dispatches_to_huggingface(self, monkeypatch):
        """_validate_provider('huggingface') dispatches to _validate_huggingface."""
        key = "hf-test-token"  # pragma: allowlist secret
        monkeypatch.setenv("HF_TOKEN", key)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            status = validator._validate_provider("huggingface")

        assert status.provider == "huggingface"
        assert status.valid is True

    def test_validate_all_hf_model_no_longer_unknown(self, monkeypatch):
        """validate_all no longer treats HF org/model as unknown provider."""
        key = "hf-test-token"  # pragma: allowlist secret
        monkeypatch.setenv("HF_TOKEN", key)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_API_KEY", raising=False)
        validator = ProviderValidator()

        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test-user"}
        mock_hf = SimpleNamespace(HfApi=Mock(return_value=mock_api))

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            results = validator.validate_all(["meta-llama/Meta-Llama-3-8B-Instruct"])

        assert "huggingface" in results
        assert results["huggingface"].valid is True
