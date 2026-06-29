"""Tests for HuggingFace environment-variable alias resolution (#1569).

Covers both resolution paths so they cannot drift:
  - APIKeyManager.get_api_key  (traigent/config/api_keys.py)
  - env_config.get_api_key     (traigent/utils/env_config.py)

Alias order: HF_TOKEN (native) -> HUGGING_FACE_HUB_TOKEN -> HF_API_KEY (back-compat).
All three vars are cleared before each test to ensure full isolation.
"""

import pytest

from traigent.config.api_keys import APIKeyManager
from traigent.utils.env_config import get_api_key as env_get_api_key

_HF_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_KEY")


@pytest.fixture(autouse=True)
def _clear_hf_env(monkeypatch):
    """Remove all three HF vars before every test in this module."""
    for var in _HF_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# APIKeyManager tests
# ---------------------------------------------------------------------------


class TestAPIKeyManagerHF:
    def setup_method(self, method):
        self.manager = APIKeyManager()

    def test_hf_token_only(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-native")
        assert self.manager.get_api_key("huggingface") == "tok-native"

    def test_hugging_face_hub_token_only(self, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        assert self.manager.get_api_key("huggingface") == "tok-hub"

    def test_hf_api_key_only(self, monkeypatch):
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert self.manager.get_api_key("huggingface") == "tok-legacy"

    def test_hf_token_wins_over_hub_and_legacy(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-wins")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert self.manager.get_api_key("huggingface") == "tok-wins"

    def test_hub_token_wins_over_legacy_when_no_hf_token(self, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert self.manager.get_api_key("huggingface") == "tok-hub"

    def test_none_when_no_hf_vars(self):
        result = self.manager.get_api_key("huggingface")
        assert result is None


# ---------------------------------------------------------------------------
# env_config.get_api_key tests
# ---------------------------------------------------------------------------


class TestEnvConfigGetApiKeyHF:
    def test_hf_token_only(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-native")
        assert env_get_api_key("huggingface") == "tok-native"

    def test_hugging_face_hub_token_only(self, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        assert env_get_api_key("huggingface") == "tok-hub"

    def test_hf_api_key_only(self, monkeypatch):
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert env_get_api_key("huggingface") == "tok-legacy"

    def test_hf_token_wins_over_hub_and_legacy(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-wins")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert env_get_api_key("huggingface") == "tok-wins"

    def test_hub_token_wins_over_legacy_when_no_hf_token(self, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-hub")
        monkeypatch.setenv("HF_API_KEY", "tok-legacy")
        assert env_get_api_key("huggingface") == "tok-hub"

    def test_none_when_no_hf_vars(self):
        result = env_get_api_key("huggingface")
        assert result is None

    def test_case_insensitive_provider(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-case")
        assert env_get_api_key("HuggingFace") == "tok-case"
