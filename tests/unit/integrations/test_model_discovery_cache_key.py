"""Regression tests for credential-scoped model-discovery cache entries."""

from unittest.mock import patch

from traigent.integrations.model_discovery.cache import ModelCache
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery


def test_openai_cache_separates_org_and_project_with_the_same_key(tmp_path):
    """OpenAI account context must not reuse another organization's model list."""
    cache = ModelCache(cache_dir=tmp_path)

    first_context = {
        "OPENAI_API_KEY": "sk-shared",  # pragma: allowlist secret
        "OPENAI_ORG_ID": "org-one",
        "OPENAI_PROJECT_ID": "project-one",
    }
    second_context = {
        "OPENAI_API_KEY": "sk-shared",  # pragma: allowlist secret
        "OPENAI_ORG_ID": "org-two",
        "OPENAI_PROJECT_ID": "project-two",
    }

    with patch.dict("os.environ", first_context, clear=True):
        first = OpenAIDiscovery(cache=cache)
        with patch.object(
            first, "_fetch_models_from_sdk", return_value=["gpt-org-one"]
        ):
            assert first.list_models() == ["gpt-org-one"]
        first_key = first._get_cache_key()

    with patch.dict("os.environ", second_context, clear=True):
        second = OpenAIDiscovery(cache=cache)
        with patch.object(
            second, "_fetch_models_from_sdk", return_value=["gpt-org-two"]
        ) as fetch_models:
            assert second.list_models() == ["gpt-org-two"]
        second_key = second._get_cache_key()

    assert first_key != second_key
    assert "sk-shared" not in first_key
    fetch_models.assert_called_once()


def test_openai_cache_key_changes_for_each_client_context(tmp_path):
    discovery = OpenAIDiscovery(cache=ModelCache(cache_dir=tmp_path))

    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-shared",  # pragma: allowlist secret
            "OPENAI_BASE_URL": "https://one.example",
            "OPENAI_ORG_ID": "org-one",
            "OPENAI_PROJECT_ID": "project-one",
        },
        clear=True,
    ):
        first_key = discovery._get_cache_key()

    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-shared",  # pragma: allowlist secret
            "OPENAI_BASE_URL": "https://two.example",
            "OPENAI_ORG_ID": "org-two",
            "OPENAI_PROJECT_ID": "project-two",
        },
        clear=True,
    ):
        second_key = discovery._get_cache_key()

    assert first_key.startswith("openai-")
    assert first_key != second_key
