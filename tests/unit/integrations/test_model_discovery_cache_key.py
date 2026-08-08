"""Regression tests for credential-scoped model-discovery cache entries."""

import hashlib
import hmac
import re
from unittest.mock import patch

import pytest

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


def test_openai_cache_reuses_same_context_across_discovery_and_cache_objects(
    tmp_path,
):
    """The same context must deterministically address a persistent cache file."""
    context = {
        "OPENAI_API_KEY": "sk-same-context",  # pragma: allowlist secret
        "OPENAI_BASE_URL": "https://one.example",
        "OPENAI_ORG_ID": "org-one",
        "OPENAI_PROJECT_ID": "project-one",
    }
    cache_dir = tmp_path / "model-cache"

    with patch.dict("os.environ", context, clear=True):
        first = OpenAIDiscovery(cache=ModelCache(cache_dir=cache_dir))
        with patch.object(first, "_fetch_models_from_sdk", return_value=["gpt-cached"]):
            assert first.list_models() == ["gpt-cached"]
        first_key = first._get_cache_key()

        second = OpenAIDiscovery(cache=ModelCache(cache_dir=cache_dir))
        with patch.object(second, "_fetch_models_from_sdk") as fetch_models:
            assert second.list_models() == ["gpt-cached"]
        second_key = second._get_cache_key()

    assert first_key == second_key
    assert all(
        "sk-same-context" not in cache_file.name for cache_file in cache_dir.iterdir()
    )
    assert all(
        "sk-same-context" not in cache_file.read_text(encoding="utf-8")
        for cache_file in cache_dir.iterdir()
    )
    fetch_models.assert_not_called()


def test_cache_partition_changes_for_credentials_and_account_contexts(tmp_path):
    """Credential and endpoint/account changes must not share model lists."""
    contexts = [
        {
            "OPENAI_API_KEY": "sk-first",  # pragma: allowlist secret
            "OPENAI_BASE_URL": "https://one.example",
            "OPENAI_ORG_ID": "org-one",
            "OPENAI_PROJECT_ID": "project-one",
        },
        {
            "OPENAI_API_KEY": "sk-second",  # pragma: allowlist secret
            "OPENAI_BASE_URL": "https://one.example",
            "OPENAI_ORG_ID": "org-one",
            "OPENAI_PROJECT_ID": "project-one",
        },
        {
            "OPENAI_API_KEY": "sk-first",  # pragma: allowlist secret
            "OPENAI_BASE_URL": "https://two.example",
            "OPENAI_ORG_ID": "org-two",
            "OPENAI_PROJECT_ID": "project-two",
        },
    ]

    keys = []
    for context in contexts:
        with patch.dict("os.environ", context, clear=True):
            discovery = OpenAIDiscovery(cache=ModelCache(cache_dir=tmp_path))
            keys.append(discovery._get_cache_key())

    assert len(set(keys)) == len(contexts)


def test_cache_partition_identifier_is_opaque_versioned_hmac(tmp_path):
    """The cache partition uses the v1 HMAC construction, not raw SHA-256."""
    identity_material = "identity-material-example"
    parts = (identity_material, "https://one.example", "org-one", "project-one")
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": parts[0],
            "OPENAI_BASE_URL": parts[1],
            "OPENAI_ORG_ID": parts[2],
            "OPENAI_PROJECT_ID": parts[3],
        },
        clear=True,
    ):
        discovery = OpenAIDiscovery(cache=ModelCache(cache_dir=tmp_path))
        partition_id = discovery._get_credential_fingerprint()

    assert partition_id is not None
    material = "\x00".join(parts).encode("utf-8")
    expected = hmac.new(
        material,
        b"traigent:model-discovery-cache:v1",
        hashlib.sha256,
    ).hexdigest()[:24]
    raw_sha256 = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]

    assert partition_id == expected
    assert partition_id != raw_sha256
    assert identity_material not in partition_id
    assert re.fullmatch(r"[0-9a-f]{24}", partition_id)


def test_cache_partition_rejects_plain_delimiter_boundary_collisions():
    """NUL-separated parts distinguish values that collide under ``|`` joining."""
    first = OpenAIDiscovery._fingerprint("alpha|beta", "gamma")
    second = OpenAIDiscovery._fingerprint("alpha", "beta|gamma")

    assert first != second


def test_cache_partition_rejects_nul_containing_parts():
    with pytest.raises(ValueError, match="must not contain NUL"):
        OpenAIDiscovery._fingerprint("alpha\x00beta")


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
