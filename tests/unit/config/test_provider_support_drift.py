"""Drift regression test for the canonical provider-support table (#1568).

The SDK historically maintained the same provider-support information in six
hand-edited registries that silently disagreed (HF/Cohere/Google/Mistral). This
test pins every registry to the single source of truth,
:mod:`traigent.config.provider_support`, so future drift fails CI.

It documents the **intended** support matrix (which provider participates in
which layer) and asserts each registry matches it:

================  =========  =========  ==========  =======  =========  =======
provider          key map    validated  prefix-det  tiered   discovery  in yaml
================  =========  =========  ==========  =======  =========  =======
openai            yes        yes        yes         yes      yes        yes
anthropic         yes        yes        yes         yes      yes        yes
google (gemini)   yes        yes        yes         yes      yes        yes
mistral           yes        yes        yes         yes      yes        yes
cohere            yes        yes        yes         no       no         yes
huggingface       yes        yes        no (*)      yes      no         yes
azure_openai      no         no         no (**)     no       yes        yes
bedrock           no         no         no (**)     no       no         yes
================  =========  =========  ==========  =======  =========  =======

(*)  HuggingFace is the last-resort bare ``org/model`` match, deliberately not a
     name prefix and absent from ``_KNOWN_MODELS``.
(**) azure_openai / bedrock are recognized LiteLLM prefixes that map to "known
     but not ours" (``None``); they are mapping-only (not validated).
"""

from __future__ import annotations

import pytest

from traigent.config import provider_support as ps
from traigent.integrations.providers import _MODEL_TIERS, list_available_providers
from traigent.providers.validation import (
    _KNOWN_MODELS,
    _PROVIDER_PATTERNS,
    ProviderValidator,
    get_provider_for_model,
)

# The intended matrix, restated as data so a reviewer can read the intersection
# /diff at a glance. Tuple order matches ProviderSpec fields used below.
# name: (key_managed, validated, prefix_detection, tiered, discovery, support_level)
INTENDED_MATRIX = {
    "openai": (True, True, True, True, True, "validated"),
    "anthropic": (True, True, True, True, True, "validated"),
    "google": (True, True, True, True, True, "validated"),
    "mistral": (True, True, True, True, True, "validated"),
    "cohere": (True, True, True, False, False, "validated"),
    "huggingface": (True, True, False, True, False, "validated"),
    "azure_openai": (False, False, False, False, True, "mapping_only"),
    "bedrock": (False, False, False, False, False, "mapping_only"),
}


@pytest.mark.unit
class TestCanonicalTableMatchesIntent:
    """The canonical table itself must encode the documented matrix."""

    def test_table_covers_exactly_the_intended_providers(self) -> None:
        assert {s.name for s in ps.PROVIDER_SPECS} == set(INTENDED_MATRIX)

    def test_each_spec_matches_intended_matrix(self) -> None:
        for spec in ps.PROVIDER_SPECS:
            expected = INTENDED_MATRIX[spec.name]
            actual = (
                spec.key_managed,
                spec.validated,
                spec.detection == ps.DETECT_PREFIX,
                spec.tiered,
                spec.discovery,
                spec.support_level,
            )
            assert actual == expected, f"{spec.name} drifted from intended matrix"


@pytest.mark.unit
class TestValidatorRegistryReconciles:
    """Registry 2: validator patterns / known-models / _validate_* methods."""

    def test_prefix_patterns_match_table(self) -> None:
        assert set(_PROVIDER_PATTERNS) == ps.prefix_detection_providers()

    def test_known_models_match_table(self) -> None:
        assert set(_KNOWN_MODELS) == ps.known_models_providers()

    def test_validate_methods_match_table(self) -> None:
        methods = {
            name[len("_validate_") :]
            for name in dir(ProviderValidator)
            if name.startswith("_validate_")
        }
        methods.discard("provider")  # _validate_provider is the dispatcher
        assert methods == ps.validated_providers()

    def test_huggingface_is_last_resort_not_prefix(self) -> None:
        # HF must not be a prefix/known-models provider; it is the catch-all.
        assert "huggingface" not in _PROVIDER_PATTERNS
        assert "huggingface" not in _KNOWN_MODELS
        assert get_provider_for_model("some-org/some-model") == "huggingface"


@pytest.mark.unit
class TestTierRegistryReconciles:
    """Registry 4: tier lists / list_available_providers."""

    def test_model_tiers_keys_match_table(self) -> None:
        assert set(_MODEL_TIERS) == ps.tiered_registry_keys()

    def test_list_available_providers_matches_tiers(self) -> None:
        assert set(list_available_providers()) == set(_MODEL_TIERS)


@pytest.mark.unit
class TestDiscoveryRegistryReconciles:
    """Registry 5: default model-discovery registrations."""

    def test_default_discoveries_match_table(self) -> None:
        from traigent.integrations.model_discovery import registry as disc

        # Snapshot the (process-global) registry, recompute the *default* set in
        # isolation, then restore — so this exact check does not pollute other
        # tests in the worker. register_discovery takes the lock internally, so
        # we only hold it for the snapshot/restore, never across registration.
        with disc._registry_lock:
            saved_classes = dict(disc._discovery_registry)
            saved_instances = dict(disc._discovery_instances)
            disc._discovery_registry.clear()
            disc._discovery_instances.clear()
        try:
            disc._register_default_discoveries()
            defaults = set(disc.list_registered_providers())
        finally:
            with disc._registry_lock:
                disc._discovery_registry.clear()
                disc._discovery_registry.update(saved_classes)
                disc._discovery_instances.clear()
                disc._discovery_instances.update(saved_instances)

        assert defaults == ps.discovery_registry_keys()


@pytest.mark.unit
class TestModelsYamlReconciles:
    """Registry 3: config/models.yaml provider set + support levels."""

    def test_yaml_provider_keys_match_table(self) -> None:
        config = ps.load_models_yaml()
        assert config, "models.yaml failed to load"
        assert set(config) == ps.models_yaml_registry_keys()

    def test_yaml_support_level_matches_table(self) -> None:
        config = ps.load_models_yaml()
        for spec in ps.PROVIDER_SPECS:
            entry = config[spec.effective_registry_key]
            assert entry.get("support_level") == spec.support_level, (
                f"{spec.effective_registry_key} support_level drift between "
                "models.yaml and provider_support.py"
            )


@pytest.mark.unit
class TestKeyMapReconciles:
    """Registries 1 + 1': both key maps cover exactly the key-managed set."""

    def test_key_managed_set(self) -> None:
        assert ps.key_managed_providers() == {
            "openai",
            "anthropic",
            "google",
            "mistral",
            "cohere",
            "huggingface",
        }

    def test_validator_env_chains_are_supersets_of_table(self) -> None:
        # The validator reads these env vars per provider; the canonical key
        # chain must include them so the key manager and validator agree.
        expected_first = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HF_TOKEN",
        }
        for provider, first in expected_first.items():
            assert ps.provider_env_keys(provider)[0] == first
        # Google/Cohere/HF carry their documented fallbacks.
        assert ps.provider_env_keys("google") == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert ps.provider_env_keys("cohere") == ("COHERE_API_KEY", "CO_API_KEY")
        assert ps.provider_env_keys("huggingface") == (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_API_KEY",
        )

    def test_mapping_only_providers_have_no_key(self) -> None:
        for provider in ps.mapping_only_providers():
            assert ps.provider_env_keys(provider) == ()


@pytest.mark.unit
class TestMappingOnlyValidatorStatus:
    """Requirement #2: recognized-but-unvalidated providers are labeled."""

    def test_mapping_only_provider_returns_not_validated(self) -> None:
        validator = ProviderValidator()
        for provider in ps.mapping_only_providers():
            status = validator._validate_provider(provider)
            assert status.valid is False
            assert status.error_type == "NotValidated"
            assert "not" in status.message.lower()

    def test_truly_unknown_provider_still_unsupported(self) -> None:
        validator = ProviderValidator()
        status = validator._validate_provider("totally-unknown-provider")
        assert status.valid is False
        assert status.error_type == "UnsupportedProvider"
        assert "No validator for provider" in status.message

    def test_litellm_known_prefixes_not_huggingface(self) -> None:
        # Mapping-only providers are recognized LiteLLM prefixes routed away
        # from the HuggingFace catch-all.
        assert get_provider_for_model("azure/my-deployment") != "huggingface"
        assert get_provider_for_model("bedrock/anthropic.claude-3") != "huggingface"
