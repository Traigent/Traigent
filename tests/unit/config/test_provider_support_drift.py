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
from traigent.integrations.providers import (
    _FALLBACK_MODELS,
    _MODEL_TIERS,
    list_available_providers,
)
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


# Known-retired / delisted model IDs that must never resurface in any of the
# hand-maintained default tables (#1937). The old drift test asserted only
# provider KEY-SETs, so every retired ID here passed CI green. The canonical
# set now lives IN THE SDK (traigent.config.retired_models) because the
# runtime pattern fallback must consult it too; the tests import it so the
# swept tables and the runtime denylist can never drift apart, and the
# EXPECTED_RETIRED_CORE literal below guards against the source set being
# quietly emptied (which would relax both runtime and tests at once).
from traigent.config.retired_models import (  # noqa: E402
    RETIRED_MODEL_IDS,
    is_retired_model,
    normalize_model_id,
)

EXPECTED_RETIRED_CORE = frozenset(
    {
        "o1-preview",
        "o1-mini",
        "o1-preview-2024-09-12",
        "o1-mini-2024-09-12",
        "claude-3-opus-20240229",
        "claude-3-opus-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "gemini-1.0-pro",
        "gemini-pro",
        "models/gemini-pro",
    }
)

# The SDK's current Anthropic default fallback IDs (parameter_ranges.Choices.model
# / providers._FALLBACK_MODELS). These must be recognized by _KNOWN_MODELS so the
# validator never warns "unknown" on the SDK's own happy-path defaults.
CURRENT_ANTHROPIC_DEFAULTS = frozenset(
    {
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
    }
)


def _all_ids(table) -> set[str]:
    """Flatten every model-ID value out of a nested tier/fallback table."""
    ids: set[str] = set()
    for value in table.values():
        if isinstance(value, dict):
            for models in value.values():
                ids.update(models)
        else:  # flat list of model IDs (e.g. _FALLBACK_MODELS)
            ids.update(value)
    return ids


@pytest.mark.unit
class TestModelCatalogCurrency:
    """#1937: assert the model-ID VALUES, not just the provider key-sets.

    The provider-drift gate historically pinned only the taxonomy (which
    provider appears in which registry). Catalog *currency* — the actual model
    strings inside each tier/frozenset — was invisible, so retired IDs passed
    CI green. These checks close that blind spot offline.
    """

    def test_model_tiers_carry_no_retired_ids(self) -> None:
        offenders = _all_ids(_MODEL_TIERS) & RETIRED_MODEL_IDS
        assert not offenders, f"_MODEL_TIERS lists retired model IDs: {offenders}"

    def test_fallback_models_carry_no_retired_ids(self) -> None:
        offenders = _all_ids(_FALLBACK_MODELS) & RETIRED_MODEL_IDS
        assert not offenders, f"_FALLBACK_MODELS lists retired model IDs: {offenders}"

    def test_known_models_drop_fully_dead_gemini_1_0(self) -> None:
        # _KNOWN_MODELS is a broad *recognition* allowlist (avoid false "unknown"
        # warnings), so it legitimately keeps older-but-referenceable IDs. Only
        # the fully-dead Gemini 1.0 IDs must be pruned (#1936).
        dead_gemini_1_0 = {"gemini-1.0-pro", "gemini-pro"}
        offenders = set(_KNOWN_MODELS.get("google", frozenset())) & dead_gemini_1_0
        assert not offenders, (
            f"_KNOWN_MODELS['google'] lists dead Gemini 1.0 IDs: {offenders}"
        )

    def test_current_anthropic_defaults_are_known(self) -> None:
        # The SDK must not warn "unknown" on its own default Anthropic models.
        missing = CURRENT_ANTHROPIC_DEFAULTS - set(_KNOWN_MODELS["anthropic"])
        assert not missing, f"_KNOWN_MODELS['anthropic'] omits SDK defaults: {missing}"

    def test_fallback_anthropic_defaults_are_current(self) -> None:
        anthropic_fallback = {
            mid
            for (provider, _tier), models in _FALLBACK_MODELS.items()
            if provider == "anthropic"
            for mid in models
        }
        assert anthropic_fallback == CURRENT_ANTHROPIC_DEFAULTS

    def test_anthropic_table_ids_are_in_models_yaml(self) -> None:
        # Every Anthropic ID emitted by the tier/fallback tables must exist in
        # the canonical config/models.yaml catalog (single source of truth).
        config = ps.load_models_yaml()
        yaml_anthropic = set(config["anthropic"]["known_models"])
        table_anthropic = {
            mid for tier in _MODEL_TIERS["anthropic"].values() for mid in tier
        } | {
            mid
            for (provider, _tier), models in _FALLBACK_MODELS.items()
            if provider == "anthropic"
            for mid in models
        }
        missing = table_anthropic - yaml_anthropic
        assert not missing, f"Anthropic IDs absent from models.yaml catalog: {missing}"

    def test_models_yaml_serves_no_retired_ids(self) -> None:
        # config/models.yaml known_models is not just a recognition allowlist:
        # BaseModelDiscovery.list_models() SERVES it as the discovery fallback
        # when SDK discovery fails, so a retired ID here is *offered* as an
        # available model (#1936/#1937). Membership is checked on the
        # NORMALIZED ID so provider-prefixed alias forms (e.g. the Bedrock
        # anthropic.claude-3-opus-20240229-v1:0) cannot dodge the sweep.
        # No silent exemptions: a back-compat entry that must stay needs an
        # inline models.yaml justification AND removal from RETIRED_MODEL_IDS
        # with the reason recorded there.
        config = ps.load_models_yaml()
        offenders: dict[str, set[str]] = {}
        for provider, provider_cfg in config.items():
            if not isinstance(provider_cfg, dict):
                continue
            known = set(provider_cfg.get("known_models") or [])
            hit = {mid for mid in known if is_retired_model(mid)}
            if hit:
                offenders[provider] = hit
        assert not offenders, f"models.yaml serves retired model IDs: {offenders}"

    def test_retired_core_ids_stay_in_canonical_denylist(self) -> None:
        # Anti-tautology guard: the sweep tests import RETIRED_MODEL_IDS from
        # the SDK source. If someone quietly removed entries there, runtime
        # denylist AND sweep tests would relax together — this literal core
        # pins the members that must never leave the set.
        missing = EXPECTED_RETIRED_CORE - RETIRED_MODEL_IDS
        assert not missing, f"canonical denylist lost core retired IDs: {missing}"

    def test_normalize_model_id_collapses_alias_forms(self) -> None:
        assert normalize_model_id("models/gemini-pro") == "gemini-pro"
        assert (
            normalize_model_id("anthropic.claude-3-opus-20240229-v1:0")
            == "claude-3-opus-20240229"
        )
        assert normalize_model_id("meta.llama3-70b-instruct-v1:0") == (
            "llama3-70b-instruct"
        )
        # Non-aliased IDs pass through untouched.
        assert normalize_model_id("gpt-4o") == "gpt-4o"

    def test_discovery_snapshots_carry_no_retired_ids(self) -> None:
        # The hardcoded per-provider discovery snapshots are SERVED by
        # list_models() exactly like models.yaml — sweep them too (#1937).
        from traigent.integrations.model_discovery.anthropic_discovery import (
            KNOWN_ANTHROPIC_MODELS,
        )
        from traigent.integrations.model_discovery.azure_discovery import (
            KNOWN_AZURE_BASE_MODELS,
        )

        for name, snapshot in {
            "KNOWN_ANTHROPIC_MODELS": KNOWN_ANTHROPIC_MODELS,
            "KNOWN_AZURE_BASE_MODELS": KNOWN_AZURE_BASE_MODELS,
        }.items():
            offenders = {mid for mid in snapshot if is_retired_model(mid)}
            assert not offenders, f"{name} serves retired model IDs: {offenders}"

    def test_every_finite_model_table_is_swept(self) -> None:
        # #1937's literal ask: value-level drift — EVERY finite model table the
        # SDK ships must contain no member of the retired denylist. This is the
        # single guard that fails if any table drifts back to serving a retired
        # id (sol's correction: the round-3 check only partially scanned
        # _KNOWN_MODELS — this scans it and the provider tables in full).
        import traigent.integrations.providers as prov
        from traigent.providers.validation import _KNOWN_MODELS

        tables: dict[str, list[str]] = {}
        for provider, ids in _KNOWN_MODELS.items():
            tables[f"_KNOWN_MODELS[{provider}]"] = list(ids)
        for key, ids in prov._FALLBACK_MODELS.items():
            tables[f"_FALLBACK_MODELS[{key}]"] = list(ids)
        for provider, tiers in prov._MODEL_TIERS.items():
            for tier, ids in tiers.items():
                tables[f"_MODEL_TIERS[{provider}][{tier}]"] = list(ids)

        offenders = {
            name: sorted(m for m in ids if is_retired_model(m))
            for name, ids in tables.items()
            if any(is_retired_model(m) for m in ids)
        }
        assert not offenders, f"finite model tables serve retired IDs: {offenders}"

    def test_pattern_fallback_rejects_retired_ids(self) -> None:
        # The regex fallback accepts unknown-but-plausible NEW ids; it must
        # not RE-ADMIT retired ids whose shape still matches. These four are
        # the confirmed re-admission cases from review; each matched its
        # provider pattern before the denylist check landed in
        # ModelDiscovery.is_valid_model.
        from traigent.integrations.model_discovery.anthropic_discovery import (
            AnthropicDiscovery,
        )
        from traigent.integrations.model_discovery.gemini_discovery import (
            GeminiDiscovery,
        )
        from traigent.integrations.model_discovery.openai_discovery import (
            OpenAIDiscovery,
        )

        assert not AnthropicDiscovery().is_valid_model("claude-3-opus-20240229")
        assert not OpenAIDiscovery().is_valid_model("o1-preview-2024-09-12")
        gemini = GeminiDiscovery()
        assert not gemini.is_valid_model("gemini-1.5-pro")
        assert not gemini.is_valid_model("models/gemini-pro")
        # Provider-prefixed Bedrock alias normalizes to the retired base ID.
        assert is_retired_model("anthropic.claude-3-opus-20240229-v1:0")
        # Plausible NEW ids must still pass the shape fallback.
        assert AnthropicDiscovery().is_valid_model("claude-sonnet-4-6")
        assert GeminiDiscovery().is_valid_model("gemini-2.0-flash")
