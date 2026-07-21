"""Canonical provider-support table for the Traigent SDK (issue #1568).

This module is the **single source of truth** for *which providers the SDK
supports and at what level*. Historically the same information was duplicated
— and silently drifted — across six hand-maintained registries:

1. Key map ......... :func:`traigent.config.api_keys.APIKeyManager.get_api_key`
                      and its twin :func:`traigent.utils.env_config.get_api_key`
2. Validator ....... :mod:`traigent.providers.validation`
                      (``_PROVIDER_PATTERNS`` / LiteLLM prefix map /
                      ``_KNOWN_MODELS`` / ``_validate_<provider>`` methods)
3. Model registry .. :mod:`traigent.config.models.yaml` (richest list of models)
4. Tiers ........... :mod:`traigent.integrations.providers`
                      (``_MODEL_TIERS`` / ``_FALLBACK_MODELS`` /
                      ``list_available_providers``)
5. Discovery ....... :mod:`traigent.integrations.model_discovery.registry`

The :data:`PROVIDER_SPECS` table below declares, per canonical provider, which
of those layers it participates in and how its API key is resolved from the
environment. The *model data* (known models + regex pattern) is sourced from
``config/models.yaml`` (see :func:`load_models_yaml`); this table adds the
support-matrix metadata that YAML does not carry (env-var chains, support
level, registry aliases).

Two things genuinely **derive** from this table at runtime:

* **Key resolution** — both ``APIKeyManager.get_api_key`` and
  ``env_config.get_api_key`` resolve provider keys through
  :func:`resolve_api_key_from_env`, so the two key maps can no longer drift and
  ``google``/``mistral`` are now first-class (the concrete #1568 bug fix).
* **The "mapping-only" validator status** — ``ProviderValidator`` consults
  :func:`get_provider_spec` so a recognized-but-unvalidated provider (e.g.
  ``azure_openai``, ``bedrock``) reports a clearly-labeled *not validated*
  status instead of the generic ``UnsupportedProvider`` error.

The remaining registries (validator patterns/known-models, tier lists, the
discovery registrations) keep their authored data — that data is intentionally
richer than this table — but a drift regression test
(``tests/unit/config/test_provider_support_drift.py``) pins every registry to
this table so future drift fails CI.

Aliases / naming: the discovery registry, ``models.yaml`` and the tier lists
historically key Google's models under ``gemini`` while the validator keys them
under ``google``. The canonical name here is ``google``; ``registry_key`` /
``aliases`` capture the ``gemini`` spelling used elsewhere.

Note: ``traigent`` (the Traigent *backend* API key, ``TRAIGENT_API_KEY``) is a
backend-auth credential, **not** an LLM provider, so it is intentionally not in
this LLM-provider support matrix; ``env_config.get_api_key`` keeps a dedicated
branch for it.
"""

# Traceability: CONC-Layer-Core FUNC-INTEGRATIONS REQ-INT-008 CONC-Security

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# NOTE: ``yaml`` (PyYAML) is intentionally NOT imported at module level. This
# module is in the import chain of the CORE public key/config API
# (``traigent.get_api_key`` -> ``APIKeyManager`` / ``env_config.get_api_key``
# via ``resolve_api_key_from_env``), and PyYAML is *not* a core dependency
# (only ``types-PyYAML`` is a dev dep). A module-level ``import yaml`` would
# make a minimal install fail importing ``get_api_key`` before any behavior
# runs (#1568 review). The canonical :data:`PROVIDER_SPECS` table below is a
# hand-authored static literal that needs no YAML load, so all runtime key
# resolution / validator support-level / tier participation work with NO yaml
# present. The optional ``models.yaml`` cross-check is loaded lazily inside
# :func:`load_models_yaml`, which imports ``yaml`` locally and is used only by
# the drift test (dev has PyYAML) and ``model_discovery`` (integrations extra).

# Path to the richest model registry; this module is "sourced from models.yaml".
MODELS_YAML_PATH = Path(__file__).parent / "models.yaml"

# Support levels (kept in lockstep with the ``support_level`` field in
# config/models.yaml; the drift test asserts they match).
SUPPORT_VALIDATED = "validated"
SUPPORT_MAPPING_ONLY = "mapping_only"

# Provider-key-detection categories used by the validator's model->provider
# resolver (``traigent.providers.validation.get_provider_for_model``):
DETECT_PREFIX = "prefix"  # in _PROVIDER_PATTERNS (e.g. gpt-*, claude-*)
DETECT_LAST_RESORT = "last_resort"  # HuggingFace bare org/model catch-all
DETECT_LITELLM_KNOWN = "litellm_known"  # known LiteLLM prefix mapped to None
DETECT_NONE = "none"


@dataclass(frozen=True)
class ProviderSpec:
    """Declared support level for a single canonical provider.

    Attributes:
        name: Canonical provider name (e.g. ``"google"``).
        support_level: ``"validated"`` or ``"mapping_only"`` — must match the
            ``support_level`` recorded in ``config/models.yaml``.
        env_keys: Ordered tuple of environment-variable names from which the
            provider's API key is resolved (first non-empty wins). Empty tuple
            means the provider is intentionally not key-managed.
        aliases: Alternate names this provider is known by in other layers or
            LiteLLM prefixes (e.g. ``("gemini",)`` for Google).
        registry_key: The name used for this provider in ``models.yaml``, the
            tier lists and the discovery registry (defaults to ``name``; e.g.
            ``"gemini"`` for Google).
        validated: True if ``ProviderValidator`` defines ``_validate_<name>``.
        detection: How the validator's model->provider resolver attributes
            models to this provider (see ``DETECT_*``).
        has_known_models: True if the provider appears in the validator's
            ``_KNOWN_MODELS`` map (used for pre-call model-name warnings).
        tiered: True if the provider appears in ``_MODEL_TIERS``.
        discovery: True if a default model-discovery class is registered.
        in_models_yaml: True if the provider has an entry in ``models.yaml``.
    """

    name: str
    support_level: str
    env_keys: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    registry_key: str = ""
    validated: bool = False
    detection: str = DETECT_NONE
    has_known_models: bool = False
    tiered: bool = False
    discovery: bool = False
    in_models_yaml: bool = True

    @property
    def key_managed(self) -> bool:
        """True if this provider resolves an API key from the environment."""
        return bool(self.env_keys)

    @property
    def effective_registry_key(self) -> str:
        """Name used in models.yaml / tiers / discovery for this provider."""
        return self.registry_key or self.name


# ---------------------------------------------------------------------------
# THE canonical table. One entry per provider the SDK recognizes.
# ---------------------------------------------------------------------------
PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="openai",
        support_level=SUPPORT_VALIDATED,
        env_keys=("OPENAI_API_KEY",),
        validated=True,
        detection=DETECT_PREFIX,
        has_known_models=True,
        tiered=True,
        discovery=True,
    ),
    ProviderSpec(
        name="anthropic",
        support_level=SUPPORT_VALIDATED,
        env_keys=("ANTHROPIC_API_KEY",),
        validated=True,
        detection=DETECT_PREFIX,
        has_known_models=True,
        tiered=True,
        discovery=True,
    ),
    ProviderSpec(
        name="google",
        support_level=SUPPORT_VALIDATED,
        # Validator reads GOOGLE_API_KEY or GEMINI_API_KEY; the key manager
        # now reconciles with it (was previously missing -> #1568 bug).
        env_keys=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        aliases=("gemini", "vertex_ai"),
        registry_key="gemini",
        validated=True,
        detection=DETECT_PREFIX,
        has_known_models=True,
        tiered=True,
        discovery=True,
    ),
    ProviderSpec(
        name="mistral",
        support_level=SUPPORT_VALIDATED,
        env_keys=("MISTRAL_API_KEY",),
        validated=True,
        detection=DETECT_PREFIX,
        has_known_models=True,
        tiered=True,
        discovery=True,
    ),
    ProviderSpec(
        name="cohere",
        support_level=SUPPORT_VALIDATED,
        # Validator reads COHERE_API_KEY or CO_API_KEY; reconcile the key map.
        env_keys=("COHERE_API_KEY", "CO_API_KEY"),
        validated=True,
        detection=DETECT_PREFIX,
        has_known_models=True,
        # Intentionally no tier list and no discovery class today.
        tiered=False,
        discovery=False,
    ),
    ProviderSpec(
        name="huggingface",
        support_level=SUPPORT_VALIDATED,
        # Native HF_TOKEN first, then hub token, then legacy HF_API_KEY.
        env_keys=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_KEY"),
        aliases=("hf", "huggingface_hub"),
        validated=True,
        # HF is the LAST-RESORT bare "org/model" match, not a prefix; it is
        # intentionally absent from _PROVIDER_PATTERNS and _KNOWN_MODELS.
        detection=DETECT_LAST_RESORT,
        has_known_models=False,
        tiered=True,
        discovery=False,
    ),
    ProviderSpec(
        name="azure_openai",
        support_level=SUPPORT_MAPPING_ONLY,
        # Azure uses endpoint+deployment+key combos; intentionally not a single
        # env-var key here. Recognized + discovered, but not validated.
        env_keys=(),
        aliases=("azure",),
        validated=False,
        detection=DETECT_LITELLM_KNOWN,
        has_known_models=False,
        tiered=False,
        discovery=True,
    ),
    ProviderSpec(
        name="bedrock",
        support_level=SUPPORT_MAPPING_ONLY,
        # Bedrock authenticates via AWS credentials, not a single API key.
        env_keys=(),
        validated=False,
        detection=DETECT_LITELLM_KNOWN,
        has_known_models=False,
        tiered=False,
        discovery=False,
    ),
    ProviderSpec(
        name="nous",
        support_level=SUPPORT_MAPPING_ONLY,
        # Nous Portal authenticates via a short-lived JWT minted from a refresh
        # token (OAuth), not a single static env-var API key -> not key-managed
        # and mapping-only (recognized for routing/discovery, not preflight
        # validated; the JWT-refresh helper lives in
        # traigent.integrations.llms.nous_auth). Discovery + tiers are declared
        # independently (Azure proves discovery + mapping_only coexist).
        env_keys=(),
        aliases=("nous_portal", "nousresearch"),
        validated=False,
        detection=DETECT_NONE,
        has_known_models=False,
        tiered=True,
        discovery=True,
    ),
)


# ---------------------------------------------------------------------------
# Lookups (canonical name and every alias resolve to the spec, case-insensitive)
# ---------------------------------------------------------------------------
def _build_index() -> dict[str, ProviderSpec]:
    index: dict[str, ProviderSpec] = {}
    for spec in PROVIDER_SPECS:
        names = {spec.name, spec.effective_registry_key, *spec.aliases}
        for alias in names:
            index[alias.lower()] = spec
    return index


_SPEC_INDEX: dict[str, ProviderSpec] = _build_index()


def get_provider_spec(provider: str | None) -> ProviderSpec | None:
    """Return the :class:`ProviderSpec` for a provider name or alias.

    Lookup is case-insensitive and resolves aliases (e.g. ``"gemini"`` and
    ``"vertex_ai"`` both resolve to the ``google`` spec). Returns ``None`` for
    providers not recognized by the SDK.
    """
    if not provider:
        return None
    return _SPEC_INDEX.get(provider.lower())


def canonical_provider_name(provider: str | None) -> str | None:
    """Return the canonical provider name for a name/alias, or ``None``."""
    spec = get_provider_spec(provider)
    return spec.name if spec is not None else None


def is_known_provider(provider: str | None) -> bool:
    """True if the provider name/alias is recognized by the SDK."""
    return get_provider_spec(provider) is not None


def provider_env_keys(provider: str | None) -> tuple[str, ...]:
    """Return the ordered env-var chain used to resolve ``provider``'s API key.

    Returns an empty tuple when the provider is unknown or intentionally not
    key-managed (e.g. ``azure_openai`` / ``bedrock``). This is the single
    resolution source shared by ``APIKeyManager.get_api_key`` and
    ``env_config.get_api_key`` so the two can no longer drift.
    """
    spec = get_provider_spec(provider)
    return spec.env_keys if spec is not None else ()


def resolve_api_key_from_env(
    provider: str | None,
    getter: Callable[[str], str | None] = os.getenv,
) -> str | None:
    """Resolve ``provider``'s API key from the environment.

    Iterates the canonical env-var chain for the provider and returns the first
    non-empty value, or ``None`` if none is set / the provider is not
    key-managed.

    Args:
        provider: Provider name or alias.
        getter: Callable used to read an env var (defaults to ``os.getenv``).
            Callers that want masked logging pass their own reader.
    """
    for env_name in provider_env_keys(provider):
        value = getter(env_name)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Set/collection helpers (used by the drift regression test and consumers)
# ---------------------------------------------------------------------------
def validated_providers() -> set[str]:
    """Canonical names of providers with a real validator."""
    return {s.name for s in PROVIDER_SPECS if s.validated}


def mapping_only_providers() -> set[str]:
    """Canonical names recognized but intentionally not validated."""
    return {s.name for s in PROVIDER_SPECS if s.support_level == SUPPORT_MAPPING_ONLY}


def key_managed_providers() -> set[str]:
    """Canonical names that resolve an API key from the environment."""
    return {s.name for s in PROVIDER_SPECS if s.key_managed}


def prefix_detection_providers() -> set[str]:
    """Canonical names attributed to models via a known name prefix."""
    return {s.name for s in PROVIDER_SPECS if s.detection == DETECT_PREFIX}


def known_models_providers() -> set[str]:
    """Canonical names that appear in the validator's ``_KNOWN_MODELS``."""
    return {s.name for s in PROVIDER_SPECS if s.has_known_models}


def tiered_registry_keys() -> set[str]:
    """Registry keys (e.g. ``gemini``) that should appear in ``_MODEL_TIERS``."""
    return {s.effective_registry_key for s in PROVIDER_SPECS if s.tiered}


def discovery_registry_keys() -> set[str]:
    """Registry keys that should have a default model-discovery registration."""
    return {s.effective_registry_key for s in PROVIDER_SPECS if s.discovery}


def models_yaml_registry_keys() -> set[str]:
    """Registry keys that should have an entry in ``config/models.yaml``."""
    return {s.effective_registry_key for s in PROVIDER_SPECS if s.in_models_yaml}


@lru_cache(maxsize=1)
def load_models_yaml() -> dict[str, Any]:
    """Load and cache ``config/models.yaml`` (the model-data source).

    ``yaml`` (PyYAML) is imported **inside** this function on purpose: PyYAML is
    not a core dependency, so importing it at module level would break the core
    key/config API on a minimal install (#1568 review). This loader is the only
    yaml-dependent path in this module and is used exclusively by the drift
    regression test (dev installs have PyYAML) and ``model_discovery`` (the
    integrations extra, which ships PyYAML). The runtime key-resolution /
    support-matrix paths never call it and therefore never need yaml.

    Returns an empty dict if the file is missing or unparseable; callers and
    the drift test treat that as a hard failure surfaced via assertions rather
    than crashing import. (Runtime never reaches this function.)
    """
    import yaml

    if not MODELS_YAML_PATH.exists():
        return {}
    with open(MODELS_YAML_PATH) as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


__all__ = [
    "ProviderSpec",
    "PROVIDER_SPECS",
    "SUPPORT_VALIDATED",
    "SUPPORT_MAPPING_ONLY",
    "DETECT_PREFIX",
    "DETECT_LAST_RESORT",
    "DETECT_LITELLM_KNOWN",
    "DETECT_NONE",
    "MODELS_YAML_PATH",
    "get_provider_spec",
    "canonical_provider_name",
    "is_known_provider",
    "provider_env_keys",
    "resolve_api_key_from_env",
    "validated_providers",
    "mapping_only_providers",
    "key_managed_providers",
    "prefix_detection_providers",
    "known_models_providers",
    "tiered_registry_keys",
    "discovery_registry_keys",
    "models_yaml_registry_keys",
    "load_models_yaml",
]
