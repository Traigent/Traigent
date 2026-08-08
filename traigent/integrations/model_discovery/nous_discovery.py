"""Nous Portal (Hermes) model discovery.

Nous Portal is OpenAI-compatible: it exposes ``/v1/models``, so discovery reuses
the ``openai`` SDK client — but it MUST pass ``base_url=NOUS_BASE_URL``
explicitly, or a verbatim copy of the OpenAI discovery would silently hit
``api.openai.com`` instead of the Nous inference API. Credentials come from the
JWT-refresh helper (:mod:`traigent.integrations.llms.nous_auth`), not a static
env var.

The portal fronts the whole Hermes family *and* hundreds of hosted third-party
models, so discovery returns **all** advertised model IDs sorted and unfiltered
— any shape filter would wrongly drop legitimate models. When no credentials are
present (the common offline case) it returns ``[]`` so ``list_models()`` falls
back to the ``config/models.yaml`` known-model list.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
from collections.abc import Callable

from traigent.integrations.llms import nous_auth
from traigent.integrations.llms.nous_auth import (
    NOUS_BASE_URL,
    _get_nous_cache_identity,
    get_nous_api_key,
    has_nous_credentials,
)
from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# OWNER: confirm the exact portal-served model-ID string form (bare vs
# "NousResearch/"-prefixed) against ``traigent models -p nous --json`` in
# Phase-0 before merge. The pattern accepts BOTH forms, so discovery validates
# either spelling; the seeded known_models (config/models.yaml) use the
# HuggingFace-canonical "NousResearch/..." form.
NOUS_MODEL_PATTERN = r"^(NousResearch/|Hermes-|DeepHermes-)"


class NousDiscovery(ModelDiscovery):
    """Model discovery for Nous Portal (Hermes family + hosted third-party models)."""

    PROVIDER = "nous"
    FRAMEWORK = Framework.NOUS

    def _prepare_model_fetch(self) -> tuple[str, Callable[[], list[str]]]:
        """Bind cache lookup and SDK fetch to one non-network auth snapshot."""
        identity = _get_nous_cache_identity()
        if identity is None:
            return self.PROVIDER, lambda: []

        source, identity_material = identity
        cache_parts = (
            (NOUS_BASE_URL, source)
            if source == "invalid-auth-file-v1"
            else (NOUS_BASE_URL, source, identity_material)
        )
        cache_key = f"{self.PROVIDER}-{self._fingerprint(*cache_parts)}"

        return cache_key, lambda: self._fetch_models_from_identity(identity)

    def _get_credential_fingerprint(self) -> str | None:
        """Partition portal models by endpoint and winning credential source."""
        identity = _get_nous_cache_identity()
        if identity is None:
            return None
        source, identity_material = identity
        if source == "invalid-auth-file-v1":
            invalid_partition_id: str = ModelDiscovery._fingerprint(
                NOUS_BASE_URL, source
            )
            return invalid_partition_id
        valid_partition_id: str = ModelDiscovery._fingerprint(
            NOUS_BASE_URL, source, identity_material
        )
        return valid_partition_id

    def _fetch_models_from_identity(self, identity: tuple[str, str]) -> list[str]:
        """Fetch using the already captured identity, without rereading sources."""
        source, identity_material = identity
        try:
            if source in {"invalid-auth-file-v1", "invalid-auth-source-v1"}:
                raise nous_auth.NousAuthError(
                    "Nous credential identity is invalid; refusing authenticated discovery"
                )
            if source == "static":
                bearer = identity_material
            else:
                with nous_auth._state_lock:
                    bearer = nous_auth._get_nous_api_key_for_identity(
                        identity_material, source
                    )
            return self._fetch_models_with_bearer(bearer)
        except ImportError:
            logger.debug("OpenAI SDK not installed")
            raise
        except Exception as exc:
            logger.warning(
                "Nous credentials present but token mint/discovery failed "
                "(%s: %s); falling back to the static model catalog",
                type(exc).__name__,
                exc,
            )
            raise

    def _fetch_models_with_bearer(self, bearer: str) -> list[str]:
        """Fetch portal models with a token selected by the caller."""
        from openai import OpenAI

        client_kwargs: dict[str, str] = {
            "api_key": bearer,
            "base_url": NOUS_BASE_URL,
        }
        client = OpenAI(**client_kwargs)
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        logger.info("Discovered %d Nous models via SDK", len(model_ids))
        return sorted(model_ids)

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models from the Nous Portal ``/v1/models`` endpoint.

        Returns:
            All advertised model IDs, sorted and unfiltered, or ``[]`` when no
            credentials are present (the designed models.yaml-fallback path).

        Raises:
            Exception: If the SDK is missing or the API call / token mint fails;
                the caller's ``list_models()`` catches it and falls back to the
                config known-model list.
        """
        if not has_nous_credentials():
            logger.debug("No Nous credentials present, skipping SDK discovery")
            return []

        try:
            # base_url is REQUIRED here — without it the client hits
            # api.openai.com instead of the Nous inference API.
            return self._fetch_models_with_bearer(get_nous_api_key())

        except ImportError:
            logger.debug("OpenAI SDK not installed")
            raise
        except Exception as exc:
            # Credentials ARE present (the no-credential path returned [] above),
            # so a mint/fetch failure here is a genuinely degraded state — a
            # broken credential that would otherwise look like a clean discovery
            # once list_models() silently falls back to the static catalog.
            # Surface it at WARNING (not debug) so the degraded state is visible,
            # then re-raise: base.list_models() catches it and falls back to the
            # models.yaml known-model list (never a mock). The shared base-class
            # fallback contract for other providers is intentionally untouched.
            logger.warning(
                "Nous credentials present but token mint/discovery failed "
                "(%s: %s); falling back to the static model catalog",
                type(exc).__name__,
                exc,
            )
            raise

    def get_pattern(self) -> str | None:
        """Return the Hermes-family regex, preferring a config-file override."""
        # Explicit annotation: the base reads the pattern out of an untyped YAML
        # dict, so the value is Any without it (warn_return_any / changed-file
        # mypy scope would flag this new module even though the shipped
        # discovery classes share the pattern).
        config_pattern: str | None = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern
        return NOUS_MODEL_PATTERN
