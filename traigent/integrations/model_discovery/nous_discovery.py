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

from traigent.integrations.llms.nous_auth import (
    NOUS_BASE_URL,
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
            from openai import OpenAI

            # base_url is REQUIRED here — without it the client hits
            # api.openai.com instead of the Nous inference API.
            client = OpenAI(api_key=get_nous_api_key(), base_url=NOUS_BASE_URL)
            models = client.models.list()

            model_ids = [model.id for model in models.data]
            logger.info("Discovered %d Nous models via SDK", len(model_ids))
            return sorted(model_ids)

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
