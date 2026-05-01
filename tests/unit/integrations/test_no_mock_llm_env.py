"""Verify dangerous env-toggles are not honored by ``MockAdapter``.

Sprint 2 cleanup (S2-B) removed the ``TRAIGENT_MOCK_LLM`` env-toggle
from the ``MockAdapter`` and ``with_mock_support`` decorator surface
in :mod:`traigent.integrations.utils.mock_adapter`. The current
contract on this surface:

* The recommended path is the in-code API
  :func:`traigent.testing.enable_mock_mode_for_quickstart`.
* The legacy ``TRAIGENT_MOCK_LLM=true`` env var is honored as a
  backward-compat path **only outside production**. ``is_mock_enabled``
  delegates to :func:`traigent.utils.env_config.is_mock_llm`, which
  hard-blocks the env-var path when ``ENVIRONMENT=production``. Those
  guarantees are exercised in
  ``tests/unit/integrations/test_mock_adapter_safety.py``.
* Provider-specific ``*_MOCK`` env vars (e.g. ``OPENAI_MOCK=true``)
  are still completely ignored — those were the worst offenders in
  the original incident.

Companion S2-B test classes for ``traigent/security/encryption.py``,
``traigent/evaluators/local.py``, and ``traigent/core/optimization_pipeline.py``
live on ``develop`` and will land here once their corresponding source
changes are cherry-picked to ``main`` (the cherry-pick of #799 was
scoped to the mock-mode API only).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from traigent.integrations.utils.mock_adapter import MockAdapter


class TestProviderSpecificMockEnvsAreIgnored:
    """Provider-specific ``*_MOCK`` env vars (e.g. ``OPENAI_MOCK=true``)
    were the worst offenders in the original prod incident — one var
    per provider, easy to miss in a deployment audit. Those are
    completely ignored regardless of ``ENVIRONMENT`` or any other
    settings."""

    @pytest.mark.parametrize(
        "var",
        [
            "OPENAI_MOCK",
            "AZURE_OPENAI_MOCK",
            "ANTHROPIC_MOCK",
            "GEMINI_MOCK",
            "COHERE_MOCK",
            "HUGGINGFACE_MOCK",
            "BEDROCK_MOCK",
        ],
    )
    def test_provider_specific_mock_envs_do_not_enable_mock(self, var: str) -> None:
        """Provider-specific *_MOCK env vars must not enable mock mode."""
        with patch.dict(os.environ, {var: "true"}, clear=True):
            # Strip the trailing _MOCK to derive the provider name.
            provider = var.removesuffix("_MOCK").lower()
            assert MockAdapter.is_mock_enabled(provider) is False

    def test_with_mock_support_decorator_is_no_longer_exported(self) -> None:
        """``with_mock_support`` was removed from the module + package API."""
        import traigent.integrations.utils as utils_pkg
        from traigent.integrations.utils import mock_adapter

        assert not hasattr(mock_adapter, "with_mock_support")
        assert not hasattr(utils_pkg, "with_mock_support")
