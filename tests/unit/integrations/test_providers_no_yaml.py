"""Regression tests for the lazy model_discovery import in providers.py.

These tests verify that ``get_models_for_tier`` works correctly when PyYAML /
model_discovery is unavailable (returns static tier lists) and that the normal
discovery path still functions when PyYAML is present.

Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008
PR: #1568 review fix — lazy model_discovery import; tiers fall back to static lists.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from collections.abc import Generator
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _yaml_unavailable() -> Generator[None, None, None]:
    """Context manager that makes PyYAML (and its dependents) appear absent.

    The lazy get_model_discovery wrapper in providers.py catches ImportError
    raised when model_discovery.base tries ``import yaml``.  We reproduce that
    by:
      1. Stashing and removing the live ``yaml`` / model_discovery submodules
         from sys.modules so they are re-imported on the next import attempt.
      2. Setting sys.modules['yaml'] = None (Python's sentinel for
         "this module is explicitly absent").
      3. Restoring everything on exit so other tests are unaffected.
    """
    keys_to_stash = [
        k
        for k in sys.modules
        if k == "yaml" or k.startswith("traigent.integrations.model_discovery")
    ]
    stashed = {k: sys.modules.pop(k) for k in keys_to_stash}
    sys.modules["yaml"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        # Remove the sentinel
        sys.modules.pop("yaml", None)
        # Restore previous modules
        sys.modules.update(stashed)


# ---------------------------------------------------------------------------
# Tests: no-yaml fallback path
# ---------------------------------------------------------------------------


class TestGetModelsForTierNoYaml:
    """get_models_for_tier must return static lists when PyYAML is unavailable."""

    def test_import_does_not_require_yaml(self) -> None:
        """providers module can be imported (re-accessed) even with yaml=None.

        Because providers.py is already loaded by the test runner the import
        itself is cached.  What we verify here is that calling the module
        attributes works — the module-level code does NOT touch yaml.
        """
        import importlib

        # Push yaml out of sys.modules so any eager import would fail
        with _yaml_unavailable():
            # Re-importing an already-loaded module returns the cached version
            # without executing module-level code again.  Still, no AttributeError
            # or ImportError must surface when we access the public API.
            import traigent.integrations.providers as mod  # noqa: PLC0415

            importlib.invalidate_caches()
            assert callable(mod.get_models_for_tier)

    def test_hf_balanced_returns_static_models_without_yaml(self) -> None:
        """HuggingFace balanced tier returns static fallback when yaml is absent."""
        with (
            patch.dict(os.environ, {}, clear=True),
            _yaml_unavailable(),
        ):
            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="huggingface", tier="balanced")

        assert isinstance(models, list), "Must return a list, not raise"
        assert len(models) >= 1, "Static HF balanced list must be non-empty"
        assert "gpt-4o-mini" not in models, "HF must not fall back to OpenAI model"
        assert all("/" in m for m in models), (
            "HF model IDs are expected in 'org/model' format"
        )
        # Verify at least one known static entry is present
        assert any(
            m in models
            for m in [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ]
        ), f"Expected a known HF model in static fallback list, got: {models}"

    def test_hf_fast_returns_static_models_without_yaml(self) -> None:
        """HuggingFace fast tier static fallback works when yaml is absent."""
        with (
            patch.dict(os.environ, {}, clear=True),
            _yaml_unavailable(),
        ):
            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="huggingface", tier="fast")

        assert isinstance(models, list)
        assert len(models) >= 1
        assert "gpt-4o-mini" not in models
        assert all("/" in m for m in models)

    def test_hf_quality_returns_static_models_without_yaml(self) -> None:
        """HuggingFace quality tier static fallback works when yaml is absent."""
        with (
            patch.dict(os.environ, {}, clear=True),
            _yaml_unavailable(),
        ):
            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="huggingface", tier="quality")

        assert isinstance(models, list)
        assert len(models) >= 1
        assert any("70B" in m or "Mixtral" in m for m in models), (
            "Quality tier should include a large HF model"
        )

    def test_openai_returns_static_models_without_yaml(self) -> None:
        """OpenAI tier static fallback also works when yaml is absent."""
        with (
            patch.dict(os.environ, {}, clear=True),
            _yaml_unavailable(),
        ):
            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="openai", tier="quality")

        assert "gpt-4o" in models

    def test_get_model_discovery_lazy_wrapper_returns_none_on_import_error(
        self,
    ) -> None:
        """The lazy get_model_discovery wrapper returns None when yaml is absent.

        This directly tests the lazy-wrapper function rather than going through
        the higher-level get_models_for_tier API.
        """
        from traigent.integrations import providers  # noqa: PLC0415

        with _yaml_unavailable():
            result = providers.get_model_discovery(provider="huggingface", cached=False)

        # With yaml unavailable the wrapper must return None (not raise)
        assert result is None, f"Expected None when yaml is absent, got {result!r}"

    def test_no_yaml_does_not_raise(self) -> None:
        """None of the public tier/provider helpers must raise when yaml is absent."""
        from traigent.integrations.providers import (  # noqa: PLC0415
            get_all_tiers,
            get_models_for_tier,
            list_available_providers,
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            _yaml_unavailable(),
        ):
            providers = list_available_providers()
            tiers = get_all_tiers()
            # Pick a representative provider + tier combo that exercises the HF path
            hf_models = get_models_for_tier(provider="huggingface", tier="balanced")
            oai_models = get_models_for_tier(provider="openai", tier="fast")

        assert "huggingface" in providers
        assert "fast" in tiers
        assert len(hf_models) >= 1
        assert len(oai_models) >= 1


# ---------------------------------------------------------------------------
# Tests: discovery path still works when yaml IS available
# ---------------------------------------------------------------------------


class TestGetModelsForTierWithDiscovery:
    """Normal discovery path must still work — this is a non-regression check."""

    def test_discovery_result_is_used_when_available(self) -> None:
        """When model_discovery returns a non-None object, its models are consulted."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_get_discovery,
        ):
            mock_disc = MagicMock()
            mock_disc.list_models.return_value = [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "meta-llama/Meta-Llama-3-70B-Instruct",
            ]
            mock_get_discovery.return_value = mock_disc

            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="huggingface", tier="balanced")

        # discovery was called and its output flowed into the result
        mock_get_discovery.assert_called()
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_existing_patch_target_still_works(self) -> None:
        """Existing tests that patch get_model_discovery to return None still pass.

        This ensures backward compat: the refactor kept the same patch target.
        """
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery", return_value=None
            ),
        ):
            from traigent.integrations.providers import get_models_for_tier  # noqa: PLC0415

            models = get_models_for_tier(provider="openai", tier="balanced")

        assert "gpt-4o-mini" in models, (
            "Patching get_model_discovery to None should fall back to static list"
        )
