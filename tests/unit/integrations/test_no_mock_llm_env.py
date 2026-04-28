"""Verify TRAIGENT_MOCK_LLM is no longer honored anywhere in the SDK.

Sprint 2 cleanup (S2-B) removed the ``TRAIGENT_MOCK_LLM`` env-toggle from
three high-severity sites that previously shipped to customers:

* ``traigent/integrations/utils/mock_adapter.py`` — ``with_mock_support``
  decorator + ``MockAdapter.is_mock_enabled`` (env-toggle replaced real
  LLM calls with canned mock text).
* ``traigent/security/encryption.py`` — encrypt() / decrypt() fallback
  branches (env-toggle produced ``b"mock_" + plaintext`` "ciphertext"
  and stripped the prefix on decrypt).
* ``traigent/evaluators/local.py`` — ``_compute_mock_accuracy`` (env-toggle
  fabricated accuracy via random.uniform() + string-length heuristics).

These tests pin the new behaviour: setting ``TRAIGENT_MOCK_LLM=true`` and
calling the affected methods must hit real code paths and either succeed
on real input or fail with a real error — never return mock data.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from traigent.integrations.utils.mock_adapter import MockAdapter
from traigent.security.encryption import EncryptionManager, KeyManager


class TestMockAdapterIgnoresEnv:
    """``MockAdapter.is_mock_enabled`` must not consult any env variable."""

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "True"])
    def test_traigent_mock_llm_does_not_enable_mock(self, value: str) -> None:
        """No truthy form of TRAIGENT_MOCK_LLM should enable mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": value}, clear=True):
            assert MockAdapter.is_mock_enabled("openai") is False
            assert MockAdapter.is_mock_enabled("anthropic") is False
            assert MockAdapter.is_mock_enabled("gemini") is False
            assert MockAdapter.is_mock_enabled("cohere") is False
            assert MockAdapter.is_mock_enabled("bedrock") is False
            assert MockAdapter.is_mock_enabled("anything") is False

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
        """Provider-specific *_MOCK env vars must not enable mock mode either."""
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


class TestEncryptionIgnoresEnv:
    """encrypt() / decrypt() must fail closed regardless of TRAIGENT_MOCK_LLM."""

    def test_encrypt_with_mock_env_still_raises_when_no_crypto(self) -> None:
        """Setting TRAIGENT_MOCK_LLM=true must NOT enable mock encryption."""
        key_manager = KeyManager()
        em = EncryptionManager(key_manager)
        em.crypto_available = False  # simulate missing cryptography lib

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            with pytest.raises(
                RuntimeError, match="requires the 'cryptography' package"
            ):
                em.encrypt("payload that must not leak as plaintext")

    def test_decrypt_with_mock_env_still_raises_when_no_crypto(self) -> None:
        """Setting TRAIGENT_MOCK_LLM=true must NOT enable mock decryption."""
        key_manager = KeyManager()
        em = EncryptionManager(key_manager)
        em.crypto_available = False
        key_id = key_manager.generate_key("AES-256")
        envelope = {
            "ciphertext": b"mock_attacker_supplied_payload",
            "iv": os.urandom(12),
            "tag": os.urandom(16),
            "key_id": key_id,
        }

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            with pytest.raises(
                RuntimeError, match="requires the 'cryptography' package"
            ):
                em.decrypt(envelope)

    def test_encrypt_does_not_emit_mock_prefix_with_real_crypto(self) -> None:
        """Real encryption never emits the legacy b'mock_' or b'encrypted_' prefix."""
        key_manager = KeyManager()
        em = EncryptionManager(key_manager)
        if not em.crypto_available:
            pytest.skip("cryptography not installed in this environment")

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            result = em.encrypt(b"sensitive payload")

        assert not result["ciphertext"].startswith(b"mock_")
        assert not result["ciphertext"].startswith(b"encrypted_")
        # Round-trip must still succeed under real crypto.
        assert em.decrypt(result) == b"sensitive payload"


class TestLocalEvaluatorIgnoresEnv:
    """LocalEvaluator must always compute real accuracy, regardless of env."""

    def test_compute_mock_accuracy_method_is_removed(self) -> None:
        """The fabricator method must no longer exist on LocalEvaluator."""
        from traigent.evaluators.local import LocalEvaluator

        assert not hasattr(LocalEvaluator, "_compute_mock_accuracy")
        assert not hasattr(LocalEvaluator, "_log_mock_mode_warning")

    def test_accuracy_metric_is_real_even_with_mock_env(self) -> None:
        """With TRAIGENT_MOCK_LLM=true, accuracy must be deterministic real comparison."""
        from traigent.evaluators.local import LocalEvaluator

        evaluator = LocalEvaluator(metrics=["accuracy"])

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            # Identical strings — real accuracy must be 1.0.
            metrics_match = evaluator._compute_example_metrics(
                actual_output="hello",
                expected_output="hello",
            )
            # Different strings — real accuracy must be 0.0.
            metrics_mismatch = evaluator._compute_example_metrics(
                actual_output="hello",
                expected_output="goodbye",
            )

        assert metrics_match["accuracy"] == 1.0
        assert metrics_mismatch["accuracy"] == 0.0

    def test_accuracy_metric_does_not_use_random_with_mock_env(self) -> None:
        """Repeat calls under TRAIGENT_MOCK_LLM=true must be deterministic.

        The old _compute_mock_accuracy used random.uniform() so identical
        inputs would yield different scores. The real path is purely
        deterministic.
        """
        from traigent.evaluators.local import LocalEvaluator

        evaluator = LocalEvaluator(metrics=["accuracy"])

        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            scores = [
                evaluator._compute_example_metrics(
                    actual_output="positive sentiment indeed", expected_output="other"
                )["accuracy"]
                for _ in range(10)
            ]

        # All identical — no random fabrication.
        assert len(set(scores)) == 1
        # And it's the real-comparison value (mismatch -> 0.0).
        assert scores[0] == 0.0


class TestMockModeConfigIsInert:
    """``mock_mode_config`` must not change optimizer or evaluator behaviour.

    Round 2 of S2-B retired the last behavioural sites that consulted this
    parameter. The parameter is still accepted on public APIs for backward
    compatibility but must be a true no-op.
    """

    def test_resolve_custom_evaluator_ignores_mock_mode_config(self) -> None:
        """``resolve_custom_evaluator`` must return the user evaluator regardless of mock_mode_config."""
        from traigent.core.optimization_pipeline import resolve_custom_evaluator

        def my_evaluator(*_args: object, **_kwargs: object) -> dict[str, float]:
            return {"accuracy": 1.0}

        # Even with TRAIGENT_MOCK_LLM=true AND a mock_mode_config that previously
        # would have triggered the override, the user-supplied evaluator wins.
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=False):
            result = resolve_custom_evaluator(
                my_evaluator,
                mock_mode_config={"enabled": True, "override_evaluator": True},
                decorator_custom_evaluator=None,
            )
        assert result is my_evaluator

        # And without the env var, same behaviour.
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_custom_evaluator(
                None,
                mock_mode_config={"enabled": True, "override_evaluator": True},
                decorator_custom_evaluator=my_evaluator,
            )
        assert result is my_evaluator

    def test_apply_mock_config_overrides_does_not_change_algorithm(self) -> None:
        """``_apply_mock_config_overrides`` must not change the algorithm or seed."""
        from traigent.core.optimized_function import OptimizedFunction

        # Build a minimal stand-in with the attribute the method reads.
        instance = OptimizedFunction.__new__(OptimizedFunction)
        instance.mock_mode_config = {  # type: ignore[attr-defined]
            "optimizer": "grid_search",  # would have rerouted in round 1
            "random_seed": 1234,
        }

        kwargs: dict[str, object] = {}
        algo = OptimizedFunction._apply_mock_config_overrides(
            instance, "bayesian", kwargs
        )

        # Algorithm unchanged, no seed injected.
        assert algo == "bayesian"
        assert "random_seed" not in kwargs
