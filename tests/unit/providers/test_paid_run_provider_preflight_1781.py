"""Issue #1781: provider key check runs automatically for paid runs.

All HTTP and SDK calls are mocked; no real provider is contacted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.providers.validation import (
    ProviderStatus,
    ProviderValidator,
    get_failed_providers,
    get_run_provider_for_model,
    preflight_run_providers,
    validate_providers,
)
from traigent.utils.exceptions import ProviderValidationError

_VALIDATION = "traigent.providers.validation"
_SECRET = "sk-or-SECRETVALUE123"  # pragma: allowlist secret


def _orchestrator(config: dict, models: list[str]) -> OptimizationOrchestrator:
    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch.config = config
    orch.optimizer = SimpleNamespace(config_space={"model": models})
    orch.traigent_config = SimpleNamespace(model=None)
    orch._cost_estimator = MagicMock()
    return orch


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "TRAIGENT_MOCK_LLM",
        "TRAIGENT_SKIP_PROVIDER_VALIDATION",
        "TRAIGENT_RUN_COST_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")


class TestOrchestratorPreflight:
    def test_paid_run_with_bad_key_fails_before_cost_approval(self):
        orch = _orchestrator({"cost_limit": 5.0}, ["gpt-4o-mini"])
        bad = {
            "openai": ProviderStatus(
                "openai",
                False,
                "Invalid key (AuthenticationError)",
                None,
                None,
                "AuthenticationError",
            )
        }
        with patch(f"{_VALIDATION}.validate_providers", return_value=bad):
            with pytest.raises(ProviderValidationError) as exc:
                orch._check_cost_approval(MagicMock())
        orch._cost_estimator.check_cost_approval.assert_not_called()
        assert "openai" in str(exc.value)
        assert exc.value.failed_providers == [("openai", "AuthenticationError")]

    def test_paid_run_with_good_key_reaches_cost_approval_once(self):
        orch = _orchestrator({"cost_approved": True}, ["gpt-4o-mini", "gpt-4o"])
        ok = {"openai": ProviderStatus("openai", True, "Available")}
        with patch(f"{_VALIDATION}.validate_providers", return_value=ok) as vp:
            orch._check_cost_approval(MagicMock())
        vp.assert_called_once()
        assert vp.call_args.args[0] == ["gpt-4o-mini", "gpt-4o"]
        orch._cost_estimator.check_cost_approval.assert_called_once()

    def test_only_run_models_are_validated(self):
        orch = _orchestrator({"cost_limit": 1.0}, ["claude-haiku-4-5-20251001"])
        ok = {"anthropic": ProviderStatus("anthropic", True, "Available")}
        with patch(f"{_VALIDATION}.validate_providers", return_value=ok) as vp:
            orch._check_cost_approval(MagicMock())
        assert vp.call_args.args[0] == ["claude-haiku-4-5-20251001"]

    def test_mock_run_never_calls_validation(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
        orch = _orchestrator({"cost_limit": 5.0}, ["gpt-4o-mini"])
        with patch(f"{_VALIDATION}.validate_providers") as vp:
            orch._check_cost_approval(MagicMock())
        vp.assert_not_called()

    def test_skip_setting_disables(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_SKIP_PROVIDER_VALIDATION", "true")
        orch = _orchestrator({"cost_limit": 5.0}, ["gpt-4o-mini"])
        with patch(f"{_VALIDATION}.validate_providers") as vp:
            orch._check_cost_approval(MagicMock())
        vp.assert_not_called()

    def test_run_without_cost_limit_or_approval_is_not_paid(self):
        orch = _orchestrator({}, ["gpt-4o-mini"])
        with patch(f"{_VALIDATION}.validate_providers") as vp:
            orch._check_cost_approval(MagicMock())
        vp.assert_not_called()

    def test_no_models_means_no_validation(self):
        orch = _orchestrator({"cost_limit": 5.0}, [])
        with patch(f"{_VALIDATION}.validate_providers") as vp:
            orch._check_cost_approval(MagicMock())
        vp.assert_not_called()


class TestBedrockSkip:
    def test_bedrock_reported_skipped_not_passed(self):
        results = validate_providers(["bedrock/anthropic.claude-haiku-4-5"])
        status = results["bedrock"]
        assert status.skipped is True
        assert status.valid is False
        assert "no key or balance check" in status.message
        assert get_failed_providers(results) == []

    def test_bedrock_does_not_block_a_run(self):
        results = preflight_run_providers(["bedrock/anthropic.claude-haiku-4-5"])
        assert results["bedrock"].skipped is True


class TestOpenRouter:
    def test_resolver_names_openrouter_and_bedrock(self):
        assert get_run_provider_for_model("openrouter/openai/gpt-4o") == "openrouter"
        assert get_run_provider_for_model("bedrock/x") == "bedrock"
        assert get_run_provider_for_model("gpt-4o-mini") == "openai"

    def _check(self, monkeypatch, response=None, exc=None):
        monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)
        with patch("requests.get") as get:
            if exc is not None:
                get.side_effect = exc
            else:
                get.return_value = SimpleNamespace(status_code=response)
            status = ProviderValidator()._check_openrouter()
        return status, get

    def test_ok(self, monkeypatch):
        status, get = self._check(monkeypatch, 200)
        assert status.valid is True
        assert get.call_args.args[0] == "https://openrouter.ai/api/v1/key"

    def test_invalid_key_fails_without_key_material(self, monkeypatch):
        status, _ = self._check(monkeypatch, 401)
        assert status.valid is False
        assert status.error_type == "AuthenticationError"
        assert _SECRET not in status.message

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        status = ProviderValidator()._check_openrouter()
        assert status.valid is False and status.error_type == "MissingKey"

    def test_network_error_is_transient_and_value_free(self, monkeypatch):
        status, _ = self._check(monkeypatch, exc=ConnectionError(f"boom {_SECRET}"))
        assert status.valid is True
        assert _SECRET not in status.message

    def test_preflight_raises_value_free_error_for_bad_openrouter_key(
        self, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)
        with patch("requests.get", return_value=SimpleNamespace(status_code=401)):
            with pytest.raises(ProviderValidationError) as exc:
                preflight_run_providers(["openrouter/openai/gpt-4o"])
        assert "openrouter" in str(exc.value)
        assert _SECRET not in str(exc.value)


class TestVertexAndFunds:
    def test_vertex_is_skipped_not_failed_even_without_google_key(self, monkeypatch):
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        results = validate_providers(["vertex_ai/gemini-2.5-flash"])
        status = results["vertex_ai"]
        assert status.skipped is True and status.valid is False
        assert "credentials not checked" in status.message
        assert get_failed_providers(results) == []
        assert "google" not in results
        # The paid-run gate must not block on it.
        preflight_run_providers(["vertex_ai/gemini-2.5-flash"])

    def _openrouter(self, monkeypatch, body):
        monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)
        response = SimpleNamespace(status_code=200, json=lambda: body)
        with patch("requests.get", return_value=response):
            return ProviderValidator()._check_openrouter()

    def test_openrouter_depleted_limit_fails_closed(self, monkeypatch):
        status = self._openrouter(
            monkeypatch,
            {"data": {"limit": 10.0, "limit_remaining": 0, "usage": 10.0}},
        )
        assert status.valid is False
        assert status.error_type == "InsufficientFunds"
        assert status.funds == "depleted"
        assert _SECRET not in status.message

    def test_openrouter_depleted_via_usage_when_remaining_absent(self, monkeypatch):
        status = self._openrouter(monkeypatch, {"data": {"limit": 5, "usage": 6}})
        assert status.valid is False and status.funds == "depleted"

    def test_openrouter_depleted_run_blocks_preflight(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)
        body = {"data": {"limit": 1, "limit_remaining": -0.5}}
        response = SimpleNamespace(status_code=200, json=lambda: body)
        with patch("requests.get", return_value=response):
            with pytest.raises(ProviderValidationError) as exc:
                preflight_run_providers(["openrouter/openai/gpt-4o"])
        assert "InsufficientFunds" in str(exc.value)
        assert _SECRET not in str(exc.value)

    def test_openrouter_with_headroom_is_available(self, monkeypatch):
        status = self._openrouter(
            monkeypatch, {"data": {"limit": 10, "limit_remaining": 4.2}}
        )
        assert status.valid is True and status.funds == "available"

    @pytest.mark.parametrize(
        "body",
        [
            {"data": {"limit": None, "limit_remaining": None, "usage": 3}},
            {"data": {}},
            {"unexpected": True},
            None,
        ],
    )
    def test_openrouter_without_limit_passes_with_funds_unknown(
        self, monkeypatch, body
    ):
        status = self._openrouter(monkeypatch, body)
        assert status.valid is True
        assert status.funds == "unknown"

    def test_openrouter_unreadable_body_is_unknown_not_a_failure(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)

        def boom():
            raise ValueError("bad json")

        response = SimpleNamespace(status_code=200, json=boom)
        with patch("requests.get", return_value=response):
            status = ProviderValidator()._check_openrouter()
        assert status.valid is True and status.funds == "unknown"

    def test_openai_status_says_funds_not_checked(self, monkeypatch, capsys):
        from traigent.providers.validation import print_provider_status

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret
        fake_client = MagicMock()
        with patch("openai.OpenAI", return_value=fake_client):
            status = ProviderValidator()._validate_openai()
        assert status.valid is True
        assert status.funds == "not_checked"
        print_provider_status({"openai": status})
        assert "funds not checked" in capsys.readouterr().out
