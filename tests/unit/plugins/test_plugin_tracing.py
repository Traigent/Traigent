"""Privacy regression tests for the traigent-tracing plugin.

The plugin lives in ``plugins/traigent-tracing/traigent_tracing`` and
``traigent.core.tracing`` imports it wholesale when installed. Codex
retry2 review flagged this as a bypass for the SDK-side privacy fix:
fixing only the embedded fallback was not enough because, in any
deployment that has the plugin installed, the plugin path is the one
that ships span attributes to OTLP.

These tests assert the plugin scrubs PII/secrets from span names and
from exported ``example.input`` / ``example.expected_output`` /
``example.actual_output`` attributes — the same surface area covered
by the in-tree fallback tests.

The tests load the plugin module by path so they run regardless of
whether ``pip install traigent-tracing`` has been done. They do not
depend on OpenTelemetry being importable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def plugin_tracing():
    """Import the plugin's tracing module from its on-disk path."""
    repo_root = Path(__file__).resolve().parents[3]
    plugin_path = (
        repo_root / "plugins" / "traigent-tracing" / "traigent_tracing" / "tracing.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_traigent_tracing_under_test", plugin_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_traigent_tracing_under_test"] = module
    spec.loader.exec_module(module)
    return module


class TestPluginPIIHelpers:
    """Direct unit tests for the plugin's redaction helpers."""

    def test_email_is_scrubbed(self, plugin_tracing) -> None:
        scrubbed = plugin_tracing._scrub_pii_text("ping me at john@example.com")
        assert "john@example.com" not in scrubbed
        assert "***EMAIL***" in scrubbed

    def test_ssn_is_scrubbed(self, plugin_tracing) -> None:
        scrubbed = plugin_tracing._scrub_pii_text("SSN: 123-45-6789")
        assert "123-45-6789" not in scrubbed
        assert "***SSN***" in scrubbed

    def test_jwt_is_scrubbed(self, plugin_tracing) -> None:
        token = ".".join(["ey" + "Jheader", "payload", "signature"])
        scrubbed = plugin_tracing._scrub_pii_text(f"bearer {token}")
        assert "eyJ" not in scrubbed
        assert "***JWT***" in scrubbed

    def test_redact_payload_handles_nested_dict(self, plugin_tracing) -> None:
        payload = {
            "user": "leak me at admin@example.org",
            "creds": {"api_key": "AKIA" + ("A" * 16)},  # pragma: allowlist secret
            "items": ["SSN 999-12-3456", {"text": "ok"}],
        }
        result = plugin_tracing._redact_payload(payload)
        assert "admin@example.org" not in json.dumps(result)
        assert result["creds"]["api_key"] == "***REDACTED***"
        assert any(
            "***SSN***" in (v if isinstance(v, str) else json.dumps(v))
            for v in result["items"]
        )

    def test_clean_text_passes_through(self, plugin_tracing) -> None:
        text = "Generated response"
        assert plugin_tracing._scrub_pii_text(text) == text


class TestPluginSpanNameScrubbing:
    """Span names go to OTLP in plaintext; they must be scrubbed."""

    def test_input_preview_scrubs_email(self, plugin_tracing) -> None:
        preview = plugin_tracing._format_input_preview(
            {"prompt": "email me at john@example.com please"}
        )
        assert "john@example.com" not in preview
        assert "***EMAIL***" in preview

    def test_input_preview_scrubs_ssn(self, plugin_tracing) -> None:
        preview = plugin_tracing._format_input_preview(
            {"text": "SSN 555-12-3456 confirmed"}
        )
        assert "555-12-3456" not in preview
        assert "***SSN***" in preview

    def test_config_summary_scrubs_pii(self, plugin_tracing) -> None:
        summary = plugin_tracing._format_config_summary(
            {"model": "gpt-4", "user": "a@b.co"}
        )
        assert "a@b.co" not in summary
        assert "***EMAIL***" in summary


class TestPluginExportedAttributeScrubbing:
    """Exported example.* attributes must not leak raw PII to OTLP."""

    @staticmethod
    def _fake_tracer_with_span(span: MagicMock) -> MagicMock:
        fake = MagicMock()
        fake.start_as_current_span.return_value.__enter__.return_value = span
        fake.start_as_current_span.return_value.__exit__.return_value = False
        return fake

    def test_example_input_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        tracer = self._fake_tracer_with_span(span)

        with patch.object(plugin_tracing, "get_tracer", return_value=tracer):
            with plugin_tracing.example_evaluation_span(
                "ex-1", 0, input_data={"prompt": "ping janet@example.org"}
            ):
                pass

        input_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "example.input"
        )
        exported = input_call.args[1]
        assert "janet@example.org" not in exported
        assert "***EMAIL***" in exported

    def test_example_expected_output_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        tracer = self._fake_tracer_with_span(span)

        with patch.object(plugin_tracing, "get_tracer", return_value=tracer):
            with plugin_tracing.example_evaluation_span(
                "ex-2", 0, expected_output="Customer SSN: 111-22-3333"
            ):
                pass

        expected_call = next(
            c
            for c in span.set_attribute.call_args_list
            if c.args[0] == "example.expected_output"
        )
        exported = expected_call.args[1]
        assert "111-22-3333" not in exported
        assert "***SSN***" in exported

    def test_example_actual_output_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        fake_access_key = "AKIA" + ("A" * 16)
        plugin_tracing.record_example_result(
            span,
            success=True,
            actual_output=(
                "Contact ops@example.io with reference 444-55-6666 "
                f"or {fake_access_key} for assistance."
            ),
        )

        actual_call = next(
            c
            for c in span.set_attribute.call_args_list
            if c.args[0] == "example.actual_output"
        )
        exported = actual_call.args[1]
        assert "ops@example.io" not in exported
        assert "444-55-6666" not in exported
        assert fake_access_key not in exported
        assert "***EMAIL***" in exported
        assert "***SSN***" in exported
        assert "***AWS_ACCESS_KEY***" in exported

    def test_trial_config_scrubs_secret_keys(self, plugin_tracing) -> None:
        span = MagicMock()
        tracer = self._fake_tracer_with_span(span)
        fake_access_key = "AKIA" + ("X" * 16)

        with patch.object(plugin_tracing, "get_tracer", return_value=tracer):
            with plugin_tracing.trial_span(
                "t-1",
                0,
                config={
                    "api_key": fake_access_key,  # pragma: allowlist secret
                    "model": "gpt-4",
                },
            ):
                pass

        config_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "trial.config"
        )
        assert fake_access_key not in config_call.args[1]
        assert "***REDACTED***" in config_call.args[1]

    def test_optimization_best_config_scrubs_secret_keys(self, plugin_tracing) -> None:
        span = MagicMock()
        fake_access_key = "AKIA" + ("X" * 16)
        plugin_tracing.record_optimization_complete(
            span,
            trial_count=5,
            best_score=0.9,
            best_config={
                "api_key": fake_access_key,  # pragma: allowlist secret
                "model": "gpt-4",
            },
        )

        best_config_call = next(
            c
            for c in span.set_attribute.call_args_list
            if c.args[0] == "optimization.best_config"
        )
        assert fake_access_key not in best_config_call.args[1]
        assert "***REDACTED***" in best_config_call.args[1]


def _status_description(call) -> str:
    """Extract the status description from a ``span.set_status`` call.

    ``_set_error_status`` (both core and plugin) has two branches: a
    real OTel ``Status`` object passed positionally when
    ``opentelemetry`` resolved a working ``trace`` module, or a
    ``status=``/``description=`` keyword fallback when it didn't. This
    helper reads the description from whichever branch actually ran, so
    the test doesn't depend on which OTel packages happen to be
    installed in the environment it runs in.
    """
    if call.args:
        return str(call.args[0].description)
    return str(call.kwargs["description"])


class TestPluginErrorScrubbing:
    """Regression tests for #1885: trial.error/example.error and the
    exported span status description must not leak raw PII to OTLP.

    Mirrors the equivalent tests in ``tests/unit/core/test_tracing.py``
    for the in-tree fallback — the plugin is a separate implementation
    (``traigent.core.tracing`` imports it wholesale when installed), so
    the scrub has to be verified here independently.
    """

    def test_trial_error_attribute_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        plugin_tracing.record_trial_result(
            span,
            "failed",
            error="AssertionError: expected SSN 123-45-6789 got ops@example.io",
        )

        error_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "trial.error"
        )
        exported = error_call.args[1]
        assert "123-45-6789" not in exported
        assert "ops@example.io" not in exported
        assert "***SSN***" in exported
        assert "***EMAIL***" in exported

    def test_trial_error_status_description_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        plugin_tracing.record_trial_result(
            span, "failed", error="Customer SSN: 123-45-6789 rejected"
        )

        description = _status_description(span.set_status.call_args)
        assert "123-45-6789" not in description
        assert "***SSN***" in description

    def test_example_error_attribute_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        jwt_like = ".".join(["ey" + "Jheader", "payload", "signature"])
        plugin_tracing.record_example_result(
            span,
            success=False,
            error=f"Provider 400: request echoed token {jwt_like}",
        )

        error_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "example.error"
        )
        exported = error_call.args[1]
        assert jwt_like not in exported
        assert "***JWT***" in exported

    def test_example_error_status_description_is_scrubbed(self, plugin_tracing) -> None:
        span = MagicMock()
        plugin_tracing.record_example_result(
            span, success=False, error="Contact ops@example.io about this failure"
        )

        description = _status_description(span.set_status.call_args)
        assert "ops@example.io" not in description
        assert "***EMAIL***" in description

    def test_exception_error_is_coerced_to_scrubbed_string(
        self, plugin_tracing
    ) -> None:
        """An Exception instance (real caller shape) must not crash or leak."""
        span = MagicMock()
        exc = ValueError("auth failed for jane@example.org with key AKIA" + "A" * 16)
        plugin_tracing.record_trial_result(span, "failed", error=exc)

        error_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "trial.error"
        )
        exported = error_call.args[1]
        assert isinstance(exported, str)
        assert "jane@example.org" not in exported
        assert "AKIA" + "A" * 16 not in exported
        assert "***EMAIL***" in exported
        assert "***AWS_ACCESS_KEY***" in exported

        description = _status_description(span.set_status.call_args)
        assert isinstance(description, str)
        assert "jane@example.org" not in description
        assert "***EMAIL***" in description

    def test_dict_error_is_coerced_to_scrubbed_string(self, plugin_tracing) -> None:
        """A dict payload gets key-aware secret redaction plus PII scrub."""
        span = MagicMock()
        plugin_tracing.record_example_result(
            span,
            success=False,
            error={
                "api_key": "sk-super-secret-value",  # pragma: allowlist secret
                "detail": "user ops@example.io rejected",
            },
        )

        error_call = next(
            c for c in span.set_attribute.call_args_list if c.args[0] == "example.error"
        )
        exported = error_call.args[1]
        assert isinstance(exported, str)
        assert "sk-super-secret-value" not in exported
        assert "***REDACTED***" in exported
        assert "ops@example.io" not in exported
        assert "***EMAIL***" in exported

        description = _status_description(span.set_status.call_args)
        assert isinstance(description, str)
        assert "sk-super-secret-value" not in description

    def test_none_error_keeps_no_attribute_no_status_path(self, plugin_tracing) -> None:
        """error=None must keep today's behavior: no error attr, no status."""
        span = MagicMock()
        plugin_tracing.record_trial_result(span, "completed", error=None)
        plugin_tracing.record_example_result(span, success=True, error=None)

        attribute_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        assert "trial.error" not in attribute_keys
        assert "example.error" not in attribute_keys
        span.set_status.assert_not_called()
