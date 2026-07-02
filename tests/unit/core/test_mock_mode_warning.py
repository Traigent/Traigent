"""Tests for #1419 mock-mode output-score warning and #1447 docstring model-ID fix.

#1447 — public @optimize / Choices docstrings must not cite decommissioned model IDs.
#1419 — create_effective_evaluator must emit a one-time WARNING when mock mode is
        active and an output-based scorer (custom_evaluator / metric_functions /
        scoring_function) is supplied, because mock LLM returns a canned constant
        string so every trial scores identically.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.optimization_pipeline import create_effective_evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_EVALUATOR = MagicMock(return_value=(MagicMock(), None))
_DUMMY_LOCAL = MagicMock(return_value=(MagicMock(), None))
_DUMMY_CUSTOM = MagicMock()  # callable used as custom_evaluator


def _call_create_evaluator(**overrides: Any) -> Any:
    """Call create_effective_evaluator with safe defaults, merging overrides."""
    defaults: dict[str, Any] = {
        "timeout": None,
        "custom_evaluator": None,
        "effective_batch_size": None,
        "effective_thread_workers": None,
        "effective_privacy_enabled": False,
        "objectives": ["accuracy"],
        "execution_mode": "local",
        "mock_mode_config": None,
        "metric_functions": None,
        "scoring_function": None,
        "decorator_custom_evaluator": None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# #1419 — warning fires in mock mode + output-based scorer
# ---------------------------------------------------------------------------


class TestMockModeOutputScoreWarning:
    """create_effective_evaluator emits a WARNING when mock mode + output scorer."""

    def _run(self, mock_llm: bool, caplog: pytest.LogCaptureFixture, **kw: Any) -> None:
        """Invoke create_effective_evaluator with both underlying builders patched out."""
        kwargs = _call_create_evaluator(**kw)
        with (
            patch(
                "traigent.core.optimization_pipeline.is_mock_llm",
                return_value=mock_llm,
            ),
            patch(
                "traigent.core.optimization_pipeline._create_wrapped_custom_evaluator",
                return_value=(MagicMock(), None),
            ),
            patch(
                "traigent.core.optimization_pipeline._create_local_evaluator",
                return_value=(MagicMock(), None),
            ),
        ):
            with caplog.at_level(
                logging.WARNING, logger="traigent.core.optimization_pipeline"
            ):
                create_effective_evaluator(**kwargs)

    # --- fires ---

    def test_warns_when_mock_and_custom_evaluator(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning fires: mock mode ON + custom_evaluator supplied."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            custom_evaluator=_DUMMY_CUSTOM,
        )
        assert any("Mock mode active" in r.message for r in caplog.records), (
            "Expected 'Mock mode active' warning in log records"
        )
        assert any(
            "output-based metrics are not meaningful" in r.message
            for r in caplog.records
        )

    def test_warns_when_mock_and_metric_functions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning fires: mock mode ON + metric_functions supplied."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            metric_functions={"accuracy": lambda pred, ref: float(pred == ref)},
        )
        assert any("Mock mode active" in r.message for r in caplog.records)

    def test_warns_when_mock_and_scoring_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning fires: mock mode ON + scoring_function supplied."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            scoring_function=lambda pred, ref: float(pred == ref),
        )
        assert any("Mock mode active" in r.message for r in caplog.records)

    def test_warns_when_mock_and_decorator_custom_evaluator(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning fires: mock mode ON + decorator-level custom_evaluator."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            decorator_custom_evaluator=_DUMMY_CUSTOM,
        )
        assert any("Mock mode active" in r.message for r in caplog.records)

    # --- does NOT fire ---

    def test_no_warn_when_not_mock_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when mock mode is OFF, even with output-based scorer."""
        self._run(
            mock_llm=False,
            caplog=caplog,
            custom_evaluator=_DUMMY_CUSTOM,
        )
        assert not any("Mock mode active" in r.message for r in caplog.records)

    def test_no_warn_when_mock_but_no_output_scorer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when mock mode is ON but no output-based scorer is supplied."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            # No custom_evaluator / metric_functions / scoring_function
        )
        assert not any("Mock mode active" in r.message for r in caplog.records)

    def test_warning_is_at_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mock-score notice must be at WARNING level (not DEBUG/INFO)."""
        self._run(
            mock_llm=True,
            caplog=caplog,
            custom_evaluator=_DUMMY_CUSTOM,
        )
        warning_records = [r for r in caplog.records if "Mock mode active" in r.message]
        assert warning_records, "Expected at least one matching log record"
        assert all(r.levelno == logging.WARNING for r in warning_records)


# ---------------------------------------------------------------------------
# #1447 — decommissioned model IDs removed from public docstrings
# ---------------------------------------------------------------------------


class TestDocstringModelIds:
    """Public-API docstrings must not reference the retired claude-2 model ID."""

    def test_decorators_module_no_claude2(self) -> None:
        """traigent/api/decorators.py docstrings must not cite claude-2."""
        import inspect

        import traigent.api.decorators as mod

        source = inspect.getsource(mod)
        assert "claude-2" not in source, (
            "Found retired model ID 'claude-2' in traigent/api/decorators.py. "
            "Replace with a current model ID (e.g. claude-haiku-4-5-20251001)."
        )

    def test_parameter_ranges_module_no_claude2(self) -> None:
        """traigent/api/parameter_ranges.py Choices docstring must not cite claude-2."""
        import traigent.api.parameter_ranges as mod
        import inspect

        source = inspect.getsource(mod)
        assert "claude-2" not in source, (
            "Found retired model ID 'claude-2' in traigent/api/parameter_ranges.py. "
            "Replace with a current model ID (e.g. claude-haiku-4-5-20251001)."
        )

    def test_choices_docstring_uses_current_model_id(self) -> None:
        """The Choices class docstring example uses a current model ID."""
        from traigent.api.parameter_ranges import Choices

        doc = Choices.__doc__ or ""
        assert "claude-haiku-4-5-20251001" in doc, (
            "Expected Choices docstring to use claude-haiku-4-5-20251001 "
            f"as the example Anthropic model ID. Got docstring:\n{doc}"
        )
