"""Mock-mode user-facing text matches how mock mode was enabled (#2796, #2816).

The deprecated ``TRAIGENT_MOCK_LLM`` variable must not be recommended to a user
who enabled mock mode through ``enable_mock_mode_for_quickstart()``, and
``traigent diagnose`` must see the in-code flag.
"""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traigent import testing as traigent_testing
from traigent.core.optimization_pipeline import create_effective_evaluator
from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics
from traigent.utils.error_handler import (
    APIKeyError,
    ErrorHandler,
)

PACKAGE = Path(__file__).resolve().parents[3] / "traigent"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    traigent_testing._reset_for_tests()
    yield
    traigent_testing._reset_for_tests()


def _banner(caplog: pytest.LogCaptureFixture) -> str:
    kwargs = {
        "timeout": None,
        "custom_evaluator": MagicMock(),
        "effective_batch_size": None,
        "effective_thread_workers": None,
        "effective_privacy_enabled": False,
        "objectives": ["accuracy"],
        "execution_mode": "local",
        "metric_functions": None,
        "scoring_function": None,
        "decorator_custom_evaluator": None,
    }
    with (
        patch(
            "traigent.core.optimization_pipeline._create_wrapped_custom_evaluator",
            return_value=(MagicMock(), None),
        ),
        patch(
            "traigent.core.optimization_pipeline._create_local_evaluator",
            return_value=(MagicMock(), None),
        ),
        caplog.at_level(logging.WARNING, logger="traigent.core.optimization_pipeline"),
    ):
        create_effective_evaluator(**kwargs)
    return " ".join(
        r.getMessage() for r in caplog.records if "Mock mode active" in r.getMessage()
    )


def test_banner_for_in_code_mock_names_the_function_not_the_variable_2796(
    caplog: pytest.LogCaptureFixture,
) -> None:
    traigent_testing.enable_mock_mode_for_quickstart()
    banner = _banner(caplog)
    assert "enable_mock_mode_for_quickstart()" in banner
    assert "TRAIGENT_MOCK_LLM" not in banner


def test_banner_for_env_mock_does_not_name_the_deprecated_variable_2796(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    banner = _banner(caplog)
    assert banner
    assert "TRAIGENT_MOCK_LLM" not in banner
    assert "legacy mock environment flag" in banner


def test_user_facing_error_text_does_not_recommend_the_variable_2796() -> None:
    texts = []
    for handler in (
        ErrorHandler.handle_zero_accuracy,
        ErrorHandler.handle_unknown_error,
    ):
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - TraigentError
            handler(RuntimeError("x"))
        texts.append(str(getattr(excinfo.value, "fix", "")))
    for mock_on in (False, True):
        with patch("traigent.utils.env_config.is_mock_llm", return_value=mock_on):
            texts.append(str(APIKeyError.from_provider_error(RuntimeError("401")).fix))
    assert all("TRAIGENT_MOCK_LLM" not in text for text in texts), texts
    assert any("enable_mock_mode_for_quickstart" in text for text in texts)


def test_modules_with_mock_hints_do_not_mention_the_variable_in_strings_2796() -> None:
    """No user-facing string in these modules recommends the deprecated variable."""
    modules = [
        "utils/error_handler.py",
        "core/optimization_pipeline.py",
        "examples/quickstart/publish_and_verify.py",
    ]
    for relative in modules:
        tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Messages only: a bare variable name used as a key is code.
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and " " in node.value
            ):
                assert "TRAIGENT_MOCK_LLM" not in node.value, (relative, node.lineno)


def test_diagnose_sees_in_code_mock_mode_2816() -> None:
    traigent_testing.enable_mock_mode_for_quickstart()
    assert "TRAIGENT_MOCK_LLM" not in os.environ

    report = DiagnosticReport()
    with patch("traigent.initialize"):
        TraigentDiagnostics._check_traigent_config(report)
    TraigentDiagnostics._add_recommendations(report)

    assert any("Mock LLM mode is enabled" in s["message"] for s in report.successes)
    assert not any("Enable mock LLM mode" in r for r in report.recommendations)


def test_diagnose_recommends_mock_mode_when_it_is_off_2816() -> None:
    report = DiagnosticReport()
    TraigentDiagnostics._add_recommendations(report)
    assert any("Enable mock LLM mode" in r for r in report.recommendations)
