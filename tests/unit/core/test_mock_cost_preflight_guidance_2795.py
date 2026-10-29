"""A refused mock-mode run is told the dry-run route, not to approve spend (#2795).

Mock mode alone does not skip the pre-run cost preflights (#2681), so a simulated
run can still be refused. The refusal must name the supported exemption and its
non-production requirement, and must not lead with the approval variables, which
also cover a later real run. A non-mock refusal keeps its approval guidance.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from traigent.core.cost_enforcement import CostEnforcer
from traigent.core.cost_estimator import CostEstimator
from traigent.utils.env_config import mock_cost_preflight_guidance
from traigent.utils.exceptions import CostLimitExceeded

EXEMPT = "TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT"


class _Dataset:
    def __len__(self) -> int:
        return 46


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (EXEMPT, "TRAIGENT_COST_APPROVED", "TRAIGENT_MOCK_LLM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")


def _declined_estimator() -> CostEstimator:
    enforcer = MagicMock()
    enforcer.check_and_approve.return_value = False
    enforcer.config.limit = 2.0
    return CostEstimator(enforcer, max_trials=4, max_total_examples=None)


def _refusal_message(monkeypatch: pytest.MonkeyPatch, *, mock: bool) -> str:
    if mock:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    with pytest.raises(CostLimitExceeded) as excinfo:
        _declined_estimator().approve_estimated_cost(4.37)
    return str(excinfo.value)


def test_mock_refusal_names_exemption_not_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _refusal_message(monkeypatch, mock=True)
    assert "simulated" in message
    assert EXEMPT in message
    assert "non-production" in message
    assert "To proceed, raise TRAIGENT_RUN_COST_LIMIT" not in message
    # The approval variable may only appear as a warning against its use.
    assert "Do not use TRAIGENT_COST_APPROVED" in message


def test_non_mock_refusal_keeps_approval_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _refusal_message(monkeypatch, mock=False)
    assert "approve after review with TRAIGENT_COST_APPROVED=true" in message
    assert EXEMPT not in message


def test_non_interactive_print_names_exemption_under_mock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    assert CostEnforcer()._request_user_approval(4.37) is False
    err = capsys.readouterr().err
    assert EXEMPT in err
    assert "To proceed, raise TRAIGENT_RUN_COST_LIMIT" not in err


def test_non_interactive_print_keeps_approval_guidance_without_mock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert CostEnforcer()._request_user_approval(4.37) is False
    err = capsys.readouterr().err
    assert "approve after review with TRAIGENT_COST_APPROVED=true" in err
    assert EXEMPT not in err


def test_guidance_is_one_place() -> None:
    text = mock_cost_preflight_guidance()
    assert EXEMPT in text and "ENVIRONMENT=development" in text
