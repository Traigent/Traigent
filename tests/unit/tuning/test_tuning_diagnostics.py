"""An optimizer failure is a closed, diagnosable reason; the exception text stays local."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.core.cost_enforcement import CostTrackingRequiredError
from traigent.tuning.controller import (
    E_CONNECTED_TUNING_REFUSED,
    ERROR_REASONS,
    _error_reason,
    _read_lines,
)
from traigent.tuning.errors import TuningError
from traigent.utils.cost_calculator import UnknownModelError
from traigent.utils.exceptions import CostLimitExceeded

from .test_cli_end_to_end import WITNESS, _authority, _holdout_text, _json
from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    _holdout,
    at,
    controller,
    e_ref,
)

SECRET = "PATIENT-ALICE-HAS-DIABETES"  # pragma: allowlist secret


class Raising:
    """An optimizer that raises ``exc`` (its message carries customer content)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __call__(self, space: dict[str, Any]) -> Any:
        raise self.exc


def _all_stored_text(store: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(store.rglob("*"))
        if p.is_file()
    )


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (
            CostLimitExceeded(accumulated=0.0, limit=2.0, estimated=3.57),
            "cost_preflight_refused",
        ),
        (CostTrackingRequiredError(SECRET), "cost_preflight_refused"),
        (UnknownModelError(SECRET), "unpriced_model"),
        (TuningError(E_CONNECTED_TUNING_REFUSED, SECRET), "connected_tuning_refused"),
        (RuntimeError(SECRET), "optimizer_raised"),
    ],
)
def test_an_optimizer_exception_is_recorded_as_a_closed_reason_without_its_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
    reason: str,
) -> None:
    assert reason in ERROR_REASONS and _error_reason(exc) == reason
    c = controller(tmp_path, Raising(exc))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    with caplog.at_level(logging.ERROR, logger="traigent.tuning.controller"):
        report = c.step(at(10))
    assert report.finished and len(report.errors) == 1
    assert report.errors[0].endswith(f":{reason}")
    facts = c.controller_facts()
    failed = [f for f in facts if f.get("fact") == "optimize_failed"]
    assert [f["reason"] for f in failed] == [reason]
    # the full exception (message and traceback) is in the local log only
    assert any(r.exc_info for r in caplog.records)
    # no stored record carries the exception text
    assert SECRET not in _all_stored_text(tmp_path / "store")


def test_a_clean_run_reports_no_errors(tmp_path: Path) -> None:
    from .test_controller import FakeOptimizer

    c = controller(tmp_path, FakeOptimizer())
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    assert c.step(at(10)).errors == []


def test_cli_step_exits_non_zero_when_an_approved_run_ended_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    midnight = datetime.now(UTC).strftime("%Y-%m-%dT00:00:00Z")
    source = (
        WITNESS.replace("HOLDOUT_ARG", '"holdout.jsonl"')
        .replace("TRIGGER", f'Every(30, "{midnight}")')
        .replace("anthropic/claude-haiku-5-5", "mock/unpriced-model")
    )
    (tmp_path / "agent_diag.py").write_text(source, encoding="utf-8")
    (tmp_path / "holdout.jsonl").write_text(_holdout_text(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("agent_diag", None)
    ctl = [
        "--target", "agent_diag.py:classify", "--store", "./store",
        "--agent", "ticket-router", *_authority(tmp_path, monkeypatch),
    ]  # fmt: skip
    try:
        run = CliRunner()
        out = run.invoke(cli, ["tune", "step", *ctl]).output
        occ = _json(out)["awaiting_approval"][0]
        assert (
            run.invoke(
                cli, ["tune", "approve", occ, "--approver", "alice", *ctl]
            ).exit_code
            == 0
        )
        monkeypatch.setenv(
            "TRAIGENT_STRICT_COST_ACCOUNTING", "true"
        )  # unpriced refuses
        result = run.invoke(cli, ["tune", "step", *ctl])
        assert result.exit_code != 0, result.output
        assert "optimization_failed" in result.output
        assert any(
            f"{r}" in result.output
            for r in ("unpriced_model", "cost_preflight_refused")
        )
        facts = _read_lines(tmp_path / "store", tmp_path / "store" / "controller.jsonl")
        assert [f["reason"] for f in facts if f.get("fact") == "optimize_failed"]
    finally:
        sys.modules.pop("agent_diag", None)
