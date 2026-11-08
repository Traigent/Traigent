"""Regression for #3263: ``traigent export`` writes a timezone-aware UTC exported_at."""

from __future__ import annotations

import json
import time
from datetime import datetime as real_datetime
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import traigent.cli.main as cli_main


def _fake_result() -> SimpleNamespace:
    trial = SimpleNamespace(
        trial_id="t1",
        config={"model": "m"},
        metrics={"accuracy": 0.9},
        status="completed",
        duration=1.0,
        timestamp=real_datetime(2026, 1, 1),
        error_message=None,
        metadata={},
    )
    return SimpleNamespace(
        best_config={"model": "m"},
        best_metrics={"accuracy": 0.9},
        function_name="fn",
        trials=[trial],
        stop_reason="max_trials",
    )


@pytest.mark.parametrize("fmt", ["slim", "full"])
def test_exported_at_is_aware_utc_under_non_utc_tz(
    fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "Pacific/Auckland")
    time.tzset()
    try:
        monkeypatch.setattr(
            cli_main, "_load_cli_result", lambda *a, **k: _fake_result()
        )
        out = tmp_path / "out.json"
        res = CliRunner().invoke(
            cli_main.cli, ["export", "run", "-o", str(out), "--format", fmt]
        )
        assert res.exit_code == 0, res.output
        stamp = real_datetime.fromisoformat(json.loads(out.read_text())["exported_at"])
        assert stamp.utcoffset() == timedelta(0)
    finally:
        monkeypatch.undo()
        time.tzset()
