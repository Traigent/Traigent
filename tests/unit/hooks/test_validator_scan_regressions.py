"""Regression tests for hooks validator scan behaviour (#2639)."""

# Traceability: CONC-Quality-Reliability REQ-ORCH-003

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.hooks.config import HooksConfig
from traigent.hooks.validator import AgentValidator


def test_module_calling_sys_exit_is_reported_not_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # discovery only loads modules inside the cwd
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nsys.exit(0)\n")
    results = AgentValidator(HooksConfig()).validate_file(bad)
    assert len(results) == 1
    assert results[0].is_valid is False
    assert results[0].issues[0].code == "DISCOVERY_ERROR"


def test_directory_scan_does_not_skip_explicit_target_with_underscore_part(
    tmp_path: Path,
) -> None:
    target = tmp_path / "_agents"
    target.mkdir()
    (target / "a.py").write_text("x = 1\n")
    seen: list[str] = []
    validator = AgentValidator(HooksConfig())
    validator.validate_file = lambda p: seen.append(Path(p).name) or []  # type: ignore[method-assign]
    validator.validate_directory(target)
    assert seen == ["a.py"]


def test_directory_scan_still_skips_hidden_subdirs_under_target(
    tmp_path: Path,
) -> None:
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "h.py").write_text("x = 1\n")
    (tmp_path / "ok.py").write_text("x = 1\n")
    seen: list[str] = []
    validator = AgentValidator(HooksConfig())
    validator.validate_file = lambda p: seen.append(Path(p).name) or []  # type: ignore[method-assign]
    validator.validate_directory(tmp_path)
    assert seen == ["ok.py"]


def test_system_exit_message_is_not_logged_or_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mod = tmp_path / "m.py"
    mod.write_text('import sys\nsys.exit("SYNTHETIC_SECRET_SENTINEL")\n')
    with caplog.at_level("DEBUG"):
        results = AgentValidator(HooksConfig()).validate_file(mod)
    text = " ".join(i.message for r in results for i in r.issues) + caplog.text
    assert "SYNTHETIC_SECRET_SENTINEL" not in text
    assert any(i.code == "DISCOVERY_ERROR" for r in results for i in r.issues)
    assert "SystemExit" in text  # class only, never the message
