"""TRAIGENT_CONFIG_BASE_DIR is read per call in apply.py, like the CLI (Traigent#3161)."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.config_generator import apply as apply_mod


def test_env_set_after_import_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    other = tmp_path / "elsewhere"
    base.mkdir()
    other.mkdir()
    source = base / "agent.py"
    source.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(apply_mod, "_SAFE_BASE_DIR", None)
    monkeypatch.setenv("TRAIGENT_CONFIG_BASE_DIR", str(base))
    monkeypatch.chdir(other)

    assert apply_mod._sanitize_source_path(source) == source.resolve()
