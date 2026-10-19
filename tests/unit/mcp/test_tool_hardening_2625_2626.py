"""MCP tool hardening: exception text (#2625) and manifest symlinks (#2626)."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.mcp import tools

_CANARY = "SYNTHETIC_CANARY sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _run_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> dict:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")

    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError(_CANARY)

    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: False)
    monkeypatch.setattr(tools, "_run_loaded_optimizations", boom)
    return tools.run_optimization_tool(
        script_path="agent.py", mode=mode, confirm=True, cost_limit=5.0
    )


def test_real_run_failure_returns_class_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_failing(tmp_path, monkeypatch, "real")
    assert result["code"] == "optimization_failed"
    assert "SYNTHETIC_CANARY" not in result["message"]
    assert "sk-proj-AAAA" not in result["message"]
    assert "ValueError" in result["message"]


def test_mock_run_failure_is_redacted_but_names_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_failing(tmp_path, monkeypatch, "mock")
    assert result["code"] == "optimization_failed"
    assert "sk-proj-AAAAAAAAAAAAAAAA" not in result["message"]
    assert result["message"].startswith("ValueError:")
    assert len(result["message"]) <= tools._MAX_ERROR_MESSAGE_CHARS


def test_scaffold_eval_rejects_dangling_manifest_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    monkeypatch.chdir(ws)
    target = outside / "manifest.json"
    link = ws / ("draft_eval.jsonl" + tools._EVAL_MANIFEST_SUFFIX)
    link.symlink_to(target)

    result = tools.scaffold_eval_tool(output_path="draft_eval.jsonl")

    assert result["ok"] is False
    assert result["code"] == "path_rejected"
    assert not target.exists()


def test_scaffold_eval_overwrite_does_not_clobber_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    monkeypatch.chdir(ws)
    target = outside / "victim.txt"
    target.write_text("keep", encoding="utf-8")
    (ws / ("draft_eval.jsonl" + tools._EVAL_MANIFEST_SUFFIX)).symlink_to(target)

    result = tools.scaffold_eval_tool(output_path="draft_eval.jsonl", overwrite=True)

    assert result["ok"] is False
    assert target.read_text(encoding="utf-8") == "keep"
