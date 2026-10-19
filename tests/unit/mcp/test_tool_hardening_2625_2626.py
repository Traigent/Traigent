"""MCP tool hardening: exception text (#2625) and manifest symlinks (#2626)."""

from __future__ import annotations

from pathlib import Path

import logging

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


_CUSTOMER = "Alice_has_diabetes and password=hunter2 secret customer text"


@pytest.mark.parametrize("mode", ["real", "mock"])
def test_failure_is_content_free_in_response_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")

    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError(f"{_CUSTOMER} {_CANARY}")

    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: False)
    monkeypatch.setattr(tools, "_run_loaded_optimizations", boom)
    with caplog.at_level(logging.DEBUG):
        result = tools.run_optimization_tool(
            script_path="agent.py", mode=mode, confirm=True, cost_limit=5.0
        )

    assert result["code"] == "optimization_failed"
    assert "ValueError" in result["message"]
    blob = result["message"] + "\n" + caplog.text
    for needle in ("Alice_has_diabetes", "hunter2", "SYNTHETIC_CANARY", "sk-proj-"):
        assert needle not in blob
    assert "ValueError" in caplog.text


def test_scaffold_write_refuses_symlink_at_open(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        tools._write_text_nofollow(link, "x", overwrite=True)
    assert not target.exists()


def test_scaffold_write_refuses_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linkdir").symlink_to(real)
    with pytest.raises(OSError):
        tools._write_text_nofollow(tmp_path / "linkdir" / "f.json", "x", overwrite=True)
    assert not (real / "f.json").exists()


def test_scaffold_eval_race_symlink_inserted_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    monkeypatch.chdir(ws)
    target = outside / "victim.txt"
    link = ws / ("eval.draft.jsonl" + tools._EVAL_MANIFEST_SUFFIX)
    real_write = tools._write_text_nofollow

    def racing(path: Path, text: str, *, overwrite: bool) -> None:
        if str(path).endswith(tools._EVAL_MANIFEST_SUFFIX):
            link.symlink_to(target)  # inserted after the pre-check
        real_write(path, text, overwrite=overwrite)

    monkeypatch.setattr(tools, "_write_text_nofollow", racing)
    result = tools.scaffold_eval_tool()
    assert result["ok"] is False
    assert result["code"] == "path_rejected"
    assert not target.exists()


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
