"""Checkpoints are written on the sequential path and never skipped (#2957)."""

from __future__ import annotations

from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"a"}\n{"input":{"q":"2"},"output":"a"}\n'
    )

    @traigent.optimize(
        configuration_space={"v": [str(i) for i in range(12)]},
        objectives=["accuracy"],
        offline=True,
        max_trials=12,
        algorithm="grid",
        evaluation=EvaluationOptions(eval_dataset="d.jsonl"),
        **kwargs,
    )
    def agent(q: str) -> str:
        return str(traigent.get_config()["v"])

    agent.optimize_sync()
    return Path(tmp_path / "res")


def _checkpoints(root: Path) -> list[Path]:
    return sorted(root.rglob("checkpoint_*.json"))


def test_sequential_run_writes_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _checkpoints(_run(tmp_path, monkeypatch))


def test_parallel_run_with_non_dividing_concurrency_writes_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _run(tmp_path, monkeypatch, parallel_config={"trial_concurrency": 3})
    assert _checkpoints(root)


def test_checkpoint_written_when_crossing_batch_then_breaks() -> None:
    import asyncio
    from types import SimpleNamespace

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = object.__new__(OptimizationOrchestrator)
    orch.parallel_trials = 3
    saved: list[int] = []
    orch._log_checkpoint = lambda **kw: saved.append(kw["trial_count"])  # type: ignore[method-assign]
    orch._trials = []
    orch.optimizer = SimpleNamespace()

    async def fake_batch(**kwargs):
        return 12, "break"

    orch._run_parallel_batch = fake_batch  # type: ignore[method-assign,assignment]
    result = asyncio.run(
        orch._dispatch_trial(None, None, None, None, 9, 5.0, None)  # type: ignore[arg-type]
    )
    assert result == (12, "break")
    assert saved == [12]
