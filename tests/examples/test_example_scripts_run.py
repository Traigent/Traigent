"""Shipped example scripts must run in mock mode on the dependency set the SDK installs."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _run(script: Path, tmp_path: Path, *, drop: tuple[str, ...] = ()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(
        TRAIGENT_MOCK_LLM="true",
        TRAIGENT_OFFLINE_MODE="true",
        PYTHONPATH=str(ROOT),
    )
    # Constructing the LangChain client needs a key even though mock mode
    # intercepts every call; this value is never sent anywhere.
    env.setdefault("OPENAI_API_KEY", "sk-mock-not-a-real-key")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def test_examples_do_not_import_removed_langchain_schema() -> None:
    offenders = []
    for path in EXAMPLES.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "langchain.schema":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


LANGCHAIN_MESSAGE_EXAMPLES = (
    "advanced/optimization-routing/ex01-local-search/local_search.py",
    "advanced/optimization-routing/ex02-offline-no-egress/offline_no_egress.py",
    "advanced/results-analysis/ex01-viewing-results/viewing_results.py",
    "advanced/results-analysis/ex02-performance-metrics/performance_metrics.py",
    "gallery/page-inline/best-practices/ex01-small-space/small_space.py",
    "gallery/page-inline/best-practices/ex02-weighted-objectives/weighted_objectives.py",
    "gallery/page-inline/common-tasks/p0-context-engineering/context_engineering.py",
    "gallery/page-inline/common-tasks/p0-few-shot-selection/few_shot_selection.py",
    "gallery/page-inline/common-tasks/p0-structured-output/structured_output.py",
)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("relative", LANGCHAIN_MESSAGE_EXAMPLES)
def test_langchain_message_examples_run_in_mock_mode(
    relative: str, tmp_path: Path
) -> None:
    """Each script builds a message LangChain accepts and finishes in mock mode (#2934).

    The scripts call their LLM function directly, outside an optimization run,
    so they switch on the in-code mock interception themselves when the
    documented ``TRAIGENT_MOCK_LLM`` switch is set; otherwise the provider-host
    guard (correctly) blocks the real call.
    """
    result = _run(EXAMPLES / relative, tmp_path)
    assert "Unsupported message type" not in result.stderr, result.stderr[-2000:]
    assert "MESSAGE_COERCION_FAILURE" not in result.stderr
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.timeout(180)
def test_text_to_sql_mock_quick_start_runs(tmp_path: Path) -> None:
    result = _run(EXAMPLES / "core/text-to-sql/run.py", tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.timeout(300)
def test_surrogate_coevolution_demo_wire_carry_proof_runs(tmp_path: Path) -> None:
    result = _run(
        EXAMPLES / "advanced/surrogate_evaluator_coevolution_demo.py", tmp_path
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.timeout(180)
def test_mock_prompt_optimization_walkthrough_needs_no_cost_approval(
    tmp_path: Path,
) -> None:
    script = ROOT / "walkthrough/mock/advanced/02_prompt_optimization.py"
    result = _run(script, tmp_path, drop=("TRAIGENT_COST_APPROVED",))
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("runner", ["basics", "column_map"])
def test_ragas_runner_exits_nonzero_when_every_trial_fails(
    runner: str, tmp_path: Path
) -> None:
    pytest.importorskip("traigent")
    try:
        import ragas  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("ragas installed; all-trials-failed path is not reachable")
    result = _run(EXAMPLES / f"advanced/ragas/{runner}/run.py", tmp_path)
    assert result.returncode != 0, result.stdout[-2000:]


def test_example_symlinks_are_not_dangling() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-s", "examples"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    dangling = []
    for line in tracked:
        meta, _, name = line.partition("\t")
        if meta.startswith("120000"):
            link = ROOT / name
            if not link.exists():
                dangling.append(name)
    assert not dangling, dangling


@pytest.mark.timeout(180)
def test_ragas_with_llm_mock_fallback_metrics_complete(tmp_path: Path) -> None:
    result = _run(EXAMPLES / "advanced/ragas/with_llm/run.py", tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "best_config: None" not in result.stdout
