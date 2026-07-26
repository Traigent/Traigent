"""Regression coverage for #2007: walkthrough latency summaries must say ``ms``.

The canonical ``latency`` metric is in MILLISECONDS (see
``_format_metric_value`` in :mod:`traigent.utils.results_table` and the #1855
evaluator contract). Five real-mode examples still printed it with
``f"{...:.3f}s"``, so an 850 ms best trial read as ``850.000s`` (~14 minutes).
``walkthrough/demo/optimize_rag.py`` had the mirror-image bug: it fed
SECONDS-valued replay data into the canonical metric slot, so the shared table
renderer showed the 1.025 s winner as ``1ms``.

The summary assertions read the example SOURCES rather than running ``main()``:
the real-mode files make live provider calls, so importing them is not free.
``optimize_rag.py`` is pure replay data, so its ``build_results()`` is loaded and
checked directly. Everything here is offline: zero LLM spend.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from traigent.utils.results_table import _format_metric_value, print_results_table

REPO_ROOT = Path(__file__).resolve().parents[3]
WALKTHROUGH = REPO_ROOT / "walkthrough"
OPTIMIZE_RAG = WALKTHROUGH / "demo" / "optimize_rag.py"

# The canonical millisecond summary shared by every real-mode example.
REAL_SUMMARY = "print(f\"  Latency: {results.best_metrics.get('latency', 0):.0f}ms\")"

REAL_MODE_EXAMPLES = [
    WALKTHROUGH / "real" / "04_multi_objective.py",
    WALKTHROUGH / "real" / "07_multi_provider.py",
    WALKTHROUGH / "real" / "advanced" / "01_tuned_variables.py",
    WALKTHROUGH / "real" / "advanced" / "02_prompt_optimization.py",
    WALKTHROUGH / "real" / "advanced" / "03_multi_agent.py",
]

# ``1.025`` seconds -> canonical milliseconds, the winning trial in the replay.
BEST_LATENCY_MS = 1025.0
BEST_DURATION_SECONDS = 1.025

# A seconds-suffixed format spec applied to a canonical latency value.
SECONDS_SUFFIXED_LATENCY = re.compile(r":\.\d+f\}s")


def _load_optimize_rag() -> ModuleType:
    """Import the replay demo by path WITHOUT running ``main()``.

    ``walkthrough`` ships no ``__init__.py``, and the module bootstraps its own
    ``sys.path`` entry for ``utils.helpers`` at import time; that bootstrap is
    part of the shipped behaviour, so it is deliberately not stubbed. Only
    module-level replay data is executed - no optimizer, no provider call.
    """
    spec = importlib.util.spec_from_file_location(
        "walkthrough_demo_optimize_rag", OPTIMIZE_RAG
    )
    assert spec is not None and spec.loader is not None, OPTIMIZE_RAG
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def optimize_rag() -> ModuleType:
    return _load_optimize_rag()


@pytest.mark.parametrize("example", REAL_MODE_EXAMPLES, ids=lambda p: p.name)
def test_real_mode_summary_prints_milliseconds(example: Path) -> None:
    source = example.read_text(encoding="utf-8")
    assert REAL_SUMMARY in source, f"{example} lost the millisecond latency summary"


def test_multi_provider_keeps_the_latency_presence_guard() -> None:
    """07 only prints latency when the metric exists; the fix must not drop it."""
    source = (WALKTHROUGH / "real" / "07_multi_provider.py").read_text(encoding="utf-8")
    guarded = f'if "latency" in results.best_metrics:\n        {REAL_SUMMARY}'
    assert guarded in source


def test_optimize_rag_summary_prints_milliseconds() -> None:
    source = OPTIMIZE_RAG.read_text(encoding="utf-8")
    assert (
        "print(f\"  Latency:  {results.best_metrics['latency']:.0f}ms\")" in source
    ), f"{OPTIMIZE_RAG} lost the millisecond latency summary"


def test_no_walkthrough_latency_metric_is_formatted_as_seconds() -> None:
    """No ``best_metrics`` latency expression may carry an ``s`` suffix.

    Local elapsed-time displays (``demo/rag_agent.py``, ``mock/07``) are
    legitimately seconds and do not read ``best_metrics``, so they are untouched
    by this scan.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
        for path in sorted(WALKTHROUGH.rglob("*.py"))
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "best_metrics" in line
        and "latency" in line
        and SECONDS_SUFFIXED_LATENCY.search(line)
    ]
    assert not offenders, "canonical latency rendered as seconds:\n" + "\n".join(
        offenders
    )


def test_optimize_rag_best_metrics_latency_is_milliseconds(
    optimize_rag: ModuleType,
) -> None:
    results = optimize_rag.build_results()
    assert results.best_metrics["latency"] == BEST_LATENCY_MS


def test_optimize_rag_trial_metrics_are_milliseconds_durations_are_seconds(
    optimize_rag: ModuleType,
) -> None:
    results = optimize_rag.build_results()
    best = results.trials[optimize_rag.BEST_IDX]

    assert best.metrics["latency"] == BEST_LATENCY_MS
    assert best.duration == BEST_DURATION_SECONDS

    for trial, row in zip(results.trials, optimize_rag.TRIALS_DATA, strict=True):
        latency_seconds = row[7]
        assert trial.metrics["latency"] == pytest.approx(latency_seconds * 1000.0)
        assert trial.duration == latency_seconds

    # The whole replay ran in ~30 s; the total stays in seconds too.
    assert results.duration == pytest.approx(
        sum(row[7] for row in optimize_rag.TRIALS_DATA)
    )


def test_optimize_rag_renders_the_winner_as_1025ms(
    optimize_rag: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shared renderer showed ``1ms`` while the replay fed it seconds."""
    results = optimize_rag.build_results()
    assert _format_metric_value("latency", results.best_metrics["latency"]) == "1025ms"

    print_results_table(
        results,
        optimize_rag.CONFIG_SPACE,
        optimize_rag.OBJECTIVES,
        mode_label="REAL",
    )
    out = capsys.readouterr().out
    assert "1025ms" in out
