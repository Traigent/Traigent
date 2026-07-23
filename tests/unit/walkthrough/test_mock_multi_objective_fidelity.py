"""Regression coverage for #2003: the mock multi-objective demos must optimize.

Before the fix, ``walkthrough/mock/04_multi_objective.py`` recorded ``cost = 0.0``
on every trial, so the declared 30% cost weight had no effect on ``best_config``
and the SDK emitted the #1832 "uniformly constant" warning. Latency varied but
was mis-ordered (``if "gpt-4o" in model`` also matched ``gpt-4o-mini``).

The fix injects per-configuration usage with :func:`traigent.with_usage` and
registers a ``latency`` metric function returning MILLISECONDS. These tests pin
the contracts that fix rides on, **against the shipped example modules**:

* ``cost`` reaches ``trial.metrics`` as the per-trial TOTAL, and it gets there
  from the injected ``total_cost`` (see
  :func:`test_shipped_09_cost_comes_from_the_injected_total_cost`), and
* ``latency`` is the example's custom millisecond value, registered on the
  shipped decorator and displayed unmodified.

The examples are loaded by *path*: ``walkthrough.mock.04_multi_objective`` is not
a legal dotted module name because the file starts with a digit. Reaching for a
local re-implementation instead is exactly how the first version of this file
ended up green against two silent reverts of the fix.

Everything here is offline (``TRAIGENT_MOCK_LLM``/``TRAIGENT_OFFLINE_MODE``, no
provider key): zero LLM spend.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.utils.cost_calculator import cost_from_tokens
from walkthrough.utils.mock_answers import (
    DEFAULT_MOCK_MODEL,
    MOCK_TASK_TOKENS,
    RAG_ANSWERS,
    get_mock_cost,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WALKTHROUGH = REPO_ROOT / "walkthrough"
EXAMPLE_04 = WALKTHROUGH / "mock" / "04_multi_objective.py"
EXAMPLE_09 = WALKTHROUGH / "mock" / "09_rag_multi_objective.py"
RAG_DATASET = WALKTHROUGH / "datasets" / "rag_questions.jsonl"

MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
DATASET_SIZE = 20
RAG_DATASET_SIZE = 13

# Models whose mock price and the SDK's own token pricing DISAGREE for rag_qa.
# On these, the injected ``total_cost`` is the only thing that can produce the
# mock number: zero it out and the SDK's implausible-under-report clamp
# back-fills its own token-derived cost instead. For ``classification`` the two
# tables agree exactly, which is why a 04-only cost assertion cannot tell the
# ``total_cost`` lever apart from the token counts.
DIVERGENT_RAG_MODELS = ["gpt-5-nano", "gpt-5.1", "gpt-5.2"]
RAG_MAX_TOKENS = 200


@functools.cache
def _load_example(path: Path) -> ModuleType:
    """Import a shipped walkthrough example by path and return the module.

    The example bootstraps its own ``sys.path`` for ``utils.helpers`` and calls
    ``traigent.initialize(offline=True, ...)`` at import time; both are the
    behaviour under test, so they are deliberately not stubbed.
    """
    spec = importlib.util.spec_from_file_location(f"walkthrough_mock_{path.stem}", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _example_04() -> ModuleType:
    return _load_example(EXAMPLE_04)


def _example_09() -> ModuleType:
    return _load_example(EXAMPLE_09)


@pytest.fixture(autouse=True)
def offline_mock_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Force offline mock mode with an isolated results folder (no spend)."""
    env = {
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "TRAIGENT_RESULTS_FOLDER": str(tmp_path / "results"),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TRAIGENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return env


def _objectives(accuracy: float, cost: float, latency: float) -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", orientation="maximize", weight=accuracy),
            ObjectiveDefinition("cost", orientation="minimize", weight=cost),
            ObjectiveDefinition("latency", orientation="minimize", weight=latency),
        ]
    )


async def _run_04(objectives: ObjectiveSchema) -> Any:
    """Run the SHIPPED example-04 function over a model-only space, offline."""
    return await _example_04().ai_agent_classify_text_sentiment.optimize(
        algorithm="grid",
        max_trials=len(MODELS),
        configuration_space={"model": MODELS},
        objectives=objectives,
        show_progress=False,
    )


async def _run_09(objectives: ObjectiveSchema) -> Any:
    """Run the SHIPPED example-09 agent over the divergent-pricing models."""
    return await _example_09().rag_agent.optimize(
        algorithm="grid",
        max_trials=len(DIVERGENT_RAG_MODELS),
        configuration_space={
            "model": DIVERGENT_RAG_MODELS,
            "prompt": ["minimal"],
            "temperature": [0.0],
            "instructions": ["direct"],
            "max_tokens": [RAG_MAX_TOKENS],
        },
        objectives=objectives,
        show_progress=False,
    )


# ---------------------------------------------------------------------------
# Shipped-symbol contracts
# ---------------------------------------------------------------------------


def test_shipped_04_registers_its_latency_metric_function() -> None:
    """Example 04 must hand its millisecond latency to the decorator."""
    example = _example_04()

    registered = example.ai_agent_classify_text_sentiment.metric_functions or {}

    assert registered.get("latency") is example.mock_latency_ms


def test_shipped_09_registers_its_latency_metric_function() -> None:
    """Example 09 must hand its millisecond latency to the decorator."""
    example = _example_09()

    registered = example.rag_agent.metric_functions or {}

    assert registered.get("latency") is example.mock_rag_latency_ms


@pytest.mark.parametrize("model", MODELS)
def test_shipped_04_latency_metric_returns_milliseconds(model: str) -> None:
    """``latency`` is the SDK's millisecond unit; seconds would render ``0ms``."""
    value = _example_04().mock_latency_ms(None, None, {"model": model})

    assert value == pytest.approx(get_mock_latency(model, "classification") * 1000.0)
    assert value > 1.0


@pytest.mark.parametrize("model", DIVERGENT_RAG_MODELS)
def test_shipped_09_latency_metric_returns_milliseconds(model: str) -> None:
    """Same unit contract for 09, including its max_tokens retrieval penalty."""
    example = _example_09()

    small = example.mock_rag_latency_ms(None, None, {"model": model, "max_tokens": 50})
    large = example.mock_rag_latency_ms(None, None, {"model": model, "max_tokens": 200})

    assert small == pytest.approx(
        (get_mock_latency(model, "rag_qa") + 0.001 * 50) * 1000.0
    )
    assert small > 1.0
    # A bigger retrieval budget must cost time, or max_tokens carries no signal.
    assert large > small


# ---------------------------------------------------------------------------
# Production-path runs of the shipped examples
# ---------------------------------------------------------------------------


async def test_shipped_04_cost_dominant_weighting_selects_cheapest() -> None:
    """A cost-dominant weighting must pick the cheapest model."""
    results = await _run_04(_objectives(accuracy=0.02, cost=0.97, latency=0.01))

    assert results.best_config["model"] == "gpt-4o-mini"


async def test_shipped_04_accuracy_dominant_weighting_selects_most_accurate() -> None:
    """The same example under an accuracy-dominant weighting picks the best model."""
    results = await _run_04(_objectives(accuracy=0.98, cost=0.01, latency=0.01))

    assert results.best_config["model"] == "gpt-4o"


async def test_shipped_04_trial_cost_is_per_trial_total() -> None:
    """``cost`` is the per-trial TOTAL, not the per-example mean."""
    results = await _run_04(_objectives(accuracy=0.5, cost=0.3, latency=0.2))

    assert len(results.trials) == len(MODELS)
    for trial in results.trials:
        model = trial.config["model"]
        assert trial.metrics["cost"] == pytest.approx(
            get_mock_cost(model, "classification", dataset_size=DATASET_SIZE)
        )


async def test_shipped_04_trial_latency_is_milliseconds() -> None:
    """``latency`` carries the custom millisecond value, not the wall-clock one."""
    results = await _run_04(_objectives(accuracy=0.5, cost=0.3, latency=0.2))

    for trial in results.trials:
        model = trial.config["model"]
        assert trial.metrics["latency"] == pytest.approx(
            get_mock_latency(model, "classification") * 1000.0
        )
        # A seconds regression would land at 0.2-0.36 and render as "0ms".
        assert trial.metrics["latency"] > 1.0


async def test_shipped_04_objectives_are_not_uniformly_constant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both weighted objectives must carry signal across trials (#1832)."""
    with caplog.at_level(logging.WARNING):
        results = await _run_04(_objectives(accuracy=0.5, cost=0.3, latency=0.2))

    assert len({trial.metrics["cost"] for trial in results.trials}) > 1
    assert len({trial.metrics["latency"] for trial in results.trials}) > 1
    assert "uniformly constant" not in caplog.text


def test_rag_mock_pricing_diverges_from_sdk_token_pricing() -> None:
    """Guard for the test below: it is only meaningful while the tables disagree.

    If the mock table is ever re-synced with the SDK's pricing for these models,
    the token counts alone would reproduce the mock cost and
    ``test_shipped_09_cost_comes_from_the_injected_total_cost`` would silently
    stop testing ``total_cost``. Pick different models then.
    """
    tokens = MOCK_TASK_TOKENS["rag_qa"]

    for model in DIVERGENT_RAG_MODELS:
        derived = sum(
            cost_from_tokens(tokens["input"], tokens["output"], model, strict=False)
        )
        assert derived > 0.0
        assert derived != pytest.approx(get_mock_cost(model, "rag_qa", 1), rel=0.05)


async def test_shipped_09_cost_comes_from_the_injected_total_cost() -> None:
    """``cost`` must equal the MOCK price, which only ``total_cost`` can supply.

    For these models the SDK's token-derived cost is 27-32% below the mock price,
    so dropping or zeroing ``with_usage(total_cost=...)`` in example 09 trips the
    implausible-under-report clamp and lands the SDK's number instead.
    """
    results = await _run_09(_objectives(accuracy=0.5, cost=0.2, latency=0.3))

    assert len(results.trials) == len(DIVERGENT_RAG_MODELS)
    for trial in results.trials:
        model = trial.config["model"]
        assert trial.metrics["cost"] == pytest.approx(
            get_mock_cost(model, "rag_qa", dataset_size=RAG_DATASET_SIZE)
        )


async def test_shipped_09_cost_dominant_weighting_selects_cheapest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """09's cost objective carries signal and moves ``best_config`` (#1832)."""
    with caplog.at_level(logging.WARNING):
        results = await _run_09(_objectives(accuracy=0.02, cost=0.97, latency=0.01))

    assert len({trial.metrics["cost"] for trial in results.trials}) > 1
    assert len({trial.metrics["latency"] for trial in results.trials}) > 1
    assert results.best_config["model"] == "gpt-5-nano"
    assert "uniformly constant" not in caplog.text


async def test_shipped_09_cost_is_inert_without_with_usage() -> None:
    """The #2003 counterfactual: no usage injection leaves cost a constant $0.

    Same scorer, same latency metric, same dataset as example 09 - only the
    ``traigent.with_usage`` wrapper is gone.
    """
    example = _example_09()

    @traigent.optimize(
        eval_dataset=str(RAG_DATASET),
        objectives=_objectives(accuracy=0.5, cost=0.2, latency=0.3),
        scoring_function=example.rag_accuracy_scorer,
        metric_functions={"latency": example.mock_rag_latency_ms},
        configuration_space={
            "model": DIVERGENT_RAG_MODELS,
            "prompt": ["minimal"],
            "temperature": [0.0],
            "instructions": ["direct"],
            "max_tokens": [RAG_MAX_TOKENS],
        },
        injection_mode="context",
        offline=True,
    )
    def rag_agent_without_usage(question: str) -> str:
        set_mock_model(traigent.get_config().get("model", DEFAULT_MOCK_MODEL))
        answer: str = RAG_ANSWERS.get(normalize_text(question), "I don't know.")
        return answer

    results = await rag_agent_without_usage.optimize(
        algorithm="grid", max_trials=len(DIVERGENT_RAG_MODELS), show_progress=False
    )

    assert {trial.metrics.get("cost", 0.0) for trial in results.trials} == {0.0}


# ---------------------------------------------------------------------------
# Shipped examples end to end
# ---------------------------------------------------------------------------

_TRIAL_ROW = re.compile(r"^│\s*(?:★\s*)?\d+\s*│")


def _parse_results_table(stdout: str) -> list[dict[str, str]]:
    """Parse the SDK results table into one ``{column: cell}`` dict per trial."""
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("│").split("│")]
        if header is None:
            if "model" in cells and "latency" in cells:
                header = cells
            continue
        if _TRIAL_ROW.match(line):
            rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _cost(row: dict[str, str]) -> float:
    return float(row["cost"].lstrip("$"))


def _latency_ms(row: dict[str, str]) -> float:
    return float(row["latency"].removesuffix("ms"))


def _run_example(
    path: Path, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **env_overrides, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=600,
    )


def test_example_04_smoke(offline_mock_env: dict[str, str]) -> None:
    """The shipped example runs clean and reports real milliseconds and dollars."""
    completed = _run_example(EXAMPLE_04, offline_mock_env)

    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "simulated from the static mock pricing/latency tables" in completed.stdout
    assert "uniformly constant" not in combined
    assert "Model: gpt-4o-mini" in completed.stdout
    # The user-visible #2003 symptom was "Latency:  0ms" under every model.
    assert "Latency:  200ms (simulated, per call)" in completed.stdout
    assert "Cost:     $0.00036 (simulated, 20 examples)" in completed.stdout
    assert re.search(r"Latency:\s+0ms", completed.stdout) is None

    rows = _parse_results_table(completed.stdout)
    assert len(rows) == 8
    for row in rows:
        model = row["model"]
        assert _latency_ms(row) == pytest.approx(
            get_mock_latency(model, "classification") * 1000.0, abs=0.5
        )
        assert _latency_ms(row) > 1.0
        assert _cost(row) == pytest.approx(
            get_mock_cost(model, "classification", DATASET_SIZE), abs=5e-6
        )


def test_example_09_smoke(offline_mock_env: dict[str, str]) -> None:
    """Example 09 shows the same numbers that drove selection, not a re-estimate.

    Every displayed latency must match the shipped ``mock_rag_latency_ms`` for
    that row's config - including the ``max_tokens`` retrieval penalty, which a
    model-only display override would silently drop.
    """
    latency_of = _example_09().mock_rag_latency_ms
    completed = _run_example(EXAMPLE_09, offline_mock_env)

    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "simulated from the static mock pricing/latency tables" in completed.stdout
    assert "uniformly constant" not in combined
    assert "Model:        gpt-4o-mini" in completed.stdout
    assert "Latency:  825ms (simulated)" in completed.stdout
    assert "Cost:     $0.004680 (simulated)" in completed.stdout
    assert re.search(r"Latency:\s+0ms", completed.stdout) is None

    rows = _parse_results_table(completed.stdout)
    assert len(rows) == 18
    for row in rows:
        config = {"model": row["model"], "max_tokens": int(row["max_tokens"])}
        assert _latency_ms(row) == pytest.approx(
            latency_of(None, None, config), abs=0.5
        )
        assert _latency_ms(row) > 1.0
        assert _cost(row) == pytest.approx(
            get_mock_cost(row["model"], "rag_qa", RAG_DATASET_SIZE), abs=5e-6
        )
