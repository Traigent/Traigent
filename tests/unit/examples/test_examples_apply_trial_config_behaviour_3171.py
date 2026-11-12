"""Changed examples must behave differently under different trial configs (#3171).

Each example is loaded from its file, the SDK config context is set the way the
SDK's own tests do (``traigent.config.context.set_config``), and the decorated
function is called in mock mode. Examples that tune through an evaluator are
checked on the arguments the evaluator receives, so a value that is silently
replaced by a function default is caught.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from traigent.config.context import set_config

ROOT = Path(__file__).resolve().parents[3]
AI_DIR = ROOT / "examples" / "advanced" / "ai-engineering-tasks"
_SIBLING_MODULES = (
    "dataset",
    "evaluator",
    "selection_config",
    "context_config",
    "budget_config",
    "function_config",
    "safety_config",
)


def _load(path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load an example file with its sibling modules, isolated from other examples."""
    for name in _SIBLING_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    spec = importlib.util.spec_from_file_location(
        f"_example_{path.parent.name}_{path.stem}", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_config():
    set_config(None)
    yield
    set_config(None)


def _record_evaluator_calls(module: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def recorder(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"accuracy": 0.0}

    monkeypatch.setattr(module, "_trial_evaluation_function", lambda: recorder)
    return calls


# (example dir, decorated function, keys the function reads as parameters,
#  keys it does NOT read as parameters but must still forward from the trial)
EVALUATOR_EXAMPLES = [
    (
        "p0_few_shot_selection",
        "optimize_few_shot_selection",
        {"selection_method": "semantic_knn", "n_examples": 5},
        {"similarity_metric": "euclidean", "diversity_weight": 0.5},
    ),
    (
        "p0_context_engineering",
        "optimize_context_engineering",
        {"retrieval_method": "bm25", "chunk_size": 1024},
        {"reranking": True, "query_expansion": True},
    ),
    (
        "p1_token_budget",
        "optimize_token_budget",
        {"allocation_strategy": "dynamic_priority", "truncation_strategy": "hard_cut"},
        {"buffer_pct": 0.11, "output_buffer": 768},
    ),
    (
        "p1_function_calling",
        "optimize_function_calling",
        {"selection_strategy": "all_tools", "max_retries": 3},
        {"include_examples": False, "progressive_hints": True},
    ),
]


@pytest.mark.parametrize(
    ("example", "fn", "read_keys", "forwarded_keys"), EVALUATOR_EXAMPLES
)
def test_evaluator_receives_trial_values_not_defaults(
    example: str,
    fn: str,
    read_keys: dict[str, Any],
    forwarded_keys: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(AI_DIR / example / "main.py", monkeypatch)
    calls = _record_evaluator_calls(module, monkeypatch)
    func = getattr(module, fn)

    trial_a = {**read_keys, **forwarded_keys}
    set_config(trial_a)
    func()
    set_config({k: v for k, v in trial_a.items() if k not in forwarded_keys})
    func()
    first, second = calls
    for key, value in trial_a.items():
        assert first[key] == value, f"{example}: {key} was replaced by a default"
    assert first != second, f"{example}: trial config did not change the call"
    for key in forwarded_keys:
        assert second[key] != forwarded_keys[key]


def test_structured_output_behaviour_follows_trial_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs with no optional dependency: pure output differences."""
    module = _load(AI_DIR / "p0_structured_output" / "main.py", monkeypatch)
    func = module.extract_entities
    text = "meeting with john smith of ACME CORP"

    set_config(
        {"output_format": "json", "validation": "strict", "extraction_style": "concise"}
    )
    strict = func(text)
    set_config(
        {
            "output_format": "json",
            "validation": "lenient",
            "extraction_style": "concise",
        }
    )
    lenient = func(text)
    set_config(
        {
            "output_format": "json",
            "validation": "lenient",
            "extraction_style": "verbose",
        }
    )
    verbose = func(text)
    set_config(
        {"output_format": "xml", "validation": "lenient", "extraction_style": "concise"}
    )
    xml = func(text)

    assert json.loads(strict) == {}
    assert json.loads(lenient) == {"person": "John Smith"} or "person" in json.loads(
        lenient
    )
    assert "evidence" in json.loads(verbose) and "evidence" not in json.loads(lenient)
    assert xml.startswith("<entities>")


def test_few_shot_output_follows_trial_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(AI_DIR / "p0_few_shot_selection" / "main.py", monkeypatch)
    set_config({"selection_method": "semantic_knn", "n_examples": 1})
    few = module.optimize_few_shot_selection()
    set_config({"selection_method": "semantic_knn", "n_examples": 5})
    many = module.optimize_few_shot_selection()
    assert few["accuracy"] != many["accuracy"]
