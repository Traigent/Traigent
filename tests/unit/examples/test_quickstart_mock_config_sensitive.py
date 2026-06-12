"""Regression tests for config-sensitive quickstart mock examples."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from traigent.config.context import ConfigurationContext
from traigent.testing import _reset_for_tests

ROOT_DIR = Path(__file__).resolve().parents[3]
QUICKSTART_DIR = ROOT_DIR / "examples" / "quickstart"
_QUICKSTART_ENV_KEYS = (
    "TRAIGENT_COST_APPROVED",
    "TRAIGENT_DATASET_ROOT",
    "TRAIGENT_MOCK_LLM",
    "TRAIGENT_RESULTS_FOLDER",
)


@pytest.fixture(autouse=True)
def isolate_quickstart_import_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_for_tests()
    for key in _QUICKSTART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    yield

    for stem in ("01_simple_qa", "03_custom_objectives"):
        sys.modules.pop(f"_quickstart_{stem}", None)
    for key in _QUICKSTART_ENV_KEYS:
        os.environ.pop(key, None)
    _reset_for_tests()


def _load_example_module(stem: str) -> ModuleType:
    path = QUICKSTART_DIR / f"{stem}.py"
    module_name = f"_quickstart_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_simple_qa_mock_varies_by_config() -> None:
    module = _load_example_module("01_simple_qa")
    hard_question = "What year did World War II end?"

    cheap_answer = module._mock_answer_for_config(
        hard_question,
        {"model": "gpt-3.5-turbo", "temperature": 0.9},
    )
    strong_answer = module._mock_answer_for_config(
        hard_question,
        {"model": "gpt-4o", "temperature": 0.1},
    )

    assert cheap_answer == "I don't know"
    assert strong_answer == "1945"


def test_simple_qa_mock_branch_reads_config_context(monkeypatch) -> None:
    module = _load_example_module("01_simple_qa")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    question = "What is the chemical symbol for water?"

    with ConfigurationContext({"model": "gpt-3.5-turbo", "temperature": 0.9}):
        assert module.simple_qa_agent.func(question) == "I don't know"

    with ConfigurationContext({"model": "gpt-4o", "temperature": 0.1}):
        assert module.simple_qa_agent.func(question) == "H2O"


def test_custom_objectives_mock_varies_by_config() -> None:
    module = _load_example_module("03_custom_objectives")
    hard_question = "Who painted the Mona Lisa?"

    weak_answer = module._mock_answer_for_config(
        hard_question,
        {"model": "gpt-3.5-turbo", "temperature": 0.95, "top_p": 0.2},
    )
    strong_answer = module._mock_answer_for_config(
        hard_question,
        {"model": "gpt-4o-mini", "temperature": 0.1, "top_p": 0.9},
    )

    assert weak_answer == "I don't know"
    assert strong_answer == "Leonardo da Vinci"


def test_custom_objectives_function_reads_config_context(monkeypatch) -> None:
    module = _load_example_module("03_custom_objectives")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    question = "What is the largest planet in our solar system?"

    with ConfigurationContext(
        {"model": "gpt-3.5-turbo", "temperature": 0.95, "top_p": 0.2}
    ):
        assert module.weighted_agent.func(question) == "I don't know"

    with ConfigurationContext(
        {"model": "gpt-4o-mini", "temperature": 0.1, "top_p": 0.9}
    ):
        assert module.weighted_agent.func(question) == "Jupiter"
