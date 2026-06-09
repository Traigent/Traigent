"""Regression tests for bundled skill snippets."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from traigent.api.types import ExampleResult
from traigent.evaluators.base import EvaluationExample


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_OPTIONS_DOC = (
    REPO_ROOT
    / "traigent"
    / "skills"
    / "traigent-decorator-setup"
    / "references"
    / "evaluation-options.md"
)


def _extract_custom_evaluator_snippet() -> str:
    markdown = EVALUATION_OPTIONS_DOC.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    for block in blocks:
        if "def my_evaluator" in block and "ExampleResult(" in block:
            return textwrap.dedent(block)
    raise AssertionError("custom evaluator ExampleResult snippet not found")


def test_custom_evaluator_snippet_executes_with_current_api() -> None:
    snippet = _extract_custom_evaluator_snippet()

    assert "score=" not in snippet
    assert "prediction=" not in snippet
    assert 'example["' not in snippet
    assert "example.input_data" in snippet
    assert "example.expected_output" in snippet
    assert "metrics=" in snippet

    namespace: dict[str, object] = {}
    exec(snippet, namespace)
    my_evaluator = namespace["my_evaluator"]

    example = EvaluationExample(
        input_data={"question": "What is Python?"},
        expected_output="programming language",
        metadata={"example_id": "qa-1", "db_path": "eval.sqlite"},
    )

    def answer(question: str) -> str:
        assert question == "What is Python?"
        return "Python is a programming language."

    result = my_evaluator(answer, {"model": "mock-model"}, example)

    assert isinstance(result, ExampleResult)
    assert result.example_id == "qa-1"
    assert result.input_data == example.input_data
    assert result.expected_output == "programming language"
    assert result.actual_output == "Python is a programming language."
    assert result.metrics == {"accuracy": 1.0}
    assert result.success is True
    assert result.error_message is None
    assert result.execution_time >= 0.0
    assert result.metadata["db_path"] == "eval.sqlite"
    assert result.metadata["model"] == "mock-model"
