"""#3130: an LLM reply of the wrong JSON shape falls back instead of raising."""

from __future__ import annotations

import pytest

from traigent.config_generator.agent_classifier import _llm_classify
from traigent.config_generator.subsystems.benchmarks import _llm_generate_examples
from traigent.config_generator.subsystems.objectives import _llm_enrich_objectives
from traigent.config_generator.subsystems.structural_constraints import (
    _llm_suggest_constraints,
)
from traigent.config_generator.subsystems.tvar_ranges import (
    _parse_llm_range_response,
)
from traigent.config_generator.subsystems.tvar_recommendations import _llm_recommend
from traigent.config_generator.types import BenchmarkSpec, ObjectiveSpec, TVarSpec


class _Stub:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        return self.reply


DEEP = "[" * 100000
_TVAR = TVarSpec(name="t", range_type="Choices", range_kwargs={"values": [1, 2]})
_OBJ = ObjectiveSpec(name="accuracy", orientation="maximize")


@pytest.mark.parametrize(
    "reply",
    ['{"agent_type": ["rag"]}', '{"agent_type": {"a": 1}}', DEEP, '["rag"]'],
)
def test_classify_wrong_shape_returns_none(reply: str) -> None:
    assert _llm_classify("def f(): pass", _Stub(reply)) is None


@pytest.mark.parametrize(
    "reply",
    [
        '["accuracy"]',
        "[null]",
        "[1]",
        '[{"name": ["a"], "orientation": "maximize"}]',
        '[{"name": "x", "orientation": ["maximize"]}]',
        DEEP,
    ],
)
def test_objectives_wrong_shape_keeps_heuristic(reply: str) -> None:
    assert _llm_enrich_objectives("src", [_OBJ], _Stub(reply)) is None


@pytest.mark.parametrize("reply", ['["c1"]', "[null]", "[1]", DEEP])
def test_constraints_wrong_shape_returns_empty(reply: str) -> None:
    assert _llm_suggest_constraints([_TVAR, _TVAR], _Stub(reply), "src") == []


@pytest.mark.parametrize(
    "reply",
    [
        '["a"]',
        "[null]",
        '[{"name": "n", "range_type": ["Range"], "kwargs": {}}]',
        '[{"name": ["n"], "range_type": "Range", "kwargs": {}}]',
        DEEP,
    ],
)
def test_recommendations_wrong_shape_returns_empty(reply: str) -> None:
    assert _llm_recommend([_TVAR], "rag", "src", _Stub(reply)) == []


@pytest.mark.parametrize(
    "reply",
    [
        '{"range_type": ["Range"], "kwargs": {}}',
        '{"range_type": {"a": 1}, "kwargs": {}}',
        DEEP,
    ],
)
def test_range_response_wrong_shape_returns_none(reply: str) -> None:
    assert _parse_llm_range_response("n", reply) is None


@pytest.mark.parametrize("reply", ["[1, 2]", '["a"]', "[null]", DEEP])
def test_benchmark_non_object_examples_return_none(reply: str) -> None:
    bench = BenchmarkSpec(name="b", example_schema={"input": "str"})
    assert _llm_generate_examples(bench, "src", _Stub(reply)) is None


def test_benchmark_keeps_only_object_examples() -> None:
    bench = BenchmarkSpec(name="b", example_schema={"input": "str"})
    spec = _llm_generate_examples(
        bench, "src", _Stub('[{"input": "a", "output": "b"}, 3, null]')
    )
    assert spec is not None
    assert spec.sample_examples == ({"input": "a", "output": "b"},)


@pytest.mark.parametrize(
    "reply",
    [
        '[{"description": 5, "constraint_code": "x"}]',
        '[{"description": "d", "constraint_code": {"a": 1}}]',
        '[{"description": ["d"], "constraint_code": ["x"]}]',
    ],
)
def test_constraints_non_string_fields_are_dropped(reply: str) -> None:
    assert _llm_suggest_constraints([_TVAR, _TVAR], _Stub(reply), "src") == []
