"""#2473: configure_ragas_defaults() reaches the @traigent.optimize evaluator.

``BaseEvaluator._get_ragas_config`` always passed an explicit RagasConfig
built only from its own ``ragas_*`` kwargs, which the SDK never sets, so the
public defaults were dropped. ragas-free.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import traigent.metrics.ragas_metrics as rm
from traigent.core.optimization_pipeline import _create_local_evaluator
from traigent.evaluators.local import LocalEvaluator


@pytest.fixture(autouse=True)
def _reset_defaults() -> Iterator[None]:
    rm.configure_ragas_defaults(column_map=None, llm=None, embeddings=None)
    yield
    rm.configure_ragas_defaults(column_map=None, llm=None, embeddings=None)


def test_global_defaults_reach_the_decorator_evaluator() -> None:
    llm, embeddings = object(), object()
    column_map = {"reference": "reference_answer", "retrieved_contexts": "gold"}
    rm.configure_ragas_defaults(column_map=column_map, llm=llm, embeddings=embeddings)

    evaluator, _ = _create_local_evaluator(
        None,
        None,
        None,
        objectives=["answer_similarity"],
        execution_mode="local",
        metric_functions=None,
        scoring_function=None,
    )
    config = evaluator._get_ragas_config()

    assert dict(config.column_map or {}) == column_map
    assert config.llm is llm
    assert config.embeddings is embeddings


def test_evaluator_kwargs_win_field_by_field() -> None:
    global_llm, own_llm, global_embeddings = object(), object(), object()
    rm.configure_ragas_defaults(
        column_map={"retrieved_contexts": "gold"},
        llm=global_llm,
        embeddings=global_embeddings,
    )
    evaluator = LocalEvaluator(
        metrics=["answer_similarity"],
        ragas_column_map={"retrieved_contexts": "own"},
        ragas_llm=own_llm,
    )
    config = evaluator._get_ragas_config()

    assert dict(config.column_map or {}) == {"retrieved_contexts": "own"}
    assert config.llm is own_llm
    assert config.embeddings is global_embeddings
