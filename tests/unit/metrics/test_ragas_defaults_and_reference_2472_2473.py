"""configure_ragas_defaults() reaches the evaluator (#2473); a mapped
``reference`` column is honoured and rows only drop when a selected metric needs
the reference (#2472)."""

from __future__ import annotations

import pytest

import traigent.metrics.ragas_metrics as rm
from traigent.api.types import ExampleResult
from traigent.core.optimization_pipeline import _create_local_evaluator


class _Sample:
    def __init__(self, **kw):
        self.fields = kw


@pytest.fixture
def clean_globals(monkeypatch):
    monkeypatch.setattr(rm, "_GLOBAL_RAGAS_CONFIG", rm.RagasConfig())
    monkeypatch.setattr(rm, "SingleTurnSample", _Sample)


def _row(expected="EXPECTED", metadata=None):
    return ExampleResult(
        example_id="e0",
        input_data={"prompt": "q"},
        expected_output=expected,
        actual_output="resp",
        metrics={},
        execution_time=0.0,
        success=True,
        error_message=None,
        metadata=metadata or {},
    )


def _evaluator():
    ev, _ = _create_local_evaluator(
        None,
        None,
        None,
        objectives=["answer_similarity"],
        execution_mode="local",
        metric_functions=None,
        scoring_function=None,
    )
    return ev


def test_global_defaults_reach_evaluator_ragas_config(clean_globals):
    llm, emb = object(), object()
    rm.configure_ragas_defaults(
        column_map={"reference": "ref_col"}, llm=llm, embeddings=emb
    )
    cfg = _evaluator()._get_ragas_config()
    assert dict(cfg.column_map) == {"reference": "ref_col"}
    assert cfg.llm is llm and cfg.embeddings is emb


def test_per_run_field_wins_and_global_fills_the_gap(clean_globals):
    g_llm, run_llm = object(), object()
    rm.configure_ragas_defaults(column_map={"reference": "g"}, llm=g_llm)
    ev = _evaluator()
    ev.config["ragas_llm"] = run_llm
    cfg = ev._get_ragas_config()
    assert cfg.llm is run_llm
    assert dict(cfg.column_map) == {"reference": "g"}


def test_mapped_reference_wins_over_expected_output(clean_globals):
    cfg = rm.RagasConfig(column_map={"reference": "reference_answer"})
    row = _row(metadata={"reference_answer": "MAPPED"})
    fields = rm._prepare_samples([row], config=cfg, required_columns={"reference"})[
        0
    ].fields
    assert fields["reference"] == "MAPPED"


def test_unmapped_reference_still_uses_expected_output(clean_globals):
    cfg = rm.RagasConfig()
    fields = rm._prepare_samples(
        [_row(metadata={"reference_answer": "MAPPED"})],
        config=cfg,
        required_columns={"reference"},
    )[0].fields
    assert fields["reference"] == "EXPECTED"


def test_missing_reference_keeps_row_for_context_only_metric(clean_globals):
    cfg = rm.RagasConfig()
    row = _row(expected=None, metadata={"retrieved_contexts": ["c"]})
    kept = rm._prepare_samples(
        [row], config=cfg, required_columns={"retrieved_contexts"}
    )
    assert len(kept) == 1
    dropped = rm._prepare_samples([row], config=cfg, required_columns={"reference"})
    assert dropped == []
