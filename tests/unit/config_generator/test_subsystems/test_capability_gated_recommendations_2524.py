"""SQL-only catalog knobs are gated on a SQL capability, not on agent type.

Traigent#2524: the classifier puts HumanEval, MBPP and subprocess runners in
the same ``code_gen`` class as text-to-SQL agents, so database-schema knobs
were recommended to every code generator. The gate is a declared or detected
capability; the agent type alone never decides it.
"""

from __future__ import annotations

import pytest

from traigent.cloud.client import RecommendationBundle
from traigent.config_generator.agent_classifier import classify_agent
from traigent.config_generator.subsystems.tvar_recommendations import (
    detect_capability_signals,
    generate_recommendations,
)

SQL_ONLY_KNOBS = {
    "schema_context",
    "evidence_usage",
    "fewshot_selector",
    "generation_path",
    "repair_policy",
}

HUMANEVAL_AGENT = """
import subprocess

def solve(problem):  # HumanEval / MBPP style code generator
    code = generate_code(problem["prompt"])
    return subprocess.run(["python", "-c", code], capture_output=True).stdout
"""

SQL_AGENT = """
import sqlite3

def answer(question, db_path):
    sql = generate_code(question)  # text-to-SQL
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql).fetchall()
"""


def _names(source: str, **kwargs) -> set[str]:
    classification = classify_agent(source)
    return {
        rec.name
        for rec in generate_recommendations(
            [],
            source_code=source,
            classification=classification,
            recommendation_bundle=RecommendationBundle.empty(),
            **kwargs,
        )
    }


def test_humaneval_style_agent_is_not_offered_sql_knobs() -> None:
    assert classify_agent(HUMANEVAL_AGENT).agent_type == "code_gen"
    names = _names(HUMANEVAL_AGENT)
    assert "schema_context" not in names
    assert SQL_ONLY_KNOBS.isdisjoint(names)
    # Knobs that are not SQL-specific are still offered to the same agent.
    assert {"fewshot_k", "candidate_count"} <= names


def test_sql_agent_in_the_same_class_is_offered_sql_knobs() -> None:
    assert classify_agent(SQL_AGENT).agent_type == "code_gen"
    assert SQL_ONLY_KNOBS <= _names(SQL_AGENT)


def test_declared_capability_overrides_detection_both_ways() -> None:
    assert SQL_ONLY_KNOBS <= _names(HUMANEVAL_AGENT, capability_signals={"sql"})
    assert SQL_ONLY_KNOBS.isdisjoint(_names(SQL_AGENT, capability_signals=()))


@pytest.mark.parametrize(
    "source",
    [
        "def f(q):\n    return run_sql(q)\n",
        'Q = "SELECT name FROM users WHERE id = ?"\n',
        "import duckdb\n",
        "from psycopg2 import connect\n",
        "DDL = 'create table t (id int)'\n",
    ],
)
def test_sql_signal_detected(source: str) -> None:
    assert detect_capability_signals(source) == frozenset({"sql"})


@pytest.mark.parametrize(
    "source",
    [
        "",
        HUMANEVAL_AGENT,
        "def select_best(items):\n    return items[0]\n",
        "from pymongo import MongoClient  # a NoSQL document store\n",
    ],
)
def test_no_sql_signal(source: str) -> None:
    assert detect_capability_signals(source) == frozenset()
