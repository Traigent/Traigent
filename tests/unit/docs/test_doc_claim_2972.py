"""Doc claim matches code (#2972)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_with_usage_multi_agent_example_runs_2972() -> None:
    import doctest

    import traigent.api.functions as fmod

    doc = fmod.with_usage.__doc__
    start = doc.index(">>> from traigent.cloud.agent_dtos")
    end = doc.index("In OSS-only builds")
    block = doc[start:end]
    # The example ends with a `return` that only makes sense inside a function;
    # drop that statement, run everything before it.
    block = block.split(">>> return", 1)[0]
    examples = doctest.DocTestParser().get_examples(block)
    ns: dict = {}
    for ex in examples:
        exec(compile(ex.source, "<with_usage doc>", "exec"), ns)  # noqa: S102
    workflow = ns["workflow"]
    assert workflow.total_cost == pytest.approx(0.003)
    assert workflow.total_input_tokens == 100
