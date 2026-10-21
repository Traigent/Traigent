"""Traigent#2420: auto_override_frameworks must apply for SYNC functions.

The override gate is a ``ContextVar`` set in the caller's context. A sync user
function runs in a worker thread, so ``BaseEvaluator._execute_sync_in_thread``
must run it inside ``copy_context()`` of the caller; otherwise the patched
framework constructor sees the gate off and silently builds the hard-coded
model for every trial. This runs ``optimize_sync()`` end to end, keyless and
offline (mock mode), and asserts each trial's constructor values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langchain_openai")

import traigent  # noqa: E402


@pytest.fixture
def qa_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    path = tmp_path / "qa.jsonl"
    rows = [
        {"input": {"question": "What is 2+2?"}, "output": "4"},
        {"input": {"question": "Capital of France?"}, "output": "Paris"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_sync_function_gets_per_trial_framework_override(
    qa_dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_openai import ChatOpenAI

    from traigent.testing import enable_mock_mode_for_quickstart

    monkeypatch.setenv("OPENAI_API_KEY", "mock-placeholder")
    enable_mock_mode_for_quickstart()
    built: list[tuple[str, float, str, float]] = []

    @traigent.optimize(
        eval_dataset=qa_dataset,
        configuration_space={
            "model": ["gpt-4o-mini", "gpt-4o"],
            "temperature": [0.0, 1.0],
        },
        objectives=["accuracy"],
        offline=True,
        algorithm="grid",
        max_trials=4,
        auto_override_frameworks=True,
        framework_targets=["langchain_openai.ChatOpenAI"],
    )
    def summarize_document(question):  # sync on purpose: the broken lane
        # Hard-coded values must be overridden by the trial config.
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        cfg = traigent.get_config()
        built.append(
            (cfg["model"], cfg["temperature"], llm.model_name, llm.temperature)
        )
        return llm.invoke(question).content

    summarize_document.optimize_sync()

    assert built, "the function never ran"
    assert {(m, t) for m, t, _, _ in built} == {
        ("gpt-4o-mini", 0.0),
        ("gpt-4o-mini", 1.0),
        ("gpt-4o", 0.0),
        ("gpt-4o", 1.0),
    }
    for cfg_model, cfg_temp, built_model, built_temp in built:
        assert (built_model, built_temp) == (cfg_model, cfg_temp)
