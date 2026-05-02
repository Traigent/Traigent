"""Unit tests for the Python optimizer adoption assistant."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.optimizer.proposer import build_decorate_plan
from traigent.optimizer.scanner import scan_path

AGENT_SOURCE = """\
from langchain_openai import ChatOpenAI


def answer_question(question: str) -> str:
    model = "gpt-4o-mini"
    temperature = 0.7
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm.invoke(question).content
"""


def _write_agent(tmp_path: Path) -> Path:
    source = tmp_path / "agent.py"
    source.write_text(AGENT_SOURCE, encoding="utf-8")
    return source


def test_scan_path_emits_schema_shaped_report(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)

    report = scan_path(source)

    assert report["report_version"] == "0.1.0"
    assert report["runtime"] == "python"
    assert len(report["candidates"]) == 1

    candidate = report["candidates"][0]
    assert candidate["fingerprint"]["candidate_id"]
    assert len(candidate["fingerprint"]["source_hash"]) == 64
    assert len(candidate["fingerprint"]["source_span_hash"]) == 64
    assert candidate["function"]["name"] == "answer_question"
    assert {signal["kind"] for signal in candidate["signals"]} == {"llm_call"}
    assert {signal["tvar"]["name"] for signal in candidate["tvar_signals"]} >= {
        "model",
        "temperature",
    }

    objectives = {objective["name"]: objective for objective in candidate["objective_candidates"]}
    assert objectives["accuracy"]["requires_confirmation"] is True
    assert objectives["accuracy"]["auto_measurable"] is False
    assert objectives["cost"]["requires_confirmation"] is False
    assert objectives["latency"]["auto_measurable"] is True


def test_build_decorate_plan_preserves_tvar_evidence_and_locator(
    tmp_path: Path,
) -> None:
    source = _write_agent(tmp_path)

    plan = build_decorate_plan(
        source,
        function_name="answer_question",
        objective_names=["accuracy"],
        dataset_ref="eval/qa.jsonl",
    )

    assert plan["requested_emit_mode"] == "auto"
    assert plan["resolved_emit_mode"] == "tvl"
    assert plan["confirmation_state"]["objectives_confirmed"] is True
    assert plan["dataset_plan"]["status"] == "present"
    assert plan["dataset_plan"]["dataset_ref"] == "eval/qa.jsonl"

    first_binding = plan["proposed_tvar_bindings"][0]
    assert first_binding["evidence"]["file"] == "agent.py"
    assert first_binding["locator"]["kind"] == "line_col"
    assert first_binding["locator"]["details"]


def test_optimizer_scan_cli_outputs_json(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["optimizer", "scan", str(source), "--json"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["candidates"][0]["function"]["name"] == "answer_question"


def test_optimizer_decorate_cli_outputs_json(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "answer_question",
            "--objective",
            "accuracy",
            "--dataset",
            "eval/qa.jsonl",
            "--json",
        ],
    )

    assert result.exit_code == 0
    plan = json.loads(result.output)
    assert plan["target"]["function"].endswith("answer_question")
    assert plan["selected_objectives"] == [
        {"name": "accuracy", "direction": "maximize"}
    ]
