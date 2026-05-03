"""Unit tests for the Python optimizer adoption assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

import traigent.cli as cli_package
from traigent.cli import optimizer_command
from traigent.cli.main import cli
from traigent.optimizer import scanner as scanner_module
from traigent.optimizer.proposer import (
    _dataset_field_warning,
    _dataset_format,
    _read_dataset_fields,
    build_decorate_plan,
)
from traigent.optimizer.scanner import infer_project_root, scan_path
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    SourceLocation,
    SuggestedRange,
    TunedVariableCandidate,
)

AGENT_SOURCE = """\
from langchain_openai import ChatOpenAI


def answer_question(question: str) -> str:
    model = "gpt-4o-mini"
    temperature = 0.7
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm.invoke(question).content
"""


def test_cli_package_lazy_exports_cli() -> None:
    assert cli_package.cli is cli


def _write_agent(tmp_path: Path) -> Path:
    source = tmp_path / "agent.py"
    source.write_text(AGENT_SOURCE, encoding="utf-8")
    return source


def test_infer_project_root_walks_up_to_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("project docs", encoding="utf-8")
    app_dir = tmp_path / "app"
    app_dir.mkdir()

    assert infer_project_root(app_dir) == tmp_path


def test_scan_path_emits_schema_shaped_report(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)

    report = scan_path(source)

    assert report["report_version"] == "0.1.0"
    assert report["runtime"] == "python"
    assert report["agent_enrichment"] is None
    assert len(report["candidates"]) == 1

    candidate = report["candidates"][0]
    assert candidate["fingerprint"]["candidate_id"]
    assert len(candidate["fingerprint"]["source_hash"]) == 64
    assert len(candidate["fingerprint"]["source_span_hash"]) == 64
    assert candidate["function"]["name"] == "answer_question"
    assert {signal["kind"] for signal in candidate["signals"]} == {"llm_call"}
    tvar_names = {signal["tvar"]["name"] for signal in candidate["tvar_signals"]}
    assert tvar_names >= {"model", "temperature"}
    assert "question" not in tvar_names
    assert {signal["domain_source"] for signal in candidate["tvar_signals"]} == {
        "static"
    }

    objectives = {objective["name"]: objective for objective in candidate["objective_candidates"]}
    assert objectives["accuracy"]["requires_confirmation"] is True
    assert objectives["accuracy"]["auto_measurable"] is False
    assert objectives["accuracy"]["source"] == "static"
    assert objectives["cost"]["requires_confirmation"] is False
    assert objectives["latency"]["auto_measurable"] is True


def test_scan_path_directory_handles_syntax_errors_and_function_filter(
    tmp_path: Path,
) -> None:
    source = _write_agent(tmp_path)
    (tmp_path / "broken.py").write_text("def nope(:\n", encoding="utf-8")

    report = scan_path(tmp_path, function_name="answer_question")
    empty_report = scan_path(source, function_name="missing_function")

    assert len(report["candidates"]) == 1
    assert empty_report["candidates"] == []


def test_scan_path_detects_sibling_dataset_and_marks_singleton_confidence_low(
    tmp_path: Path,
) -> None:
    source = _write_agent(tmp_path)
    (tmp_path / "eval.jsonl").write_text(
        '{"question":"q","expected_output":"a"}\n',
        encoding="utf-8",
    )

    report = scan_path(source)
    plan = build_decorate_plan(source, function_name="answer_question")

    candidate = report["candidates"][0]
    assert candidate["dataset_status"] == {
        "status": "present",
        "candidate_path": "eval.jsonl",
    }
    by_name = {
        signal["tvar"]["name"]: signal for signal in candidate["tvar_signals"]
    }
    assert by_name["model"]["confidence"] == "low"
    assert plan["dataset_plan"]["status"] == "present"
    assert plan["dataset_plan"]["dataset_ref"] == "eval.jsonl"
    assert any("singleton enum" in warning for warning in plan["warnings"])
    assert any("missing expected fields" in warning for warning in plan["warnings"])


def test_scan_path_detects_retrieval_and_provider_call_shapes(tmp_path: Path) -> None:
    source = tmp_path / "providers.py"
    source.write_text(
        dedent(
            """\
            import litellm

            class SupportAgent:
                async def retrieve(self, query):
                    top_k = 4
                    return self.vectorstore.similarity_search(query, k=top_k)

            def call_openai(client, prompt):
                return client.chat.completions.create(model="gpt-4o-mini", input=prompt)

            def call_anthropic(client, prompt):
                return client.messages.create(model="claude-sonnet-4-6", messages=[])

            def call_litellm(prompt):
                return litellm.completion(model="gpt-4o-mini", messages=[])
            """
        ),
        encoding="utf-8",
    )

    report = scan_path(source)

    by_name = {
        candidate["function"]["name"]: candidate for candidate in report["candidates"]
    }
    assert by_name["retrieve"]["function"]["qualified_name"].endswith(
        "SupportAgent.retrieve"
    )
    assert {
        objective["name"] for objective in by_name["retrieve"]["objective_candidates"]
    } >= {"recall_at_k", "latency"}
    frameworks = {
        signal["framework"]
        for candidate in report["candidates"]
        for signal in candidate["signals"]
    }
    assert frameworks >= {"openai", "anthropic", "litellm"}


def test_scanner_candidate_conversion_covers_domain_shapes() -> None:
    def candidate(
        name: str,
        candidate_type: CandidateType,
        *,
        current_value=None,
        suggested_range: SuggestedRange | None = None,
    ) -> TunedVariableCandidate:
        return TunedVariableCandidate(
            name=name,
            candidate_type=candidate_type,
            confidence=DetectionConfidence.HIGH,
            location=SourceLocation(line=1, col_offset=0),
            current_value=current_value,
            suggested_range=suggested_range,
        )

    int_tvar = scanner_module._candidate_to_tvar(
        candidate(
            "top_k",
            CandidateType.NUMERIC_INTEGER,
            suggested_range=SuggestedRange("IntRange", {"low": 1, "high": 10}),
        )
    )
    bool_tvar = scanner_module._candidate_to_tvar(
        candidate("rerank", CandidateType.BOOLEAN, current_value=True)
    )
    enum_tvar = scanner_module._candidate_to_tvar(
        candidate(
            "model",
            CandidateType.CATEGORICAL,
            suggested_range=SuggestedRange(
                "Choices",
                {"values": ["gpt-4o-mini", "claude-sonnet-4-6"]},
            ),
        )
    )
    log_tvar = scanner_module._candidate_to_tvar(
        candidate(
            "threshold",
            CandidateType.NUMERIC_CONTINUOUS,
            suggested_range=SuggestedRange("LogRange", {"low": 0.001, "high": 1.0}),
        )
    )
    malformed = scanner_module._candidate_to_tvar(
        candidate(
            "bad_range",
            CandidateType.NUMERIC_CONTINUOUS,
            suggested_range=SuggestedRange("Range", {"low": 0.0}),
        )
    )

    assert int_tvar["type"] == "int"
    assert int_tvar["domain"] == {"range": [1, 10]}
    assert bool_tvar["type"] == "bool"
    assert bool_tvar["default"] is True
    assert enum_tvar["type"] == "enum"
    assert enum_tvar["domain"]["values"] == ["gpt-4o-mini", "claude-sonnet-4-6"]
    assert log_tvar["scale"] == "log"
    assert "domain" not in malformed
    assert not scanner_module._is_actionable_tvar_candidate(
        candidate("unused", CandidateType.CATEGORICAL)
    )


def test_dataset_helpers_cover_formats_and_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("question,expected_output\nq,a\n", encoding="utf-8")
    jsonl_path = tmp_path / "eval.jsonl"
    jsonl_path.write_text('\n{"question":"q","expected_output":"a"}\n', encoding="utf-8")
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("{not-json}\n", encoding="utf-8")

    assert _dataset_format("eval.csv") == "csv"
    assert _dataset_format("eval.parquet") == "parquet"
    assert _dataset_format("hf://org/dataset") == "hf_dataset"
    assert _dataset_format("eval.txt") == "other"
    assert _read_dataset_fields(csv_path) == ["question", "expected_output"]
    assert _read_dataset_fields(jsonl_path) == ["question", "expected_output"]
    assert _read_dataset_fields(bad_path) == []
    assert _dataset_field_warning(
        {
            "status": "present",
            "dataset_ref": "eval.csv",
            "expected_fields": ["question", "relevant_doc_ids"],
        },
        tmp_path,
    )
    assert (
        _dataset_field_warning(
            {
                "status": "present",
                "dataset_ref": "hf://org/dataset",
                "expected_fields": ["question"],
            },
            tmp_path,
        )
        is None
    )


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
    assert plan["agent_enrichment"] is None
    assert plan["confirmation_state"]["objectives_confirmed"] is True
    assert plan["dataset_plan"]["status"] == "present"
    assert plan["dataset_plan"]["dataset_ref"] == "eval/qa.jsonl"

    first_binding = plan["proposed_tvar_bindings"][0]
    assert first_binding["domain_source"] == "static"
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
    assert report["candidates"][0]["tvar_signals"][0]["domain_source"] == "static"


def test_optimizer_scan_cli_prints_summary_and_writes_output(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    output_path = tmp_path / "scan.json"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["optimizer", "scan", str(source), "--top", "1", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote scan report" in result.output
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(written["candidates"]) == 1


def test_optimizer_scan_cli_accepts_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _write_agent(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "scan",
            str(source),
            "--agent",
            "auto",
            "--project-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["agent_enrichment"][0]["status"] == "skipped"


def test_optimizer_scan_cli_top_zero_prints_empty_summary(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["optimizer", "scan", str(source), "--top", "0"])

    assert result.exit_code == 0, result.output
    assert "No optimizer candidates detected" in result.output


def test_optimizer_scan_cli_reports_value_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _write_agent(tmp_path)

    def fail_scan(*_args, **_kwargs):
        raise ValueError("bad scan")

    monkeypatch.setattr(optimizer_command, "scan_path", fail_scan)
    result = CliRunner().invoke(cli, ["optimizer", "scan", str(source)])

    assert result.exit_code != 0
    assert "bad scan" in result.output


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
    assert plan["agent_enrichment"] is None
    assert plan["selected_objectives"] == [
        {"name": "accuracy", "direction": "maximize"}
    ]


def test_optimizer_decorate_cli_prints_summary_and_writes_output(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    output_path = tmp_path / "decorate.json"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "answer_question",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Decorate plan" in result.output
    assert "Warning" in result.output
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["target"]["function"].endswith("answer_question")


def test_optimizer_decorate_cli_prints_agent_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _write_agent(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "answer_question",
            "--agent",
            "auto",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Agent:" in result.output
    assert "Agent warning:" in result.output


def test_optimizer_decorate_cli_write_is_rejected(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "answer_question",
            "--write",
        ],
    )

    assert result.exit_code != 0
    assert "--write is not implemented" in result.output


def test_optimizer_decorate_cli_reports_missing_candidate(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "missing_function",
        ],
    )

    assert result.exit_code != 0
    assert "No optimizer candidate found" in result.output


def test_optimizer_decorate_auto_emit_defaults_to_tvl(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)

    plan = build_decorate_plan(source, function_name="answer_question")

    assert plan["requested_emit_mode"] == "auto"
    assert plan["resolved_emit_mode"] == "tvl"


def test_optimizer_decorate_cli_runs_command_agent(tmp_path: Path) -> None:
    source = _write_agent(tmp_path)
    agent = tmp_path / "agent_response.py"
    agent.write_text(
        dedent(
            """\
            import json

            print(json.dumps({
                "response_version": "0.1.0",
                "context_confidence": "high",
                "tvar_recommendations": [{
                    "tvar": {
                        "name": "temperature",
                        "type": "float",
                        "domain": {"range": [0.1, 1.0], "resolution": 0.1},
                        "default": 0.7,
                    },
                    "confidence": "high",
                    "domain_intent": "search_space",
                    "current_value": 0.7,
                    "evidence": {
                        "file": "agent.py",
                        "line": 5,
                        "snippet": "temperature = 0.7",
                        "category": "literal_assignment",
                    },
                    "locator": {
                        "kind": "line_col",
                        "details": {
                            "function": "answer_question",
                            "line": 5,
                            "tvar": "temperature",
                        },
                    },
                    "rationale": "temperature affects output variability",
                }],
                "objective_recommendations": [],
                "warnings": [],
            }))
            """
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "decorate",
            str(source),
            "--function",
            "answer_question",
            "--agent",
            "command",
            "--agent-command",
            f"{sys.executable} {agent}",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["agent_enrichment"]["provider"] == "command"
    assert plan["proposed_tvar_bindings"][1]["domain_source"] == "agent"


def test_optimizer_scan_cli_falls_back_when_auto_agent_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _write_agent(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "optimizer",
            "scan",
            str(source),
            "--agent",
            "auto",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["agent_enrichment"][0]["status"] == "skipped"
