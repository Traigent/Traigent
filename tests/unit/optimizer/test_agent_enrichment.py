"""Unit tests for optimizer coding-agent enrichment adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from traigent.optimizer.agent_enrichment import (
    AgentRunConfig,
    ClaudeCodeAdapter,
    CodexAdapter,
    CommandAdapter,
    GitHubModelsAdapter,
    build_agent_prompt,
    enrich_decorate_plan,
    enrich_scan_report,
    load_agent_recommendation_schema,
    merge_agent_recommendations,
    parse_and_validate_agent_response,
    select_adapter,
)

VALID_RESPONSE: dict[str, Any] = {
    "response_version": "0.1.0",
    "context_confidence": "high",
    "tvar_recommendations": [
        {
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
            "rationale": "temperature affects answer variability.",
        }
    ],
    "objective_recommendations": [
        {
            "name": "groundedness",
            "direction": "maximize",
            "confidence": "medium",
            "rationale": "The answer should be grounded in retrieved context.",
            "required_dataset_fields": ["input", "expected_output"],
            "auto_measurable": False,
            "requires_confirmation": True,
        }
    ],
    "warnings": [],
}


def _plan() -> dict[str, Any]:
    return {
        "plan_version": "0.1.0",
        "runtime": "python",
        "tool_version": "traigent==test",
        "generated_at": "2026-05-03T00:00:00Z",
        "target": {
            "file": "agent.py",
            "function": "agent.answer_question",
            "line": 4,
            "candidate_id": "abc-agent_answer_question",
            "source_hash": "0" * 64,
            "source_span_hash": "1" * 64,
        },
        "requested_emit_mode": "auto",
        "resolved_emit_mode": "tvl",
        "injection_mode": "context",
        "proposed_tvar_bindings": [
            {
                "tvar": {
                    "name": "temperature",
                    "type": "float",
                    "domain": {"range": [0.0, 2.0]},
                    "default": 0.7,
                },
                "confidence": "high",
                "domain_source": "static",
                "evidence": {
                    "file": "agent.py",
                    "line": 5,
                    "snippet": "temperature = 0.7",
                    "category": "literal_assignment",
                },
                "injection_mode": "context",
                "current_value": 0.7,
                "locator": {
                    "kind": "line_col",
                    "details": {
                        "function": "answer_question",
                        "line": 5,
                        "tvar": "temperature",
                    },
                },
            }
        ],
        "selected_objectives": [],
        "objective_candidates": [
            {
                "name": "accuracy",
                "direction": "maximize",
                "confidence": "medium",
                "source": "static",
                "rationale": "LLM quality should be measured.",
                "required_dataset_fields": ["input", "expected_output"],
                "auto_measurable": False,
                "requires_confirmation": True,
            }
        ],
        "dataset_plan": {
            "status": "stub_required",
            "format": "jsonl",
            "stub_path": "answer_question_dataset.jsonl",
            "expected_fields": ["input", "expected_output"],
        },
        "agent_enrichment": None,
        "emitted_files": [],
        "confirmation_state": {
            "objectives_confirmed": False,
            "dataset_confirmed": False,
            "write_authorized": False,
        },
        "warnings": [],
    }


class RecordingRunner:
    def __init__(self, stdout: str | None = None, returncode: int = 0) -> None:
        self.stdout = stdout if stdout is not None else json.dumps(VALID_RESPONSE)
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self,
        args: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        stdin: int | None,
        text: bool,
        capture_output: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            (
                args,
                {
                    "cwd": cwd,
                    "env": env,
                    "stdin": stdin,
                    "text": text,
                    "capture_output": capture_output,
                    "timeout": timeout,
                    "check": check,
                },
            )
        )
        return subprocess.CompletedProcess(
            args,
            self.returncode,
            stdout=self.stdout,
            stderr="" if self.returncode == 0 else "failed",
        )


def test_parse_and_validate_agent_response_accepts_wrapper_json() -> None:
    wrapped = json.dumps({"result": json.dumps(VALID_RESPONSE)})

    payload, errors = parse_and_validate_agent_response(wrapped)

    assert errors == []
    assert payload is not None
    assert payload["context_confidence"] == "high"


def test_parse_and_validate_agent_response_accepts_claude_structured_output() -> None:
    wrapped = json.dumps({"structured_output": VALID_RESPONSE})

    payload, errors = parse_and_validate_agent_response(wrapped)

    assert errors == []
    assert payload is not None
    assert payload["context_confidence"] == "high"


def test_parse_and_validate_agent_response_rejects_bad_payload() -> None:
    payload, errors = parse_and_validate_agent_response('{"context_confidence":"none"}')

    assert payload is None
    assert errors


def test_parse_and_validate_agent_response_rejects_missing_evidence_line() -> None:
    response = {
        **VALID_RESPONSE,
        "tvar_recommendations": [
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "evidence": {
                    "file": "agent.py",
                    "snippet": "temperature = 0.7",
                    "category": "literal_assignment",
                },
            }
        ],
        "objective_recommendations": [],
    }

    payload, errors = parse_and_validate_agent_response(json.dumps(response))

    assert payload is None
    assert any("line" in error for error in errors)


def test_parse_and_validate_agent_response_normalizes_unknown_evidence_category() -> None:
    response = {
        **VALID_RESPONSE,
        "tvar_recommendations": [
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "evidence": {
                    **VALID_RESPONSE["tvar_recommendations"][0]["evidence"],
                    "category": "project_context",
                },
            }
        ],
        "objective_recommendations": [],
    }

    payload, errors = parse_and_validate_agent_response(json.dumps(response))

    assert errors == []
    assert payload is not None
    evidence = payload["tvar_recommendations"][0]["evidence"]
    assert evidence["category"] == "other"
    assert evidence["detail"] == "project_context"


def test_parse_and_validate_agent_response_normalizes_domain_step() -> None:
    response = {
        **VALID_RESPONSE,
        "tvar_recommendations": [
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "tvar": {
                    **VALID_RESPONSE["tvar_recommendations"][0]["tvar"],
                    "domain": {"range": [0.0, 1.0], "step": 0.1},
                },
            }
        ],
        "objective_recommendations": [],
    }

    payload, errors = parse_and_validate_agent_response(json.dumps(response))

    assert errors == []
    assert payload is not None
    assert payload["tvar_recommendations"][0]["tvar"]["domain"]["resolution"] == 0.1


def test_parse_and_validate_agent_response_normalizes_tvar_level_resolution() -> None:
    response = {
        **VALID_RESPONSE,
        "tvar_recommendations": [
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "tvar": {
                    "name": "temperature",
                    "type": "float",
                    "domain": {"range": [0.0, 1.0]},
                    "resolution": 0.1,
                    "default": 0.7,
                },
            }
        ],
        "objective_recommendations": [],
    }

    payload, errors = parse_and_validate_agent_response(json.dumps(response))

    assert errors == []
    assert payload is not None
    tvar = payload["tvar_recommendations"][0]["tvar"]
    assert "resolution" not in tvar
    assert tvar["domain"]["resolution"] == 0.1


def test_merge_agent_recommendations_updates_domains_and_objectives() -> None:
    plan = _plan()

    warnings = merge_agent_recommendations(plan, VALID_RESPONSE)

    assert warnings == []
    binding = plan["proposed_tvar_bindings"][0]
    assert binding["domain_source"] == "agent"
    assert binding["tvar"]["domain"] == {"range": [0.1, 1.0], "resolution": 0.1}
    objectives = {item["name"]: item for item in plan["objective_candidates"]}
    assert objectives["groundedness"]["source"] == "agent"


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        (
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "domain_intent": "current_only",
            },
            "current_only",
        ),
        (
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "tvar": {
                    "name": "temperature",
                    "type": "float",
                    "default": 0.7,
                },
            },
            "empty or degenerate search domain",
        ),
        (
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "tvar": {
                    "name": "temperature",
                    "type": "float",
                    "domain": {"range": [0.7, 0.7]},
                    "default": 0.7,
                },
            },
            "empty or degenerate search domain",
        ),
        (
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "current_value": 2.0,
            },
            "outside proposed domain",
        ),
        (
            {
                **VALID_RESPONSE["tvar_recommendations"][0],
                "tvar": {
                    "name": "model",
                    "type": "enum",
                    "domain": {"values": ["gpt-4o-mini"]},
                    "default": "gpt-4o-mini",
                },
                "current_value": "gpt-4o-mini",
            },
            "singleton enum",
        ),
    ],
)
def test_merge_agent_recommendations_rejects_unsafe_tvars(
    recommendation: dict[str, Any],
    expected: str,
) -> None:
    payload = {
        **VALID_RESPONSE,
        "tvar_recommendations": [recommendation],
        "objective_recommendations": [],
    }

    warnings = merge_agent_recommendations(_plan(), payload)

    assert any(expected in warning for warning in warnings)


def test_merge_agent_recommendations_rejects_unseen_model_values() -> None:
    plan = _plan()
    plan["proposed_tvar_bindings"].append(
        {
            "tvar": {
                "name": "model",
                "type": "str",
                "default": "gpt-4o-mini",
            },
            "confidence": "high",
            "domain_source": "static",
            "evidence": {
                "file": "agent.py",
                "line": 4,
                "snippet": 'model = "gpt-4o-mini"',
                "category": "literal_assignment",
            },
            "injection_mode": "context",
            "current_value": "gpt-4o-mini",
            "locator": {
                "kind": "line_col",
                "details": {"function": "answer_question", "line": 4, "tvar": "model"},
            },
        }
    )
    payload = {
        **VALID_RESPONSE,
        "tvar_recommendations": [
            {
                "tvar": {
                    "name": "model",
                    "type": "enum",
                    "domain": {"values": ["gpt-4o-mini", "gpt-5-mega"]},
                    "default": "gpt-4o-mini",
                },
                "confidence": "high",
                "domain_intent": "search_space",
                "current_value": "gpt-4o-mini",
                "evidence": {
                    "file": "agent.py",
                    "line": 4,
                    "snippet": 'model = "gpt-4o-mini"',
                    "category": "literal_assignment",
                },
                "locator": {
                    "kind": "line_col",
                    "details": {
                        "function": "answer_question",
                        "line": 4,
                        "tvar": "model",
                    },
                },
                "rationale": "Model choice affects quality and cost.",
            }
        ],
        "objective_recommendations": [],
    }

    warnings = merge_agent_recommendations(plan, payload)

    assert any("model values were not seen" in warning for warning in warnings)
    model_binding = plan["proposed_tvar_bindings"][1]
    assert model_binding["domain_source"] == "static"


def test_merge_agent_recommendations_allows_model_values_from_context() -> None:
    plan = _plan()
    plan["proposed_tvar_bindings"].append(
        {
            "tvar": {
                "name": "model",
                "type": "str",
                "default": "gpt-4o-mini",
            },
            "confidence": "high",
            "domain_source": "static",
            "evidence": {
                "file": "agent.py",
                "line": 4,
                "snippet": 'model = "gpt-4o-mini"',
                "category": "literal_assignment",
            },
            "injection_mode": "context",
            "current_value": "gpt-4o-mini",
            "locator": {
                "kind": "line_col",
                "details": {"function": "answer_question", "line": 4, "tvar": "model"},
            },
        }
    )
    recommendation = {
        "tvar": {
            "name": "model",
            "type": "enum",
            "domain": {"values": ["gpt-4o-mini", "claude-sonnet-4-6"]},
            "default": "gpt-4o-mini",
        },
        "confidence": "high",
        "domain_intent": "search_space",
        "current_value": "gpt-4o-mini",
        "evidence": {
            "file": "agent.py",
            "line": 4,
            "snippet": 'model = "gpt-4o-mini"',
            "category": "literal_assignment",
        },
        "locator": {
            "kind": "line_col",
            "details": {"function": "answer_question", "line": 4, "tvar": "model"},
        },
        "rationale": "Both model names appear in project context.",
    }
    payload = {
        **VALID_RESPONSE,
        "tvar_recommendations": [recommendation],
        "objective_recommendations": [],
    }

    warnings = merge_agent_recommendations(
        plan,
        payload,
        context_text="README mentions claude-sonnet-4-6 as supported.",
    )

    assert warnings == []
    assert plan["proposed_tvar_bindings"][1]["domain_source"] == "agent"


def test_build_agent_prompt_truncates_to_budget(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x" * 1000, encoding="utf-8")

    prompt, summary = build_agent_prompt(
        static_payload=_plan(),
        function_source="def answer_question(): pass",
        source_path=tmp_path / "agent.py",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            budget_tokens=40,
        ),
    )

    assert len(prompt) < 220
    assert "budget_tokens=40" in summary


def test_command_adapter_passes_prompt_and_schema_env(tmp_path: Path) -> None:
    runner = RecordingRunner()
    adapter = CommandAdapter(runner)
    config = AgentRunConfig(mode="command", project_root=tmp_path, command="agent-bin")

    raw = adapter.run(prompt="PROMPT", schema={"type": "object"}, config=config)

    assert raw.provider == "command"
    args, kwargs = runner.calls[0]
    assert args == ["agent-bin"]
    assert kwargs["env"]["TRAIGENT_OPTIMIZER_AGENT_PROMPT"] == "PROMPT"
    assert "TRAIGENT_OPTIMIZER_AGENT_SCHEMA_JSON" in kwargs["env"]
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_command_adapter_reports_missing_executable(tmp_path: Path) -> None:
    adapter = CommandAdapter()

    raw = adapter.run(
        prompt="PROMPT",
        schema={"type": "object"},
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="definitely-not-a-traigent-agent",
        ),
    )

    assert raw.provider == "command"
    assert raw.response_text == ""
    assert raw.warnings


def test_claude_code_adapter_builds_read_only_structured_command(tmp_path: Path) -> None:
    runner = RecordingRunner()
    adapter = ClaudeCodeAdapter(runner)

    adapter.run(
        prompt="PROMPT",
        schema={"type": "object"},
        config=AgentRunConfig(mode="claude-code", project_root=tmp_path, model="sonnet"),
    )

    args = runner.calls[0][0]
    assert args[:2] == ["claude", "-p"]
    assert "--json-schema" not in args
    assert "--permission-mode" in args
    assert "--effort" in args
    assert "medium" in args
    assert "--allowedTools=Read,Grep,Glob" in args
    assert f"--add-dir={tmp_path}" in args
    assert args[-1] == "PROMPT"


def test_codex_adapter_builds_read_only_structured_command(tmp_path: Path) -> None:
    runner = RecordingRunner()
    adapter = CodexAdapter(runner)

    raw = adapter.run(
        prompt="PROMPT",
        schema={"type": "object"},
        config=AgentRunConfig(mode="codex", project_root=tmp_path, model="gpt-5.4"),
    )

    args = runner.calls[0][0]
    assert args[:2] == ["codex", "exec"]
    assert "--sandbox" in args
    assert "read-only" in args
    assert "--ephemeral" in args
    assert "--output-schema" in args
    assert "--ask-for-approval" not in args
    assert raw.provider == "codex"


def test_github_models_adapter_uses_gh_models_structured_prompt(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner(
        stdout="gh copilot\tgithub/gh-copilot\tv1.2.0\n"
        "gh models\tgithub/gh-models\tv0.0.25"
    )
    adapter = GitHubModelsAdapter(runner)

    raw = adapter.run(
        prompt="PROMPT",
        schema={"type": "object"},
        config=AgentRunConfig(mode="github-models", project_root=tmp_path),
    )

    args = runner.calls[0][0]
    assert args[:3] == ["gh", "models", "run"]
    assert "--temperature" in args
    assert raw.provider == "github-models"
    assert raw.agent_version == "gh models\tgithub/gh-models\tv0.0.25"
    assert raw.warnings


def test_enrich_decorate_plan_with_command_adapter_merges_and_records_provenance(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()

    plan = enrich_decorate_plan(
        _plan(),
        source_path=tmp_path / "agent.py",
        function_source="def answer_question(): pass",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
        ),
        runner=runner,
    )

    assert plan["agent_enrichment"]["status"] == "completed"
    assert plan["agent_enrichment"]["provider"] == "command"
    assert plan["agent_enrichment"]["validation_status"] == "valid"
    assert plan["proposed_tvar_bindings"][0]["domain_source"] == "agent"


def test_enrich_decorate_plan_marks_partial_when_policy_filters_some_recommendations(
    tmp_path: Path,
) -> None:
    skipped = {
        **VALID_RESPONSE["tvar_recommendations"][0],
        "tvar": {
            "name": "model",
            "type": "enum",
            "domain": {"values": ["gpt-4o-mini"]},
            "default": "gpt-4o-mini",
        },
        "current_value": "gpt-4o-mini",
    }
    runner = RecordingRunner(
        stdout=json.dumps(
            {
                **VALID_RESPONSE,
                "tvar_recommendations": [
                    VALID_RESPONSE["tvar_recommendations"][0],
                    skipped,
                ],
                "objective_recommendations": [],
            }
        )
    )

    plan = enrich_decorate_plan(
        _plan(),
        source_path=tmp_path / "agent.py",
        function_source="def answer_question(): pass",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
        ),
        runner=runner,
    )

    assert plan["agent_enrichment"]["validation_status"] == "partial"
    assert plan["proposed_tvar_bindings"][0]["domain_source"] == "agent"
    assert any("singleton enum" in warning for warning in plan["agent_enrichment"]["warnings"])


def test_enrich_decorate_plan_marks_rejected_by_policy_when_nothing_applies(
    tmp_path: Path,
) -> None:
    response = {
        **VALID_RESPONSE,
        "context_confidence": "low",
        "objective_recommendations": [],
    }
    runner = RecordingRunner(stdout=json.dumps(response))

    plan = enrich_decorate_plan(
        _plan(),
        source_path=tmp_path / "agent.py",
        function_source="def answer_question(): pass",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
        ),
        runner=runner,
    )

    assert plan["agent_enrichment"]["validation_status"] == "rejected_by_policy"
    assert plan["proposed_tvar_bindings"][0]["domain_source"] == "static"


def test_enrich_decorate_plan_rejects_invalid_agent_output(tmp_path: Path) -> None:
    runner = RecordingRunner(stdout="not-json")

    plan = enrich_decorate_plan(
        _plan(),
        source_path=tmp_path / "agent.py",
        function_source="def answer_question(): pass",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
        ),
        runner=runner,
    )

    assert plan["agent_enrichment"]["status"] == "rejected"
    assert plan["agent_enrichment"]["validation_status"] == "invalid"
    assert plan["proposed_tvar_bindings"][0]["domain_source"] == "static"


def test_enrich_decorate_plan_caps_large_agent_response(tmp_path: Path) -> None:
    runner = RecordingRunner(stdout="{" + ("x" * 600_000))

    plan = enrich_decorate_plan(
        _plan(),
        source_path=tmp_path / "agent.py",
        function_source="def answer_question(): pass",
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
        ),
        runner=runner,
    )

    warnings = plan["agent_enrichment"]["warnings"]
    assert any("exceeded" in warning for warning in warnings)
    assert plan["agent_enrichment"]["status"] == "rejected"


def test_enrich_scan_report_stops_at_total_timeout(tmp_path: Path) -> None:
    source = tmp_path / "agent.py"
    source.write_text("def answer_question():\n    return 'ok'\n", encoding="utf-8")
    report = {
        "report_version": "0.1.0",
        "runtime": "python",
        "scan_root": str(tmp_path),
        "generated_at": "2026-05-03T00:00:00Z",
        "tool_version": "traigent==test",
        "agent_enrichment": None,
        "candidates": [
            {
                "fingerprint": {
                    "candidate_id": "abc-agent_answer_question",
                    "source_hash": "0" * 64,
                    "source_span_hash": "1" * 64,
                },
                "function": {
                    "file": "agent.py",
                    "line": 1,
                    "end_line": 2,
                    "name": "answer_question",
                    "qualified_name": "agent.answer_question",
                },
                "score": 1.0,
                "signals": [],
                "tvar_signals": [],
                "objective_candidates": [],
                "dataset_status": {"status": "stub_required"},
            }
        ],
    }

    enriched = enrich_scan_report(
        report,
        config=AgentRunConfig(
            mode="command",
            project_root=tmp_path,
            command="agent-bin",
            total_timeout_seconds=0,
        ),
        runner=RecordingRunner(),
    )

    assert enriched["agent_enrichment"][0]["status"] == "skipped"


def test_select_adapter_auto_prefers_claude_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/bin/tool" if name == "claude" else None)

    adapter = select_adapter(AgentRunConfig(mode="auto"))

    assert isinstance(adapter, ClaudeCodeAdapter)


def test_select_adapter_static_returns_none() -> None:
    assert select_adapter(AgentRunConfig(mode="static")) is None


def test_load_agent_recommendation_schema() -> None:
    schema = load_agent_recommendation_schema()

    assert schema["title"] == "Optimizer Agent Recommendation Schema"
