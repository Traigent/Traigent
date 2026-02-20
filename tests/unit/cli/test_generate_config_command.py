"""Tests for CLI command: traigent generate-config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from traigent.cli.generate_config_command import generate_config
from traigent.config_generator.types import (
    AutoConfigResult,
    BenchmarkSpec,
    ObjectiveSpec,
    SafetySpec,
    TVarRecommendation,
    TVarSpec,
)

SAMPLE_SOURCE = """\
from langchain_openai import ChatOpenAI

def my_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return llm.invoke(query)
"""


def _make_result() -> AutoConfigResult:
    """Build a realistic AutoConfigResult for test output."""
    return AutoConfigResult(
        tvars=(
            TVarSpec(
                name="temperature",
                range_type="Range",
                range_kwargs={"low": 0.0, "high": 2.0},
                source="preset",
            ),
        ),
        objectives=(
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ),
        safety_constraints=(
            SafetySpec(
                metric_name="hallucination_rate",
                operator="<=",
                threshold=0.15,
                agent_type="general_llm",
            ),
        ),
        benchmarks=(
            BenchmarkSpec(name="General LLM Evaluation", description="General eval"),
        ),
        recommendations=(
            TVarRecommendation(
                name="prompting_strategy",
                range_type="Choices",
                range_kwargs={"values": ["direct", "chain_of_thought"]},
                category="prompting",
                reasoning="Different strategies can improve output quality",
                impact_estimate="medium",
            ),
        ),
        agent_type="general_llm",
    )


class TestGenerateConfigCLI:
    def test_table_output(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file)])
            assert result.exit_code == 0
            assert "temperature" in result.output
            assert "accuracy" in result.output
            assert "Tunable Variables" in result.output

    def test_python_output(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file), "-o", "python"])
            assert result.exit_code == 0
            assert "@traigent.optimize(" in result.output
            assert "temperature" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file), "-o", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["agent_type"] == "general_llm"
            assert len(data["tvars"]) == 1
            assert data["tvars"][0]["name"] == "temperature"

    def test_invalid_subsystem(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        runner = CliRunner()
        result = runner.invoke(
            generate_config, [str(source_file), "--only", "invalid_sub"]
        )
        assert result.exit_code == 1
        assert "Unknown subsystems" in result.output

    def test_only_tvars_subsystem(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=AutoConfigResult(
                tvars=(
                    TVarSpec(
                        name="temperature",
                        range_type="Range",
                        range_kwargs={"low": 0.0, "high": 2.0},
                    ),
                ),
            ),
        ) as mock_gen:
            runner = CliRunner()
            result = runner.invoke(
                generate_config, [str(source_file), "--only", "tvars"]
            )
            assert result.exit_code == 0
            # Verify subsystems kwarg was passed
            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs["subsystems"] == frozenset({"tvars"})

    def test_function_option(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=AutoConfigResult(),
        ) as mock_gen:
            runner = CliRunner()
            result = runner.invoke(
                generate_config,
                [str(source_file), "--function", "my_agent"],
            )
            assert result.exit_code == 0
            assert mock_gen.call_args.kwargs["function_name"] == "my_agent"

    def test_enrich_flag(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=AutoConfigResult(),
        ) as mock_gen:
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file), "--enrich"])
            assert result.exit_code == 0
            assert mock_gen.call_args.kwargs["enrich"] is True

    def test_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(generate_config, ["/nonexistent/path.py"])
        assert result.exit_code != 0
