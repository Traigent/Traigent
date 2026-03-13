"""Tests for CLI command: traigent generate-config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.generate_config_command import _output_tvl, generate_config
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

    def test_tvl_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)
        monkeypatch.chdir(tmp_path)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file), "-o", "tvl"])
            assert result.exit_code == 0
            assert "TVL spec written to:" in result.output
            tvl_file = tmp_path / "agent.tvl.yml"
            assert tvl_file.exists()
            content = tvl_file.read_text()
            assert "temperature" in content

    def test_tvl_output_rejects_path_outside_working_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        monkeypatch.chdir(safe_dir)

        with pytest.raises(ValueError, match="outside the allowed base directory"):
            _output_tvl(_make_result(), source_file)

    def test_apply_decorator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)
        monkeypatch.chdir(tmp_path)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(
                generate_config,
                [str(source_file), "-f", "my_agent", "--apply"],
            )
            assert result.exit_code == 0
            assert "Applied @traigent.optimize()" in result.output
            assert "Backup saved to" in result.output
            # Verify .bak file created
            bak_file = tmp_path / "agent.py.bak"
            assert bak_file.exists()
            # Verify decorator inserted
            modified = source_file.read_text()
            assert "@traigent.optimize(" in modified

    def test_apply_no_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)
        monkeypatch.chdir(tmp_path)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(
                generate_config,
                [str(source_file), "-f", "my_agent", "--apply", "--no-backup"],
            )
            assert result.exit_code == 0
            assert "Applied @traigent.optimize()" in result.output
            assert "Backup saved to" not in result.output
            bak_file = tmp_path / "agent.py.bak"
            assert not bak_file.exists()

    def test_apply_requires_function(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=_make_result(),
        ):
            runner = CliRunner()
            result = runner.invoke(
                generate_config,
                [str(source_file), "--apply"],
            )
            assert result.exit_code == 1
            assert "--apply requires --function" in result.output

    def test_generate_error_handling(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            side_effect=RuntimeError("parse failure"),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file)])
            assert result.exit_code == 1
            assert "Error: parse failure" in result.output

    def test_generate_file_not_found(self, tmp_path: Path) -> None:
        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            side_effect=FileNotFoundError("missing"),
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file)])
            assert result.exit_code == 1
            assert "File not found" in result.output

    def test_table_output_full_sections(self, tmp_path: Path) -> None:
        """Test table output renders all sections including structural constraints."""
        from traigent.config_generator.types import StructuralConstraintSpec

        full_result = AutoConfigResult(
            tvars=(
                TVarSpec(
                    name="temperature",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 2.0},
                ),
            ),
            objectives=(
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ),
            safety_constraints=(
                SafetySpec(
                    metric_name="hallucination_rate",
                    operator="<=",
                    threshold=0.15,
                ),
            ),
            structural_constraints=(
                StructuralConstraintSpec(
                    description="Low temp constraint",
                    constraint_code="temperature.lte(1.0)",
                ),
            ),
            benchmarks=(
                BenchmarkSpec(name="QA Eval", description="Standard QA benchmark"),
            ),
            recommendations=(
                TVarRecommendation(
                    name="top_p",
                    range_type="Range",
                    range_kwargs={"low": 0.0, "high": 1.0},
                    reasoning="Explore top-p sampling",
                    impact_estimate="medium",
                ),
            ),
            warnings=("Watch out for cost",),
            agent_type="general_llm",
            llm_calls_made=3,
            llm_cost_usd=0.0042,
        )

        source_file = tmp_path / "agent.py"
        source_file.write_text(SAMPLE_SOURCE)

        with patch(
            "traigent.config_generator.generate_config",
            return_value=full_result,
        ):
            runner = CliRunner()
            result = runner.invoke(generate_config, [str(source_file)])
            assert result.exit_code == 0
            assert "Tunable Variables" in result.output
            assert "Objectives" in result.output
            assert "Safety Constraints" in result.output
            assert "Structural Constraints" in result.output
            assert "Benchmarks" in result.output
            assert "Recommended Additional TVars" in result.output
            assert "Warnings" in result.output
            assert "Watch out for cost" in result.output
            assert "LLM: 3 calls" in result.output
            assert "Agent type: general_llm" in result.output
