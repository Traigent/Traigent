"""CLI tests for the ``traigent detect-tvars`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.detect_tvars_command import detect_tvars

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_AGENT_SOURCE = """\
def answer_question(query: str) -> str:
    temperature = 0.7
    model = "gpt-4o"
    max_tokens = 1024
    return f"answer to {query}"
"""

SAMPLE_DATAFLOW_SOURCE = """\
def rag_pipeline(query: str) -> str:
    t = 0.7
    llm = ChatOpenAI(temperature=t, model="gpt-4o")
    docs = db.similarity_search(query, k=5)
    return llm.invoke(query)
"""

EMPTY_SOURCE = """\
# Nothing tunable here
x = 1 + 2
"""

SYNTAX_ERROR_SOURCE = """\
def broken(:
    pass
"""


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def agent_file(tmp_path: Path) -> Path:
    p = tmp_path / "agent.py"
    p.write_text(SAMPLE_AGENT_SOURCE)
    return p


@pytest.fixture()
def dataflow_file(tmp_path: Path) -> Path:
    p = tmp_path / "dataflow_agent.py"
    p.write_text(SAMPLE_DATAFLOW_SOURCE)
    return p


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    """Directory with two Python files."""
    d = tmp_path / "agents"
    d.mkdir()
    (d / "agent_a.py").write_text(SAMPLE_AGENT_SOURCE)
    (d / "agent_b.py").write_text(SAMPLE_DATAFLOW_SOURCE)
    return d


# ---------------------------------------------------------------------------
# Table output tests
# ---------------------------------------------------------------------------


class TestTableOutput:
    """Tests for the default Rich table output mode."""

    def test_scan_single_file(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_file)])
        assert result.exit_code == 0
        assert "temperature" in result.output
        assert "answer_question" in result.output

    def test_scan_directory_recursive(self, runner: CliRunner, agent_dir: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_dir)])
        assert result.exit_code == 0
        # Both files should produce output
        assert "answer_question" in result.output
        assert "rag_pipeline" in result.output

    def test_specific_function_flag(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(
            detect_tvars, [str(agent_file), "--function", "answer_question"]
        )
        assert result.exit_code == 0
        assert "temperature" in result.output

    def test_min_confidence_high(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(
            detect_tvars, [str(agent_file), "--min-confidence", "high"]
        )
        assert result.exit_code == 0
        # HIGH confidence candidates should be present
        # (temperature, model, max_tokens are exact name matches → HIGH)
        assert "temperature" in result.output

    def test_shows_suggested_range(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_file)])
        assert result.exit_code == 0
        # The config snippet should appear
        assert (
            "configuration_space" in result.output.lower() or "Range" in result.output
        )

    def test_nonexistent_function(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(
            detect_tvars, [str(agent_file), "--function", "nonexistent"]
        )
        # Should succeed but find nothing
        assert result.exit_code == 0
        assert "No tunable variable candidates" in result.output


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Tests for the --json output mode."""

    def test_json_valid(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_json_required_fields(self, runner: CliRunner, agent_file: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_file), "--json"])
        data = json.loads(result.output)
        required = {
            "function",
            "name",
            "type",
            "confidence",
            "line",
            "current_value",
            "reasoning",
            "detection_source",
        }
        for entry in data:
            assert required.issubset(
                entry.keys()
            ), f"Missing keys: {required - entry.keys()}"

    def test_json_has_suggested_range(
        self, runner: CliRunner, agent_file: Path
    ) -> None:
        result = runner.invoke(detect_tvars, [str(agent_file), "--json"])
        data = json.loads(result.output)
        # At least one candidate should have a suggested range
        ranges = [e for e in data if "suggested_range" in e]
        assert len(ranges) > 0
        sr = ranges[0]["suggested_range"]
        assert "range_type" in sr
        assert "kwargs" in sr
        assert "code" in sr

    def test_json_min_confidence_filter(
        self, runner: CliRunner, agent_file: Path
    ) -> None:
        result = runner.invoke(
            detect_tvars, [str(agent_file), "--json", "--min-confidence", "high"]
        )
        data = json.loads(result.output)
        for entry in data:
            assert entry["confidence"] == "high"

    def test_json_directory_scan(self, runner: CliRunner, agent_dir: Path) -> None:
        result = runner.invoke(detect_tvars, [str(agent_dir), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        functions = {e["function"] for e in data}
        assert "answer_question" in functions
        assert "rag_pipeline" in functions


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_nonexistent_path(self, runner: CliRunner) -> None:
        result = runner.invoke(detect_tvars, ["/nonexistent/path.py"])
        assert result.exit_code != 0

    def test_non_python_file(self, runner: CliRunner, tmp_path: Path) -> None:
        txt = tmp_path / "data.txt"
        txt.write_text("not python")
        result = runner.invoke(detect_tvars, [str(txt)])
        assert result.exit_code != 0
        assert ".py" in result.output

    def test_empty_file_no_crash(self, runner: CliRunner, tmp_path: Path) -> None:
        p = tmp_path / "empty.py"
        p.write_text("")
        result = runner.invoke(detect_tvars, [str(p)])
        assert result.exit_code == 0

    def test_syntax_error_no_crash(self, runner: CliRunner, tmp_path: Path) -> None:
        p = tmp_path / "bad.py"
        p.write_text(SYNTAX_ERROR_SOURCE)
        result = runner.invoke(detect_tvars, [str(p)])
        assert result.exit_code == 0

    def test_no_candidates_message(self, runner: CliRunner, tmp_path: Path) -> None:
        p = tmp_path / "boring.py"
        p.write_text(EMPTY_SOURCE)
        result = runner.invoke(detect_tvars, [str(p)])
        assert result.exit_code == 0
        assert "No tunable variable candidates" in result.output

    def test_empty_directory(self, runner: CliRunner, tmp_path: Path) -> None:
        d = tmp_path / "empty_dir"
        d.mkdir()
        result = runner.invoke(detect_tvars, [str(d)])
        assert result.exit_code == 0

    def test_json_empty_result(self, runner: CliRunner, tmp_path: Path) -> None:
        p = tmp_path / "boring.py"
        p.write_text(EMPTY_SOURCE)
        result = runner.invoke(detect_tvars, [str(p), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


# ---------------------------------------------------------------------------
# DataFlow-specific CLI tests
# ---------------------------------------------------------------------------


class TestDataFlowCLI:
    """Tests verifying DataFlow strategy results appear in CLI output."""

    def test_dataflow_candidates_in_json(
        self, runner: CliRunner, dataflow_file: Path
    ) -> None:
        result = runner.invoke(detect_tvars, [str(dataflow_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        names = {e["name"] for e in data}
        # DataFlow should detect 't' flowing to ChatOpenAI(temperature=t)
        assert "t" in names

    def test_dataflow_detection_source(
        self, runner: CliRunner, dataflow_file: Path
    ) -> None:
        result = runner.invoke(detect_tvars, [str(dataflow_file), "--json"])
        data = json.loads(result.output)
        sources = {e["detection_source"] for e in data}
        # Should have dataflow-detected candidates
        assert sources & {"dataflow", "combined"}, f"Got sources: {sources}"
