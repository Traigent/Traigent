"""Tests for CLI results commands (list, show, compare, rerank, export).

Tests cover:
- results list: List all optimization results
- results show: Show detailed view of a run
- results compare: Compare two runs
- results rerank: Re-score with different weights
- export: Export config for deployment
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_result():
    """Create a mock OptimizationResult with proper attribute access."""
    trial1 = Mock()
    trial1.trial_id = "trial_1"
    trial1.config = {"temperature": 0.5, "model": "gpt-4"}
    trial1.metrics = {"accuracy": 0.85, "cost": 0.01, "overall": 0.85}
    trial1.status = "completed"

    trial2 = Mock()
    trial2.trial_id = "trial_2"
    trial2.config = {"temperature": 0.7, "model": "gpt-3.5"}
    trial2.metrics = {"accuracy": 0.75, "cost": 0.005, "overall": 0.75}
    trial2.status = "completed"

    mock = Mock()
    mock.trials = [trial1, trial2]
    mock.successful_trials = [trial1, trial2]
    mock.best_config = {"temperature": 0.5, "model": "gpt-4"}
    mock.best_score = 0.85
    mock.best_metrics = {"accuracy": 0.85, "cost": 0.01}
    mock.stop_reason = "max_trials_reached"
    mock.duration = 10.5
    mock.algorithm = "tpe"
    mock.function_name = "test_func"
    return mock


class TestResultsList:
    """Tests for 'traigent results list' command."""

    def test_results_list_empty(self, runner):
        """Test results list with no results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, ["results", "list", "-d", tmpdir])
            # Should not error, just show empty or message
            assert result.exit_code == 0

    def test_results_list_with_results(self, runner, mock_result):
        """Test results list with existing results."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("traigent.cli.main.PersistenceManager") as mock_persistence_class,
        ):
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            # Return list of result metadata dictionaries
            mock_persistence.list_results.return_value = [
                {
                    "name": "test_run",
                    "function_name": "test_func",
                    "algorithm": "tpe",
                    "best_score": 0.85,
                    "total_trials": 2,
                    "success_rate": 1.0,
                    "created_at": "2024-01-01T00:00:00",
                }
            ]
            mock_persistence.load_result.return_value = mock_result

            result = runner.invoke(cli, ["results", "list", "-d", tmpdir])
            # Exit code 0 means success
            assert result.exit_code == 0


class TestResultsShow:
    """Tests for 'traigent results show' command."""

    def test_results_show_basic(self, runner, mock_result):
        """Test showing a specific result."""
        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.return_value = mock_result

            result = runner.invoke(cli, ["results", "show", "test_run"])

            assert result.exit_code == 0
            assert "Summary" in result.output
            assert "Best Configuration" in result.output

    def test_results_show_with_trials(self, runner, mock_result):
        """Test showing result with --trials flag."""
        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.return_value = mock_result

            result = runner.invoke(cli, ["results", "show", "test_run", "--trials"])

            assert result.exit_code == 0
            assert "All Trials" in result.output

    def test_results_show_not_found(self, runner):
        """Test showing non-existent result."""
        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.side_effect = FileNotFoundError()

            result = runner.invoke(cli, ["results", "show", "nonexistent"])

            assert result.exit_code == 0  # CLI handles error gracefully
            assert "not found" in result.output.lower()


class TestResultsCompare:
    """Tests for 'traigent results compare' command."""

    def test_results_compare_two_runs(self, runner, mock_result):
        """Test comparing two optimization runs."""
        mock_result2 = Mock()
        mock_result2.trials = mock_result.trials
        mock_result2.successful_trials = mock_result.trials
        mock_result2.best_config = {"temperature": 0.7, "model": "gpt-3.5"}
        mock_result2.best_score = 0.78
        mock_result2.best_metrics = {"accuracy": 0.78, "cost": 0.008}

        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.side_effect = [mock_result, mock_result2]

            result = runner.invoke(cli, ["results", "compare", "run1", "run2"])

            assert result.exit_code == 0
            assert "Comparing" in result.output
            assert "Best Score" in result.output


class TestResultsRerank:
    """Tests for 'traigent results rerank' command."""

    def test_results_rerank_basic(self, runner, mock_result):
        """Test reranking with custom weights."""
        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.return_value = mock_result

            result = runner.invoke(
                cli,
                ["results", "rerank", "test_run", "--weights", "accuracy=0.8,cost=0.2"],
            )

            assert result.exit_code == 0
            assert "Re-ranking" in result.output
            assert "Weights" in result.output

    def test_results_rerank_invalid_weights(self, runner):
        """Test reranking with invalid weights format."""
        result = runner.invoke(
            cli, ["results", "rerank", "test_run", "--weights", "invalid"]
        )

        # Should fail gracefully with error message
        assert "Invalid" in result.output or result.exit_code != 0


class TestExportCommand:
    """Tests for 'traigent export' command."""

    def test_export_slim_format(self, runner, mock_result):
        """Test exporting config in slim format."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("traigent.cli.main.PersistenceManager") as mock_persistence_class,
            patch("traigent.cli.main.WORKSPACE_ROOT", Path(tmpdir)),
        ):
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.return_value = mock_result

            output_path = Path(tmpdir) / "exported_config.json"
            result = runner.invoke(
                cli,
                ["export", "test_run", "-o", str(output_path)],
            )

            assert result.exit_code == 0
            assert "Exported" in result.output
            assert output_path.exists()

            # Verify exported content
            with open(output_path) as f:
                data = json.load(f)
            assert "config" in data

    def test_export_full_format(self, runner, mock_result):
        """Test exporting config in full format."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("traigent.cli.main.PersistenceManager") as mock_persistence_class,
            patch("traigent.cli.main.WORKSPACE_ROOT", Path(tmpdir)),
        ):
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.return_value = mock_result

            output_path = Path(tmpdir) / "full_config.json"
            result = runner.invoke(
                cli,
                ["export", "test_run", "-o", str(output_path), "--format", "full"],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_export_not_found(self, runner):
        """Test exporting non-existent result."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("traigent.cli.main.PersistenceManager") as mock_persistence_class,
        ):
            mock_persistence = Mock()
            mock_persistence_class.return_value = mock_persistence
            mock_persistence.load_result.side_effect = FileNotFoundError()

            output_path = Path(tmpdir) / "config.json"
            result = runner.invoke(
                cli,
                ["export", "nonexistent", "-o", str(output_path)],
            )

            assert "not found" in result.output.lower()


class TestDecoratorParameters:
    """Tests for new decorator parameters (auto_load_best, load_from)."""

    def test_auto_load_best_parameter_exists(self):
        """Verify auto_load_best parameter is available in decorator."""
        import traigent

        # The parameter should be accepted without error
        @traigent.optimize(
            configuration_space={"x": [1, 2, 3]},
            objectives=["score"],
            auto_load_best=True,
        )
        def test_func(x: int) -> int:
            return x

        assert hasattr(test_func, "_auto_load_best")

    def test_load_from_parameter_exists(self):
        """Verify load_from parameter is available in decorator."""
        import traigent

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"config": {"x": 2}}, f)
            config_path = f.name

        try:

            @traigent.optimize(
                configuration_space={"x": [1, 2, 3]},
                objectives=["score"],
                load_from=config_path,
            )
            def test_func(x: int) -> int:
                return x

            assert hasattr(test_func, "_load_from")
        finally:
            Path(config_path).unlink(missing_ok=True)
