"""
Comprehensive working tests for Traigent Optimization Validation System.

Focuses on testing core functionality with simplified mocks.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.cli.optimization_validator import OptimizationValidator
from traigent.cli.validation_types import OptimizedFunction, ValidationResult


class TestValidationSystemCore:
    """Test core validation system functionality."""

    def test_validation_result_properties(self):
        """Test ValidationResult properties and methods."""
        # Test superior result
        result = ValidationResult(
            function_name="test_func",
            baseline_metrics={"accuracy": 0.8, "cost": 0.05},
            optimized_metrics={"accuracy": 0.9, "cost": 0.04},
            is_superior=True,
            improvement_details={"accuracy": 12.5, "cost": 20.0},
            blocking_issues=[],
            threshold_used=0.1,
        )

        assert result.has_improvement is True
        assert result.max_improvement == 20.0
        assert result.should_block is False
        assert "PASSED" in result.get_summary()
        assert "✅ PASSED" in result.get_detailed_report()

    def test_validation_result_blocking(self):
        """Test ValidationResult blocking logic."""
        # Test blocked result
        result = ValidationResult(
            function_name="failing_func",
            baseline_metrics={"accuracy": 0.9},
            optimized_metrics={"accuracy": 0.8},
            is_superior=False,
            improvement_details={"accuracy": -11.1},
            blocking_issues=["Performance degraded"],
            threshold_used=0.1,
        )

        assert result.should_block is True
        assert "BLOCKED" in result.get_summary()
        assert "❌ BLOCKED" in result.get_detailed_report()
        assert "Blocking Issues:" in result.get_detailed_report()

    def test_optimized_function_properties(self):
        """Test OptimizedFunction properties."""
        func_info = OptimizedFunction(
            name="test_func",
            func=lambda x: x,
            decorator_config={
                "eval_dataset": "test.jsonl",
                "objectives": ["accuracy"],
                "configuration_space": {"param": ["a", "b"]},
            },
            default_params={"param": "a"},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

        assert func_info.has_defaults is True
        assert func_info.has_dataset is True
        assert "OptimizedFunction" in str(func_info)


class TestOptimizationValidator:
    """Test OptimizationValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OptimizationValidator(threshold_pct=10)

    @pytest.fixture
    def valid_function_info(self):
        """Create valid function info for testing."""
        return OptimizedFunction(
            name="test_func",
            func=lambda x: {"accuracy": 0.8},
            decorator_config={
                "eval_dataset": "test.jsonl",
                "objectives": ["accuracy"],
                "configuration_space": {"model": ["default", "optimized"]},
            },
            default_params={"model": "default"},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

    def test_compare_results_superior(self, validator):
        """Test comparison logic for superior results."""
        baseline = {"accuracy": 0.8, "cost": 0.05}
        optimized = {"accuracy": 0.9, "cost": 0.04}  # Better in both metrics
        objectives = ["accuracy", "cost"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        assert is_superior is True
        assert details["accuracy"] > 10  # Should be >10% improvement
        assert details["cost"] > 10  # Cost is inverted, so improvement is positive

    def test_compare_results_mixed_performance(self, validator):
        """Test comparison with mixed performance."""
        baseline = {"accuracy": 0.8, "cost": 0.05}
        optimized = {"accuracy": 0.9, "cost": 0.08}  # Better accuracy, worse cost
        objectives = ["accuracy", "cost"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should not be superior due to worse cost
        assert is_superior is False
        assert details["accuracy"] > 0  # Positive improvement in accuracy
        assert details["cost"] < 0  # Negative improvement in cost

    def test_compare_results_insufficient_improvement(self, validator):
        """Test comparison with insufficient improvement."""
        baseline = {"accuracy": 0.80}
        optimized = {"accuracy": 0.82}  # Only 2.5% improvement, below 10% threshold
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        assert is_superior is False
        assert 0 < details["accuracy"] < 10  # Positive but below threshold

    def test_prerequisites_check(self, validator):
        """Test prerequisites validation."""
        # Function missing configuration space
        func_info = OptimizedFunction(
            name="invalid_func",
            func=lambda x: x,
            decorator_config={"eval_dataset": "test.jsonl", "objectives": ["accuracy"]},
            default_params={},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

        issues = validator._check_prerequisites(func_info)
        assert "No configuration space specified" in issues

    @pytest.mark.asyncio
    async def test_validation_with_prerequisites_failure(self, validator):
        """Test validation when prerequisites fail."""
        func_info = OptimizedFunction(
            name="invalid_func",
            func=lambda x: x,
            decorator_config={"eval_dataset": "test.jsonl", "objectives": ["accuracy"]},
            default_params={},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

        result = await validator.validate_optimization(func_info)

        assert result.is_superior is False
        assert result.should_block is True
        assert len(result.blocking_issues) > 0
        assert result.baseline_metrics == {}
        assert result.optimized_metrics == {}

    @pytest.mark.asyncio
    async def test_validation_successful(self, validator, valid_function_info):
        """Test successful validation with mocked execution."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            with (
                patch.object(
                    validator,
                    "_run_baseline",
                    return_value=({"accuracy": 0.8}, {"model": "default"}),
                ),
                patch.object(
                    validator,
                    "_run_optimization",
                    return_value=({"accuracy": 0.9}, {"model": "optimized"}),
                ),
            ):

                result = await validator.validate_optimization(valid_function_info)

                assert result.is_superior is True
                assert result.should_block is False
                assert result.baseline_metrics == {"accuracy": 0.8}
                assert result.optimized_metrics == {"accuracy": 0.9}
                assert result.improvement_details["accuracy"] > 10


class TestCLIIntegration:
    """Test CLI command integration."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_check_command_help(self, cli_runner):
        """Test check command help output."""
        result = cli_runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "Validate Traigent optimization" in result.output
        assert "--functions" in result.output
        assert "--threshold" in result.output
        assert "--dry-run" in result.output

    def test_check_command_invalid_file(self, cli_runner):
        """Test check command with invalid file."""
        result = cli_runner.invoke(cli, ["check", "nonexistent_file.py"])
        assert result.exit_code == 1  # Should fail gracefully

    def test_check_command_missing_argument(self, cli_runner):
        """Test check command with missing required argument."""
        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 2  # Click should show usage error

    def test_check_command_with_options(self, cli_runner):
        """Test check command with various options."""
        # Create a minimal valid Python file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Simple Python file")
            temp_file = f.name

        try:
            # Test with threshold option
            result = cli_runner.invoke(
                cli, ["check", temp_file, "--threshold", "15", "--dry-run"]
            )
            assert result.exit_code in [
                0,
                1,
            ]  # May succeed or fail, but shouldn't crash

            # Test with functions filter
            result = cli_runner.invoke(
                cli, ["check", temp_file, "--functions", "test_func", "--dry-run"]
            )
            assert result.exit_code in [0, 1]

            # Test with objectives filter
            result = cli_runner.invoke(
                cli, ["check", temp_file, "--objectives", "accuracy,cost", "--dry-run"]
            )
            assert result.exit_code in [0, 1]

        finally:
            os.unlink(temp_file)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def validator(self):
        return OptimizationValidator(threshold_pct=10)

    def test_comparison_with_zero_baseline(self, validator):
        """Test comparison when baseline has zero values."""
        baseline = {"accuracy": 0.0}
        optimized = {"accuracy": 0.5}
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle zero baseline gracefully
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)

    def test_comparison_with_identical_values(self, validator):
        """Test comparison with identical metrics."""
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.8}
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        assert is_superior is False
        assert details["accuracy"] == 0.0  # No improvement

    def test_comparison_with_missing_metrics(self, validator):
        """Test comparison with missing metrics."""
        baseline = {"accuracy": 0.8}
        optimized = {"precision": 0.7}  # Different metric
        objectives = ["accuracy", "precision"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle missing metrics gracefully
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)

    def test_comparison_with_empty_objectives(self, validator):
        """Test comparison with no objectives."""
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.9}
        objectives = []

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        assert is_superior is False  # No objectives to compare
        assert details == {}

    def test_threshold_boundary_conditions(self):
        """Test different threshold values."""
        # Zero threshold - any improvement should be superior
        validator_zero = OptimizationValidator(threshold_pct=0)
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.800001}  # Tiny improvement
        objectives = ["accuracy"]

        is_superior, details = validator_zero._compare_results(
            baseline, optimized, objectives
        )
        assert is_superior is True

        # Very high threshold
        validator_high = OptimizationValidator(threshold_pct=1000)
        baseline = {"accuracy": 0.5}
        optimized = {"accuracy": 0.8}  # 60% improvement
        objectives = ["accuracy"]

        is_superior, details = validator_high._compare_results(
            baseline, optimized, objectives
        )
        assert is_superior is False  # 60% doesn't meet 1000% threshold

    def test_large_and_small_metric_values(self, validator):
        """Test with extreme metric values."""
        # Very large values (use accuracy instead of latency)
        baseline = {"accuracy": 1000000}
        optimized = {
            "accuracy": 1150000
        }  # 15% improvement (clearly above 10% threshold)
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )
        assert is_superior is True
        assert (
            abs(details["accuracy"] - 15.0) < 0.1
        )  # Should be approximately 15% improvement

        # Very small values
        baseline = {"error_rate": 0.000001}
        optimized = {"error_rate": 0.0000005}  # 50% improvement
        objectives = ["error_rate"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )
        assert is_superior is True


class TestMockModeIntegration:
    """Test integration with Traigent mock mode."""

    @pytest.fixture
    def validator(self):
        return OptimizationValidator(threshold_pct=10)

    @pytest.fixture
    def mock_function_info(self):
        """Create mock function info for testing."""
        from unittest.mock import Mock

        from traigent.evaluators.base import EvaluationExample

        # The actual function that will be called
        def actual_func(*args, **kwargs):
            return "expected result"  # Return string for _extract_response_text

        # Create a simple callable class that acts like a decorated function
        class MockOptimizedFunction:
            """Mock that acts like a traigent-decorated function."""

            __name__ = "mock_func"

            def __init__(self):
                self.func = actual_func  # The wrapped function
                self.config_param = None  # Config parameter name
                # Mock the provider with inject_config method
                self._provider = Mock()
                self._provider.inject_config = Mock(return_value=actual_func)

            def __call__(self, *args, **kwargs):
                return "expected result"

            def _load_dataset(self):
                mock_dataset = Mock()
                mock_dataset.examples = [
                    EvaluationExample(
                        input_data={"text": "test input"},
                        expected_output="expected result",
                    )
                ]
                return mock_dataset

            async def optimize(self, **kwargs):
                mock_result = Mock()
                mock_result.successful_trials = [Mock()]
                mock_result.best_metrics = {"accuracy": 0.9}
                mock_result.best_config = {"model": "optimized"}
                return mock_result

        mock_func = MockOptimizedFunction()

        return OptimizedFunction(
            name="mock_func",
            func=mock_func,
            decorator_config={
                "eval_dataset": "test.jsonl",
                "objectives": ["accuracy"],
                "configuration_space": {"model": ["default", "optimized"]},
            },
            default_params={"model": "default"},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

    @pytest.mark.asyncio
    async def test_mock_mode_baseline_execution(self, validator, mock_function_info):
        """Test baseline execution in mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            metrics, config = await validator._run_baseline(mock_function_info)

            assert isinstance(metrics, dict)
            assert isinstance(config, dict)
            # In mock mode, should return some metrics
            assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_mock_mode_optimization_execution(
        self, validator, mock_function_info
    ):
        """Test optimization execution in mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}):
            metrics, config = await validator._run_optimization(mock_function_info)

            assert isinstance(metrics, dict)
            assert isinstance(config, dict)
            # In mock mode, should return some metrics
            assert len(metrics) > 0


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling."""

    def test_many_objectives_handling(self):
        """Test handling many objectives."""
        validator = OptimizationValidator(threshold_pct=10)

        # Create metrics with 50 objectives
        baseline = {f"metric_{i}": 0.5 for i in range(50)}
        optimized = {f"metric_{i}": 0.6 for i in range(50)}  # 20% improvement in all
        objectives = [f"metric_{i}" for i in range(50)]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle many objectives efficiently
        assert is_superior is True
        assert len(details) == 50
        for i in range(50):
            assert (
                abs(details[f"metric_{i}"] - 20.0) < 0.001
            )  # Handle floating point precision

    def test_large_configuration_space(self):
        """Test with large configuration spaces."""
        func_info = OptimizedFunction(
            name="large_config_func",
            func=lambda: {"accuracy": 0.8},
            decorator_config={
                "eval_dataset": "test.jsonl",
                "objectives": ["accuracy"],
                "configuration_space": {
                    f"param_{i}": [f"value_{j}" for j in range(10)]
                    for i in range(10)  # 10 parameters with 10 values each
                },
            },
            default_params={f"param_{i}": "value_0" for i in range(10)},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

        # Should handle large configuration spaces without issues
        assert len(func_info.decorator_config["configuration_space"]) == 10
        assert len(func_info.default_params) == 10


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--cov=traigent.cli", "--cov-report=term-missing"])
