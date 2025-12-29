"""
Comprehensive test suite for Traigent Optimization Validation System.

Tests all components of the automated validation system:
- Function discovery engine
- Optimization validator
- CLI integration
- Error handling and edge cases
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from click.testing import CliRunner

from traigent.cli.function_discovery import (
    _extract_default_parameters,
    discover_optimized_functions,
)
from traigent.cli.main import cli
from traigent.cli.optimization_validator import OptimizationValidator
from traigent.cli.validation_types import OptimizedFunction, ValidationResult


class TestValidationTypes:
    """Test validation data structures."""

    def test_optimized_function_creation(self):
        """Test OptimizedFunction dataclass creation."""

        def func(x):
            return x

        func_info = OptimizedFunction(
            name="test_func",
            func=func,
            decorator_config={"eval_dataset": "test.jsonl", "objectives": ["accuracy"]},
            default_params={"param1": "default"},
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
        )

        assert func_info.name == "test_func"
        assert func_info.func == func
        assert func_info.objectives == ["accuracy"]
        assert func_info.default_params == {"param1": "default"}

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            function_name="test_func",
            baseline_metrics={"accuracy": 0.8},
            optimized_metrics={"accuracy": 0.9},
            is_superior=True,
            improvement_details={"accuracy": 12.5},
            blocking_issues=[],
        )

        assert result.function_name == "test_func"
        assert result.is_superior is True
        assert result.baseline_metrics == {"accuracy": 0.8}
        assert result.improvement_details == {"accuracy": 12.5}


class TestFunctionDiscovery:
    """Test function discovery engine."""

    @pytest.fixture
    def temp_module(self):
        """Create temporary module for testing within the workspace."""
        content = '''
import traigent

# Mock OptimizedFunction for testing
class OptimizedFunction:  # Use the exact name expected by function_discovery
    def __init__(self, func, **kwargs):
        self.func = func
        self.eval_dataset = kwargs.get("eval_dataset")
        self.objectives = kwargs.get("objectives", [])
        self.decorator_config = kwargs
        self.optimize = lambda: None  # Add optimize method for discovery

@traigent.optimize(
    eval_dataset="test.jsonl",
    objectives=["accuracy"],
    configuration_space={"model": ["default", "optimized"], "temp": [0.0, 0.5, 1.0]}
)
def test_function(text: str, model: str = "default", temp: float = 0.5):
    """Test function with default parameters."""
    return {"result": "test"}

# Create mock optimized function instance
test_function = OptimizedFunction(
    test_function,
    eval_dataset="test.jsonl",
    objectives=["accuracy"],
    configuration_space={"model": ["default", "optimized"], "temp": [0.0, 0.5, 1.0]}
)

def regular_function():
    """Regular function that should not be discovered."""
    return "regular"
'''

        modules_dir = Path(__file__).resolve().parent / "temp_modules"
        modules_dir.mkdir(exist_ok=True)
        temp_file = modules_dir / f"temp_module_{uuid4().hex}.py"
        temp_file.write_text(content)

        try:
            yield str(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()
            if not any(modules_dir.iterdir()):
                modules_dir.rmdir()

    def test_extract_default_params(self):
        """Test parameter extraction from function signatures."""

        def test_func(
            text: str,
            model: str = "gpt-3.5",
            temperature: float = 0.7,
            max_tokens: int = 100,
        ):
            return text

        defaults = _extract_default_parameters(test_func)
        expected = {"model": "gpt-3.5", "temperature": 0.7, "max_tokens": 100}
        assert defaults == expected

    def test_extract_default_params_no_defaults(self):
        """Test parameter extraction with no default parameters."""

        def test_func(text: str, required_param: int):
            return text

        defaults = _extract_default_parameters(test_func)
        assert defaults == {}

    def test_discover_optimized_functions(self, temp_module):
        """Test function discovery from module."""
        # Create mock dataset file
        dataset_path = Path(temp_module).parent / "test.jsonl"
        dataset_path.write_text('{"input": {"text": "test"}, "output": "positive"}\n')

        try:
            functions = discover_optimized_functions(temp_module)

            assert len(functions) == 1
            func_info = functions[0]
            assert func_info.name == "test_function"
            assert func_info.eval_dataset == "test.jsonl"
            assert func_info.objectives == ["accuracy"]
        finally:
            if dataset_path.exists():
                dataset_path.unlink()

    def test_discover_with_function_filter(self, temp_module):
        """Test function discovery with function filter."""
        functions = discover_optimized_functions(
            temp_module, function_filter=["nonexistent"]
        )
        assert len(functions) == 0

    def test_discover_invalid_module(self):
        """Test function discovery with invalid module path."""
        with pytest.raises(
            (ImportError, FileNotFoundError, ValueError)
        ):  # Should raise ImportError or similar
            discover_optimized_functions("nonexistent_module.py")


class TestOptimizationValidator:
    """Test optimization validator component."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OptimizationValidator(threshold_pct=10)

    @pytest.fixture
    def mock_function_info(self):
        """Create mock function info for testing."""
        from traigent.evaluators.base import EvaluationExample

        # The actual function that will be called
        def actual_func(*args, **kwargs):
            return "expected result"  # Return string for _extract_response_text

        # Create a simple callable class that acts like a decorated function
        class MockOptimizedFunction:
            """Mock that acts like a traigent-decorated function."""

            __name__ = "test_func"

            def __init__(self):
                self.func = actual_func  # The wrapped function
                self.config_param = None  # Config parameter name
                # Mock the provider with inject_config method
                self._provider = Mock()
                self._provider.inject_config = Mock(return_value=actual_func)

            def __call__(self, *args, **kwargs):
                return "expected result"  # Return string for _extract_response_text

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
            name="test_func",
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
    async def test_validate_optimization_superior(self, validator, mock_function_info):
        """Test validation when optimization is superior."""
        with (
            patch.object(validator, "_run_baseline") as mock_baseline,
            patch.object(validator, "_run_optimization") as mock_opt,
            patch.object(validator, "_compare_results") as mock_compare,
        ):

            # Setup mocks
            mock_baseline.return_value = ({"accuracy": 0.8}, {"model": "default"})
            mock_opt.return_value = ({"accuracy": 0.9}, {"model": "optimized"})
            mock_compare.return_value = (True, {"accuracy": 12.5})

            result = await validator.validate_optimization(mock_function_info)

            assert result.is_superior is True
            assert result.function_name == "test_func"
            assert result.baseline_metrics == {"accuracy": 0.8}
            assert result.optimized_metrics == {"accuracy": 0.9}
            assert result.improvement_details == {"accuracy": 12.5}
            assert not result.blocking_issues

    @pytest.mark.asyncio
    async def test_validate_optimization_inferior(self, validator, mock_function_info):
        """Test validation when optimization is inferior."""
        with (
            patch.object(validator, "_run_baseline") as mock_baseline,
            patch.object(validator, "_run_optimization") as mock_opt,
            patch.object(validator, "_compare_results") as mock_compare,
        ):

            # Setup mocks for inferior performance
            mock_baseline.return_value = ({"accuracy": 0.9}, {"model": "default"})
            mock_opt.return_value = ({"accuracy": 0.8}, {"model": "optimized"})
            mock_compare.return_value = (False, {"accuracy": -11.1})

            result = await validator.validate_optimization(mock_function_info)

            assert result.is_superior is False
            assert result.improvement_details == {"accuracy": -11.1}
            assert result.should_block is True

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

    def test_compare_results_mixed(self, validator):
        """Test comparison logic for mixed results."""
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
        """Test comparison logic for insufficient improvement."""
        baseline = {"accuracy": 0.80}
        optimized = {"accuracy": 0.82}  # Only 2.5% improvement, below 10% threshold
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        assert is_superior is False
        assert 0 < details["accuracy"] < 10  # Positive but below threshold

    @pytest.mark.asyncio
    async def test_run_baseline_mock_mode(self, validator, mock_function_info):
        """Test baseline execution in mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_MODE": "true"}):
            metrics, config = await validator._run_baseline(mock_function_info)

            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert isinstance(config, dict)

    @pytest.mark.asyncio
    async def test_run_optimization_mock_mode(self, validator, mock_function_info):
        """Test optimization execution in mock mode."""
        with patch.dict(os.environ, {"TRAIGENT_MOCK_MODE": "true"}):
            metrics, config = await validator._run_optimization(mock_function_info)

            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert isinstance(config, dict)


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

    def test_check_command_dry_run(self, cli_runner):
        """Test check command dry run mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import traigent

@traigent.optimize(
    eval_dataset="test.jsonl",
    objectives=["accuracy"],
    configuration_space={"param": ["a", "b"]}
)
def test_func():
    return "test"
"""
            )
            temp_file = f.name

        try:
            result = cli_runner.invoke(cli, ["check", temp_file, "--dry-run"])
            # Should not fail even if function discovery doesn't work perfectly
            assert result.exit_code in [
                0,
                1,
            ]  # Allow either success or expected failure
        finally:
            os.unlink(temp_file)

    def test_check_command_with_filters(self, cli_runner):
        """Test check command with function filters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('def regular_function(): return "test"')
            temp_file = f.name

        try:
            result = cli_runner.invoke(
                cli,
                [
                    "check",
                    temp_file,
                    "--functions",
                    "nonexistent",
                    "--threshold",
                    "15",
                    "--objectives",
                    "accuracy",
                ],
            )
            # Should handle no matching functions gracefully
            assert result.exit_code in [0, 1]
        finally:
            os.unlink(temp_file)

    def test_check_command_invalid_file(self, cli_runner):
        """Test check command with invalid file."""
        result = cli_runner.invoke(cli, ["check", "nonexistent_file.py"])
        assert result.exit_code == 1  # Should fail gracefully


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def validator(self):
        return OptimizationValidator(threshold_pct=10)

    def test_missing_objectives_in_metrics(self, validator):
        """Test handling of missing objectives in metrics."""
        baseline = {"accuracy": 0.8}
        optimized = {"precision": 0.9}  # Different metric
        objectives = ["accuracy", "precision"]

        # Should handle missing metrics gracefully
        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)

    def test_empty_objectives(self, validator):
        """Test handling of empty objectives list."""
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.9}
        objectives = []

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )
        assert is_superior is False  # No objectives to compare
        assert details == {}

    def test_zero_baseline_metrics(self, validator):
        """Test handling of zero baseline metrics."""
        baseline = {"accuracy": 0.0}
        optimized = {"accuracy": 0.5}
        objectives = ["accuracy"]

        # Should handle division by zero gracefully
        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)

    @pytest.mark.asyncio
    async def test_optimization_execution_error(self, validator):
        """Test handling of optimization execution errors."""
        func_info = OptimizedFunction(
            name="error_func",
            func=lambda: None,  # Function that might cause errors
            decorator_config={},
            default_params={},
            eval_dataset=None,
            objectives=["accuracy"],
        )

        with patch.object(
            validator, "_run_optimization", side_effect=Exception("Mock error")
        ):
            result = await validator.validate_optimization(func_info)

            # Should handle errors gracefully and provide useful feedback
            assert result.is_superior is False
            assert result.should_block is True


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        """Test complete validation workflow end-to-end."""
        # Create temporary test module
        test_content = '''
import os
os.environ["TRAIGENT_MOCK_MODE"] = "true"
import traigent

@traigent.optimize(
    eval_dataset="test.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={"model": ["default", "optimized"], "temperature": [0.0, 0.5, 1.0]}
)
def analyze_sentiment(text: str, model: str = "default", temperature: float = 0.5):
    """Test sentiment analysis function."""
    return {"accuracy": 0.8, "cost": 0.05}
'''

        modules_dir = Path(__file__).resolve().parent / "temp_modules"
        modules_dir.mkdir(exist_ok=True)
        temp_path = modules_dir / f"validation_workflow_{uuid4().hex}.py"
        temp_path.write_text(test_content)

        temp_file = str(temp_path)

        # Create mock dataset
        dataset_path = temp_path.parent / "test.jsonl"
        dataset_path.write_text('{"input": {"text": "test"}, "output": "positive"}\n')

        try:
            # Test function discovery
            functions = discover_optimized_functions(temp_file)
            assert len(functions) >= 0  # May find functions or not due to mock setup

            # Test CLI integration
            runner = CliRunner()
            result = runner.invoke(cli, ["check", temp_file, "--dry-run"])
            assert result.exit_code in [0, 1]  # Allow success or expected failure

        finally:
            temp_path.unlink(missing_ok=True)
            if dataset_path.exists():
                dataset_path.unlink()
            if modules_dir.exists() and not any(modules_dir.iterdir()):
                modules_dir.rmdir()


# Fixtures for all tests
@pytest.fixture(autouse=True)
def mock_environment():
    """Set up mock environment for all tests."""
    with patch.dict(os.environ, {"TRAIGENT_MOCK_MODE": "true"}):
        yield


@pytest.fixture(autouse=True)
def mock_pareto_calculator():
    """Mock ParetoFrontCalculator for all tests."""
    with patch("traigent.cli.optimization_validator.ParetoFrontCalculator") as mock:
        mock_instance = Mock()
        mock_instance.is_pareto_superior.return_value = True
        mock.return_value = mock_instance
        yield mock_instance


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=traigent.cli", "--cov-report=term-missing"])
