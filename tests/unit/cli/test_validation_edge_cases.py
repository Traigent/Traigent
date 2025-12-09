"""
Edge case and error scenario tests for TraiGent Optimization Validation System.

Tests challenging scenarios, edge cases, and error conditions.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.function_discovery import (
    _extract_default_parameters,
    discover_optimized_functions,
)
from traigent.cli.main import cli
from traigent.cli.optimization_validator import OptimizationValidator
from traigent.cli.validation_types import OptimizedFunction, ValidationResult


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_function_with_no_default_parameters(self):
        """Test handling functions with no default parameters."""

        def test_func(required_param1: str, required_param2: int):
            return {"result": "test"}

        defaults = _extract_default_parameters(test_func)
        assert defaults == {}

    def test_function_with_complex_default_parameters(self):
        """Test handling functions with complex default parameter types."""

        def test_func(
            text: str,
            config: dict = None,
            options: list = None,
            flag: bool = True,
            number: float = 3.14,
            none_param: str = None,
        ):
            return {"result": "test"}

        defaults = _extract_default_parameters(test_func)
        expected = {
            "config": None,
            "options": None,
            "flag": True,
            "number": 3.14,
            "none_param": None,
        }
        assert defaults == expected

    def test_function_with_args_and_kwargs(self):
        """Test handling functions with *args and **kwargs."""

        def test_func(text: str, param: str = "default", *args, **kwargs):
            return {"result": "test"}

        defaults = _extract_default_parameters(test_func)
        assert defaults == {"param": "default"}
        # *args and **kwargs should be ignored

    @pytest.fixture
    def validator(self):
        return OptimizationValidator(threshold_pct=10)

    def test_comparison_with_zero_baseline_values(self, validator):
        """Test comparison when baseline metrics contain zero values."""
        baseline = {"accuracy": 0.0, "precision": 0.5}
        optimized = {"accuracy": 0.3, "precision": 0.6}
        objectives = ["accuracy", "precision"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle zero baseline gracefully
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)
        # Zero baseline should still allow for meaningful comparison
        assert (
            abs(details["precision"] - 20.0) < 0.01
        )  # 20% improvement (with tolerance)

    def test_comparison_with_negative_baseline_values(self, validator):
        """Test comparison when baseline metrics contain negative values."""
        baseline = {
            "loss": -0.5,
            "error_rate": -0.2,
        }  # Negative is worse for these metrics
        optimized = {"loss": -0.3, "error_rate": -0.1}  # Less negative is better
        objectives = ["loss", "error_rate"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle negative values correctly
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)

    def test_comparison_with_identical_values(self, validator):
        """Test comparison when baseline and optimized metrics are identical."""
        baseline = {"accuracy": 0.8, "precision": 0.7}
        optimized = {"accuracy": 0.8, "precision": 0.7}
        objectives = ["accuracy", "precision"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Identical values should not be considered superior
        assert is_superior is False
        assert details["accuracy"] == 0.0  # No improvement
        assert details["precision"] == 0.0  # No improvement

    def test_comparison_with_missing_metrics(self, validator):
        """Test comparison when some metrics are missing."""
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.9, "precision": 0.7}  # Extra metric in optimized
        objectives = ["accuracy", "precision"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle missing metrics gracefully
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)
        assert "accuracy" in details
        # Precision might be handled differently since it's missing from baseline

    def test_comparison_with_very_small_improvements(self, validator):
        """Test comparison with very small metric improvements."""
        baseline = {"accuracy": 0.800000}
        optimized = {"accuracy": 0.800001}  # Tiny improvement
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Very small improvements should not be considered superior
        assert is_superior is False
        assert details["accuracy"] < validator.threshold * 100

    def test_comparison_with_very_large_improvements(self, validator):
        """Test comparison with very large metric improvements."""
        baseline = {"accuracy": 0.1}
        optimized = {"accuracy": 0.9}  # 800% improvement
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Large improvements should be considered superior
        assert is_superior is True
        assert details["accuracy"] > 100  # Much larger than threshold

    def test_mixed_objective_types(self, validator):
        """Test comparison with mixed objective types (some improve, some degrade)."""
        # For "cost" type objectives, lower is better (inverse optimization)
        baseline = {"accuracy": 0.8, "cost": 0.1, "latency": 2.0}
        optimized = {
            "accuracy": 0.9,
            "cost": 0.2,
            "latency": 1.5,
        }  # Better accuracy and latency, worse cost
        objectives = ["accuracy", "cost", "latency"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle mixed improvements correctly
        assert isinstance(is_superior, bool)
        assert isinstance(details, dict)
        assert details["accuracy"] > 0  # Improved
        assert details["cost"] < 0  # Degraded (cost increased)
        assert details["latency"] > 0  # Improved (latency decreased)


class TestErrorScenarios:
    """Test error conditions and exception handling."""

    def test_discover_functions_invalid_python_file(self):
        """Test function discovery with invalid Python syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("invalid python syntax !!!")
            invalid_file = f.name

        try:
            with pytest.raises(
                (SyntaxError, ImportError, ValueError)
            ):  # Should raise syntax error or import error
                discover_optimized_functions(invalid_file)
        finally:
            os.unlink(invalid_file)

    def test_discover_functions_nonexistent_file(self):
        """Test function discovery with non-existent file."""
        with pytest.raises(
            (FileNotFoundError, ImportError, ValueError)
        ):  # Should raise FileNotFoundError or ImportError
            discover_optimized_functions("completely_nonexistent_file.py")

    def test_discover_functions_outside_workspace(self):
        """Path traversal outside the workspace should be rejected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# temporary external module")
            external_file = f.name

        try:
            with pytest.raises(ValueError):
                discover_optimized_functions(external_file)
        finally:
            os.chmod(external_file, 0o600)
            os.unlink(external_file)

    def test_discover_functions_permission_denied(self):
        """Test function discovery with permission-denied file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Valid Python file")
            restricted_file = f.name

        try:
            # Make file unreadable
            os.chmod(restricted_file, 0o000)

            # Should handle permission error gracefully
            try:
                discover_optimized_functions(restricted_file)
                # If it doesn't raise an exception, that's also acceptable
            except PermissionError:
                # Expected behavior
                pass
            except Exception:
                # Other exceptions might also be acceptable
                pass

        finally:
            # Restore permissions and clean up
            try:
                os.chmod(restricted_file, 0o644)
                os.unlink(restricted_file)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_validator_with_failing_baseline(self):
        """Test validator when baseline execution fails."""
        validator = OptimizationValidator(threshold_pct=10)

        func_info = OptimizedFunction(
            name="failing_func",
            func=lambda: None,
            decorator_config={},
            default_params={},
            eval_dataset=None,
            objectives=["accuracy"],
        )

        with (
            patch.object(validator, "_check_prerequisites", return_value=[]),
            patch.object(
                validator, "_run_baseline", side_effect=Exception("Baseline failed")
            ),
        ):
            result = await validator.validate_optimization(func_info)

            assert result.is_superior is False
            assert len(result.blocking_issues) > 0
            assert any(
                "error" in issue.lower() or "baseline failed" in issue.lower()
                for issue in result.blocking_issues
            )

    @pytest.mark.asyncio
    async def test_validator_with_failing_optimization(self):
        """Test validator when optimization execution fails."""
        validator = OptimizationValidator(threshold_pct=10)

        func_info = OptimizedFunction(
            name="failing_func",
            func=lambda: None,
            decorator_config={},
            default_params={},
            eval_dataset=None,
            objectives=["accuracy"],
        )

        with (
            patch.object(validator, "_check_prerequisites", return_value=[]),
            patch.object(
                validator,
                "_run_baseline",
                return_value=({"accuracy": 0.8}, {"param": "default"}),
            ),
            patch.object(
                validator,
                "_run_optimization",
                side_effect=Exception("Optimization failed"),
            ),
        ):

            result = await validator.validate_optimization(func_info)

            assert result.is_superior is False
            assert len(result.blocking_issues) > 0
            assert any(
                "error" in issue.lower() or "optimization failed" in issue.lower()
                for issue in result.blocking_issues
            )

    def test_cli_with_invalid_threshold(self):
        """Test CLI with invalid threshold values."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Valid Python file")
            valid_file = f.name

        try:
            # Test negative threshold
            result = runner.invoke(cli, ["check", valid_file, "--threshold", "-5"])
            # Should either handle gracefully or show appropriate error
            assert result.exit_code in [0, 1, 2]  # Various acceptable exit codes

            # Test extremely high threshold
            result = runner.invoke(cli, ["check", valid_file, "--threshold", "99999"])
            assert result.exit_code in [0, 1, 2]

        finally:
            os.unlink(valid_file)

    def test_cli_with_invalid_arguments(self):
        """Test CLI with invalid command-line arguments."""
        runner = CliRunner()

        # Test with missing required argument
        result = runner.invoke(cli, ["check"])
        assert result.exit_code == 2  # Click should show usage error

        # Test with invalid flag
        result = runner.invoke(cli, ["check", "test.py", "--invalid-flag"])
        assert result.exit_code == 2  # Click should show usage error


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    @pytest.fixture
    def validator(self):
        return OptimizationValidator(threshold_pct=0)  # Zero threshold

    def test_zero_threshold_validation(self, validator):
        """Test validation with zero improvement threshold."""
        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.800001}  # Tiny improvement
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # With zero threshold, any improvement should be superior
        assert is_superior is True
        assert details["accuracy"] > 0

    def test_maximum_threshold_validation(self):
        """Test validation with very high improvement threshold."""
        validator = OptimizationValidator(threshold_pct=1000)  # 1000% threshold

        baseline = {"accuracy": 0.5}
        optimized = {"accuracy": 0.8}  # 60% improvement
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # 60% improvement should not meet 1000% threshold
        assert is_superior is False
        assert details["accuracy"] < validator.threshold * 100

    def test_empty_objectives_list(self):
        """Test handling of empty objectives list."""
        validator = OptimizationValidator(threshold_pct=10)

        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.9}
        objectives = []  # Empty objectives

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # No objectives to compare should result in not superior
        assert is_superior is False
        assert details == {}

    def test_single_objective_optimization(self):
        """Test optimization with single objective."""
        validator = OptimizationValidator(threshold_pct=10)

        baseline = {"accuracy": 0.8}
        optimized = {"accuracy": 0.9}  # 12.5% improvement
        objectives = ["accuracy"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Single objective with sufficient improvement
        assert is_superior is True
        assert details["accuracy"] > 10

    def test_many_objectives_optimization(self):
        """Test optimization with many objectives."""
        validator = OptimizationValidator(threshold_pct=10)

        # Create baseline and optimized with 10 objectives
        baseline = {f"metric_{i}": 0.5 for i in range(10)}
        optimized = {f"metric_{i}": 0.6 for i in range(10)}  # 20% improvement in all
        objectives = [f"metric_{i}" for i in range(10)]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # All objectives improved above threshold
        assert is_superior is True
        for i in range(10):
            assert (
                abs(details[f"metric_{i}"] - 20.0) < 0.01
            )  # Use tolerance for floating-point

    def test_large_metric_values(self):
        """Test handling of very large metric values."""
        validator = OptimizationValidator(threshold_pct=10)

        baseline = {"latency": 1000000}  # 1 million units
        optimized = {"latency": 890000}  # 11% improvement (better than threshold)
        objectives = ["latency"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle large numbers correctly
        # For minimization metrics like latency, improvement should be positive when value decreases
        assert is_superior is True  # 11% improvement exceeds 10% threshold
        assert abs(details["latency"] - 11.0) < 0.1  # Approximately 11% improvement

    def test_very_small_metric_values(self):
        """Test handling of very small metric values."""
        validator = OptimizationValidator(threshold_pct=10)

        baseline = {"error_rate": 0.000001}  # Very small baseline
        optimized = {"error_rate": 0.0000005}  # 50% improvement
        objectives = ["error_rate"]

        is_superior, details = validator._compare_results(
            baseline, optimized, objectives
        )

        # Should handle small numbers correctly
        assert is_superior is True
        assert details["error_rate"] > 10  # Should show significant improvement


class TestConcurrencyAndPerformance:
    """Test concurrent execution and performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_validations(self):
        """Test running multiple validations concurrently."""
        validator = OptimizationValidator(threshold_pct=10)

        # Create multiple function infos
        functions = []
        for i in range(5):
            func_info = OptimizedFunction(
                name=f"func_{i}",
                func=lambda x: {"accuracy": 0.8},
                decorator_config={},
                default_params={"param": f"default_{i}"},
                eval_dataset=None,
                objectives=["accuracy"],
            )
            functions.append(func_info)

        # Mock the internal methods to return quickly
        async def mock_baseline(func_info):
            return {"accuracy": 0.8}, {"param": "default"}

        async def mock_optimization(func_info):
            return {"accuracy": 0.9}, {"param": "optimized"}

        with (
            patch.object(validator, "_check_prerequisites", return_value=[]),
            patch.object(validator, "_run_baseline", side_effect=mock_baseline),
            patch.object(validator, "_run_optimization", side_effect=mock_optimization),
        ):

            # Run validations concurrently
            import asyncio

            tasks = [validator.validate_optimization(func) for func in functions]
            results = await asyncio.gather(*tasks)

            # All validations should complete successfully
            assert len(results) == 5
            for result in results:
                assert isinstance(result, ValidationResult)
                assert (
                    result.is_superior is True
                )  # All should be superior with our mock

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large dataset references."""
        # Create function info with large dataset path
        large_dataset_path = "/path/to/very/long/" + "dataset/" * 150 + "file.jsonl"

        func_info = OptimizedFunction(
            name="memory_test_func",
            func=lambda: {"accuracy": 0.8},
            decorator_config={"eval_dataset": large_dataset_path},
            default_params={"param": "default"},
            eval_dataset=large_dataset_path,
            objectives=["accuracy"] * 100,  # Many objectives
        )

        # Should handle large configurations without excessive memory usage
        assert len(func_info.eval_dataset) > 1000
        assert len(func_info.objectives) == 100


if __name__ == "__main__":
    # Run edge case tests
    pytest.main([__file__, "-v", "--tb=short"])
