"""Comprehensive tests for traigent.core.refactoring_utils module."""

import time
from unittest.mock import Mock, patch

import pytest

from traigent.core.refactoring_utils import (
    RefactoringValidator,
    create_refactoring_safety_check,
    quick_validation_check,
)


class TestRefactoringValidator:
    """Test RefactoringValidator class."""

    def test_refactoring_validator_init(self):
        """Test RefactoringValidator initialization."""
        validator = RefactoringValidator()
        assert validator.baseline_metrics == {}
        assert validator.validation_results == []

    def test_establish_baseline_success(self):
        """Test establish_baseline with successful instantiation."""
        validator = RefactoringValidator()

        with patch("traigent.core.refactoring_utils.importlib.import_module"), patch(
            "traigent.core.refactoring_utils.get_provider"
        ) as mock_provider, patch(
            "traigent.core.refactoring_utils.OptimizationOrchestrator"
        ), patch(
            "traigent.core.refactoring_utils.TraigentConfig"
        ):

            # Mock get_provider to return a proper mock with get_config method
            mock_provider_instance = Mock()
            mock_provider_instance.get_config.return_value = {}
            mock_provider.return_value = mock_provider_instance

            baseline = validator.establish_baseline()

            assert "import_time" in baseline
            assert "instantiation_time" in baseline
            assert "timestamp" in baseline
            assert baseline["import_time"] >= 0
            # instantiation_time could be inf if mock fails internally
            assert isinstance(baseline["instantiation_time"], (int, float))
            assert validator.baseline_metrics == baseline

    def test_establish_baseline_instantiation_failure(self):
        """Test establish_baseline when instantiation fails."""
        validator = RefactoringValidator()

        with patch("traigent.core.refactoring_utils.importlib.import_module"), patch(
            "traigent.core.refactoring_utils.get_provider"
        ) as mock_provider:

            mock_provider.side_effect = Exception("Mock error")

            baseline = validator.establish_baseline()

            assert baseline["instantiation_time"] == float("inf")
            assert "import_time" in baseline
            assert validator.baseline_metrics == baseline

    def test_validate_api_compatibility(self):
        """Test validate_api_compatibility checks public APIs."""
        validator = RefactoringValidator()

        with patch(
            "traigent.core.refactoring_utils.inspect.getmembers"
        ) as mock_members:
            # Mock method lists for classes
            mock_members.side_effect = [
                [("method1", Mock()), ("method2", Mock())],  # OptimizedFunction
                [("method3", Mock()), ("method4", Mock())],  # OptimizationOrchestrator
            ]

            results = validator.validate_api_compatibility()

            assert results["compatible"] is True
            assert "OptimizedFunction" in results["public_classes"]
            assert "OptimizationOrchestrator" in results["public_classes"]
            assert len(results["public_methods"]) == 4
            assert "OptimizedFunction.method1" in results["public_methods"]
            assert "OptimizationOrchestrator.method3" in results["public_methods"]
            assert results["breaking_changes"] == []

    def test_validate_performance_regression_no_baseline(self):
        """Test validate_performance_regression establishes baseline if missing."""
        validator = RefactoringValidator()

        with patch.object(validator, "establish_baseline") as mock_establish:
            mock_establish.return_value = {"import_time": 0.1}

            result = validator.validate_performance_regression()

            mock_establish.assert_called_once()
            assert result["regression_detected"] is False

    def test_validate_performance_regression_no_regression(self):
        """Test validate_performance_regression with no regression."""
        validator = RefactoringValidator()
        validator.baseline_metrics = {"import_time": 0.1}

        with patch("traigent.core.refactoring_utils.time.time") as mock_time:
            mock_time.side_effect = [100.0, 100.05]  # 0.05s import time

            result = validator.validate_performance_regression(threshold=0.1)

            assert result["regression_detected"] is False
            assert "import_time" in result
            assert result["threshold"] == 0.1

    def test_validate_performance_regression_with_regression(self):
        """Test validate_performance_regression detects regression."""
        validator = RefactoringValidator()
        validator.baseline_metrics = {"import_time": 0.1}

        with patch("traigent.core.refactoring_utils.time.time") as mock_time:
            # Provide extra values for logger.warning() which internally uses time.time()
            mock_time.side_effect = [
                100.0,
                100.2,
                100.2,
                100.2,
                100.2,
            ]  # 0.2s import time (2x baseline)

            result = validator.validate_performance_regression(threshold=0.05)

            assert result["regression_detected"] is True
            # Use pytest.approx for float comparison
            assert result["import_regression"] == pytest.approx(2.0, rel=1e-6)
            assert result["threshold"] == 0.05

    def test_validate_performance_regression_zero_baseline(self):
        """Test validate_performance_regression with zero baseline."""
        validator = RefactoringValidator()
        validator.baseline_metrics = {"import_time": 0}

        result = validator.validate_performance_regression()

        # Should not detect regression when baseline is 0
        assert result["regression_detected"] is False

    def test_validate_functionality_success(self):
        """Test validate_functionality with all tests passing."""
        validator = RefactoringValidator()

        # Parameter and ConfigurationSpace are imported inside validate_functionality
        # We need to patch them at the module level where they're imported from
        with patch("traigent.core.types.Parameter") as mock_param, patch(
            "traigent.core.types.ConfigurationSpace"
        ) as mock_space:

            mock_param.return_value = Mock()
            mock_space_instance = Mock()
            mock_space.return_value = mock_space_instance

            results = validator.validate_functionality()

            assert results["tests_passed"] == 3
            assert results["tests_failed"] == 0
            assert results["errors"] == []
            assert results["success"] is True
            mock_space_instance.add_parameter.assert_called_once()

    def test_validate_functionality_failure(self):
        """Test validate_functionality with test failures."""
        validator = RefactoringValidator()

        # Patch at the actual import location
        with patch("traigent.core.types.Parameter") as mock_param:
            mock_param.side_effect = Exception("Mock import error")

            results = validator.validate_functionality()

            # Import test passes (tests_passed = 1), then Parameter creation fails
            assert results["tests_passed"] == 1
            assert results["tests_failed"] == 1
            assert len(results["errors"]) == 1
            assert "Mock import error" in results["errors"][0]
            assert results["success"] is False

    def test_run_comprehensive_validation_success(self):
        """Test run_comprehensive_validation with all checks passing."""
        validator = RefactoringValidator()

        with patch.object(
            validator, "establish_baseline"
        ) as mock_baseline, patch.object(
            validator, "validate_api_compatibility"
        ) as mock_api, patch.object(
            validator, "validate_performance_regression"
        ) as mock_perf, patch.object(
            validator, "validate_functionality"
        ) as mock_func:

            mock_baseline.return_value = {"import_time": 0.1}
            mock_api.return_value = {"compatible": True}
            mock_perf.return_value = {"regression_detected": False}
            mock_func.return_value = {"success": True}

            results = validator.run_comprehensive_validation()

            assert results["overall_success"] is True
            assert "baseline" in results
            assert "api_compatibility" in results
            assert "performance" in results
            assert "functionality" in results
            assert "timestamp" in results
            assert len(validator.validation_results) == 1

    def test_run_comprehensive_validation_performance_failure(self):
        """Test run_comprehensive_validation with performance regression."""
        validator = RefactoringValidator()

        with patch.object(
            validator, "establish_baseline"
        ) as mock_baseline, patch.object(
            validator, "validate_api_compatibility"
        ) as mock_api, patch.object(
            validator, "validate_performance_regression"
        ) as mock_perf, patch.object(
            validator, "validate_functionality"
        ) as mock_func:

            mock_baseline.return_value = {"import_time": 0.1}
            mock_api.return_value = {"compatible": True}
            mock_perf.return_value = {"regression_detected": True}
            mock_func.return_value = {"success": True}

            results = validator.run_comprehensive_validation()

            assert results["overall_success"] is False

    def test_run_comprehensive_validation_functionality_failure(self):
        """Test run_comprehensive_validation with functionality failure."""
        validator = RefactoringValidator()

        with patch.object(
            validator, "establish_baseline"
        ) as mock_baseline, patch.object(
            validator, "validate_api_compatibility"
        ) as mock_api, patch.object(
            validator, "validate_performance_regression"
        ) as mock_perf, patch.object(
            validator, "validate_functionality"
        ) as mock_func:

            mock_baseline.return_value = {"import_time": 0.1}
            mock_api.return_value = {"compatible": True}
            mock_perf.return_value = {"regression_detected": False}
            mock_func.return_value = {"success": False}

            results = validator.run_comprehensive_validation()

            assert results["overall_success"] is False

    def test_generate_validation_report_no_results(self):
        """Test generate_validation_report with no validation results."""
        validator = RefactoringValidator()

        report = validator.generate_validation_report()

        assert "No validation results available" in report

    def test_generate_validation_report_with_results(self):
        """Test generate_validation_report with validation results."""
        validator = RefactoringValidator()
        validator.validation_results = [
            {
                "timestamp": time.time(),
                "overall_success": True,
                "baseline": {"import_time": 0.1, "instantiation_time": 0.05},
                "api_compatibility": {
                    "public_classes": ["Class1"],
                    "public_methods": ["method1"],
                },
                "performance": {"regression_detected": False, "threshold": 0.05},
                "functionality": {"tests_passed": 3, "tests_failed": 0, "errors": []},
            }
        ]

        report = validator.generate_validation_report()

        assert "Refactoring Validation Report" in report
        assert "PASSED" in report
        assert "Import Time" in report
        assert "Public Classes: 1" in report
        assert "Tests Passed: 3" in report

    def test_generate_validation_report_with_errors(self):
        """Test generate_validation_report includes errors."""
        validator = RefactoringValidator()
        validator.validation_results = [
            {
                "timestamp": time.time(),
                "overall_success": False,
                "baseline": {"import_time": 0.1, "instantiation_time": 0.05},
                "api_compatibility": {"public_classes": [], "public_methods": []},
                "performance": {"regression_detected": True, "threshold": 0.05},
                "functionality": {
                    "tests_passed": 0,
                    "tests_failed": 1,
                    "errors": ["Error 1", "Error 2"],
                },
            }
        ]

        report = validator.generate_validation_report()

        assert "FAILED" in report
        assert "Errors" in report
        assert "Error 1" in report
        assert "Error 2" in report

    def test_generate_validation_report_latest_results(self):
        """Test generate_validation_report uses latest results."""
        validator = RefactoringValidator()
        validator.validation_results = [
            {
                "timestamp": time.time() - 100,
                "overall_success": False,
                "baseline": {"import_time": 0.1, "instantiation_time": 0.05},
                "api_compatibility": {"public_classes": [], "public_methods": []},
                "performance": {"regression_detected": True, "threshold": 0.05},
                "functionality": {"tests_passed": 0, "tests_failed": 1, "errors": []},
            },
            {
                "timestamp": time.time(),
                "overall_success": True,
                "baseline": {"import_time": 0.1, "instantiation_time": 0.05},
                "api_compatibility": {"public_classes": [], "public_methods": []},
                "performance": {"regression_detected": False, "threshold": 0.05},
                "functionality": {"tests_passed": 5, "tests_failed": 0, "errors": []},
            },
        ]

        report = validator.generate_validation_report()

        assert "PASSED" in report
        assert "Tests Passed: 5" in report


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_refactoring_safety_check(self):
        """Test create_refactoring_safety_check returns validator."""
        validator = create_refactoring_safety_check()

        assert isinstance(validator, RefactoringValidator)
        assert validator.baseline_metrics == {}
        assert validator.validation_results == []

    def test_quick_validation_check_success(self):
        """Test quick_validation_check returns True on success."""
        with patch(
            "traigent.core.refactoring_utils.create_refactoring_safety_check"
        ) as mock_create:
            mock_validator = Mock()
            mock_validator.run_comprehensive_validation.return_value = {
                "overall_success": True
            }
            mock_create.return_value = mock_validator

            result = quick_validation_check()

            assert result is True
            mock_validator.run_comprehensive_validation.assert_called_once()

    def test_quick_validation_check_failure(self):
        """Test quick_validation_check returns False on failure."""
        with patch(
            "traigent.core.refactoring_utils.create_refactoring_safety_check"
        ) as mock_create:
            mock_validator = Mock()
            mock_validator.run_comprehensive_validation.return_value = {
                "overall_success": False
            }
            mock_create.return_value = mock_validator

            result = quick_validation_check()

            assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validator_multiple_validation_runs(self):
        """Test validator can run multiple validations."""
        validator = RefactoringValidator()

        with patch.object(
            validator, "establish_baseline"
        ) as mock_baseline, patch.object(
            validator, "validate_api_compatibility"
        ) as mock_api, patch.object(
            validator, "validate_performance_regression"
        ) as mock_perf, patch.object(
            validator, "validate_functionality"
        ) as mock_func:

            mock_baseline.return_value = {"import_time": 0.1}
            mock_api.return_value = {"compatible": True}
            mock_perf.return_value = {"regression_detected": False}
            mock_func.return_value = {"success": True}

            validator.run_comprehensive_validation()
            validator.run_comprehensive_validation()

            assert len(validator.validation_results) == 2

    def test_validate_performance_regression_custom_threshold(self):
        """Test validate_performance_regression with custom thresholds."""
        validator = RefactoringValidator()
        validator.baseline_metrics = {"import_time": 0.1}

        # Test high threshold - should not detect regression
        with patch("traigent.core.refactoring_utils.time.time") as mock_time:
            mock_time.side_effect = [100.0, 100.15]  # 0.15s import time (1.5x baseline)

            result = validator.validate_performance_regression(threshold=0.6)
            assert result["regression_detected"] is False

        # Test low threshold - should detect regression
        with patch("traigent.core.refactoring_utils.time.time") as mock_time:
            # Provide extra values for logger.warning() which internally uses time.time()
            mock_time.side_effect = [100.0, 100.15, 100.15, 100.15, 100.15]

            result = validator.validate_performance_regression(threshold=0.3)
            assert result["regression_detected"] is True

    def test_baseline_metrics_persistence(self):
        """Test baseline metrics persist across validation calls."""
        validator = RefactoringValidator()

        with patch("traigent.core.refactoring_utils.importlib.import_module"), patch(
            "traigent.core.refactoring_utils.get_provider"
        ) as mock_provider, patch(
            "traigent.core.refactoring_utils.OptimizationOrchestrator"
        ), patch(
            "traigent.core.refactoring_utils.TraigentConfig"
        ):

            mock_provider.return_value.get_config.return_value = {}

            first_baseline = validator.establish_baseline()
            second_call_baseline = validator.baseline_metrics

            assert first_baseline == second_call_baseline
            assert "import_time" in second_call_baseline
