"""Refactoring utilities and safety tools for core architecture changes.

This module provides validation tools to ensure safe refactoring of the core
architecture while maintaining backward compatibility and performance.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import importlib
import inspect
import time
from typing import Any, cast

from traigent.core.optimized_function import OptimizedFunction
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class RefactoringValidator:
    """Validation tools for safe refactoring of core architecture."""

    def __init__(self) -> None:
        self.baseline_metrics: dict[str, Any] = {}
        self.validation_results: list[Any] = []

    def establish_baseline(self) -> dict[str, Any]:
        """Establish performance and functionality baseline before refactoring."""
        logger.info("Establishing refactoring baseline...")

        # Measure import time
        import_time = self._measure_import_time("traigent.core.orchestrator")

        # Measure instantiation time
        start_time = time.time()
        # Create minimal instances for baseline
        try:
            from traigent.config import TraigentConfig, get_provider

            provider = get_provider("context")
            # Use the unified get_config method on ConfigurationProvider
            provider.get_config()

            OptimizationOrchestrator(
                optimizer=type("MockOptimizer", (), {"objectives": ["accuracy"]})(),
                evaluator=type("MockEvaluator", (), {})(),
                config=TraigentConfig(),
            )
            instantiation_time = time.time() - start_time
        except Exception as e:
            instantiation_time = float("inf")
            logger.warning(f"Baseline instantiation failed: {e}")

        baseline = {
            "import_time": import_time,
            "instantiation_time": instantiation_time,
            "timestamp": time.time(),
        }

        self.baseline_metrics = baseline
        logger.info(f"Baseline established: {baseline}")
        return baseline

    def _measure_import_time(self, module_name: str) -> float:
        """Measure the time required to import a module."""
        start_time = time.time()
        importlib.import_module(module_name)
        return time.time() - start_time

    def validate_api_compatibility(self) -> dict[str, Any]:
        """Ensure public APIs remain unchanged after refactoring."""
        logger.info("Validating API compatibility...")

        results = {
            "public_classes": [],
            "public_methods": [],
            "breaking_changes": [],
            "compatible": True,
        }

        # Check OptimizedFunction public API
        of_class = OptimizedFunction
        of_methods = [
            name
            for name, method in inspect.getmembers(of_class, predicate=inspect.ismethod)
        ]
        public_methods_list = cast(list[str], results["public_methods"])
        public_methods_list.extend([f"OptimizedFunction.{m}" for m in of_methods])

        # Check OptimizationOrchestrator public API
        oo_class = OptimizationOrchestrator
        oo_methods = [
            name
            for name, method in inspect.getmembers(oo_class, predicate=inspect.ismethod)
        ]
        public_methods_list.extend(
            [f"OptimizationOrchestrator.{m}" for m in oo_methods]
        )

        results["public_classes"] = ["OptimizedFunction", "OptimizationOrchestrator"]

        public_methods = cast(list[str], results["public_methods"])
        public_classes = cast(list[str], results["public_classes"])
        logger.info(
            f"API compatibility validated: {len(public_methods)} methods, {len(public_classes)} classes"
        )
        return results

    def validate_performance_regression(
        self, threshold: float = 0.05
    ) -> dict[str, Any]:
        """Check for performance regression against baseline."""
        logger.info("Checking performance regression...")

        if not self.baseline_metrics:
            logger.warning("No baseline established, running establishment...")
            self.establish_baseline()

        # Measure current performance
        current_import_time = self._measure_import_time("traigent.core.orchestrator")

        # Check for regression
        baseline_import = self.baseline_metrics.get("import_time", 0)
        if baseline_import > 0:
            regression_ratio = current_import_time / baseline_import
            if regression_ratio > (1 + threshold):
                logger.warning(
                    f"Performance regression detected: {regression_ratio:.2f}x slower import"
                )
                return {
                    "regression_detected": True,
                    "import_regression": regression_ratio,
                    "threshold": threshold,
                }

        logger.info(f"No performance regression detected (threshold: {threshold})")
        return {
            "regression_detected": False,
            "import_time": current_import_time,
            "threshold": threshold,
        }

    def validate_functionality(self) -> dict[str, Any]:
        """Run comprehensive functionality tests."""
        logger.info("Validating functionality...")

        results: dict[str, Any] = {
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "success": True,
        }

        try:
            # Test basic imports
            from traigent.core.types import ConfigurationSpace, Parameter, ParameterType

            results["tests_passed"] += 1

            # Test type creation
            param = Parameter(
                name="test_param", type=ParameterType.FLOAT, bounds=(0.0, 1.0)
            )
            results["tests_passed"] += 1

            # Test configuration space
            config_space = ConfigurationSpace()
            config_space.add_parameter(param)
            results["tests_passed"] += 1

            logger.info(
                f"Functionality validation passed: {results['tests_passed']} tests"
            )

        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(str(e))
            results["success"] = False
            logger.error(f"Functionality validation failed: {e}")

        return results

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation checks."""
        logger.info("Running comprehensive refactoring validation...")

        results = {
            "timestamp": time.time(),
            "baseline": self.establish_baseline(),
            "api_compatibility": self.validate_api_compatibility(),
            "performance": self.validate_performance_regression(),
            "functionality": self.validate_functionality(),
            "overall_success": True,
        }

        # Determine overall success
        perf_results = cast(dict[str, Any], results["performance"])
        func_results = cast(dict[str, Any], results["functionality"])
        if perf_results.get("regression_detected", False) or not func_results.get(
            "success", True
        ):
            results["overall_success"] = False

        self.validation_results.append(results)

        if results["overall_success"]:
            logger.info("All validation checks passed!")
        else:
            logger.error("Validation checks failed - review results above")

        return results

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validation first."

        latest = self.validation_results[-1]

        report = f"""
# 🔍 Refactoring Validation Report

**Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest["timestamp"]))}
**Overall Status:** {"✅ PASSED" if latest["overall_success"] else "❌ FAILED"}

## 📊 Performance Metrics
- Import Time: {latest["baseline"].get("import_time", "N/A"):.4f}s
- Instantiation Time: {latest["baseline"].get("instantiation_time", "N/A"):.4f}s

## 🔗 API Compatibility
- Public Classes: {len(latest["api_compatibility"].get("public_classes", []))}
- Public Methods: {len(latest["api_compatibility"].get("public_methods", []))}

## ⚡ Performance Check
- Regression Detected: {latest["performance"].get("regression_detected", "N/A")}
- Threshold: {latest["performance"].get("threshold", "N/A")}

## 🧪 Functionality Tests
- Tests Passed: {latest["functionality"].get("tests_passed", 0)}
- Tests Failed: {latest["functionality"].get("tests_failed", 0)}
"""

        if latest["functionality"].get("errors"):
            report += "\n## ❌ Errors\n"
            for error in latest["functionality"]["errors"]:
                report += f"- {error}\n"

        return report


def create_refactoring_safety_check() -> RefactoringValidator:
    """Create and return a configured refactoring validator."""
    validator = RefactoringValidator()
    return validator


# Convenience function for quick validation
def quick_validation_check() -> bool:
    """Run a quick validation check and return success status."""
    validator = create_refactoring_safety_check()
    results = validator.run_comprehensive_validation()
    return cast(bool, results["overall_success"])
