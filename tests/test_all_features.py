"""Test all Traigent SDK features to ensure everything is working correctly."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import traigent
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.utils.constraints import ConstraintManager, temperature_constraint
from traigent.utils.importance import ParameterImportanceAnalyzer
from traigent.utils.multi_objective import ParetoFrontCalculator
from traigent.utils.persistence import PersistenceManager
from traigent.utils.validation import OptimizationValidator
from traigent.visualization.plots import PlotGenerator


def test_imports():
    """Test that all imports work correctly."""
    print("✅ All imports successful")
    # Verify key imports are available
    assert traigent is not None, "traigent module should be importable"
    assert (
        OptimizationOrchestrator is not None
    ), "OptimizationOrchestrator should be importable"


def test_basic_decorator():
    """Test basic decorator functionality."""

    @traigent.optimize(
        objectives=["accuracy"],
        configuration_space={"param1": [1, 2, 3], "param2": (0.0, 1.0)},
    )
    def simple_function(x: int, **config) -> int:
        return x * config.get("param1", 1)

    # Test function is still callable
    result = simple_function(5, param1=2)
    assert result == 10, f"Expected 10, got {result}"

    # Test it has optimization methods
    assert hasattr(simple_function, "optimize")
    assert hasattr(simple_function, "get_best_config")

    print("✅ Basic decorator works correctly")


def test_validation():
    """Test validation functionality."""
    # Valid configuration
    valid_config = {"model": ["gpt-4o-mini", "GPT-4o"], "temperature": (0.0, 1.0)}

    result = OptimizationValidator.validate_optimization_config(
        valid_config, ["accuracy"], None, "grid"
    )

    assert result.is_valid, "Valid configuration should pass validation"

    # Invalid configuration
    invalid_config = {
        "": ["value1"],  # Empty parameter name
        "temperature": (1.0, 0.0),  # Invalid range
    }

    result = OptimizationValidator.validate_optimization_config(
        invalid_config,
        [],  # No objectives
        None,
        "unknown",  # Unknown algorithm
    )

    assert not result.is_valid, "Invalid configuration should fail validation"
    assert len(result.errors) > 0, "Should have validation errors"

    print("✅ Validation works correctly")


def test_constraints():
    """Test constraint system."""
    manager = ConstraintManager()
    manager.add_constraint(temperature_constraint(0.0, 1.0))

    # Valid configuration
    valid_config = {"temperature": 0.5}
    is_valid, violations = manager.validate_configuration(valid_config)
    assert is_valid, "Valid temperature should pass constraint"

    # Invalid configuration
    invalid_config = {"temperature": 1.5}
    is_valid, violations = manager.validate_configuration(invalid_config)
    assert not is_valid, "Invalid temperature should fail constraint"
    assert len(violations) == 1, "Should have one violation"

    print("✅ Constraints work correctly")


def test_visualization():
    """Test visualization functionality."""
    plotter = PlotGenerator(use_matplotlib=False)  # Use ASCII

    # Create mock result
    from datetime import datetime

    trials = []
    for i in range(5):
        trial = TrialResult(
            trial_id=f"trial_{i}",
            config={"param": i},
            metrics={"accuracy": 0.5 + i * 0.1},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        trials.append(trial)

    result = OptimizationResult(
        trials=trials,
        best_config={"param": 4},
        best_score=0.9,
        optimization_id="test_opt",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
    )

    # Generate plot
    plot = plotter.plot_optimization_progress(result)
    assert isinstance(plot, str), "Plot should be a string"
    assert len(plot) > 0, "Plot should not be empty"

    print("✅ Visualization works correctly")


def test_persistence():
    """Test persistence functionality."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = PersistenceManager(tmpdir)

        # Create mock result
        from datetime import datetime

        result = OptimizationResult(
            trials=[],
            best_config={"param": 1},
            best_score=0.9,
            optimization_id="test_save",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid",
            timestamp=datetime.now(),
            metadata={
                "function_name": "test_function",
                "configuration_space": {"param": [1, 2, 3]},
            },
        )

        # Save result
        saved_path = persistence.save_result(result, "test_result")
        assert Path(saved_path).exists(), "Saved result should exist"

        # Load result
        loaded = persistence.load_result("test_result")
        assert loaded.best_score == 0.9, "Loaded result should match"

        # List results
        results = persistence.list_results()
        assert len(results) == 1, "Should have one saved result"

        print("✅ Persistence works correctly")


def test_multi_objective():
    """Test multi-objective functionality."""
    from datetime import datetime

    # Create trials with multiple objectives
    trials = []
    for i in range(10):
        trial = TrialResult(
            trial_id=f"trial_{i}",
            config={"param": i},
            metrics={"accuracy": 0.5 + i * 0.05, "speed": 1.0 - i * 0.1},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        trials.append(trial)

    # Calculate Pareto front
    calculator = ParetoFrontCalculator(maximize={"accuracy": True, "speed": True})
    pareto_front = calculator.calculate_pareto_front(trials, ["accuracy", "speed"])

    assert len(pareto_front) > 0, "Should have Pareto-optimal points"
    assert len(pareto_front) <= len(trials), "Pareto front should be subset of trials"

    print("✅ Multi-objective works correctly")


def test_importance_analysis():
    """Test parameter importance analysis."""
    from datetime import datetime

    # Create trials with varying parameters
    trials = []
    for i in range(20):
        config = {"param1": i % 3, "param2": (i % 5) * 0.2}
        score = config["param1"] * 0.3 + config["param2"] * 0.7

        trial = TrialResult(
            trial_id=f"trial_{i}",
            config=config,
            metrics={"accuracy": score},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        trials.append(trial)

    # Analyze importance
    analyzer = ParameterImportanceAnalyzer("accuracy")
    results = analyzer.analyze_variance_based(trials)

    assert len(results) > 0, "Should have importance results"
    assert "param2" in results, "Should analyze param2"

    print("✅ Importance analysis works correctly")


async def main():
    """Run all tests."""
    print("🧪 Testing Traigent SDK Features")
    print("=" * 50)

    test_imports()
    test_basic_decorator()
    test_validation()
    test_constraints()
    test_visualization()
    test_persistence()
    test_multi_objective()
    test_importance_analysis()

    print("\n✅ All tests passed!")
    print("Traigent SDK is working correctly!")


if __name__ == "__main__":
    asyncio.run(main())
