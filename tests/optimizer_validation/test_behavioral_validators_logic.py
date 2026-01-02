"""Unit tests for behavioral validator logic.

These tests verify that validators correctly DETECT violations (false negative protection).
The main test suite verifies validators don't have false positives (passing valid tests).
This file ensures the "watchmen are actually watching."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tests.optimizer_validation.specs.behavioral_validators import (
    BayesianValidator,
    ConfigSpaceValidator,
    GridSearchValidator,
    ParallelValidator,
    RandomSearchValidator,
    SequentialValidator,
)
from tests.optimizer_validation.specs.scenario import ExpectedOutcome, ExpectedResult

# =============================================================================
# Mock Data Structures
# =============================================================================


@dataclass
class MockTrialResult:
    """Mock trial result for testing validators."""

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime | None = None
    duration: float | None = None


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing validators."""

    trials: list[MockTrialResult] = field(default_factory=list)
    best_config: dict[str, Any] | None = None
    best_score: float | None = None


@dataclass
class MockTestScenario:
    """Mock test scenario for testing validators."""

    name: str = "test_scenario"
    config_space: dict[str, Any] = field(default_factory=dict)
    max_trials: int = 10
    mock_mode_config: dict[str, Any] | None = None
    parallel_config: dict[str, Any] | None = None
    expected: ExpectedResult = field(default_factory=lambda: ExpectedResult())


# =============================================================================
# ConfigSpaceValidator Tests
# =============================================================================


class TestConfigSpaceValidatorDetection:
    """Test that ConfigSpaceValidator catches violations."""

    def test_detects_categorical_violation(self):
        """Validator should detect value outside categorical list."""
        validator = ConfigSpaceValidator()

        scenario = MockTestScenario(
            config_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.3, 0.7]}
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id="t1",
                    config={"model": "gpt-5", "temperature": 0.3},  # gpt-5 not in list
                )
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect categorical violation"
        assert len(validation.errors) == 1
        assert "categorical_bounds" in validation.errors[0].category
        assert "gpt-5" in validation.errors[0].message

    def test_detects_continuous_bounds_violation(self):
        """Validator should detect value outside continuous range."""
        validator = ConfigSpaceValidator()

        scenario = MockTestScenario(
            config_space={"temperature": (0.0, 1.0)}  # Continuous range
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id="t1",
                    config={"temperature": 1.5},  # Outside [0.0, 1.0]
                )
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect continuous bounds violation"
        assert len(validation.errors) == 1
        assert "continuous_bounds" in validation.errors[0].category

    def test_passes_valid_configs(self):
        """Validator should pass valid configurations."""
        validator = ConfigSpaceValidator()

        scenario = MockTestScenario(
            config_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.3, 0.7]}
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id="t1", config={"model": "gpt-4", "temperature": 0.7}
                ),
                MockTrialResult(
                    trial_id="t2", config={"model": "gpt-3.5", "temperature": 0.3}
                ),
            ]
        )

        validation = validator.validate(result, scenario)

        assert validation.passed, f"Should pass valid configs: {validation.errors}"


# =============================================================================
# GridSearchValidator Tests
# =============================================================================


class TestGridSearchValidatorDetection:
    """Test that GridSearchValidator catches violations."""

    def test_detects_duplicate_configs(self):
        """Validator should detect duplicate configurations in grid search."""
        validator = GridSearchValidator()

        scenario = MockTestScenario(
            config_space={"model": ["a", "b"], "temp": [0.3, 0.7]},
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(trial_id="t1", config={"model": "a", "temp": 0.3}),
                MockTrialResult(trial_id="t2", config={"model": "a", "temp": 0.7}),
                MockTrialResult(
                    trial_id="t3", config={"model": "a", "temp": 0.3}
                ),  # Duplicate!
                MockTrialResult(trial_id="t4", config={"model": "b", "temp": 0.7}),
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect duplicate configs"
        assert any("uniqueness" in e.category for e in validation.errors)

    def test_detects_wrong_trial_count(self):
        """Validator should detect incorrect trial count for grid."""
        validator = GridSearchValidator()

        scenario = MockTestScenario(
            config_space={"model": ["a", "b"], "temp": [0.3, 0.7]},  # 4 combinations
            max_trials=10,  # More than cardinality
            mock_mode_config={"optimizer": "grid"},
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(trial_id="t1", config={"model": "a", "temp": 0.3}),
                MockTrialResult(trial_id="t2", config={"model": "a", "temp": 0.7}),
                # Missing 2 trials!
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect wrong trial count"
        assert any("trial_count" in e.category for e in validation.errors)


# =============================================================================
# RandomSearchValidator Tests
# =============================================================================


class TestRandomSearchValidatorDetection:
    """Test that RandomSearchValidator catches violations."""

    def test_detects_duplicates_in_unexphausted_space(self):
        """Validator should detect duplicates when space isn't exhausted.

        The validator only checks for duplicates when:
        1. Discrete space size >= 10 (small spaces naturally have duplicates)
        2. Number of trials <= 30% of space size

        So we need a space of at least 10 options with few trials.
        """
        validator = RandomSearchValidator()

        # 12 options (>= 10), and 3 trials (<= 30% of 12 = 3.6)
        scenario = MockTestScenario(
            config_space={
                "model": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
            },
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(trial_id="t1", config={"model": "a"}),
                MockTrialResult(trial_id="t2", config={"model": "a"}),  # Duplicate!
                MockTrialResult(trial_id="t3", config={"model": "b"}),
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect duplicates in non-exhausted space"
        assert any("duplicates" in e.category for e in validation.errors)


# =============================================================================
# BayesianValidator Tests
# =============================================================================


class TestBayesianValidatorDetection:
    """Test that BayesianValidator catches violations."""

    def test_detects_all_identical_configs(self):
        """Validator should detect when TPE produces identical configs."""
        validator = BayesianValidator()

        scenario = MockTestScenario(
            config_space={"model": ["a", "b"], "temp": [0.3, 0.7]},
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(trial_id="t1", config={"model": "a", "temp": 0.3}),
                MockTrialResult(trial_id="t2", config={"model": "a", "temp": 0.3}),
                MockTrialResult(trial_id="t3", config={"model": "a", "temp": 0.3}),
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect all identical configs"
        assert any("diversity" in e.category for e in validation.errors)

    def test_passes_diverse_configs(self):
        """Validator should pass when configs are diverse."""
        validator = BayesianValidator()

        scenario = MockTestScenario(
            config_space={"model": ["a", "b"], "temp": [0.3, 0.7]},
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(trial_id="t1", config={"model": "a", "temp": 0.3}),
                MockTrialResult(trial_id="t2", config={"model": "b", "temp": 0.7}),
                MockTrialResult(trial_id="t3", config={"model": "a", "temp": 0.7}),
            ]
        )

        validation = validator.validate(result, scenario)

        assert validation.passed, f"Should pass diverse configs: {validation.errors}"


# =============================================================================
# SequentialValidator Tests
# =============================================================================


class TestSequentialValidatorDetection:
    """Test that SequentialValidator catches violations."""

    def test_detects_non_monotonic_timestamps(self):
        """Validator should detect out-of-order timestamps."""
        validator = SequentialValidator()

        now = datetime.now(timezone.utc)
        scenario = MockTestScenario(
            config_space={"model": ["a"]},
            parallel_config=None,  # Sequential mode
        )
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id="t1",
                    config={"model": "a"},
                    timestamp=now,
                    duration=1.0,
                ),
                MockTrialResult(
                    trial_id="t2",
                    config={"model": "a"},
                    timestamp=now - timedelta(seconds=5),  # Earlier than t1!
                    duration=1.0,
                ),
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect non-monotonic timestamps"
        assert any("ordering" in e.category for e in validation.errors)


# =============================================================================
# ParallelValidator Tests
# =============================================================================


class TestParallelValidatorDetection:
    """Test that ParallelValidator catches violations."""

    def test_detects_concurrency_exceeded(self):
        """Validator should detect when max concurrent exceeds limit."""
        validator = ParallelValidator()

        now = datetime.now(timezone.utc)
        scenario = MockTestScenario(
            config_space={"model": ["a"]},
            parallel_config={"trial_concurrency": 2},  # Max 2 concurrent
        )
        # Create 4 trials all overlapping (all start at same time, long duration)
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id=f"t{i}",
                    config={"model": "a"},
                    timestamp=now,
                    duration=10.0,  # Long enough to overlap
                )
                for i in range(4)
            ]
        )

        validation = validator.validate(result, scenario)

        assert not validation.passed, "Should detect concurrency exceeded"
        assert any("concurrency" in e.category for e in validation.errors)

    def test_passes_within_concurrency_limit(self):
        """Validator should pass when concurrency is within limit."""
        validator = ParallelValidator()

        now = datetime.now(timezone.utc)
        scenario = MockTestScenario(
            config_space={"model": ["a"]},
            parallel_config={"trial_concurrency": 4},  # Allow 4 concurrent
        )
        # Create 4 trials all overlapping - within limit
        result = MockOptimizationResult(
            trials=[
                MockTrialResult(
                    trial_id=f"t{i}",
                    config={"model": "a"},
                    timestamp=now,
                    duration=10.0,
                )
                for i in range(4)
            ]
        )

        validation = validator.validate(result, scenario)

        assert validation.passed, f"Should pass within limit: {validation.errors}"


# =============================================================================
# Applicability Tests
# =============================================================================


class TestValidatorApplicability:
    """Test that validators correctly determine when they apply."""

    def test_grid_validator_only_applies_to_grid(self):
        """GridSearchValidator should only apply to grid optimizer."""
        validator = GridSearchValidator()

        grid_scenario = MockTestScenario(
            mock_mode_config={"optimizer": "grid"},
        )
        random_scenario = MockTestScenario(
            mock_mode_config={"optimizer": "random"},
        )
        no_config_scenario = MockTestScenario()

        assert validator.applies_to(grid_scenario)
        assert not validator.applies_to(random_scenario)
        assert not validator.applies_to(no_config_scenario)

    def test_sequential_validator_applies_without_parallel_config(self):
        """SequentialValidator should apply when no parallel config."""
        validator = SequentialValidator()

        sequential = MockTestScenario(parallel_config=None)
        sequential_explicit = MockTestScenario(parallel_config={"trial_concurrency": 1})
        parallel = MockTestScenario(parallel_config={"trial_concurrency": 4})

        assert validator.applies_to(sequential)
        assert validator.applies_to(sequential_explicit)
        assert not validator.applies_to(parallel)

    def test_sequential_validator_skips_async_scenarios(self):
        """SequentialValidator should skip async scenarios.

        Async I/O naturally allows concurrent execution even without
        explicit parallelism, so the sequential validator should not
        apply to async scenarios.
        """
        validator = SequentialValidator()

        # Create a mock scenario with is_async=True
        async_scenario = MockTestScenario(parallel_config=None)
        async_scenario.is_async = True

        sync_scenario = MockTestScenario(parallel_config=None)
        sync_scenario.is_async = False

        assert not validator.applies_to(async_scenario), "Should skip async scenarios"
        assert validator.applies_to(sync_scenario), "Should apply to sync scenarios"

    def test_parallel_validator_applies_with_concurrency(self):
        """ParallelValidator should apply when concurrency > 1."""
        validator = ParallelValidator()

        parallel = MockTestScenario(parallel_config={"trial_concurrency": 4})
        parallel_mode = MockTestScenario(parallel_config={"mode": "parallel"})
        sequential = MockTestScenario(parallel_config={"trial_concurrency": 1})

        assert validator.applies_to(parallel)
        assert validator.applies_to(parallel_mode)
        assert not validator.applies_to(sequential)

    def test_validators_skip_failure_scenarios(self):
        """All validators should skip expected failure scenarios."""
        failure_scenario = MockTestScenario(
            expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
            config_space={"model": ["a"]},
            mock_mode_config={"optimizer": "grid"},
        )

        validators = [
            ConfigSpaceValidator(),
            GridSearchValidator(),
        ]

        for validator in validators:
            assert not validator.applies_to(
                failure_scenario
            ), f"{validator.name} should skip failure scenarios"
