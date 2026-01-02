"""Behavioral validators for optimizer validation tests.

This module provides reusable validators that check optimization results
exhibit correct behavioral properties for the configured mode/algorithm.

Validators are automatically applied based on scenario dimensions and
will fail tests if violations are detected.

Example:
    # Validators are auto-applied via result_validator fixture
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()

    # Skip specific validators if needed
    validation = result_validator(scenario, result, skip_behavioral=["grid_search"])
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from functools import reduce
from itertools import product
from operator import mul
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

    from .scenario import TestScenario


@dataclass
class BehavioralValidationError:
    """A single behavioral validation error."""

    validator: str  # Name of the validator that produced this error
    category: str  # Error category (e.g., "bounds", "ordering", "uniqueness")
    message: str  # Human-readable error message
    expected: str | None = None  # What was expected
    actual: str | None = None  # What was observed
    trial_ids: list[str] | None = None  # Affected trial IDs if applicable


@dataclass
class BehavioralValidationResult:
    """Result of behavioral validation."""

    passed: bool
    validator_name: str
    errors: list[BehavioralValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)

    def add_error(
        self,
        category: str,
        message: str,
        expected: str | None = None,
        actual: str | None = None,
        trial_ids: list[str] | None = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            BehavioralValidationError(
                validator=self.validator_name,
                category=category,
                message=message,
                expected=expected,
                actual=actual,
                trial_ids=trial_ids,
            )
        )
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning (does not fail)."""
        self.warnings.append(message)

    def add_check(self, check_name: str) -> None:
        """Record a check that was performed."""
        self.checks_performed.append(check_name)


class BehavioralValidator(ABC):
    """Base class for behavioral validators.

    Validators check that optimization results exhibit correct behavior
    for the configured mode/algorithm. They are auto-applied based on
    scenario dimensions.
    """

    # Unique name for this validator (used in error messages and skip lists)
    name: str = "base"

    # Priority for ordering (lower = earlier)
    priority: int = 100

    @abstractmethod
    def applies_to(self, scenario: TestScenario) -> bool:
        """Check if this validator should be applied to the scenario.

        Args:
            scenario: The test scenario specification

        Returns:
            True if this validator should run for this scenario
        """

    @abstractmethod
    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        """Validate optimization result against expected behaviors.

        Args:
            result: The optimization result to validate
            scenario: The test scenario specification

        Returns:
            BehavioralValidationResult with any errors found
        """

    def _create_result(self) -> BehavioralValidationResult:
        """Create a new validation result for this validator."""
        return BehavioralValidationResult(
            passed=True,
            validator_name=self.name,
        )

    def _hash_config(self, config: dict[str, Any]) -> str:
        """Create deterministic hash of config."""
        sorted_config = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(sorted_config.encode()).hexdigest()[:16]

    def _is_failure_scenario(self, scenario: TestScenario) -> bool:
        """Check if scenario expects failure."""
        from .scenario import ExpectedOutcome

        return scenario.expected.outcome == ExpectedOutcome.FAILURE

    def _is_mock_mode(self, scenario: TestScenario) -> bool:
        """Check if we're running in mock mode.

        Uses explicit signals (scenario config, environment) rather than
        heuristics to determine if timing-based checks should be relaxed.
        """
        # Check scenario's mock_mode_config
        if scenario.mock_mode_config:
            return True
        # Check environment variable
        if os.environ.get("TRAIGENT_MOCK_MODE", "").lower() == "true":
            return True
        return False


# =============================================================================
# Algorithm Validators
# =============================================================================


class ConfigSpaceValidator(BehavioralValidator):
    """Validates that all trial configs respect the configuration space.

    Universal checks applied to ALL tests:
    - Categorical values are from defined lists (ERROR if violated)
    - Continuous values are within (min, max) range (ERROR if violated)
    - Missing config keys generate warnings (not errors, since some scenarios
      may intentionally have sparse configs)
    """

    name = "config_space"
    priority = 1  # Run first

    def applies_to(self, scenario: TestScenario) -> bool:
        # Always applies if there's a config space and not a failure scenario
        if self._is_failure_scenario(scenario):
            return False
        return bool(scenario.config_space)

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()
        config_space = scenario.config_space

        validation.add_check(f"space_keys={list(config_space.keys())}")

        for trial in result.trials:
            if not trial.config:
                validation.add_error(
                    category="missing_config",
                    message="Trial has no configuration",
                    trial_ids=[trial.trial_id],
                )
                continue

            # Check each config value
            for key, space_def in config_space.items():
                if key not in trial.config:
                    # Key might be optional - warn but don't fail
                    validation.add_warning(
                        f"Trial {trial.trial_id} missing config key: {key}"
                    )
                    continue

                value = trial.config[key]

                if isinstance(space_def, list):
                    # Categorical check
                    if value not in space_def:
                        validation.add_error(
                            category="categorical_bounds",
                            message=f"Value '{value}' not in allowed categorical values",
                            expected=f"one of {space_def}",
                            actual=str(value),
                            trial_ids=[trial.trial_id],
                        )

                elif isinstance(space_def, tuple) and len(space_def) == 2:
                    # Continuous range check
                    min_val, max_val = space_def
                    if not isinstance(value, (int, float)):
                        validation.add_error(
                            category="type_error",
                            message=f"Expected numeric for continuous param '{key}'",
                            expected=f"number in [{min_val}, {max_val}]",
                            actual=f"{type(value).__name__}: {value}",
                            trial_ids=[trial.trial_id],
                        )
                    elif not (min_val <= value <= max_val):
                        validation.add_error(
                            category="continuous_bounds",
                            message=f"Value {value} outside allowed range",
                            expected=f"[{min_val}, {max_val}]",
                            actual=str(value),
                            trial_ids=[trial.trial_id],
                        )

        return validation


class GridSearchValidator(BehavioralValidator):
    """Validates grid search behavioral invariants.

    Grid search must:
    - Have trial count == expected grid cardinality (or max_trials if smaller)
    - Produce all unique configs
    - Be deterministic (same space = same order)
    """

    name = "grid_search"
    priority = 10

    def applies_to(self, scenario: TestScenario) -> bool:
        if self._is_failure_scenario(scenario):
            return False
        if not scenario.mock_mode_config:
            return False
        return scenario.mock_mode_config.get("optimizer") == "grid"

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()

        # Extract trial configs
        trial_configs = [t.config for t in result.trials if t.config]

        # Check 1: Calculate expected cardinality
        cardinality = self._calculate_grid_cardinality(scenario.config_space)
        validation.add_check(f"grid_cardinality={cardinality}")

        if cardinality == float("inf"):
            validation.add_warning(
                "Grid search with continuous params - cardinality check skipped"
            )
            return validation

        # Check 2: Trial count should equal cardinality (or max_trials if lower)
        expected_count = min(cardinality, scenario.max_trials)
        actual_count = len(trial_configs)

        if actual_count != expected_count:
            validation.add_error(
                category="trial_count",
                message="Grid search trial count mismatch",
                expected=str(expected_count),
                actual=str(actual_count),
            )

        # Check 3: All configs are unique
        config_hashes = [self._hash_config(c) for c in trial_configs]
        unique_hashes = set(config_hashes)

        if len(unique_hashes) != len(config_hashes):
            duplicates = len(config_hashes) - len(unique_hashes)
            validation.add_error(
                category="uniqueness",
                message="Grid search produced duplicate configurations",
                expected="all unique",
                actual=f"{duplicates} duplicates",
            )

        # Check 4: Deterministic order (optional - check against itertools.product)
        if len(trial_configs) > 0:
            expected_order = self._get_expected_grid_order(scenario.config_space)
            if expected_order and len(expected_order) >= len(trial_configs):
                for i, (actual, expected) in enumerate(
                    zip(trial_configs, expected_order)
                ):
                    if actual != expected:
                        validation.add_warning(
                            f"Grid search order differs at position {i}"
                        )
                        break

        validation.add_check("uniqueness")
        validation.add_check("deterministic_order")

        return validation

    def _calculate_grid_cardinality(self, config_space: dict[str, Any]) -> int | float:
        """Calculate total grid combinations."""
        sizes = []
        for value in config_space.values():
            if isinstance(value, list):
                sizes.append(len(value))
            elif isinstance(value, tuple):
                # Continuous ranges are not valid for pure grid search
                return float("inf")
            else:
                sizes.append(1)
        return reduce(mul, sizes, 1) if sizes else 1

    def _get_expected_grid_order(
        self, config_space: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get expected config order from itertools.product."""
        keys = sorted(config_space.keys())
        value_lists = []
        for k in keys:
            v = config_space[k]
            if isinstance(v, list):
                value_lists.append(v)
            else:
                value_lists.append([v])

        expected = []
        for combo in product(*value_lists):
            expected.append(dict(zip(keys, combo)))
        return expected


class RandomSearchValidator(BehavioralValidator):
    """Validates random search behavioral invariants.

    Random search must:
    - Produce no duplicate configs (unless space exhausted)
    - Respect configuration space bounds
    """

    name = "random_search"
    priority = 10

    def applies_to(self, scenario: TestScenario) -> bool:
        if self._is_failure_scenario(scenario):
            return False
        if not scenario.mock_mode_config:
            return False
        return scenario.mock_mode_config.get("optimizer") == "random"

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()

        trial_configs = [t.config for t in result.trials if t.config]

        # Extract discrete keys for uniqueness checking
        discrete_keys = self._get_discrete_keys(scenario.config_space)
        validation.add_check(f"discrete_keys={discrete_keys}")

        # Check 1: No duplicates in discrete dimensions (unless space exhausted/small)
        if discrete_keys:
            discrete_space_size = self._estimate_discrete_space_size(
                scenario.config_space, discrete_keys
            )
            validation.add_check(f"discrete_space_size={discrete_space_size}")

            # Only check duplicates if:
            # 1. Space is large enough that duplicates would be suspicious
            # 2. Number of trials is well under the space size
            # With random sampling, birthday paradox means high collision chance
            # when trials approaches sqrt(space_size). We only flag duplicates
            # when trials < 30% of space size AND space has at least 10 options.
            min_space_for_check = 10  # Small spaces naturally have duplicates
            exhaustion_ratio = 0.3

            should_check = discrete_space_size >= min_space_for_check and len(
                trial_configs
            ) <= int(discrete_space_size * exhaustion_ratio)

            if should_check:
                # Hash only discrete keys for comparison
                config_hashes = [
                    self._hash_discrete_config(c, discrete_keys) for c in trial_configs
                ]
                if len(set(config_hashes)) != len(config_hashes):
                    duplicates = len(config_hashes) - len(set(config_hashes))
                    validation.add_error(
                        category="duplicates",
                        message="Random search produced duplicate discrete configs",
                        expected="unique discrete configs (space not exhausted)",
                        actual=f"{duplicates} duplicates in discrete dimensions",
                    )

        validation.add_check("no_duplicates")

        return validation

    def _get_discrete_keys(self, config_space: dict[str, Any]) -> list[str]:
        """Get keys that have discrete (categorical) values."""
        return [k for k, v in config_space.items() if isinstance(v, list)]

    def _estimate_discrete_space_size(
        self, config_space: dict[str, Any], discrete_keys: list[str]
    ) -> int:
        """Estimate size of discrete subspace."""
        size = 1
        for key in discrete_keys:
            value = config_space.get(key)
            if isinstance(value, list):
                size *= len(value)
        return size

    def _hash_discrete_config(
        self, config: dict[str, Any], discrete_keys: list[str]
    ) -> int:
        """Hash only discrete dimensions of a config."""
        discrete_items = tuple(
            sorted((k, config.get(k)) for k in discrete_keys if k in config)
        )
        return hash(discrete_items)


class BayesianValidator(BehavioralValidator):
    """Validates Bayesian (TPE) optimization behavioral invariants.

    Bayesian optimization should:
    - Have an initial random exploration phase (n_startup_trials)
    - Show trial diversity (not repeating same config)
    - Respect configuration space bounds (delegated to ConfigSpaceValidator)

    Note: Full Bayesian behavior (adaptive sampling, acquisition function)
    cannot be verified in mock mode. This validator focuses on structural
    properties that would indicate wrong sampler was used.
    """

    name = "bayesian_tpe"
    priority = 10

    def applies_to(self, scenario: TestScenario) -> bool:
        if self._is_failure_scenario(scenario):
            return False
        if not scenario.mock_mode_config:
            return False
        optimizer = scenario.mock_mode_config.get("optimizer")
        sampler = scenario.mock_mode_config.get("sampler", "").lower()
        return optimizer == "optuna" and sampler == "tpe"

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()

        trials = result.trials
        trial_configs = [t.config for t in trials if t.config]

        # Check 1: Record startup info
        n_startup = scenario.mock_mode_config.get("n_startup_trials", 10)
        validation.add_check(f"expected_startup={n_startup}")
        validation.add_check(f"actual_trials={len(trials)}")

        # Check 2: Configs should be diverse (not all identical)
        # TPE explores the space - if all configs are identical, something is wrong
        if len(trial_configs) >= 3:
            unique_configs = len({hash(frozenset(c.items())) for c in trial_configs})
            if unique_configs == 1:
                validation.add_error(
                    category="diversity",
                    message="TPE sampler produced identical configs for all trials",
                    expected="diverse exploration",
                    actual="all configs identical",
                )
            validation.add_check(
                f"unique_configs={unique_configs}/{len(trial_configs)}"
            )

        return validation


# =============================================================================
# Execution Mode Validators
# =============================================================================


class SequentialValidator(BehavioralValidator):
    """Validates sequential execution behavioral invariants.

    Sequential execution must:
    - Have monotonically increasing trial timestamps
    - Have no overlapping execution windows
    """

    name = "sequential_execution"
    priority = 20

    def applies_to(self, scenario: TestScenario) -> bool:
        if self._is_failure_scenario(scenario):
            return False
        # Sequential if no parallel config or explicit sequential
        if not scenario.parallel_config:
            return True

        mode = scenario.parallel_config.get("mode")
        if mode == "sequential":
            return True

        concurrency = scenario.parallel_config.get("trial_concurrency", 1)
        return concurrency <= 1

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()

        trials = result.trials
        if len(trials) < 2:
            validation.add_check("skipped_single_trial")
            return validation

        # Check 1: Timestamps are monotonically increasing
        prev_timestamp = None
        for trial in trials:
            if not hasattr(trial, "timestamp") or trial.timestamp is None:
                validation.add_warning(
                    f"Trial {trial.trial_id} has no timestamp - ordering check skipped"
                )
                continue

            if prev_timestamp is not None:
                if trial.timestamp < prev_timestamp:
                    validation.add_error(
                        category="ordering",
                        message="Trial timestamps not monotonically increasing",
                        expected="timestamp >= previous",
                        actual=f"{trial.timestamp} < {prev_timestamp}",
                        trial_ids=[trial.trial_id],
                    )
            prev_timestamp = trial.timestamp

        validation.add_check("timestamp_ordering")

        # Check 2: No overlapping execution windows
        overlaps = self._check_for_overlaps(trials, scenario)
        if overlaps:
            validation.add_error(
                category="concurrency",
                message="Found overlapping execution windows in sequential mode",
                expected="no overlap",
                actual=f"{len(overlaps)} overlaps detected",
                trial_ids=overlaps,
            )

        validation.add_check("overlap_check")

        return validation

    def _check_for_overlaps(self, trials: list, scenario: TestScenario) -> list[str]:
        """Check for overlapping trial execution windows.

        Uses explicit mock mode detection plus a timing heuristic to avoid
        false positives when trials execute too quickly for reliable timing.
        """
        overlapping_trials: list[str] = []

        # Extract execution windows (start_time, end_time)
        windows = []
        for trial in trials:
            if not hasattr(trial, "timestamp") or trial.timestamp is None:
                continue
            start = trial.timestamp
            duration = getattr(trial, "duration", None)
            if duration is not None:
                end = start + timedelta(seconds=duration)
                windows.append((start, end, duration, trial.trial_id))

        # Skip overlap check in mock mode when durations are very short
        # Use explicit mock mode detection first, then timing heuristic as secondary check
        is_mock = self._is_mock_mode(scenario)
        if is_mock and windows and all(d < 0.1 for _, _, d, _ in windows):
            return []

        # Tolerance: overlaps smaller than 50ms are ignored (timing granularity)
        tolerance = timedelta(milliseconds=50)

        # Check for overlaps with tolerance
        for i, (start1, end1, dur1, id1) in enumerate(windows):
            for start2, end2, dur2, id2 in windows[i + 1 :]:
                # Calculate overlap duration
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    # Only flag as overlap if it exceeds tolerance
                    if overlap_duration > tolerance:
                        if id1 not in overlapping_trials:
                            overlapping_trials.append(id1)
                        if id2 not in overlapping_trials:
                            overlapping_trials.append(id2)

        return overlapping_trials


class ParallelValidator(BehavioralValidator):
    """Validates parallel execution behavioral invariants.

    Parallel execution should:
    - Respect batch size limits (max concurrent <= trial_concurrency)
    - Show evidence of parallelism (overlapping timestamps)
    """

    name = "parallel_execution"
    priority = 20

    def applies_to(self, scenario: TestScenario) -> bool:
        if self._is_failure_scenario(scenario):
            return False
        if not scenario.parallel_config:
            return False

        mode = scenario.parallel_config.get("mode")
        if mode == "parallel":
            return True

        concurrency = scenario.parallel_config.get("trial_concurrency", 1)
        return concurrency > 1

    def validate(
        self,
        result: OptimizationResult,
        scenario: TestScenario,
    ) -> BehavioralValidationResult:
        validation = self._create_result()

        trials = result.trials
        concurrency = scenario.parallel_config.get("trial_concurrency", 1)
        validation.add_check(f"expected_concurrency={concurrency}")

        if len(trials) < 2:
            validation.add_check("skipped_single_trial")
            return validation

        # Check 1: Max concurrent doesn't exceed limit
        max_concurrent = self._find_max_concurrent(trials, scenario)

        if max_concurrent == -1:
            # Mock mode with very short durations - skip check
            validation.add_check("concurrency_check_skipped_mock_mode")
        else:
            validation.add_check(f"max_concurrent_observed={max_concurrent}")

            if max_concurrent > concurrency:
                validation.add_error(
                    category="concurrency",
                    message="More concurrent trials than allowed",
                    expected=f"<= {concurrency}",
                    actual=str(max_concurrent),
                )

            # Check 2: Evidence of parallelism (soft check)
            if max_concurrent <= 1 and len(trials) > concurrency:
                validation.add_warning(
                    "No overlapping execution windows detected in parallel mode. "
                    "This may be acceptable for fast mock executions."
                )

        return validation

    def _find_max_concurrent(self, trials: list, scenario: TestScenario) -> int:
        """Find maximum number of concurrently running trials.

        Uses explicit mock mode detection plus timing heuristic to avoid
        false positives when trials execute too quickly for reliable timing.
        Returns -1 to signal this check should be skipped.
        """
        durations = []
        events: list[tuple[Any, int]] = []

        for trial in trials:
            if not hasattr(trial, "timestamp") or trial.timestamp is None:
                continue
            start = trial.timestamp
            duration = getattr(trial, "duration", None)
            if duration is not None:
                durations.append(duration)
                end = start + timedelta(seconds=duration)
            else:
                end = start
            events.append((start, 1))  # Trial start
            events.append((end, -1))  # Trial end

        if not events:
            return 0

        # Skip concurrency check in mock mode when durations are very short
        # Use explicit mock mode detection first, then timing heuristic as secondary check
        is_mock = self._is_mock_mode(scenario)
        if is_mock and durations and all(d < 0.1 for d in durations):
            return -1  # Signal to skip check

        # Sort by time, then by delta (end=-1 before start=+1 at same timestamp)
        # This prevents back-to-back trials from being counted as concurrent
        events.sort(key=lambda x: (x[0], x[1]))

        current = 0
        max_concurrent = 0
        for _, delta in events:
            current += delta
            max_concurrent = max(max_concurrent, current)

        return max_concurrent


# =============================================================================
# Registry and Entry Point
# =============================================================================


class ValidatorRegistry:
    """Registry for behavioral validators.

    Provides auto-discovery and scenario-based filtering of validators.
    """

    _validators: list[BehavioralValidator] = []

    @classmethod
    def register(cls, validator: BehavioralValidator) -> None:
        """Register a validator."""
        cls._validators.append(validator)
        # Sort by priority
        cls._validators.sort(key=lambda v: v.priority)

    @classmethod
    def get_validators_for(cls, scenario: TestScenario) -> list[BehavioralValidator]:
        """Get all validators that apply to a scenario."""
        return [v for v in cls._validators if v.applies_to(scenario)]

    @classmethod
    def get_all_validators(cls) -> list[BehavioralValidator]:
        """Get all registered validators."""
        return list(cls._validators)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered validators (for testing)."""
        cls._validators = []


def _register_default_validators() -> None:
    """Register all default validators."""
    validators = [
        ConfigSpaceValidator(),
        GridSearchValidator(),
        RandomSearchValidator(),
        BayesianValidator(),
        SequentialValidator(),
        ParallelValidator(),
    ]
    for v in validators:
        ValidatorRegistry.register(v)


# Auto-register on import
_register_default_validators()


def apply_behavioral_validators(
    scenario: TestScenario,
    result: OptimizationResult,
    skip_validators: list[str] | None = None,
) -> list[BehavioralValidationResult]:
    """Apply all applicable behavioral validators to a result.

    Args:
        scenario: The test scenario specification
        result: The optimization result to validate
        skip_validators: List of validator names to skip

    Returns:
        List of validation results from all applicable validators
    """
    skip = set(skip_validators or [])

    applicable_validators = ValidatorRegistry.get_validators_for(scenario)

    results = []
    for validator in applicable_validators:
        if validator.name in skip:
            continue

        validation_result = validator.validate(result, scenario)
        results.append(validation_result)

    return results
