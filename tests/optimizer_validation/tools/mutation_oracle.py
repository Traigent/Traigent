#!/usr/bin/env python3
"""Mutation-based oracle for detecting weak tests.

This tool creates semantic mutants of optimizer behavior and checks
if tests catch the mutations. Surviving mutants indicate weak tests.

Key Mutation Operators:
1. MUT-SKIP-TRIALS: Skip trial execution entirely
2. MUT-IGNORE-MAX: Ignore max_trials stop condition
3. MUT-SWAP-DIRECTION: Swap objective direction (maximize <-> minimize)
4. MUT-DROP-METRICS: Drop required metrics from results
5. MUT-FORCE-VALIDATOR: Force validator to always pass
6. MUT-EMPTY-CONFIG: Return empty config in trials

Usage:
    python -m tests.optimizer_validation.tools.mutation_oracle
    python -m tests.optimizer_validation.tools.mutation_oracle --mutant MUT-SKIP-TRIALS
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MutantType(Enum):
    """Types of semantic mutations."""

    MUT_SKIP_TRIALS = "MUT-SKIP-TRIALS"
    MUT_IGNORE_MAX = "MUT-IGNORE-MAX"
    MUT_SWAP_DIRECTION = "MUT-SWAP-DIRECTION"
    MUT_DROP_METRICS = "MUT-DROP-METRICS"
    MUT_FORCE_VALIDATOR = "MUT-FORCE-VALIDATOR"
    MUT_EMPTY_CONFIG = "MUT-EMPTY-CONFIG"
    MUT_WRONG_STOP_REASON = "MUT-WRONG-STOP-REASON"


@dataclass
class MutantResult:
    """Result of running tests against a mutant."""

    mutant_type: MutantType
    tests_run: int
    tests_passed: int
    tests_failed: int
    surviving_tests: list[str] = field(default_factory=list)
    killed_tests: list[str] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        """Percentage of tests that survived (didn't catch the mutant)."""
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run * 100


class MutationOracle:
    """Oracle for running mutation tests."""

    def __init__(self, test_dir: Path) -> None:
        self.test_dir = test_dir
        self.results: list[MutantResult] = []

    def create_mutant_conftest(self, mutant_type: MutantType) -> str:
        """Generate conftest code for a specific mutant."""
        base_code = '''
"""Mutant conftest for mutation testing."""

from __future__ import annotations
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
import pytest

# Mutation flag
MUTATION_ACTIVE = "{mutant_type}"

'''

        mutant_code = {
            MutantType.MUT_SKIP_TRIALS: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Returns empty trials."""
    async def runner(scenario):
        func, result = await scenario_runner(scenario)
        if not isinstance(result, Exception):
            # MUTATION: Clear all trials
            result.trials = []
        return func, result
    return runner
''',
            MutantType.MUT_IGNORE_MAX: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Ignores max_trials, runs double."""
    async def runner(scenario):
        # MUTATION: Double the max_trials
        original_max = scenario.max_trials
        scenario.max_trials = (scenario.max_trials or 10) * 2
        func, result = await scenario_runner(scenario)
        scenario.max_trials = original_max
        return func, result
    return runner
''',
            MutantType.MUT_SWAP_DIRECTION: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Swaps objective directions."""
    async def runner(scenario):
        func, result = await scenario_runner(scenario)
        if not isinstance(result, Exception) and hasattr(result, 'best_score'):
            # MUTATION: Negate best score
            if result.best_score is not None:
                result.best_score = -result.best_score
        return func, result
    return runner
''',
            MutantType.MUT_DROP_METRICS: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Drops all metrics from trials."""
    async def runner(scenario):
        func, result = await scenario_runner(scenario)
        if not isinstance(result, Exception):
            # MUTATION: Clear metrics from all trials
            for trial in result.trials:
                trial.metrics = {}
        return func, result
    return runner
''',
            MutantType.MUT_FORCE_VALIDATOR: '''
@pytest.fixture
def result_validator():
    """Mutant: Validator always passes."""
    from tests.optimizer_validation.specs.validators import ValidationResult

    def validator(scenario, result):
        # MUTATION: Always return passed
        return ValidationResult(passed=True)
    return validator
''',
            MutantType.MUT_EMPTY_CONFIG: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Returns empty configs in trials."""
    async def runner(scenario):
        func, result = await scenario_runner(scenario)
        if not isinstance(result, Exception):
            # MUTATION: Clear configs from all trials
            for trial in result.trials:
                trial.config = {}
        return func, result
    return runner
''',
            MutantType.MUT_WRONG_STOP_REASON: '''
@pytest.fixture
def mutant_scenario_runner(scenario_runner):
    """Mutant: Always returns wrong stop_reason."""
    async def runner(scenario):
        func, result = await scenario_runner(scenario)
        if not isinstance(result, Exception):
            # MUTATION: Set wrong stop reason
            result.stop_reason = "mutation_injected"
        return func, result
    return runner
''',
        }

        return base_code.format(mutant_type=mutant_type.value) + mutant_code.get(
            mutant_type, ""
        )

    def run_mutant(
        self, mutant_type: MutantType, test_pattern: str = "**/test_*.py"
    ) -> MutantResult:
        """Run tests with a specific mutant active."""
        # Create temporary conftest with mutation
        mutant_conftest = self.test_dir / "_mutant_conftest.py"
        mutant_conftest.write_text(self.create_mutant_conftest(mutant_type))

        try:
            # Run pytest with mutant conftest
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(self.test_dir),
                    "-v",
                    "--tb=no",
                    "-q",
                    f"--conftest={mutant_conftest}",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                env={"TRAIGENT_MOCK_MODE": "true", "MUTATION_TESTING": "true"},
            )

            # Parse results
            return self._parse_pytest_output(mutant_type, result.stdout + result.stderr)

        finally:
            # Cleanup
            if mutant_conftest.exists():
                mutant_conftest.unlink()

    def _parse_pytest_output(
        self, mutant_type: MutantType, output: str
    ) -> MutantResult:
        """Parse pytest output to extract test results."""
        surviving = []
        killed = []

        lines = output.split("\n")
        for line in lines:
            if "PASSED" in line:
                test_name = line.split("::")[1].split()[0] if "::" in line else line
                surviving.append(test_name.strip())
            elif "FAILED" in line:
                test_name = line.split("::")[1].split()[0] if "::" in line else line
                killed.append(test_name.strip())

        return MutantResult(
            mutant_type=mutant_type,
            tests_run=len(surviving) + len(killed),
            tests_passed=len(surviving),
            tests_failed=len(killed),
            surviving_tests=surviving,
            killed_tests=killed,
        )

    def run_all_mutants(self) -> list[MutantResult]:
        """Run all mutation types and collect results."""
        for mutant_type in MutantType:
            print(f"Running mutant: {mutant_type.value}")
            result = self.run_mutant(mutant_type)
            self.results.append(result)
            print(f"  Survival rate: {result.survival_rate:.1f}%")

        return self.results

    def generate_report(self) -> dict:
        """Generate mutation testing report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_mutants": len(self.results),
            "results": [],
        }

        for result in self.results:
            report["results"].append(
                {
                    "mutant_type": result.mutant_type.value,
                    "tests_run": result.tests_run,
                    "survival_rate": result.survival_rate,
                    "surviving_tests": result.surviving_tests[:20],  # Top 20
                    "killed_tests_count": len(result.killed_tests),
                }
            )

        # Identify weakest tests (survived most mutants)
        test_survival: dict[str, int] = {}
        for result in self.results:
            for test in result.surviving_tests:
                test_survival[test] = test_survival.get(test, 0) + 1

        report["weakest_tests"] = sorted(
            [{"test": t, "mutations_survived": c} for t, c in test_survival.items()],
            key=lambda x: -x["mutations_survived"],
        )[:50]

        return report


def create_metamorphic_tests() -> str:
    """Generate metamorphic test code for detecting weak oracles."""
    return '''
"""Metamorphic tests for detecting weak oracles.

These tests run the same scenario with controlled changes and verify
that the output changes appropriately.
"""

import pytest
from tests.optimizer_validation.specs import TestScenario, ObjectiveSpec


class TestMetamorphicOracles:
    """Metamorphic tests to detect weak assertions."""

    @pytest.mark.asyncio
    async def test_direction_change_affects_best_config(self, scenario_runner):
        """Changing objective direction should change best_config."""
        # Maximize scenario
        scenario_max = TestScenario(
            name="metamorphic_max",
            objectives=[ObjectiveSpec(name="accuracy", orientation="maximize")],
            config_space={"temperature": [0.1, 0.5, 0.9]},
            max_trials=3,
        )

        # Minimize scenario
        scenario_min = TestScenario(
            name="metamorphic_min",
            objectives=[ObjectiveSpec(name="accuracy", orientation="minimize")],
            config_space={"temperature": [0.1, 0.5, 0.9]},
            max_trials=3,
        )

        _, result_max = await scenario_runner(scenario_max)
        _, result_min = await scenario_runner(scenario_min)

        # METAMORPHIC PROPERTY: Different directions should yield different best configs
        if not isinstance(result_max, Exception) and not isinstance(result_min, Exception):
            if result_max.best_config and result_min.best_config:
                # This assertion catches weak tests that don't verify optimization direction
                assert result_max.best_score != result_min.best_score or \\
                       result_max.best_config != result_min.best_config, \\
                       "Direction change should affect optimization result"

    @pytest.mark.asyncio
    async def test_constraint_affects_trial_configs(self, scenario_runner):
        """Adding a constraint should reduce valid configurations."""
        # Unconstrained scenario
        scenario_free = TestScenario(
            name="metamorphic_free",
            config_space={"temperature": [0.1, 0.5, 0.9]},
            max_trials=3,
        )

        # Constrained scenario (only low temp allowed)
        scenario_constrained = TestScenario(
            name="metamorphic_constrained",
            config_space={"temperature": [0.1, 0.5, 0.9]},
            constraints=[lambda config: config["temperature"] < 0.3],
            max_trials=3,
        )

        _, result_free = await scenario_runner(scenario_free)
        _, result_constrained = await scenario_runner(scenario_constrained)

        # METAMORPHIC PROPERTY: Constraint should limit configs
        if not isinstance(result_free, Exception) and not isinstance(result_constrained, Exception):
            constrained_temps = [t.config.get("temperature") for t in result_constrained.trials]
            assert all(t < 0.3 for t in constrained_temps if t is not None), \\
                   "Constraint should limit trial configurations"

    @pytest.mark.asyncio
    async def test_smaller_space_exhausts_faster(self, scenario_runner):
        """Smaller config space should exhaust before larger one."""
        # Large space
        scenario_large = TestScenario(
            name="metamorphic_large",
            config_space={"model": ["a", "b", "c", "d", "e"]},
            max_trials=10,
            mock_mode_config={"optimizer": "grid"},
        )

        # Small space
        scenario_small = TestScenario(
            name="metamorphic_small",
            config_space={"model": ["a", "b"]},
            max_trials=10,
            mock_mode_config={"optimizer": "grid"},
        )

        _, result_large = await scenario_runner(scenario_large)
        _, result_small = await scenario_runner(scenario_small)

        # METAMORPHIC PROPERTY: Small space exhausts, large doesn't
        if not isinstance(result_large, Exception) and not isinstance(result_small, Exception):
            assert len(result_small.trials) <= len(result_large.trials), \\
                   "Smaller config space should not produce more trials"
'''


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run mutation tests")
    parser.add_argument(
        "--mutant",
        "-m",
        type=str,
        choices=[m.value for m in MutantType],
        help="Run specific mutant",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        help="Save report to file",
    )
    parser.add_argument(
        "--generate-metamorphic",
        "-g",
        action="store_true",
        help="Generate metamorphic test file",
    )
    args = parser.parse_args()

    if args.generate_metamorphic:
        output_path = Path("tests/optimizer_validation/test_metamorphic_oracles.py")
        output_path.write_text(create_metamorphic_tests())
        print(f"Generated metamorphic tests: {output_path}")
        return

    test_dir = Path("tests/optimizer_validation")
    oracle = MutationOracle(test_dir)

    if args.mutant:
        mutant_type = MutantType(args.mutant)
        result = oracle.run_mutant(mutant_type)
        print(f"Mutant: {mutant_type.value}")
        print(f"Survival rate: {result.survival_rate:.1f}%")
        print(f"Surviving tests: {len(result.surviving_tests)}")
    else:
        oracle.run_all_mutants()
        report = oracle.generate_report()

        if args.report:
            args.report.write_text(json.dumps(report, indent=2))
            print(f"Report saved to {args.report}")
        else:
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
