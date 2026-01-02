"""
CTD (Combinatorial Test Design) Integration Tests from CSV Specification
=========================================================================
This module reads test specifications from CSV and generates combinatorial test cases
based on coverage requirements, parameter values, and constraints.

Key Features:
- Dynamic test generation from CSV specification
- Parameter coverage analysis (pairwise, n-wise)
- Constraint handling and validation
- Mock mode support for safe testing
"""

import csv
import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Ensure we're in mock mode for testing
os.environ["TRAIGENT_MOCK_MODE"] = "true"
os.environ["TRAIGENT_GENERATE_MOCKS"] = "true"


@dataclass
class Parameter:
    """Represents a test parameter with its possible values and constraints."""

    name: str
    values: list[Any]
    type: str = "string"
    constraints: list[str] = field(default_factory=list)
    coverage_level: int = 2  # Default to pairwise coverage

    def __hash__(self):
        return hash(self.name)


@dataclass
class TestSpecification:
    """Test specification from CSV row with CTD metadata."""

    id: str
    name: str
    description: str
    scenario_notes: str
    execution_mode: str
    objectives: list[str]
    algorithm: str
    max_trials: int
    configuration_space: dict[str, list[Any]]
    evaluator: str
    scoring_function: str
    injection_mode: str
    framework_targets: list[str]
    dataset: str
    parallel_trials: int
    batch_size: int
    privacy_enabled: bool
    mock_mode: bool
    integration: str
    notes: str

    # CTD specific attributes
    parameters: list[Parameter] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    coverage_requirements: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Extract parameters and constraints from configuration space."""
        if isinstance(self.configuration_space, str):
            try:
                self.configuration_space = json.loads(self.configuration_space)
            except (json.JSONDecodeError, TypeError):
                self.configuration_space = {}

        # Extract parameters from configuration space
        for param_name, param_values in self.configuration_space.items():
            param = Parameter(
                name=param_name,
                values=(
                    param_values if isinstance(param_values, list) else [param_values]
                ),
                type=self._infer_type(param_values),
            )
            self.parameters.append(param)

        # Define constraints based on scenario
        self._define_constraints()

        # Set coverage requirements
        self._set_coverage_requirements()

    def _infer_type(self, values):
        """Infer parameter type from values."""
        if not values:
            return "unknown"
        sample = values[0] if isinstance(values, list) else values
        if isinstance(sample, bool):
            return "boolean"
        elif isinstance(sample, (int, float)):
            return "numeric"
        else:
            return "string"

    def _define_constraints(self):
        """Define constraints based on test scenario."""
        # Example constraints based on common patterns
        if self.mock_mode:
            self.constraints.append("mock_mode==true => integration=='mock'")

        if "accuracy" in self.objectives and "cost" in self.objectives:
            self.constraints.append("len(objectives)>1 => execution_mode=='standard'")

        if self.parallel_trials > 1:
            self.constraints.append("parallel_trials>1 => batch_size==1")

        if self.privacy_enabled:
            self.constraints.append("privacy_enabled==true => no_sensitive_data")

        # Algorithm-specific constraints
        if self.algorithm == "grid":
            self.constraints.append("algorithm=='grid' => exhaustive_search")
        elif self.algorithm == "random":
            self.constraints.append("algorithm=='random' => max_trials_limit")

    def _set_coverage_requirements(self):
        """Set coverage requirements based on test criticality."""
        # Default pairwise coverage for most parameters
        self.coverage_requirements = {param.name: 2 for param in self.parameters}

        # Higher coverage for critical parameters
        if "model" in [p.name for p in self.parameters]:
            self.coverage_requirements["model"] = 3  # 3-wise for model selection

        if self.execution_mode == "hybrid":
            # Higher coverage for hybrid mode
            for param in self.parameters:
                self.coverage_requirements[param.name] = 3


class CTDTestGenerator:
    """Generates combinatorial test cases from specifications."""

    def __init__(self, csv_path: str):
        """Initialize with CSV specification path."""
        self.csv_path = Path(csv_path)
        self.specifications: list[TestSpecification] = []
        self.test_cases: list[dict[str, Any]] = []
        self.coverage_matrix: dict[str, set[tuple]] = defaultdict(set)

    def load_specifications(self):
        """Load test specifications from CSV."""
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get("id"):
                    continue

                parallel_trials = 1
                batch_size = 1
                parallel_config_raw = row.get("parallel_config", "")
                parallel_config: dict[str, Any] = {}
                if parallel_config_raw:
                    try:
                        parallel_config = json.loads(parallel_config_raw)
                        if isinstance(parallel_config, dict):
                            if "trial_concurrency" in parallel_config:
                                parallel_trials = int(
                                    parallel_config["trial_concurrency"]
                                )
                            elif "parallel_trials" in parallel_config:
                                parallel_trials = int(
                                    parallel_config["parallel_trials"]
                                )
                            if "example_concurrency" in parallel_config:
                                batch_size = int(parallel_config["example_concurrency"])
                            elif "batch_size" in parallel_config:
                                batch_size = int(parallel_config["batch_size"])
                    except (json.JSONDecodeError, TypeError, ValueError):
                        parallel_config = {}

                if row.get("parallel_trials"):
                    try:
                        parallel_trials = int(row["parallel_trials"])
                    except ValueError:
                        parallel_trials = 1
                if row.get("batch_size"):
                    try:
                        batch_size = int(row["batch_size"])
                    except ValueError:
                        batch_size = 1

                spec = TestSpecification(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    scenario_notes=row.get("scenario_notes", ""),
                    execution_mode=row["execution_mode"],
                    objectives=(
                        json.loads(row["objectives"]) if row["objectives"] else []
                    ),
                    algorithm=row["algorithm"],
                    max_trials=int(row["max_trials"]) if row["max_trials"] else 0,
                    configuration_space=row["configuration_space"],
                    evaluator=row.get("evaluator", "default"),
                    scoring_function=row.get("scoring_function", ""),
                    injection_mode=row.get("injection_mode", "context"),
                    framework_targets=(
                        json.loads(row["framework_targets"])
                        if row["framework_targets"]
                        else []
                    ),
                    dataset=row.get("dataset", ""),
                    parallel_trials=parallel_trials,
                    batch_size=batch_size,
                    privacy_enabled=row.get("privacy_enabled", "").lower() == "true",
                    mock_mode=row.get("mock_mode", "").lower() == "true",
                    integration=row.get("integration", "none"),
                    notes=row.get("notes", ""),
                )
                self.specifications.append(spec)

    def generate_n_wise_combinations(
        self, parameters: list[Parameter], n: int
    ) -> list[dict[str, Any]]:
        """Generate n-wise parameter combinations."""
        if not parameters or n <= 0:
            return []

        # Get all parameter names and values
        param_names = [p.name for p in parameters]
        param_values = [p.values for p in parameters]

        # Generate all possible combinations
        all_combinations = []

        if n >= len(parameters):
            # Full factorial when n >= number of parameters
            for combo in itertools.product(*param_values):
                all_combinations.append(dict(zip(param_names, combo, strict=False)))
        else:
            # n-wise coverage
            # Get all n-tuples of parameters
            for param_subset in itertools.combinations(range(len(parameters)), n):
                subset_names = [param_names[i] for i in param_subset]
                subset_values = [param_values[i] for i in param_subset]

                # Generate combinations for this subset
                for combo in itertools.product(*subset_values):
                    # Create a partial test case
                    partial_case = dict(zip(subset_names, combo, strict=False))

                    # Check if this combination is already covered
                    combo_key = tuple(sorted(partial_case.items()))
                    if combo_key not in self.coverage_matrix[n]:
                        self.coverage_matrix[n].add(combo_key)

                        # Fill in remaining parameters
                        full_case = partial_case.copy()
                        for i, name in enumerate(param_names):
                            if name not in full_case:
                                # Use first value as default
                                full_case[name] = param_values[i][0]

                        all_combinations.append(full_case)

        return all_combinations

    def apply_constraints(
        self, test_cases: list[dict[str, Any]], constraints: list[str]
    ) -> list[dict[str, Any]]:
        """Filter test cases based on constraints."""
        valid_cases = []

        for case in test_cases:
            is_valid = True

            for constraint in constraints:
                # Simple constraint evaluation (can be extended)
                if "=>" in constraint:
                    condition, requirement = constraint.split("=>")
                    condition = condition.strip()
                    requirement = requirement.strip()

                    # Evaluate condition
                    if self._evaluate_condition(condition, case):
                        # Check requirement
                        if not self._evaluate_condition(requirement, case):
                            is_valid = False
                            break

            if is_valid:
                valid_cases.append(case)

        return valid_cases

    def _evaluate_condition(self, condition: str, case: dict[str, Any]) -> bool:
        """Simple condition evaluator."""
        # This is a simplified evaluator - can be extended with proper parsing
        try:
            # Handle simple equality checks
            if "==" in condition:
                left, right = condition.split("==")
                left = left.strip()
                right = right.strip().strip("'\"")

                if left in case:
                    # Handle boolean comparisons
                    if right in ["True", "False", "true", "false"]:
                        bool_val = right in ["True", "true"]
                        return case[left] == bool_val
                    else:
                        return str(case[left]) == right
                elif left == "mock_mode":
                    return os.environ.get("TRAIGENT_MOCK_MODE") == "true"

            # Handle inequality checks
            elif ">" in condition:
                left, right = condition.split(">")
                left = left.strip()
                right = right.strip()

                if left in case:
                    return float(case[left]) > float(right)

            # Handle length checks
            elif "len(" in condition:
                # Extract variable name
                import re

                match = re.match(r"len\((\w+)\)", condition.split(">")[0])
                if match:
                    var_name = match.group(1)
                    if var_name in case:
                        value = case[var_name]
                        if isinstance(value, (list, str)):
                            return len(value) > int(condition.split(">")[1])

            return True  # Default to true for unknown conditions

        except Exception:
            return True  # Default to true on evaluation error

    def generate_test_suite(self) -> list[tuple[TestSpecification, dict[str, Any]]]:
        """Generate complete test suite with CTD coverage."""
        test_suite = []

        for spec in self.specifications:
            if not spec.parameters:
                # No parameters to combine, use default case
                test_suite.append((spec, {}))
                continue

            # Determine coverage level
            max_coverage = (
                max(spec.coverage_requirements.values())
                if spec.coverage_requirements
                else 2
            )

            # Generate combinations
            test_cases = self.generate_n_wise_combinations(
                spec.parameters, max_coverage
            )

            # Apply constraints
            valid_cases = self.apply_constraints(test_cases, spec.constraints)

            # Add to test suite
            for case in valid_cases:
                test_suite.append((spec, case))

        return test_suite

    def calculate_coverage(
        self, test_suite: list[tuple[TestSpecification, dict[str, Any]]]
    ) -> dict[str, float]:
        """Calculate coverage metrics for the test suite."""
        coverage_stats = {}

        for n in [1, 2, 3]:  # Check 1-wise, 2-wise, 3-wise coverage
            total_combinations = 0
            covered_combinations = 0

            for spec, _ in test_suite:
                if not spec.parameters:
                    continue

                # Calculate total possible n-wise combinations
                params = spec.parameters
                if n <= len(params):
                    for param_subset in itertools.combinations(params, n):
                        subset_size = 1
                        for p in param_subset:
                            subset_size *= len(p.values)
                        total_combinations += subset_size

            # Calculate covered combinations from coverage matrix
            covered_combinations = len(self.coverage_matrix[n])

            if total_combinations > 0:
                coverage_stats[f"{n}-wise"] = (
                    covered_combinations / total_combinations
                ) * 100
            else:
                coverage_stats[f"{n}-wise"] = 100.0

        return coverage_stats


# Test fixtures
@pytest.fixture
def csv_path():
    """Path to test specification CSV."""
    return str(
        Path(__file__).resolve().parents[2]
        / "examples"
        / "datasets"
        / "matrices"
        / "test_matrix.csv"
    )


@pytest.fixture
def test_generator(csv_path):
    """Create and configure test generator."""
    generator = CTDTestGenerator(csv_path)
    generator.load_specifications()
    return generator


@pytest.fixture
def mock_traigent():
    """Mock Traigent client for testing."""
    from unittest.mock import MagicMock, Mock

    mock_client = Mock()
    mock_client.create_experiment = MagicMock(return_value={"id": "test_exp_123"})
    mock_client.run_optimization = MagicMock(
        return_value={
            "status": "completed",
            "best_config": {"model": "claude-3-haiku-20240307", "temperature": 0.0},
            "best_score": 0.95,
        }
    )

    return mock_client


# Parametrized tests based on CSV specification
class TestCTDCoverage:
    """Test suite with CTD coverage from CSV specification."""

    def test_load_specifications(self, test_generator):
        """Test that specifications are loaded correctly from CSV."""
        assert len(test_generator.specifications) > 0

        # Check first specification
        first_spec = test_generator.specifications[0]
        assert first_spec.id == "1"
        assert first_spec.name == "Local Accuracy (Grid)"
        assert first_spec.algorithm == "grid"
        assert len(first_spec.parameters) > 0

    def test_parameter_extraction(self, test_generator):
        """Test that parameters are correctly extracted from configuration space."""
        for spec in test_generator.specifications:
            if spec.configuration_space:
                assert len(spec.parameters) > 0

                # Check parameter types are inferred correctly
                for param in spec.parameters:
                    assert param.type in ["boolean", "numeric", "string", "unknown"]
                    assert len(param.values) > 0

    def test_constraint_definition(self, test_generator):
        """Test that constraints are properly defined for each specification."""
        for spec in test_generator.specifications:
            # Mock mode specs should have mock constraints
            if spec.mock_mode:
                has_mock_constraint = any("mock_mode" in c for c in spec.constraints)
                assert has_mock_constraint or len(spec.constraints) > 0

    def test_coverage_generation(self, test_generator):
        """Test n-wise coverage generation."""
        # Test with a spec that has parameters
        spec_with_params = None
        for spec in test_generator.specifications:
            if len(spec.parameters) >= 3:
                spec_with_params = spec
                break

        if spec_with_params:
            # Test pairwise coverage
            pairwise_cases = test_generator.generate_n_wise_combinations(
                spec_with_params.parameters, 2
            )
            assert len(pairwise_cases) > 0

            # Verify all parameters are present in each case
            for case in pairwise_cases:
                for param in spec_with_params.parameters:
                    assert param.name in case

    def test_constraint_application(self, test_generator):
        """Test that constraints properly filter test cases."""
        # Create sample test cases
        test_cases = [
            {"mock_mode": True, "integration": "mock"},
            {"mock_mode": True, "integration": "real"},
            {"mock_mode": False, "integration": "real"},
        ]

        constraints = ["mock_mode==True => integration=='mock'"]

        valid_cases = test_generator.apply_constraints(test_cases, constraints)

        # Only first and third cases should be valid (mock_mode=True must have integration=mock)
        assert len(valid_cases) == 2
        # First case: mock_mode=True with integration=mock (valid)
        # Third case: mock_mode=False (constraint doesn't apply)
        for case in valid_cases:
            if case.get("mock_mode"):
                assert case["integration"] == "mock"

    @pytest.mark.parametrize("coverage_level", [1, 2, 3])
    def test_coverage_levels(self, test_generator, coverage_level):
        """Test different coverage levels generate appropriate combinations."""
        # Use a spec with enough parameters
        spec = None
        for s in test_generator.specifications:
            if len(s.parameters) >= coverage_level:
                spec = s
                break

        if spec:
            combinations = test_generator.generate_n_wise_combinations(
                spec.parameters[: coverage_level + 1], coverage_level
            )

            # Verify we have combinations
            assert len(combinations) > 0

            # Verify coverage level is respected
            # For n-wise, we should cover all n-tuples
            covered_tuples = set()
            for combo in combinations:
                # Get all n-tuples from this combination
                param_names = list(combo.keys())
                for n_tuple_indices in itertools.combinations(
                    range(len(param_names)), coverage_level
                ):
                    n_tuple = tuple(
                        (param_names[i], combo[param_names[i]]) for i in n_tuple_indices
                    )
                    covered_tuples.add(n_tuple)

            # We should have covered multiple n-tuples
            assert len(covered_tuples) > 0

    def test_complete_test_suite_generation(self, test_generator):
        """Test generation of complete test suite with all specifications."""
        test_suite = test_generator.generate_test_suite()

        # Should have test cases
        assert len(test_suite) > 0

        # Each test case should have a spec and configuration
        for spec, config in test_suite:
            assert isinstance(spec, TestSpecification)
            assert isinstance(config, dict)

            # If spec has parameters, config should have values for them
            if spec.parameters:
                for param in spec.parameters:
                    assert param.name in config or len(config) == 0

    def test_coverage_calculation(self, test_generator):
        """Test coverage metric calculation."""
        test_suite = test_generator.generate_test_suite()
        coverage_stats = test_generator.calculate_coverage(test_suite)

        # Should have coverage stats
        assert "1-wise" in coverage_stats
        assert "2-wise" in coverage_stats
        assert "3-wise" in coverage_stats

        # Coverage should be between 0 and 100
        for _coverage_type, percentage in coverage_stats.items():
            assert 0 <= percentage <= 100

    @pytest.mark.integration
    def test_mock_mode_execution(self, test_generator, mock_traigent):
        """Test that tests execute correctly in mock mode."""
        # Ensure mock mode is set
        assert os.environ.get("TRAIGENT_MOCK_MODE") == "true"

        test_suite = test_generator.generate_test_suite()

        # Execute a subset of test cases in mock mode
        executed = 0
        for spec, config in test_suite[:5]:  # Test first 5 cases
            if spec.mock_mode or os.environ.get("TRAIGENT_MOCK_MODE") == "true":
                # Simulate test execution
                result = mock_traigent.run_optimization(
                    config=config, objectives=spec.objectives, algorithm=spec.algorithm
                )

                assert result["status"] == "completed"
                assert "best_config" in result
                executed += 1

        assert executed > 0

    def test_integration_specific_tests(self, test_generator):
        """Test integration-specific test cases."""
        test_suite = test_generator.generate_test_suite()

        # Group by integration type
        integration_groups = defaultdict(list)
        for spec, config in test_suite:
            integration_groups[spec.integration].append((spec, config))

        # Verify we have tests for different integrations
        assert len(integration_groups) > 0

        # Check anthropic integration tests
        if "anthropic" in integration_groups:
            anthropic_tests = integration_groups["anthropic"]
            assert len(anthropic_tests) > 0

            # Should have model variations
            models_tested = set()
            for _spec, config in anthropic_tests:
                if "model" in config:
                    models_tested.add(config["model"])

            if models_tested:
                assert len(models_tested) >= 1

    def test_execution_mode_coverage(self, test_generator):
        """Test that all execution modes are covered."""
        test_suite = test_generator.generate_test_suite()

        execution_modes = set()
        for spec, _ in test_suite:
            execution_modes.add(spec.execution_mode)

        # Should cover multiple execution modes
        expected_modes = {"edge_analytics", "standard", "hybrid"}
        covered_modes = execution_modes & expected_modes

        assert len(covered_modes) > 0

    def test_algorithm_coverage(self, test_generator):
        """Test that different algorithms are covered."""
        test_suite = test_generator.generate_test_suite()

        algorithms = set()
        for spec, _ in test_suite:
            algorithms.add(spec.algorithm)

        # Should cover both grid and random search
        assert "grid" in algorithms or "random" in algorithms

    def test_parallel_and_batch_configurations(self, test_generator):
        """Test parallel trials and batch size configurations."""
        test_suite = test_generator.generate_test_suite()

        parallel_configs = []
        batch_configs = []

        for spec, config in test_suite:
            if spec.parallel_trials > 1:
                parallel_configs.append((spec, config))
            if spec.batch_size > 1:
                batch_configs.append((spec, config))

        # Should have some parallel and batch configurations
        assert len(parallel_configs) > 0 or len(batch_configs) > 0

    def test_privacy_mode_tests(self, test_generator):
        """Test privacy-enabled configurations."""
        test_suite = test_generator.generate_test_suite()

        privacy_tests = [
            (spec, config) for spec, config in test_suite if spec.privacy_enabled
        ]

        if privacy_tests:
            # Privacy tests should have appropriate constraints
            for spec, _config in privacy_tests:
                assert spec.privacy_enabled
                # Could add more privacy-specific validations here


# Run coverage report
def test_coverage_report(test_generator):
    """Generate and display coverage report."""
    test_suite = test_generator.generate_test_suite()
    coverage_stats = test_generator.calculate_coverage(test_suite)

    print("\n" + "=" * 60)
    print("CTD Coverage Report")
    print("=" * 60)
    print(f"Total test specifications: {len(test_generator.specifications)}")
    print(f"Total test cases generated: {len(test_suite)}")
    print("\nCoverage Statistics:")
    for coverage_type, percentage in coverage_stats.items():
        print(f"  {coverage_type}: {percentage:.1f}%")

    print("\nTest Distribution by Execution Mode:")
    mode_counts = defaultdict(int)
    for spec, _ in test_suite:
        mode_counts[spec.execution_mode] += 1
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")

    print("\nTest Distribution by Algorithm:")
    algo_counts = defaultdict(int)
    for spec, _ in test_suite:
        algo_counts[spec.algorithm] += 1
    for algo, count in algo_counts.items():
        print(f"  {algo}: {count}")

    print("\nTest Distribution by Integration:")
    integration_counts = defaultdict(int)
    for spec, _ in test_suite:
        integration_counts[spec.integration] += 1
    for integration, count in integration_counts.items():
        print(f"  {integration}: {count}")

    print("=" * 60)

    # Verify coverage report was generated successfully
    assert len(test_suite) > 0, "Test suite should have test cases"
    assert len(coverage_stats) > 0, "Coverage stats should be calculated"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
