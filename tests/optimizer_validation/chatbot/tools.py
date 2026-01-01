"""Tool functions for knowledge graph interaction.

These functions are exposed as tools to the Claude agent for
querying the test knowledge graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Default paths
DEFAULT_GRAPH_PATH = Path(__file__).parent.parent / "viewer" / "graph_data.json"


def _load_graph() -> Any:
    """Load the knowledge graph."""
    from tests.optimizer_validation.viewer.knowledge_graph import TestKnowledgeGraph

    graph_path = DEFAULT_GRAPH_PATH
    if graph_path.exists():
        return TestKnowledgeGraph.load(str(graph_path))
    return TestKnowledgeGraph()


def query_knowledge_graph(
    query_name: str,
    parameters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Query the test knowledge graph using a premade SPARQL query.

    Available queries:
    - tests_by_algorithm: Find tests using a specific algorithm
      params: algorithm (random|grid|optuna_tpe|optuna_cmaes|bayesian)
    - tests_by_injection_execution: Find tests by injection + execution mode
      params: injection (context|seamless|parameter|attribute),
              execution (edge_analytics|privacy|hybrid|cloud|standard)
    - failed_tests: List all failed tests (no params)
    - tests_with_constraints: Find tests using constraints (no params)
    - multi_objective_tests: Find multi-objective optimization tests (no params)
    - weighted_objective_tests: Find weighted objective tests (no params)
    - parallel_tests: Find tests running in parallel mode (no params)
    - failure_mode_tests: Find tests for specific failure modes (no params)
    - tests_by_stop_condition: Find tests by stop condition
      params: condition (max_trials|timeout|convergence|config_exhaustion)
    - high_trial_tests: Find tests with high trial counts
      params: min_trials (integer)
    - dimension_distribution: Show test count per dimension value
      params: dimension (hasInjectionMode|hasAlgorithm|hasExecutionMode|...)
    - coverage_matrix: Get pairwise coverage matrix
      params: dim1, dim2 (dimension property names)
    - tests_by_config_space: Find tests by config space type
      params: config_type (categorical_only|continuous_only|mixed|single_value)
    - search_by_name: Search tests by name pattern
      params: pattern (search string)

    Args:
        query_name: Name of the premade query to execute
        parameters: Dictionary of parameters for the query

    Returns:
        List of result dictionaries from the query
    """
    from tests.optimizer_validation.chatbot.sparql_engine import SPARQLEngine

    graph = _load_graph()

    # Check if graph has SPARQL support
    if not hasattr(graph, "query_sparql"):
        # Fallback to dict-based queries
        return _fallback_query(graph, query_name, parameters or {})

    engine = SPARQLEngine(graph)
    return engine.execute(query_name, **(parameters or {}))


def _fallback_query(
    graph: Any,
    query_name: str,
    params: dict[str, str],
) -> list[dict[str, Any]]:
    """Fallback queries when SPARQL is not available.

    Uses the dict-based knowledge graph directly.
    """
    results: list[dict[str, Any]] = []

    if query_name == "tests_by_algorithm":
        algorithm = params.get("algorithm", "")
        for test_id, test in graph.tests.items():
            dims = test.get("dimensions", {})
            if dims.get("Algorithm") == algorithm:
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "status": test.get("result", {}).get("status"),
                    }
                )

    elif query_name == "failed_tests":
        for test_id, test in graph.tests.items():
            result = test.get("result", {})
            if result.get("status") == "FAIL":
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "errorType": result.get("error_type"),
                    }
                )

    elif query_name == "tests_with_constraints":
        for test_id, test in graph.tests.items():
            dims = test.get("dimensions", {})
            constraint = dims.get("ConstraintUsage")
            if constraint and constraint != "none":
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "constraint": constraint,
                    }
                )

    elif query_name == "multi_objective_tests":
        for test_id, test in graph.tests.items():
            dims = test.get("dimensions", {})
            if dims.get("ObjectiveConfig") == "multi_objective":
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "status": test.get("result", {}).get("status"),
                    }
                )

    elif query_name == "parallel_tests":
        for test_id, test in graph.tests.items():
            dims = test.get("dimensions", {})
            if dims.get("ParallelMode") == "parallel":
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "status": test.get("result", {}).get("status"),
                    }
                )

    elif query_name == "search_by_name":
        pattern = params.get("pattern", "").lower()
        for test_id, test in graph.tests.items():
            name = test.get("name", test_id).lower()
            if pattern in name:
                results.append(
                    {
                        "test": test_id,
                        "name": test.get("name", test_id),
                        "file": test.get("test_file"),
                    }
                )

    elif query_name == "dimension_distribution":
        dim = params.get("dimension", "").replace("has", "")
        counts: dict[str, int] = {}
        for test in graph.tests.values():
            dims = test.get("dimensions", {})
            value = dims.get(dim)
            if value:
                counts[value] = counts.get(value, 0) + 1
        for value, count in sorted(counts.items(), key=lambda x: -x[1]):
            results.append({"value": value, "count": count})

    return results


def list_dimensions() -> dict[str, list[str]]:
    """List all test dimensions and their possible values.

    Returns a dictionary mapping dimension names to their valid values.
    Use this to understand what values can be used when filtering tests.

    Returns:
        Dictionary mapping dimension names to lists of valid values
    """
    return {
        "InjectionMode": ["context", "seamless", "parameter", "attribute"],
        "ExecutionMode": [
            "edge_analytics",
            "privacy",
            "hybrid",
            "cloud",
            "standard",
        ],
        "Algorithm": [
            "random",
            "grid",
            "optuna_tpe",
            "optuna_cmaes",
            "optuna_random",
            "bayesian",
        ],
        "ConfigSpaceType": [
            "categorical_only",
            "continuous_only",
            "mixed",
            "single_value",
        ],
        "ObjectiveConfig": [
            "single_maximize",
            "single_minimize",
            "multi_objective",
            "weighted",
        ],
        "StopCondition": [
            "max_trials",
            "timeout",
            "convergence",
            "config_exhaustion",
        ],
        "ParallelMode": ["sequential", "parallel"],
        "ConstraintUsage": ["none", "config_only", "metric_based"],
        "FailureMode": [
            "function_raises",
            "evaluator_bug",
            "invalid_config",
            "dataset_issue",
        ],
        "ExpectedOutcome": ["success", "partial", "failure", "error"],
    }


def get_test_details(test_id: str) -> dict[str, Any]:
    """Get full details for a specific test by ID or name.

    Args:
        test_id: The test ID or name to look up

    Returns:
        Dictionary with all test details including:
        - name, description
        - dimensions (injection mode, algorithm, etc.)
        - result (status, trial count, best score, etc.)
        - metadata (file, class, method)
    """
    graph = _load_graph()

    # Try exact match first
    if test_id in graph.tests:
        return graph.tests[test_id]

    # Try matching by name
    for _tid, test in graph.tests.items():
        if test.get("name") == test_id:
            return test

    # Try partial match
    test_id_lower = test_id.lower()
    for tid, test in graph.tests.items():
        if (
            test_id_lower in tid.lower()
            or test_id_lower in test.get("name", "").lower()
        ):
            return test

    return {"error": f"Test not found: {test_id}"}


def search_tests(keyword: str) -> list[dict[str, str]]:
    """Search tests by name or description containing keyword.

    Args:
        keyword: Search term to match against test names and descriptions

    Returns:
        List of matching tests with their ID, name, and file
    """
    graph = _load_graph()
    keyword_lower = keyword.lower()
    results: list[dict[str, str]] = []

    for test_id, test in graph.tests.items():
        name = test.get("name", "") or ""
        description = test.get("description", "") or ""

        if keyword_lower in name.lower() or keyword_lower in description.lower():
            results.append(
                {
                    "id": test_id,
                    "name": name,
                    "file": test.get("test_file", "") or "",
                    "description": (
                        description[:100] + "..."
                        if description and len(description) > 100
                        else (description or "")
                    ),
                }
            )

    return results


def get_coverage_matrix(dim1: str, dim2: str) -> dict[str, dict[str, int]]:
    """Get pairwise coverage matrix for two dimensions.

    Shows how many tests cover each combination of values for the
    specified dimensions. Useful for finding coverage gaps.

    Args:
        dim1: First dimension name (e.g., "InjectionMode")
        dim2: Second dimension name (e.g., "Algorithm")

    Returns:
        Nested dictionary: matrix[dim1_value][dim2_value] = count

    Example:
        >>> get_coverage_matrix("InjectionMode", "Algorithm")
        {
            "context": {"random": 5, "grid": 3, "optuna_tpe": 2, ...},
            "seamless": {"random": 1, "grid": 0, ...},
            ...
        }
    """
    graph = _load_graph()

    # Get valid values for each dimension
    all_dims = list_dimensions()
    dim1_values = all_dims.get(dim1, [])
    dim2_values = all_dims.get(dim2, [])

    # Initialize matrix with zeros
    matrix: dict[str, dict[str, int]] = {}
    for v1 in dim1_values:
        matrix[v1] = dict.fromkeys(dim2_values, 0)

    # Count tests for each combination
    for test in graph.tests.values():
        dims = test.get("dimensions", {})
        v1 = dims.get(dim1)
        v2 = dims.get(dim2)
        if v1 and v2 and v1 in matrix and v2 in matrix.get(v1, {}):
            matrix[v1][v2] += 1

    return matrix


def get_coverage_gaps(dim1: str, dim2: str) -> list[dict[str, str]]:
    """Find dimension value pairs with no test coverage.

    Args:
        dim1: First dimension name
        dim2: Second dimension name

    Returns:
        List of gaps, each with dim1_value and dim2_value
    """
    matrix = get_coverage_matrix(dim1, dim2)
    gaps: list[dict[str, str]] = []

    for v1, row in matrix.items():
        for v2, count in row.items():
            if count == 0:
                gaps.append({dim1: v1, dim2: v2})

    return gaps


def get_test_stats() -> dict[str, Any]:
    """Get overall statistics about the test suite.

    Returns:
        Dictionary with:
        - total_tests: Total number of tests
        - passed: Number of passed tests
        - failed: Number of failed tests
        - not_run: Tests without results
        - by_dimension: Counts per dimension value
    """
    graph = _load_graph()

    stats: dict[str, Any] = {
        "total_tests": len(graph.tests),
        "passed": 0,
        "failed": 0,
        "not_run": 0,
        "by_dimension": {},
    }

    for test in graph.tests.values():
        result = test.get("result", {})
        status = result.get("status")
        if status == "PASS":
            stats["passed"] += 1
        elif status == "FAIL":
            stats["failed"] += 1
        else:
            stats["not_run"] += 1

    # Count by dimension
    all_dims = list_dimensions()
    for dim_name in all_dims:
        dim_counts: dict[str, int] = {}
        for test in graph.tests.values():
            dims = test.get("dimensions", {})
            value = dims.get(dim_name)
            if value:
                dim_counts[value] = dim_counts.get(value, 0) + 1
        stats["by_dimension"][dim_name] = dim_counts

    return stats
