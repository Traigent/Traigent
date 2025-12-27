"""SPARQL query engine with premade templates.

This module provides a query engine for the test knowledge graph,
with both premade templates for common queries and support for
dynamic query generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.optimizer_validation.viewer.knowledge_graph import TestKnowledgeGraph

# Namespace prefix for SPARQL queries
PREFIX = """
PREFIX : <http://traigent.io/test#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Premade query templates
QUERIES: dict[str, str] = {
    # Find tests by algorithm
    "tests_by_algorithm": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasAlgorithm :algorithm_{algorithm} .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find tests by injection mode + execution mode combination
    "tests_by_injection_execution": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasInjectionMode :injection_{injection} ;
                  :hasExecutionMode :execution_{execution} .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # List all failed tests
    "failed_tests": """
        SELECT ?test ?name ?errorType WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasResult ?result .
            ?result :hasStatus "FAIL" .
            OPTIONAL { ?result :hasErrorType ?errorType . }
        }
        ORDER BY ?name
    """,
    # Find tests with constraints
    "tests_with_constraints": """
        SELECT ?test ?name ?constraint WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasConstraintUsage ?constraint .
            FILTER(?constraint != :constraint_none)
        }
        ORDER BY ?name
    """,
    # Find multi-objective tests
    "multi_objective_tests": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasObjectiveConfig :objective_multi_objective .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find weighted objective tests
    "weighted_objective_tests": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasObjectiveConfig :objective_weighted .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find parallel execution tests
    "parallel_tests": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasParallelMode :parallel_parallel .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find failure mode tests
    "failure_mode_tests": """
        SELECT ?test ?name ?mode ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasFailureMode ?mode .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find tests by stop condition
    "tests_by_stop_condition": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasStopCondition :stop_{condition} .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Find tests with high trial counts
    "high_trial_tests": """
        SELECT ?test ?name ?trials WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasResult ?result .
            ?result :hasTrialCount ?trials .
            FILTER(?trials > {min_trials})
        }
        ORDER BY DESC(?trials)
    """,
    # Get dimension distribution (count tests per value)
    "dimension_distribution": """
        SELECT ?value (COUNT(?test) as ?count) WHERE {
            ?test a :TestScenario ;
                  :{dimension} ?value .
        }
        GROUP BY ?value
        ORDER BY DESC(?count)
    """,
    # Get test details
    "test_details": """
        SELECT ?prop ?value WHERE {
            :{test_id} ?prop ?value .
        }
    """,
    # Find coverage gaps for two dimensions
    "coverage_matrix": """
        SELECT ?dim1Value ?dim2Value (COUNT(?test) as ?count) WHERE {
            ?test a :TestScenario ;
                  :{dim1} ?dim1Value ;
                  :{dim2} ?dim2Value .
        }
        GROUP BY ?dim1Value ?dim2Value
        ORDER BY ?dim1Value ?dim2Value
    """,
    # Find tests by config space type
    "tests_by_config_space": """
        SELECT ?test ?name ?status WHERE {
            ?test a :TestScenario ;
                  :hasName ?name ;
                  :hasConfigSpaceType :config_{config_type} .
            OPTIONAL { ?test :hasResult ?result . ?result :hasStatus ?status . }
        }
        ORDER BY ?name
    """,
    # Search tests by name pattern (uses CONTAINS)
    "search_by_name": """
        SELECT ?test ?name ?file WHERE {
            ?test a :TestScenario ;
                  :hasName ?name .
            OPTIONAL { ?test :hasTestFile ?file . }
            FILTER(CONTAINS(LCASE(?name), LCASE("{pattern}")))
        }
        ORDER BY ?name
    """,
    # Get all tests with their dimensions
    "all_tests_summary": """
        SELECT ?test ?name ?injection ?execution ?algorithm ?outcome WHERE {
            ?test a :TestScenario ;
                  :hasName ?name .
            OPTIONAL { ?test :hasInjectionMode ?injection . }
            OPTIONAL { ?test :hasExecutionMode ?execution . }
            OPTIONAL { ?test :hasAlgorithm ?algorithm . }
            OPTIONAL { ?test :hasResult ?result . ?result :hasActualOutcome ?outcome . }
        }
        ORDER BY ?name
        LIMIT 100
    """,
}


class SPARQLEngine:
    """Execute SPARQL queries against the knowledge graph."""

    def __init__(self, graph: TestKnowledgeGraph) -> None:
        """Initialize with a knowledge graph.

        Args:
            graph: The knowledge graph to query
        """
        self.graph = graph

    def execute(self, query_name: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a premade query with parameters.

        Args:
            query_name: Name of the premade query
            **params: Parameters to substitute in the query

        Returns:
            List of result dictionaries

        Raises:
            ValueError: If query_name is not found
        """
        template = QUERIES.get(query_name)
        if not template:
            available = ", ".join(sorted(QUERIES.keys()))
            raise ValueError(f"Unknown query: {query_name}. Available: {available}")

        query = PREFIX + template.format(**params)
        return self.graph.query_sparql(query)

    def execute_raw(self, query: str) -> list[dict[str, Any]]:
        """Execute a raw SPARQL query.

        Args:
            query: The SPARQL query string

        Returns:
            List of result dictionaries
        """
        # Add prefix if not present
        if "PREFIX" not in query:
            query = PREFIX + query
        return self.graph.query_sparql(query)

    def list_queries(self) -> dict[str, str]:
        """List available premade queries with their descriptions.

        Returns:
            Dictionary mapping query names to their templates
        """
        return dict(QUERIES)

    def get_query_params(self, query_name: str) -> list[str]:
        """Get required parameters for a query.

        Args:
            query_name: Name of the query

        Returns:
            List of parameter names

        Raises:
            ValueError: If query_name is not found
        """
        template = QUERIES.get(query_name)
        if not template:
            raise ValueError(f"Unknown query: {query_name}")

        # Extract parameter names from {param} placeholders
        import re

        params = re.findall(r"\{(\w+)\}", template)
        # Filter out SPARQL built-in patterns
        return [p for p in set(params) if not p.startswith("?")]
