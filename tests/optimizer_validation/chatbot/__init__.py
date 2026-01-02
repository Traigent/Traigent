"""Chatbot module for test knowledge graph interaction.

This module provides a natural language interface for querying the test suite
using Claude API with tool use. It integrates with the knowledge graph and
supports both premade and dynamic SPARQL queries.

Usage:
    # Interactive mode
    python -m tests.optimizer_validation.chatbot.cli

    # Single query
    python -m tests.optimizer_validation.chatbot.cli -q "Which tests use optuna?"
"""

from tests.optimizer_validation.chatbot.sparql_engine import SPARQLEngine
from tests.optimizer_validation.chatbot.tools import (
    get_coverage_matrix,
    get_test_details,
    list_dimensions,
    query_knowledge_graph,
    search_tests,
)

__all__ = [
    "SPARQLEngine",
    "query_knowledge_graph",
    "list_dimensions",
    "get_test_details",
    "search_tests",
    "get_coverage_matrix",
]
