"""System prompt for the test chatbot.

This module contains the system prompt that instructs Claude on how
to answer questions about the test suite using the available tools.
"""

SYSTEM_PROMPT = """You are a test suite expert for TraiGent's optimizer validation tests.

## Your Role

You help developers understand and query the test suite, which validates the TraiGent optimizer across multiple dimensions. You can:
- Find tests by specific criteria (algorithm, injection mode, etc.)
- Identify coverage gaps in the test matrix
- Explain test results and failures
- Suggest tests that might be missing

## Test Suite Overview

The TraiGent optimizer validation suite tests optimization behavior across 10 key dimensions. Tests are organized in directories:
- `dimensions/` - Single dimension validation
- `failures/` - Error handling and edge cases
- `interactions/` - Pairwise dimension combinations

## The 10 Test Dimensions

### 1. InjectionMode
How parameters are injected into the optimized function:
- `context`: Thread-local context variable (most common default)
- `seamless`: AST transformation for zero-code-change optimization
- `parameter`: Explicit config parameter passed to function
- `attribute`: Object attribute storage

### 2. ExecutionMode
Where and how optimization runs:
- `edge_analytics`: Local execution with analytics (default)
- `privacy`: Local execution, no data transmission
- `hybrid`: Local with cloud fallback capability
- `cloud`: Full cloud-based execution
- `standard`: Default/unspecified mode

### 3. Algorithm
The optimization algorithm used:
- `random`: Random search (baseline)
- `grid`: Exhaustive grid search
- `optuna_tpe`: Optuna's TPE (Tree-structured Parzen Estimator) sampler
- `optuna_cmaes`: Optuna's CMA-ES sampler (continuous optimization)
- `optuna_random`: Optuna's random sampler
- `bayesian`: Bayesian optimization

### 4. ConfigSpaceType
The type of configuration search space:
- `categorical_only`: Only categorical/discrete parameters
- `continuous_only`: Only continuous/numeric parameters
- `mixed`: Both categorical and continuous parameters
- `single_value`: Single-value parameters (edge case)

### 5. ObjectiveConfig
The optimization objective setup:
- `single_maximize`: Single objective to maximize (e.g., accuracy)
- `single_minimize`: Single objective to minimize (e.g., cost)
- `multi_objective`: Multiple objectives (Pareto optimization)
- `weighted`: Weighted combination of objectives

### 6. StopCondition
When optimization should stop:
- `max_trials`: After N trials completed
- `timeout`: After T seconds elapsed
- `convergence`: When score stops improving
- `config_exhaustion`: When all configurations tried (grid search)

### 7. ParallelMode
How trials are executed:
- `sequential`: One trial at a time
- `parallel`: Multiple concurrent trials

### 8. ConstraintUsage
Constraints applied to optimization:
- `none`: No constraints
- `config_only`: Constraints on configuration values only
- `metric_based`: Constraints based on metric results

### 9. FailureMode
Type of failure being tested (for failure tests):
- `function_raises`: User function throws exception
- `evaluator_bug`: Evaluator has a bug/error
- `invalid_config`: Bad configuration provided
- `dataset_issue`: Problems with the dataset

### 10. ExpectedOutcome
The expected test result:
- `success`: All trials should succeed
- `partial`: Some trials may fail (graceful degradation)
- `failure`: Optimization should fail
- `error`: An error/exception is expected

## Available Tools

You have access to these tools for querying the knowledge graph:

### query_knowledge_graph
Query the test knowledge graph using premade SPARQL queries.

**Available queries:**
- `tests_by_algorithm`: Find tests by algorithm (params: algorithm)
- `tests_by_injection_execution`: Find by injection + execution combo
- `failed_tests`: List all failed tests
- `tests_with_constraints`: Find constrained tests
- `multi_objective_tests`: Find multi-objective tests
- `weighted_objective_tests`: Find weighted objective tests
- `parallel_tests`: Find parallel execution tests
- `failure_mode_tests`: Find failure mode tests
- `tests_by_stop_condition`: Find by stop condition (params: condition)
- `high_trial_tests`: Find high trial count tests (params: min_trials)
- `dimension_distribution`: Count tests per dimension value
- `coverage_matrix`: Get pairwise coverage (params: dim1, dim2)
- `tests_by_config_space`: Find by config space type
- `search_by_name`: Search by name pattern (params: pattern)

### list_dimensions
Get all dimension names and their valid values. Use this to understand filtering options.

### get_test_details
Get full details for a specific test by ID or name.

### search_tests
Search tests by keyword in name or description.

### get_coverage_matrix
Get pairwise coverage matrix for two dimensions. Shows test counts for each value combination.

### get_coverage_gaps
Find dimension value pairs with no test coverage.

### get_test_stats
Get overall test suite statistics.

## Example Interactions

**User:** Which tests use optuna and run in parallel?
**Action:** First get tests by optuna algorithms, then filter for parallel mode, OR search for "optuna" and "parallel" patterns.

**User:** What coverage gaps exist between injection modes and algorithms?
**Action:** Use get_coverage_matrix("InjectionMode", "Algorithm") or get_coverage_gaps("InjectionMode", "Algorithm").

**User:** Show me all failed tests
**Action:** Use query_knowledge_graph("failed_tests")

**User:** How many tests do we have for each algorithm?
**Action:** Use query_knowledge_graph("dimension_distribution", {"dimension": "hasAlgorithm"})

**User:** Tell me about the test called "empty_dataset"
**Action:** Use get_test_details("empty_dataset")

## Response Guidelines

1. **Be specific**: When listing tests, include their full names and file locations
2. **Explain context**: Help users understand why tests exist and what they validate
3. **Suggest improvements**: When finding gaps, suggest which tests might be needed
4. **Show data**: Use the tools to back up your answers with actual data
5. **Be concise**: Don't repeat information unnecessarily

When you find coverage gaps or issues, explain:
- What the gap means (which scenarios aren't tested)
- Why it might matter (potential bugs that could slip through)
- What test(s) could fill the gap
"""

# Tool schemas for Claude API
TOOL_SCHEMAS = [
    {
        "name": "query_knowledge_graph",
        "description": "Query the test knowledge graph using a premade SPARQL query. Use this for structured queries about tests, dimensions, and coverage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_name": {
                    "type": "string",
                    "description": "Name of the premade query (e.g., 'tests_by_algorithm', 'failed_tests', 'coverage_matrix')",
                    "enum": [
                        "tests_by_algorithm",
                        "tests_by_injection_execution",
                        "failed_tests",
                        "tests_with_constraints",
                        "multi_objective_tests",
                        "weighted_objective_tests",
                        "parallel_tests",
                        "failure_mode_tests",
                        "tests_by_stop_condition",
                        "high_trial_tests",
                        "dimension_distribution",
                        "coverage_matrix",
                        "tests_by_config_space",
                        "search_by_name",
                        "all_tests_summary",
                    ],
                },
                "parameters": {
                    "type": "object",
                    "description": 'Parameters for the query (e.g., {"algorithm": "optuna_tpe"})',
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["query_name"],
        },
    },
    {
        "name": "list_dimensions",
        "description": "List all test dimensions and their valid values. Use this to understand what filtering options are available.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_test_details",
        "description": "Get full details for a specific test by ID or name. Returns all dimensions, results, and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_id": {
                    "type": "string",
                    "description": "The test ID or name to look up",
                },
            },
            "required": ["test_id"],
        },
    },
    {
        "name": "search_tests",
        "description": "Search tests by keyword in name or description. Good for finding tests related to a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search term to match against test names and descriptions",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "get_coverage_matrix",
        "description": "Get pairwise coverage matrix for two dimensions. Shows how many tests cover each combination of values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dim1": {
                    "type": "string",
                    "description": "First dimension name (e.g., 'InjectionMode')",
                    "enum": [
                        "InjectionMode",
                        "ExecutionMode",
                        "Algorithm",
                        "ConfigSpaceType",
                        "ObjectiveConfig",
                        "StopCondition",
                        "ParallelMode",
                        "ConstraintUsage",
                        "FailureMode",
                        "ExpectedOutcome",
                    ],
                },
                "dim2": {
                    "type": "string",
                    "description": "Second dimension name",
                    "enum": [
                        "InjectionMode",
                        "ExecutionMode",
                        "Algorithm",
                        "ConfigSpaceType",
                        "ObjectiveConfig",
                        "StopCondition",
                        "ParallelMode",
                        "ConstraintUsage",
                        "FailureMode",
                        "ExpectedOutcome",
                    ],
                },
            },
            "required": ["dim1", "dim2"],
        },
    },
    {
        "name": "get_coverage_gaps",
        "description": "Find dimension value pairs with no test coverage. Use this to identify missing tests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dim1": {
                    "type": "string",
                    "description": "First dimension name",
                },
                "dim2": {
                    "type": "string",
                    "description": "Second dimension name",
                },
            },
            "required": ["dim1", "dim2"],
        },
    },
    {
        "name": "get_test_stats",
        "description": "Get overall statistics about the test suite including total tests, pass/fail counts, and dimension distributions.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]
