"""Knowledge Graph builder for optimizer validation tests.

This module extracts test metadata and builds an RDF knowledge graph
that enables semantic queries for coverage analysis and gap detection.

Usage:
    python -m tests.optimizer_validation.viewer.knowledge_graph --output graph.ttl
    python -m tests.optimizer_validation.viewer.knowledge_graph --query gaps
    python -m tests.optimizer_validation.viewer.knowledge_graph --serve
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Use simple dict-based graph if rdflib not available
try:
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import RDF, RDFS, SKOS

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

# Namespace for our ontology
TRAIGENT_NS = "http://traigent.ai/ontology/test#"
UNRESOLVED = object()


# ==============================================================================
# Dimension Definitions
# ==============================================================================

# Core values that SHOULD be tested - used for coverage gap detection
CORE_DIMENSIONS = {
    "InjectionMode": {
        "values": ["context", "parameter", "seamless"],
        "description": "How config values are passed to the function",
    },
    "ExecutionMode": {
        "values": ["edge_analytics", "privacy", "hybrid", "cloud", "standard"],
        "description": "Execution environment mode",
    },
    "Algorithm": {
        "values": ["random", "grid", "optuna_tpe", "optuna_cmaes", "optuna_random"],
        "description": "Optimizer algorithm",
    },
    "ConfigSpaceType": {
        "values": ["categorical", "continuous", "mixed", "single_value"],
        "description": "Type of configuration space",
    },
    "ObjectiveConfig": {
        "values": ["single_maximize", "single_minimize", "multi_objective", "weighted"],
        "description": "Objective configuration",
    },
    "StopCondition": {
        "values": ["max_trials", "timeout", "config_exhaustion"],
        "description": "What causes optimization to stop",
    },
    "ParallelMode": {
        "values": ["sequential", "parallel"],
        "description": "Parallel execution configuration",
    },
    "ConstraintUsage": {
        "values": ["none", "config_only", "metric_based"],
        "description": "Constraint configuration",
    },
    "FailureMode": {
        "values": ["function_raises", "evaluator_bug", "invalid_config"],
        "description": "Expected failure type",
    },
    "Reproducibility": {
        "values": ["seeded", "deterministic"],
        "description": "Reproducibility guarantees",
    },
    "EvaluatorType": {
        "values": ["default", "custom", "scoring_function"],
        "description": "Evaluator configuration type",
    },
    "ExpectedOutcome": {
        "values": ["success", "failure", "partial"],
        "description": "Expected optimization outcome",
    },
}

# Extended values including edge cases - used for full tracking
DIMENSIONS = {
    "InjectionMode": {
        "values": [
            "context",
            "parameter",
            "seamless",
            # Edge cases (validation/error handling tests)
            "none",
            "empty",
            "invalid",
            "uppercase",
            "whitespace",
        ],
        "description": "How config values are passed to the function",
        "core_count": 3,
    },
    "ExecutionMode": {
        "values": [
            "edge_analytics",
            "privacy",
            "hybrid",
            "cloud",
            "standard",
            "invalid",
        ],
        "description": "Execution environment mode",
        "core_count": 5,
    },
    "Algorithm": {
        "values": [
            "random",
            "grid",
            "optuna_tpe",
            "optuna_cmaes",
            "optuna_random",
            "optuna_grid",
            "optuna_nsga2",
            "default",
        ],
        "description": "Optimizer algorithm",
        "core_count": 5,
    },
    "ConfigSpaceType": {
        "values": [
            "categorical",
            "continuous",
            "mixed",
            "single_value",
            "empty",
            "invalid",
        ],
        "description": "Type of configuration space",
        "core_count": 4,
    },
    "ObjectiveConfig": {
        "values": [
            "single_maximize",
            "single_minimize",
            "multi_objective",
            "weighted",
            "bounded",
            "empty",
        ],
        "description": "Objective configuration",
        "core_count": 4,
    },
    "StopCondition": {
        "values": [
            "max_trials",
            "timeout",
            "config_exhaustion",
            "plateau",
            "max_samples",
            "cost_limit",
            "optimizer",
            "condition",
            "error",
        ],
        "description": "What causes optimization to stop",
        "core_count": 3,
    },
    "ParallelMode": {
        "values": ["sequential", "parallel", "auto"],
        "description": "Parallel execution configuration",
        "core_count": 2,
    },
    "ConstraintUsage": {
        "values": ["none", "config_only", "metric_based", "mixed"],
        "description": "Constraint configuration",
        "core_count": 3,
    },
    "FailureMode": {
        "values": [
            "function_raises",
            "evaluator_bug",
            "dataset_issue",
            "invalid_config",
            "invocation_failure",
            "timeout",
        ],
        "description": "Expected failure type",
        "core_count": 3,
    },
    "Reproducibility": {
        "values": ["seeded", "deterministic", "non_deterministic"],
        "description": "Reproducibility guarantees",
        "core_count": 2,
    },
    "EvaluatorType": {
        "values": ["default", "custom", "scoring_function", "metric_functions"],
        "description": "Evaluator configuration type",
        "core_count": 3,
    },
    "ExpectedOutcome": {
        "values": ["success", "failure", "partial"],
        "description": "Expected optimization outcome",
        "core_count": 3,
    },
}

# Critical pairs that should have comprehensive coverage
CRITICAL_PAIRS = [
    ("InjectionMode", "ExecutionMode"),
    ("Algorithm", "ConfigSpaceType"),
    ("ParallelMode", "StopCondition"),
    ("ObjectiveConfig", "ConstraintUsage"),
    ("Algorithm", "FailureMode"),
    ("ExecutionMode", "ParallelMode"),
]


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class TestInfo:
    """Extracted information about a single test."""

    file_path: str
    class_name: str
    method_name: str
    docstring: str | None = None
    markers: list[str] = field(default_factory=list)
    dimensions: dict[str, str] = field(default_factory=dict)
    config_space: dict[str, Any] | None = None
    expected_outcome: str = "success"
    intent: str | None = None
    param_values: dict[str, Any] = field(default_factory=dict)
    scenario_index: int = 0
    test_id: str | None = None
    display_name: str | None = None


@dataclass
class CoverageGap:
    """Represents a missing test combination."""

    dimension1: str
    value1: str
    dimension2: str
    value2: str
    priority: str = "medium"  # low, medium, high, critical


# ==============================================================================
# Test Parser
# ==============================================================================


class TestParser:
    """Parse Python test files and extract test metadata."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.tests: list[TestInfo] = []

    def parse_all(self) -> list[TestInfo]:
        """Parse all test files in the directory."""
        for test_file in self.test_dir.glob("**/*.py"):
            if test_file.name.startswith("test_"):
                self._parse_file(test_file)
        return self.tests

    def _parse_file(self, file_path: Path) -> None:
        """Parse a single test file."""
        try:
            source = file_path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return

        module_context = self._build_module_context(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                self._parse_class(file_path, node, source, module_context)

    def _build_module_context(self, tree: ast.AST) -> dict[str, Any]:
        """Collect module-level constants for evaluation."""
        context: dict[str, Any] = {}
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    value = self._eval_node(node.value, context)
                    if value is not UNRESOLVED:
                        context[target.id] = value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    value = self._eval_node(node.value, context)
                    if value is not UNRESOLVED:
                        context[node.target.id] = value
        return context

    def _parse_class(
        self,
        file_path: Path,
        class_node: ast.ClassDef,
        source: str,
        module_context: dict[str, Any],
    ) -> None:
        """Parse a test class."""
        class_docstring = ast.get_docstring(class_node)

        for node in class_node.body:
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                if node.name.startswith("test_"):
                    test_infos = self._extract_test_infos(
                        file_path,
                        class_node.name,
                        node,
                        class_docstring,
                        source,
                        module_context,
                    )
                    self.tests.extend(test_infos)

    def _extract_test_infos(
        self,
        file_path: Path,
        class_name: str,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_docstring: str | None,
        source: str,
        module_context: dict[str, Any],
    ) -> list[TestInfo]:
        """Extract test information from a method node."""
        docstring = ast.get_docstring(method_node)
        markers = self._extract_markers(method_node)

        method_source = ast.get_source_segment(source, method_node) or ""

        param_sets = self._extract_param_sets(method_node, module_context)
        scenario_calls = self._find_scenario_calls(method_node)
        if not scenario_calls:
            scenario_calls = [None]

        test_infos: list[TestInfo] = []

        for param_values in param_sets:
            base_context = dict(module_context)
            base_context.update(param_values)
            local_context = self._build_local_context(method_node, base_context)

            scenario_fields_list = [
                self._extract_scenario_fields(call, local_context)
                for call in scenario_calls
            ]
            if not scenario_fields_list:
                scenario_fields_list = [{}]

            for scenario_index, scenario_fields in enumerate(scenario_fields_list):
                dimensions = self._infer_dimensions(
                    method_node.name,
                    docstring,
                    class_docstring,
                    method_source,
                    scenario_fields,
                    param_values,
                    file_path,
                )

                intent = self._infer_intent(
                    docstring, class_docstring, method_node.name
                )
                expected_outcome = self._infer_outcome(
                    method_source, docstring, scenario_fields
                )

                test_id, display_name = self._build_test_id(
                    file_path,
                    class_name,
                    method_node.name,
                    param_values,
                    scenario_index,
                )

                test_infos.append(
                    TestInfo(
                        file_path=str(
                            file_path.relative_to(self.test_dir.parent.parent)
                        ),
                        class_name=class_name,
                        method_name=method_node.name,
                        docstring=docstring,
                        markers=markers,
                        dimensions=dimensions,
                        expected_outcome=expected_outcome,
                        intent=intent,
                        param_values=param_values,
                        scenario_index=scenario_index,
                        test_id=test_id,
                        display_name=display_name,
                    )
                )

        return test_infos

    def _extract_markers(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Extract pytest markers from decorators."""
        markers = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Attribute):
                    if decorator.value.attr == "mark":
                        markers.append(decorator.attr)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if isinstance(decorator.func.value, ast.Attribute):
                        if decorator.func.value.attr == "mark":
                            markers.append(decorator.func.attr)
        return markers

    def _extract_param_sets(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract parameter sets from @pytest.mark.parametrize decorators."""
        param_sets = [{}]

        for decorator in node.decorator_list:
            if not self._is_parametrize(decorator):
                continue
            new_sets = self._parse_parametrize(decorator, context)
            if not new_sets:
                continue
            combined = []
            for base in param_sets:
                for new_values in new_sets:
                    merged = dict(base)
                    merged.update(new_values)
                    combined.append(merged)
            param_sets = combined

        return param_sets

    def _is_parametrize(self, decorator: ast.AST) -> bool:
        if not isinstance(decorator, ast.Call):
            return False
        return self._get_call_name(decorator) == "parametrize"

    def _parse_parametrize(
        self, decorator: ast.Call, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if len(decorator.args) < 2:
            return []

        param_names = self._eval_node(decorator.args[0], context)
        values = self._eval_node(decorator.args[1], context)

        if param_names is UNRESOLVED or values is UNRESOLVED:
            return []

        if isinstance(param_names, str):
            param_names = [name.strip() for name in param_names.split(",")]
        elif isinstance(param_names, (list, tuple)):
            param_names = list(param_names)
        else:
            return []

        if isinstance(values, dict):
            values = list(values.keys())

        if not isinstance(values, (list, tuple)):
            return []

        param_value_sets: list[dict[str, Any]] = []
        for value in values:
            if len(param_names) == 1:
                param_value_sets.append({param_names[0]: value})
            else:
                if isinstance(value, (list, tuple)) and len(value) == len(param_names):
                    param_value_sets.append(dict(zip(param_names, value, strict=False)))

        return param_value_sets

    def _find_scenario_calls(
        self, method_node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[ast.Call]:
        """Find scenario builder/TestScenario calls inside a test."""
        scenario_calls = {
            "TestScenario",
            "basic_scenario",
            "multi_objective_scenario",
            "constrained_scenario",
            "failure_scenario",
            "evaluator_scenario",
            "config_space_scenario",
        }
        calls = []
        for node in ast.walk(method_node):
            if isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name in scenario_calls:
                    calls.append(node)
        return calls

    def _build_local_context(
        self,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        base_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate simple local assignments inside a test method."""
        context = dict(base_context)
        for node in method_node.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    value = self._eval_node(node.value, context)
                    if value is not UNRESOLVED:
                        context[target.id] = value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    value = self._eval_node(node.value, context)
                    if value is not UNRESOLVED:
                        context[node.target.id] = value
        return context

    def _extract_scenario_fields(
        self, call_node: ast.Call | None, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract scenario fields from a TestScenario/builder call."""
        if call_node is None:
            return {}

        func_name = self._get_call_name(call_node)
        fields: dict[str, Any] = {}

        defaults: dict[str, Any] = {
            "TestScenario": {
                "injection_mode": "context",
                "execution_mode": "edge_analytics",
            },
            "basic_scenario": {
                "injection_mode": "context",
                "execution_mode": "edge_analytics",
                "config_space": {
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.3, 0.7],
                },
            },
            "multi_objective_scenario": {
                "config_space": {
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.1, 0.5, 0.9],
                },
            },
            "constrained_scenario": {
                "config_space": {
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.1, 0.5, 0.9],
                },
            },
            "failure_scenario": {"expected": {"outcome": "failure"}},
        }

        fields.update(defaults.get(func_name, {}))

        positional_args = {
            "TestScenario": ["name", "description"],
            "basic_scenario": ["name"],
            "multi_objective_scenario": ["name", "objectives"],
            "constrained_scenario": ["name", "constraints"],
            "failure_scenario": ["name", "expected_error"],
            "evaluator_scenario": ["name", "evaluator_type"],
            "config_space_scenario": ["name", "config_space"],
        }

        for arg_name, arg in zip(
            positional_args.get(func_name, []), call_node.args, strict=False
        ):
            value = self._eval_node(arg, context)
            if value is not UNRESOLVED:
                fields[arg_name] = value

        for kw in call_node.keywords:
            if kw.arg is None:
                continue
            value = self._eval_node(kw.value, context)
            if value is not UNRESOLVED:
                fields[kw.arg] = value

        if func_name == "evaluator_scenario" and "evaluator_type" in fields:
            fields.setdefault("evaluator", {"type": fields["evaluator_type"]})

        return fields

    def _get_call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _eval_node(self, node: ast.AST, context: dict[str, Any]) -> Any:
        """Evaluate a simple AST node to a Python value if possible."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context.get(node.id, UNRESOLVED)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval_node(node.operand, context)
            if value is UNRESOLVED:
                return UNRESOLVED
            if isinstance(value, (int, float)):
                return -value
            return UNRESOLVED
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key, value in zip(node.keys, node.values, strict=False):
                key_val = self._eval_node(key, context)
                value_val = self._eval_node(value, context)
                if key_val is UNRESOLVED or value_val is UNRESOLVED:
                    continue
                result[key_val] = value_val
            return result
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = []
            for elt in node.elts:
                value = self._eval_node(elt, context)
                if value is UNRESOLVED:
                    return UNRESOLVED
                items.append(value)
            return items if isinstance(node, ast.List) else tuple(items)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return UNRESOLVED
        if isinstance(node, ast.Call):
            call_name = self._get_call_name(node)
            if call_name in {"ObjectiveSpec", "ConstraintSpec", "ExpectedResult"}:
                return self._eval_spec_call(call_name, node, context)
            if call_name == "EvaluatorSpec":
                return self._eval_evaluator_spec(node, context)
            if isinstance(node.func, ast.Attribute):
                base = self._eval_node(node.func.value, context)
                if isinstance(base, dict) and node.func.attr == "keys":
                    return list(base.keys())
                if isinstance(base, dict) and node.func.attr == "values":
                    return list(base.values())
                if isinstance(base, dict) and node.func.attr == "items":
                    return list(base.items())
            return UNRESOLVED
        return UNRESOLVED

    def _eval_spec_call(
        self, name: str, node: ast.Call, context: dict[str, Any]
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"type": name}
        positional = {
            "ObjectiveSpec": ["name"],
            "ConstraintSpec": ["name", "constraint_fn"],
            "ExpectedResult": [],
        }
        for arg_name, arg in zip(positional.get(name, []), node.args, strict=False):
            value = self._eval_node(arg, context)
            if value is not UNRESOLVED:
                data[arg_name] = value
        for kw in node.keywords:
            if kw.arg is None:
                continue
            value = self._eval_node(kw.value, context)
            if value is not UNRESOLVED:
                data[kw.arg] = value

        if name == "ExpectedResult":
            outcome = data.get("outcome")
            normalized = self._normalize_expected_outcome(outcome)
            if normalized:
                data["outcome"] = normalized
        return data

    def _eval_evaluator_spec(
        self, node: ast.Call, context: dict[str, Any]
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"type": "EvaluatorSpec"}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            value = self._eval_node(kw.value, context)
            if value is not UNRESOLVED:
                data[kw.arg] = value
        return data

    def _infer_dimensions(
        self,
        method_name: str,
        docstring: str | None,
        class_docstring: str | None,
        source: str,
        scenario_fields: dict[str, Any],
        param_values: dict[str, Any],
        file_path: Path,
    ) -> dict[str, str]:
        """Infer dimension values from test content."""
        dimensions: dict[str, str] = {}

        dimensions.update(
            self._infer_dimensions_from_scenario(
                scenario_fields, param_values, file_path
            )
        )
        dimensions.update(self._infer_dimensions_from_params(param_values, dimensions))

        text_dims = self._infer_dimensions_from_text(
            method_name, docstring, class_docstring, source
        )
        for key, value in text_dims.items():
            dimensions.setdefault(key, value)

        overrides = self._parse_dimension_overrides(docstring, param_values)
        dimensions.update(overrides)

        return dimensions

    def _infer_dimensions_from_scenario(
        self,
        scenario_fields: dict[str, Any],
        param_values: dict[str, Any],
        file_path: Path,
    ) -> dict[str, str]:
        dimensions: dict[str, str] = {}

        injection_mode = scenario_fields.get("injection_mode")
        normalized = self._normalize_dimension_value("InjectionMode", injection_mode)
        if normalized:
            dimensions["InjectionMode"] = normalized

        execution_mode = scenario_fields.get("execution_mode")
        normalized = self._normalize_dimension_value("ExecutionMode", execution_mode)
        if normalized:
            dimensions["ExecutionMode"] = normalized

        mock_mode = scenario_fields.get("mock_mode_config")
        optimizer = None
        sampler = None
        if isinstance(mock_mode, dict):
            optimizer = mock_mode.get("optimizer")
            sampler = mock_mode.get("sampler")
        if optimizer or sampler:
            algo = self._normalize_algorithm(optimizer, sampler)
            if algo:
                dimensions["Algorithm"] = algo
        if "Algorithm" not in dimensions:
            algo = self._normalize_algorithm(
                scenario_fields.get("algorithm") or scenario_fields.get("optimizer"),
                scenario_fields.get("sampler"),
            )
            if algo:
                dimensions["Algorithm"] = algo

        config_space = scenario_fields.get("config_space")
        config_type = self._infer_config_space_type(config_space)
        if config_type:
            dimensions["ConfigSpaceType"] = config_type

        objectives = scenario_fields.get("objectives")
        objective_config = self._infer_objective_config(objectives)
        if objective_config:
            dimensions["ObjectiveConfig"] = objective_config

        constraints = scenario_fields.get("constraints")
        constraint_usage = self._infer_constraint_usage(constraints)
        if constraint_usage:
            dimensions["ConstraintUsage"] = constraint_usage

        parallel_config = scenario_fields.get("parallel_config")
        parallel_mode = self._infer_parallel_mode(parallel_config, param_values)
        if parallel_mode:
            dimensions["ParallelMode"] = parallel_mode

        evaluator = scenario_fields.get("evaluator")
        evaluator_type = None
        if isinstance(evaluator, dict):
            evaluator_type = evaluator.get("type")
        evaluator_value = self._normalize_dimension_value(
            "EvaluatorType", evaluator_type
        )
        if evaluator_value:
            dimensions["EvaluatorType"] = evaluator_value

        expected_outcome = self._extract_expected_outcome(scenario_fields)
        if expected_outcome:
            dimensions["ExpectedOutcome"] = expected_outcome

        expected = scenario_fields.get("expected")
        if isinstance(expected, dict):
            stop_reason = expected.get("expected_stop_reason")
            stop_value = self._normalize_dimension_value("StopCondition", stop_reason)
            if stop_value:
                dimensions["StopCondition"] = stop_value

        if expected_outcome == "failure":
            dimensions.setdefault("StopCondition", "error")

        if isinstance(mock_mode, dict):
            if "random_seed" in mock_mode:
                dimensions["Reproducibility"] = "seeded"
            elif mock_mode.get("optimizer") == "grid":
                dimensions["Reproducibility"] = "deterministic"

        file_str = str(file_path).lower()
        if "failures" in file_str:
            if "dataset_issues" in file_str:
                dimensions["FailureMode"] = "dataset_issue"
            elif "evaluator_bugs" in file_str:
                dimensions["FailureMode"] = "evaluator_bug"
            elif "function_bugs" in file_str:
                dimensions["FailureMode"] = "function_raises"
            elif "invocation_failures" in file_str:
                dimensions["FailureMode"] = "invocation_failure"

        return dimensions

    def _infer_dimensions_from_params(
        self, param_values: dict[str, Any], current: dict[str, str]
    ) -> dict[str, str]:
        dimensions: dict[str, str] = {}

        injection_mode = param_values.get("injection_mode")
        normalized = self._normalize_dimension_value("InjectionMode", injection_mode)
        if normalized and "InjectionMode" not in current:
            dimensions["InjectionMode"] = normalized

        execution_mode = param_values.get("execution_mode")
        normalized = self._normalize_dimension_value("ExecutionMode", execution_mode)
        if normalized and "ExecutionMode" not in current:
            dimensions["ExecutionMode"] = normalized

        space_type = param_values.get("space_type")
        normalized = self._normalize_dimension_value("ConfigSpaceType", space_type)
        if normalized and "ConfigSpaceType" not in current:
            dimensions["ConfigSpaceType"] = normalized

        objective_count = param_values.get("objective_count")
        if isinstance(objective_count, int) and "ObjectiveConfig" not in current:
            dimensions["ObjectiveConfig"] = (
                "multi_objective" if objective_count > 1 else "single_maximize"
            )

        orientation = param_values.get("orientation")
        normalized = self._normalize_dimension_value("ObjectiveConfig", orientation)
        if normalized and "ObjectiveConfig" not in current:
            if orientation == "minimize":
                dimensions["ObjectiveConfig"] = "single_minimize"
            elif orientation == "maximize":
                dimensions["ObjectiveConfig"] = "single_maximize"

        trial_concurrency = param_values.get("trial_concurrency")
        example_concurrency = param_values.get("example_concurrency")
        if "ParallelMode" not in current:
            if isinstance(trial_concurrency, int) and trial_concurrency > 1:
                dimensions["ParallelMode"] = "parallel"
            elif isinstance(example_concurrency, int) and example_concurrency > 1:
                dimensions["ParallelMode"] = "parallel"

        optimizer = param_values.get("optimizer") or param_values.get("algorithm")
        sampler = param_values.get("sampler")
        algo = self._normalize_algorithm(optimizer, sampler)
        if algo and "Algorithm" not in current:
            dimensions["Algorithm"] = algo

        evaluator_type = param_values.get("evaluator_type")
        if evaluator_type and "EvaluatorType" not in current:
            normalized = self._normalize_dimension_value(
                "EvaluatorType", evaluator_type
            )
            if normalized:
                dimensions["EvaluatorType"] = normalized

        return dimensions

    def _infer_dimensions_from_text(
        self,
        method_name: str,
        docstring: str | None,
        class_docstring: str | None,
        source: str,
    ) -> dict[str, str]:
        """Infer dimension values from test content."""
        dimensions: dict[str, str] = {}
        combined_text = (
            f"{method_name} {docstring or ''} {class_docstring or ''} {source}"
        ).lower()

        if "seamless" in combined_text:
            dimensions["InjectionMode"] = "seamless"
        elif "parameter" in combined_text and "injection" in combined_text:
            dimensions["InjectionMode"] = "parameter"
        elif "context" in combined_text:
            dimensions["InjectionMode"] = "context"

        if "edge_analytics" in combined_text:
            dimensions["ExecutionMode"] = "edge_analytics"
        elif "privacy" in combined_text and "mode" in combined_text:
            dimensions["ExecutionMode"] = "privacy"
        elif "hybrid" in combined_text and "mode" in combined_text:
            dimensions["ExecutionMode"] = "hybrid"
        elif "cloud" in combined_text and "mode" in combined_text:
            dimensions["ExecutionMode"] = "cloud"
        elif "standard" in combined_text and "mode" in combined_text:
            dimensions["ExecutionMode"] = "standard"

        if "cmaes" in combined_text or "cma-es" in combined_text:
            dimensions["Algorithm"] = "optuna_cmaes"
        elif "tpe" in combined_text:
            dimensions["Algorithm"] = "optuna_tpe"
        elif "nsga2" in combined_text:
            dimensions["Algorithm"] = "optuna_nsga2"
        elif "grid" in combined_text:
            dimensions["Algorithm"] = "grid"
        elif "random" in combined_text and "sampler" not in combined_text:
            dimensions["Algorithm"] = "random"

        if "continuous" in combined_text:
            dimensions["ConfigSpaceType"] = "continuous"
        elif "categorical" in combined_text:
            dimensions["ConfigSpaceType"] = "categorical"
        elif "mixed" in combined_text and "config" in combined_text:
            dimensions["ConfigSpaceType"] = "mixed"
        elif "single" in combined_text and (
            "value" in combined_text or "config" in combined_text
        ):
            dimensions["ConfigSpaceType"] = "single_value"

        if "multi" in combined_text and "objective" in combined_text:
            dimensions["ObjectiveConfig"] = "multi_objective"
        elif "weighted" in combined_text and "objective" in combined_text:
            dimensions["ObjectiveConfig"] = "weighted"
        elif "minimize" in combined_text:
            dimensions["ObjectiveConfig"] = "single_minimize"
        elif "maximize" in combined_text:
            dimensions["ObjectiveConfig"] = "single_maximize"

        if "timeout" in combined_text:
            dimensions["StopCondition"] = "timeout"
        elif "max_trials" in combined_text or "max trials" in combined_text:
            dimensions["StopCondition"] = "max_trials"
        elif "max_samples" in combined_text or "max samples" in combined_text:
            dimensions["StopCondition"] = "max_samples"
        elif "cost_limit" in combined_text or "cost limit" in combined_text:
            dimensions["StopCondition"] = "cost_limit"
        elif "exhaust" in combined_text:
            dimensions["StopCondition"] = "config_exhaustion"
        elif "plateau" in combined_text:
            dimensions["StopCondition"] = "plateau"

        if "parallel" in combined_text:
            if "sequential" in combined_text:
                dimensions["ParallelMode"] = "sequential"
            else:
                dimensions["ParallelMode"] = "parallel"

        if "constraint" in combined_text:
            if "metric" in combined_text:
                dimensions["ConstraintUsage"] = "metric_based"
            elif "config" in combined_text:
                dimensions["ConstraintUsage"] = "config_only"
            else:
                dimensions["ConstraintUsage"] = "mixed"

        if "seed" in combined_text or "reproducib" in combined_text:
            dimensions["Reproducibility"] = "seeded"
        elif "deterministic" in combined_text:
            dimensions["Reproducibility"] = "deterministic"

        if "failure" in combined_text or "raises" in combined_text:
            if "function" in combined_text and "raises" in combined_text:
                dimensions["FailureMode"] = "function_raises"
            elif "evaluator" in combined_text:
                dimensions["FailureMode"] = "evaluator_bug"
            elif "dataset" in combined_text:
                dimensions["FailureMode"] = "dataset_issue"
            elif "invalid" in combined_text:
                dimensions["FailureMode"] = "invalid_config"

        # EvaluatorType inference
        if "scoring" in combined_text and "function" in combined_text:
            dimensions["EvaluatorType"] = "scoring_function"
        elif "metric" in combined_text and "function" in combined_text:
            dimensions["EvaluatorType"] = "metric_functions"
        elif "custom" in combined_text and "evaluator" in combined_text:
            dimensions["EvaluatorType"] = "custom"
        elif "default" in combined_text and "evaluator" in combined_text:
            dimensions["EvaluatorType"] = "default"
        # Also detect from class names
        elif "defaultevaluator" in combined_text.replace("_", ""):
            dimensions["EvaluatorType"] = "default"
        elif "customevaluator" in combined_text.replace("_", ""):
            dimensions["EvaluatorType"] = "custom"
        elif "scoringfunction" in combined_text.replace("_", ""):
            dimensions["EvaluatorType"] = "scoring_function"
        elif "metricfunction" in combined_text.replace("_", ""):
            dimensions["EvaluatorType"] = "metric_functions"

        return dimensions

    def _parse_dimension_overrides(
        self, docstring: str | None, param_values: dict[str, Any]
    ) -> dict[str, str]:
        """Parse explicit dimension annotations from docstrings."""
        if not docstring:
            return {}

        dim_keys = {key.lower(): key for key in DIMENSIONS}
        overrides: dict[str, str] = {}
        lines = docstring.splitlines()

        for idx, line in enumerate(lines):
            if "dimensions:" not in line.lower():
                continue
            payload = line.split("Dimensions:", 1)[1] if "Dimensions:" in line else ""
            payload = payload.strip()
            if payload:
                overrides.update(
                    self._parse_dimension_line(payload, dim_keys, param_values)
                )
            for next_line in lines[idx + 1 :]:
                if not next_line.strip():
                    break
                if "=" not in next_line:
                    continue
                overrides.update(
                    self._parse_dimension_line(next_line, dim_keys, param_values)
                )

        return overrides

    def _parse_dimension_line(
        self,
        text: str,
        dim_keys: dict[str, str],
        param_values: dict[str, Any],
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for match in re.finditer(r"(\w+)\s*=\s*([^,]+)", text):
            raw_dim = match.group(1).strip().lower()
            raw_value = match.group(2).strip()
            dim_name = dim_keys.get(raw_dim)
            if not dim_name:
                continue
            value = raw_value.strip()
            if value.startswith("{") and value.endswith("}"):
                key = value[1:-1].strip()
                if key in param_values:
                    value = str(param_values[key])
                else:
                    continue
            value = value.split("(")[0].strip()
            normalized = self._normalize_dimension_value(dim_name, value)
            if normalized:
                overrides[dim_name] = normalized
        return overrides

    def _normalize_dimension_value(self, dim: str, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return "empty" if dim in {"InjectionMode", "ConfigSpaceType"} else None
        text = text.replace("-", "_").replace(" ", "_")
        aliases = {
            "ConfigSpaceType": {
                "categorical_only": "categorical",
                "continuous_only": "continuous",
                "single": "single_value",
            },
            "FailureMode": {
                "validation": "invalid_config",
                "validation_error": "invalid_config",
                "invalid": "invalid_config",
            },
            "StopCondition": {
                "max_trials_reached": "max_trials",
                "max_samples_reached": "max_samples",
            },
        }
        text = aliases.get(dim, {}).get(text, text)
        if dim == "InjectionMode" and text in {"none", "null"}:
            text = "none"
        if dim == "ExecutionMode" and text == "edgeanalytics":
            text = "edge_analytics"
        if text in DIMENSIONS.get(dim, {}).get("values", []):
            return text
        return None

    def _normalize_expected_outcome(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).lower()
        if "expectedoutcome.failure" in text or text == "failure":
            return "failure"
        if "expectedoutcome.partial" in text or text == "partial":
            return "partial"
        if "expectedoutcome.success" in text or text == "success":
            return "success"
        return None

    def _normalize_algorithm(self, optimizer: Any, sampler: Any) -> str | None:
        if optimizer is None and sampler is None:
            return None
        opt = str(optimizer).lower() if optimizer is not None else ""
        samp = str(sampler).lower() if sampler is not None else ""
        if opt == "optuna":
            if samp:
                return f"optuna_{samp}"
            return "optuna_random"
        if opt in {"random", "grid"}:
            return opt
        if opt:
            return opt if opt in DIMENSIONS["Algorithm"]["values"] else None
        return None

    def _infer_config_space_type(self, config_space: Any) -> str | None:
        if config_space is None:
            return None
        if isinstance(config_space, dict):
            if not config_space:
                return "empty"
            has_cat = False
            has_cont = False
            invalid = False
            single_value = True
            for value in config_space.values():
                if isinstance(value, tuple):
                    has_cont = True
                    single_value = False
                elif isinstance(value, list):
                    has_cat = True
                    if len(value) != 1:
                        single_value = False
                else:
                    invalid = True
            if invalid:
                return "invalid"
            if has_cont and has_cat:
                return "mixed"
            if has_cont:
                return "continuous"
            if has_cat:
                return "single_value" if single_value else "categorical"
        return None

    def _infer_objective_config(self, objectives: Any) -> str | None:
        if objectives is UNRESOLVED:
            return None
        if objectives == []:
            return "empty"
        if isinstance(objectives, list):
            specs = []
            for obj in objectives:
                if isinstance(obj, dict) and obj.get("type") == "ObjectiveSpec":
                    specs.append(obj)
                elif isinstance(obj, str):
                    specs.append({"name": obj})
            if not specs:
                return None
            if len(specs) > 1:
                if any(spec.get("weight", 1.0) != 1.0 for spec in specs):
                    return "weighted"
                if any(spec.get("bounds") for spec in specs):
                    return "bounded"
                return "multi_objective"
            orientation = specs[0].get("orientation")
            if orientation == "minimize":
                return "single_minimize"
            if orientation == "band":
                return "bounded"
            return "single_maximize"
        return None

    def _infer_constraint_usage(self, constraints: Any) -> str | None:
        if constraints is UNRESOLVED:
            return None
        if constraints is None:
            return None
        if isinstance(constraints, list):
            if not constraints:
                return "none"
            requires_metrics = []
            for constraint in constraints:
                if isinstance(constraint, dict):
                    requires_metrics.append(bool(constraint.get("requires_metrics")))
            if requires_metrics:
                if all(requires_metrics):
                    return "metric_based"
                if any(requires_metrics):
                    return "mixed"
                return "config_only"
            return "mixed"
        return None

    def _infer_parallel_mode(
        self, parallel_config: Any, param_values: dict[str, Any]
    ) -> str | None:
        if isinstance(parallel_config, dict):
            mode = parallel_config.get("mode")
            normalized = self._normalize_dimension_value("ParallelMode", mode)
            if normalized:
                return normalized
            trial_concurrency = parallel_config.get("trial_concurrency")
            example_concurrency = parallel_config.get("example_concurrency")
            if isinstance(trial_concurrency, int) and trial_concurrency > 1:
                return "parallel"
            if isinstance(example_concurrency, int) and example_concurrency > 1:
                return "parallel"
        trial_concurrency = param_values.get("trial_concurrency")
        example_concurrency = param_values.get("example_concurrency")
        if isinstance(trial_concurrency, int) and trial_concurrency > 1:
            return "parallel"
        if isinstance(example_concurrency, int) and example_concurrency > 1:
            return "parallel"
        return None

    def _extract_expected_outcome(self, scenario_fields: dict[str, Any]) -> str | None:
        expected = scenario_fields.get("expected")
        if isinstance(expected, dict):
            normalized = self._normalize_expected_outcome(expected.get("outcome"))
            if normalized:
                return normalized
        return None

    def _infer_intent(
        self, docstring: str | None, class_docstring: str | None, method_name: str
    ) -> str:
        """Infer the test intent from docstrings."""
        combined = f"{docstring or ''} {class_docstring or ''} {method_name}".lower()

        if "edge case" in combined or "boundary" in combined:
            return "edge_case"
        if "error" in combined or "fail" in combined or "exception" in combined:
            return "error_handling"
        if "reproducib" in combined or "deterministic" in combined:
            return "reproducibility"
        if "interaction" in combined or "combination" in combined:
            return "integration"
        if "performance" in combined or "timing" in combined:
            return "performance"
        return "correctness"

    def _infer_outcome(
        self, source: str, docstring: str | None, scenario_fields: dict[str, Any]
    ) -> str:
        """Infer expected outcome from test source."""
        expected_outcome = self._extract_expected_outcome(scenario_fields)
        if expected_outcome:
            return expected_outcome

        evaluator = scenario_fields.get("evaluator")
        if isinstance(evaluator, dict) and evaluator.get("should_fail") is True:
            return "failure"

        if scenario_fields.get("function_should_raise") or scenario_fields.get(
            "function_raise_on_call"
        ):
            return "failure"

        combined = f"{source} {docstring or ''}".lower()

        if "expectedoutcome.failure" in combined or "expects_failure" in combined:
            return "failure"
        if "expectedoutcome.partial" in combined:
            return "partial"
        if "isinstance(result, exception)" in combined:
            return "failure"
        return "success"

    def _build_test_id(
        self,
        file_path: Path,
        class_name: str,
        method_name: str,
        param_values: dict[str, Any],
        scenario_index: int,
    ) -> tuple[str, str]:
        base_path = str(file_path.relative_to(self.test_dir.parent.parent))
        base_id = f"{base_path}::{class_name}::{method_name}"
        display = method_name

        if param_values:
            # pytest uses values only, joined by '-', in alphabetical key order
            # e.g., [edge_analytics-context] not [execution_mode=edge_analytics,injection_mode=context]
            param_parts = "-".join(
                str(param_values[key]) for key in sorted(param_values)
            )
            base_id = f"{base_id}[{param_parts}]"
            # Display name keeps the key=value format for clarity
            display_parts = ",".join(
                f"{key}={param_values[key]}" for key in sorted(param_values)
            )
            display = f"{display}[{display_parts}]"

        if scenario_index:
            base_id = f"{base_id}#{scenario_index + 1}"
            display = f"{display}#{scenario_index + 1}"

        return base_id, display


# ==============================================================================
# Knowledge Graph Builder
# ==============================================================================
# ==============================================================================
# Knowledge Graph Builder
# ==============================================================================


class TestKnowledgeGraph:
    """Build and query a knowledge graph of tests."""

    def __init__(self):
        self._test_list: list[TestInfo] = []
        self._tests_dict: dict[str, dict[str, Any]] = {}
        self._rdf_graph: Graph | None = None

    @property
    def tests(self) -> dict[str, dict[str, Any]]:
        """Get tests as a dictionary keyed by test ID.

        This property provides a dict interface for the chatbot tools,
        while maintaining backward compatibility with the list-based API.
        """
        if self._tests_dict:
            return self._tests_dict
        # Build dict from list
        for test in self._test_list:
            test_id = test.test_id or f"{test.class_name}.{test.method_name}"
            self._tests_dict[test_id] = {
                "name": test.display_name or test.method_name,
                "description": test.docstring or "",
                "test_file": test.file_path,
                "class": test.class_name,
                "method": test.method_name,
                "dimensions": test.dimensions,
                "expected_outcome": test.expected_outcome,
                "intent": test.intent,
                "markers": test.markers,
                "params": test.param_values,
            }
        return self._tests_dict

    @tests.setter
    def tests(self, value: list[TestInfo]) -> None:
        """Set tests from a list (backward compatibility)."""
        self._test_list = value
        self._tests_dict = {}  # Clear dict cache

    @classmethod
    def load(cls, json_path: str) -> TestKnowledgeGraph:
        """Load knowledge graph from a JSON file.

        Args:
            json_path: Path to the graph_data.json file

        Returns:
            TestKnowledgeGraph instance with loaded data
        """
        kg = cls()
        path = Path(json_path)

        if not path.exists():
            return kg

        with open(path) as f:
            data = json.load(f)

        # Load tests from JSON format
        for test_data in data.get("tests", []):
            test_id = test_data.get("id", "")
            kg._tests_dict[test_id] = {
                "name": test_data.get("name", ""),
                "description": test_data.get("docstring", ""),
                "test_file": test_data.get("file", ""),
                "class": test_data.get("class", ""),
                "method": test_data.get("method", ""),
                "dimensions": test_data.get("dimensions", {}),
                "expected_outcome": test_data.get("outcome", "success"),
                "intent": test_data.get("intent", ""),
                "markers": test_data.get("markers", []),
                "params": test_data.get("params", {}),
                "result": test_data.get("result", {}),
            }

        return kg

    def load_tests(self, test_dir: Path) -> None:
        """Load tests from directory."""
        parser = TestParser(test_dir)
        self.tests = parser.parse_all()

    def _get_test_list(self) -> list[TestInfo]:
        """Get tests as a list of TestInfo objects."""
        if self._test_list:
            return self._test_list
        # Convert from dict if needed
        tests = []
        for test_id, data in self._tests_dict.items():
            tests.append(
                TestInfo(
                    file_path=data.get("test_file", ""),
                    class_name=data.get("class", ""),
                    method_name=data.get("method", ""),
                    docstring=data.get("description"),
                    markers=data.get("markers", []),
                    dimensions=data.get("dimensions", {}),
                    expected_outcome=data.get("expected_outcome", "success"),
                    intent=data.get("intent"),
                    param_values=data.get("params", {}),
                    test_id=test_id,
                    display_name=data.get("name"),
                )
            )
        return tests

    def build_rdf_graph(self) -> Graph | None:
        """Build RDF graph from tests (requires rdflib)."""
        if not HAS_RDFLIB:
            print("rdflib not installed. Use pip install rdflib")
            return None

        g = Graph()
        TRAIGENT = Namespace(TRAIGENT_NS)
        g.bind("traigent", TRAIGENT)
        g.bind("rdfs", RDFS)
        g.bind("skos", SKOS)

        # Add dimension definitions
        for dim_name, dim_info in DIMENSIONS.items():
            dim_uri = TRAIGENT[dim_name]
            g.add((dim_uri, RDF.type, TRAIGENT.Dimension))
            g.add((dim_uri, RDFS.label, Literal(dim_name)))
            g.add((dim_uri, SKOS.definition, Literal(dim_info["description"])))

            for value in dim_info["values"]:
                value_uri = TRAIGENT[f"{dim_name}_{value}"]
                g.add((value_uri, RDF.type, TRAIGENT.DimensionValue))
                g.add((value_uri, RDFS.label, Literal(value)))
                g.add((value_uri, TRAIGENT.belongsToDimension, dim_uri))

        # Add tests
        for test in self._get_test_list():
            test_id = test.test_id or f"{test.class_name}_{test.method_name}"
            safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", test_id)
            test_uri = TRAIGENT[f"test_{safe_id}"]
            g.add((test_uri, RDF.type, TRAIGENT.TestCase))
            g.add(
                (
                    test_uri,
                    RDFS.label,
                    Literal(test.display_name or test.method_name),
                )
            )

            if test.docstring:
                g.add((test_uri, RDFS.comment, Literal(test.docstring[:500])))

            # Add dimension values
            for dim_name, value in test.dimensions.items():
                value_uri = TRAIGENT[f"{dim_name}_{value}"]
                g.add((test_uri, TRAIGENT.hasDimensionValue, value_uri))

            # Add file reference
            file_uri = TRAIGENT[f"file_{test.file_path.replace('/', '_')}"]
            g.add((file_uri, RDF.type, TRAIGENT.TestFile))
            g.add((file_uri, RDFS.label, Literal(test.file_path)))

            class_uri = TRAIGENT[f"class_{test.class_name}"]
            g.add((class_uri, RDF.type, TRAIGENT.TestClass))
            g.add((class_uri, RDFS.label, Literal(test.class_name)))
            g.add((class_uri, TRAIGENT.inFile, file_uri))
            g.add((test_uri, TRAIGENT.inClass, class_uri))

            # Add intent
            intent_uri = TRAIGENT[f"intent_{test.intent}"]
            g.add((test_uri, TRAIGENT.verifiesIntent, intent_uri))

            # Add expected outcome
            outcome_uri = TRAIGENT[f"outcome_{test.expected_outcome}"]
            g.add((test_uri, TRAIGENT.expectsOutcome, outcome_uri))

        self._rdf_graph = g
        return g

    def export_turtle(self, output_path: Path) -> None:
        """Export graph to Turtle format."""
        if self._rdf_graph is None:
            self.build_rdf_graph()
        if self._rdf_graph:
            self._rdf_graph.serialize(destination=str(output_path), format="turtle")

    def get_coverage_matrix(self) -> dict[str, dict[str, int]]:
        """Get coverage count for each dimension value."""
        matrix: dict[str, dict[str, int]] = {}

        for dim_name in DIMENSIONS:
            matrix[dim_name] = dict.fromkeys(DIMENSIONS[dim_name]["values"], 0)

        for test in self._get_test_list():
            for dim_name, value in test.dimensions.items():
                if dim_name in matrix and value in matrix[dim_name]:
                    matrix[dim_name][value] += 1

        return matrix

    def get_pairwise_coverage(
        self, dim1: str, dim2: str, core_only: bool = False
    ) -> dict[tuple[str, str], list[str]]:
        """Get tests covering each pair of dimension values.

        Args:
            dim1: First dimension name
            dim2: Second dimension name
            core_only: If True, only use CORE_DIMENSIONS values (excludes edge cases)
        """
        coverage: dict[tuple[str, str], list[str]] = {}

        # Choose dimension source based on core_only flag
        dims = CORE_DIMENSIONS if core_only else DIMENSIONS

        # Initialize all pairs
        for v1 in dims.get(dim1, {}).get("values", []):
            for v2 in dims.get(dim2, {}).get("values", []):
                coverage[(v1, v2)] = []

        # Find tests covering each pair
        for test in self._get_test_list():
            v1 = test.dimensions.get(dim1)
            v2 = test.dimensions.get(dim2)
            if v1 and v2 and (v1, v2) in coverage:
                coverage[(v1, v2)].append(test.display_name or test.method_name)

        return coverage

    def find_coverage_gaps(self, core_only: bool = True) -> list[CoverageGap]:
        """Find missing pairwise combinations for critical pairs.

        Args:
            core_only: If True, only check CORE_DIMENSIONS (excludes edge case gaps)
        """
        gaps = []

        for dim1, dim2 in CRITICAL_PAIRS:
            coverage = self.get_pairwise_coverage(dim1, dim2, core_only=core_only)
            for (v1, v2), tests in coverage.items():
                if not tests:
                    priority = (
                        "high" if (dim1, dim2) in CRITICAL_PAIRS[:2] else "medium"
                    )
                    gaps.append(
                        CoverageGap(
                            dimension1=dim1,
                            value1=v1,
                            dimension2=dim2,
                            value2=v2,
                            priority=priority,
                        )
                    )

        return gaps

    def suggest_new_tests(self, max_suggestions: int = 10) -> list[dict[str, Any]]:
        """Suggest new tests to improve coverage."""
        gaps = self.find_coverage_gaps()

        # Prioritize by combining multiple gaps
        suggestions = []
        seen_combos = set()

        for gap in sorted(gaps, key=lambda g: (g.priority != "high", g.dimension1)):
            combo_key = (gap.value1, gap.value2)
            if combo_key not in seen_combos:
                suggestions.append(
                    {
                        "name": f"test_{gap.value1}_{gap.value2}",
                        "dimensions": {
                            gap.dimension1: gap.value1,
                            gap.dimension2: gap.value2,
                        },
                        "priority": gap.priority,
                        "reason": f"Missing coverage for {gap.dimension1}={gap.value1} + {gap.dimension2}={gap.value2}",
                    }
                )
                seen_combos.add(combo_key)

            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def to_json(self) -> dict[str, Any]:
        """Export graph data as JSON for visualization."""
        return {
            "dimensions": DIMENSIONS,
            "tests": [
                {
                    "id": t.test_id or f"{t.class_name}.{t.method_name}",
                    "name": t.display_name or t.method_name,
                    "file": t.file_path,
                    "class": t.class_name,
                    "method": t.method_name,
                    "docstring": t.docstring,
                    "dimensions": t.dimensions,
                    "outcome": t.expected_outcome,
                    "intent": t.intent,
                    "markers": t.markers,
                    "params": t.param_values,
                    "scenario_index": t.scenario_index,
                }
                for t in self._get_test_list()
            ],
            "coverage_matrix": self.get_coverage_matrix(),
            "gaps": [
                {
                    "dim1": g.dimension1,
                    "val1": g.value1,
                    "dim2": g.dimension2,
                    "val2": g.value2,
                    "priority": g.priority,
                }
                for g in self.find_coverage_gaps()
            ],
            "suggestions": self.suggest_new_tests(),
            "critical_pairs": CRITICAL_PAIRS,
        }

    def query_sparql(self, query: str) -> list[dict[str, Any]]:
        """Run a SPARQL query on the graph."""
        if not HAS_RDFLIB or self._rdf_graph is None:
            return []

        results = []
        for row in self._rdf_graph.query(query):
            results.append({str(k): str(v) for k, v in row.asdict().items()})
        return results


# ==============================================================================
# CLI
# ==============================================================================


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Graph for Optimizer Validation Tests"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Test directory to analyze",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file (ttl for Turtle, json for JSON)"
    )
    parser.add_argument(
        "--query",
        choices=["gaps", "coverage", "suggestions", "all"],
        help="Run predefined query",
    )
    parser.add_argument("--sparql", type=str, help="Run custom SPARQL query")
    parser.add_argument(
        "--serve", action="store_true", help="Start visualization server"
    )

    args = parser.parse_args()

    # Build graph
    kg = TestKnowledgeGraph()
    kg.load_tests(args.test_dir)
    print(f"Loaded {len(kg._get_test_list())} tests")

    # Handle output
    if args.output:
        if args.output.suffix == ".ttl":
            kg.build_rdf_graph()
            kg.export_turtle(args.output)
            print(f"Exported Turtle to {args.output}")
        else:
            with open(args.output, "w") as f:
                json.dump(kg.to_json(), f, indent=2)
            print(f"Exported JSON to {args.output}")

    # Handle queries
    if args.query:
        if args.query == "gaps":
            gaps = kg.find_coverage_gaps()
            print(f"\n=== Coverage Gaps ({len(gaps)}) ===")
            for gap in gaps[:20]:
                print(
                    f"  [{gap.priority}] {gap.dimension1}={gap.value1} + {gap.dimension2}={gap.value2}"
                )

        elif args.query == "coverage":
            matrix = kg.get_coverage_matrix()
            print("\n=== Coverage Matrix ===")
            for dim, values in matrix.items():
                print(f"\n{dim}:")
                for value, count in sorted(values.items(), key=lambda x: -x[1]):
                    bar = "█" * min(count, 50)
                    print(f"  {value:20} {count:3} {bar}")

        elif args.query == "suggestions":
            suggestions = kg.suggest_new_tests()
            print("\n=== Suggested New Tests ===")
            for s in suggestions:
                print(f"\n  {s['name']}")
                print(f"    Reason: {s['reason']}")
                print(f"    Priority: {s['priority']}")

        elif args.query == "all":
            data = kg.to_json()
            print(json.dumps(data, indent=2))

    if args.sparql:
        kg.build_rdf_graph()
        results = kg.query_sparql(args.sparql)
        print("\n=== SPARQL Results ===")
        for row in results:
            print(row)

    if args.serve:
        print("\nStarting visualization server...")
        # Integration with existing serve.py
        from . import serve

        serve.run_with_knowledge_graph(kg)


if __name__ == "__main__":
    main()
