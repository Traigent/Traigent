"""Detection strategies for tuned variable identification.

Provides pluggable strategies following the DetectionStrategy protocol:

- ``ASTDetectionStrategy``: Pure static analysis using AST pattern matching
  against known LLM parameter names and value-type heuristics.
- ``LLMDetectionStrategy``: Semantic analysis using an injected LLM callable
  for deeper understanding of tunable variables.

Both strategies are stateless and thread-safe.
"""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    SourceLocation,
    SuggestedRange,
    TunedVariableCandidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class DetectionStrategy(Protocol):
    """Protocol for tuned variable detection strategies.

    Strategies are stateless: they receive source code and return candidates.
    """

    def detect(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[TunedVariableCandidate]:
        """Detect tuned variable candidates from source code.

        Args:
            source: Python source code string.
            function_name: Name of the function to analyze.
            context: Optional context (e.g., existing config space).

        Returns:
            List of detected candidates.
        """
        ...


# ---------------------------------------------------------------------------
# Known-parameter knowledge base
# ---------------------------------------------------------------------------

# Canonical -> variant mappings for LLM parameters.
_UNIVERSAL_MAPPING: dict[str, list[str]] = {
    "model": ["model", "model_name", "model_id", "engine"],
    "temperature": ["temperature", "temp"],
    "max_tokens": [
        "max_tokens",
        "max_length",
        "max_new_tokens",
        "max_tokens_to_sample",
    ],
    "top_p": ["top_p", "top_p_sampling"],
    "top_k": ["top_k"],
    "frequency_penalty": ["frequency_penalty", "freq_penalty"],
    "presence_penalty": ["presence_penalty", "pres_penalty"],
    "stop": ["stop", "stop_sequences", "stop_words"],
    "stream": ["stream", "streaming"],
    "seed": ["seed", "random_seed"],
    "n": ["n", "num_completions"],
    "timeout": ["timeout", "request_timeout"],
}

# Reverse mapping: variant name -> canonical name
_REVERSE_MAPPING: dict[str, str] = {}
for _canonical, _variants in _UNIVERSAL_MAPPING.items():
    for _variant in _variants:
        _REVERSE_MAPPING[_variant] = _canonical

# All known parameter names (flat set for fast membership checks)
_KNOWN_PARAMS: frozenset[str] = frozenset(_REVERSE_MAPPING.keys())

# Common model name substrings for detecting model string literals
_MODEL_NAME_HINTS: frozenset[str] = frozenset(
    {
        "gpt-",
        "gpt4",
        "claude",
        "llama",
        "mistral",
        "gemini",
        "o1-",
        "o3-",
        "command-r",
        "deepseek",
    }
)

# Canonical name -> suggested range metadata
_CANONICAL_RANGES: dict[str, dict[str, Any]] = {
    "temperature": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 2.0, "default": 0.7},
    },
    "top_p": {
        "range_type": "Range",
        "kwargs": {"low": 0.1, "high": 1.0, "default": 0.9},
    },
    "frequency_penalty": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 2.0, "default": 0.0},
    },
    "presence_penalty": {
        "range_type": "Range",
        "kwargs": {"low": 0.0, "high": 2.0, "default": 0.0},
    },
    "max_tokens": {
        "range_type": "IntRange",
        "kwargs": {"low": 256, "high": 4096},
    },
    "top_k": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 100},
    },
    "n": {
        "range_type": "IntRange",
        "kwargs": {"low": 1, "high": 5},
    },
    "seed": {
        "range_type": "IntRange",
        "kwargs": {"low": 0, "high": 1000},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_literal_value(node: ast.expr) -> Any | None:
    """Extract a Python literal from an AST expression node.

    Returns the value for Constant, List-of-Constants, or None if
    the expression is too complex.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                values.append(elt.value)
            else:
                return None
        return values
    if isinstance(node, (ast.Tuple)):
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                values.append(elt.value)
            else:
                return None
        return tuple(values)
    return None


def _is_parameter_range_call(node: ast.expr) -> bool:
    """Check if an AST Call node is to a known ParameterRange constructor."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id in {
        "Range",
        "IntRange",
        "LogRange",
        "Choices",
    }:
        return True
    if isinstance(func, ast.Attribute) and func.attr in {
        "Range",
        "IntRange",
        "LogRange",
        "Choices",
    }:
        return True
    return False


def _infer_candidate_type(value: Any) -> CandidateType:
    """Infer CandidateType from a Python literal value."""
    if isinstance(value, bool):
        return CandidateType.BOOLEAN
    if isinstance(value, float):
        return CandidateType.NUMERIC_CONTINUOUS
    if isinstance(value, int):
        return CandidateType.NUMERIC_INTEGER
    if isinstance(value, str):
        return CandidateType.CATEGORICAL
    if isinstance(value, list):
        return CandidateType.CATEGORICAL
    return CandidateType.CATEGORICAL


def _looks_like_model_string(value: str) -> bool:
    """Check if a string value looks like a model name."""
    lower = value.lower()
    return any(hint in lower for hint in _MODEL_NAME_HINTS)


def _suggest_range(
    canonical_name: str | None, candidate_type: CandidateType, value: Any
) -> SuggestedRange | None:
    """Suggest a ParameterRange based on canonical name and current value."""
    # Use known canonical ranges first
    if canonical_name and canonical_name in _CANONICAL_RANGES:
        meta = _CANONICAL_RANGES[canonical_name]
        return SuggestedRange(
            range_type=meta["range_type"], kwargs=dict(meta["kwargs"])
        )

    # Fallback heuristics based on value type
    if candidate_type == CandidateType.NUMERIC_CONTINUOUS and isinstance(value, float):
        low = round(value * 0.5, 4) if value > 0 else 0.0
        high = round(value * 2.0, 4) if value > 0 else 1.0
        return SuggestedRange(range_type="Range", kwargs={"low": low, "high": high})

    if candidate_type == CandidateType.NUMERIC_INTEGER and isinstance(value, int):
        low = max(1, value // 2)
        high = value * 2
        return SuggestedRange(range_type="IntRange", kwargs={"low": low, "high": high})

    if candidate_type == CandidateType.CATEGORICAL and isinstance(value, str):
        return SuggestedRange(range_type="Choices", kwargs={"values": [value]})

    if candidate_type == CandidateType.CATEGORICAL and isinstance(value, list):
        return SuggestedRange(range_type="Choices", kwargs={"values": list(value)})

    if candidate_type == CandidateType.BOOLEAN:
        return SuggestedRange(range_type="Choices", kwargs={"values": [True, False]})

    return None


def _make_location(node: ast.AST) -> SourceLocation:
    """Build a SourceLocation from an AST node."""
    return SourceLocation(
        line=getattr(node, "lineno", 0),
        col_offset=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", None),
        end_col_offset=getattr(node, "end_col_offset", None),
    )


# ---------------------------------------------------------------------------
# AST Detection Strategy
# ---------------------------------------------------------------------------


class _AssignmentCollector(ast.NodeVisitor):
    """Collect assignments within a single function body.

    Tracks scope depth to ignore assignments inside nested functions,
    list comprehensions, and other nested scopes.
    """

    def __init__(self, existing_tvars: frozenset[str]) -> None:
        self.candidates: list[TunedVariableCandidate] = []
        self._existing_tvars = existing_tvars
        self._scope_depth = 0

    # -- Scope tracking ----------------------------------------------------

    def _visit_any_function(self, node: ast.AST) -> None:
        """Shared scope-depth handler for sync and async function defs."""
        if self._scope_depth > 0:
            return
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Skip nested function bodies
        self._visit_any_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_any_function(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass  # Skip assignments inside comprehensions

    def visit_SetComp(self, node: ast.SetComp) -> None:
        pass  # Skip

    def visit_DictComp(self, node: ast.DictComp) -> None:
        pass  # Skip

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        pass  # Skip

    # -- Assignment patterns -----------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.generic_visit(node)
            return

        name = node.targets[0].id
        self._process_assignment(name, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name) or node.value is None:
            self.generic_visit(node)
            return

        name = node.target.id
        self._process_assignment(name, node.value, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect tunable kwargs in function calls (e.g., create(temperature=0.7))."""
        for kw in node.keywords:
            if kw.arg and kw.arg in _KNOWN_PARAMS:
                # Skip if it's a ParameterRange call
                if _is_parameter_range_call(kw.value):
                    continue

                value = _extract_literal_value(kw.value)
                if value is None:
                    continue

                canonical = _REVERSE_MAPPING.get(kw.arg)
                ctype = _infer_candidate_type(value)
                self.candidates.append(
                    TunedVariableCandidate(
                        name=kw.arg,
                        candidate_type=ctype,
                        confidence=DetectionConfidence.HIGH,
                        location=_make_location(kw),
                        current_value=value,
                        suggested_range=_suggest_range(canonical, ctype, value),
                        detection_source="ast",
                        reasoning=f"Keyword argument '{kw.arg}' in function call matches known LLM parameter",
                        canonical_name=canonical,
                    )
                )

        self.generic_visit(node)

    def _process_assignment(
        self, name: str, value_node: ast.expr, stmt_node: ast.AST
    ) -> None:
        """Process a single assignment and potentially add a candidate."""
        # Skip if already a ParameterRange (user already tuned this)
        if _is_parameter_range_call(value_node):
            return

        # Skip if name is in existing config space
        if name in self._existing_tvars:
            return

        value = _extract_literal_value(value_node)

        # Pattern 1: Direct name match
        if name in _KNOWN_PARAMS:
            canonical = _REVERSE_MAPPING.get(name)
            if value is not None:
                ctype = _infer_candidate_type(value)
                self.candidates.append(
                    TunedVariableCandidate(
                        name=name,
                        candidate_type=ctype,
                        confidence=DetectionConfidence.HIGH,
                        location=_make_location(stmt_node),
                        current_value=value,
                        suggested_range=_suggest_range(canonical, ctype, value),
                        detection_source="ast",
                        reasoning=f"Variable '{name}' matches known LLM parameter '{canonical or name}'",
                        canonical_name=canonical,
                    )
                )
            return

        # Pattern 2: Fuzzy name match (substring check)
        fuzzy_canonical = self._fuzzy_match(name)
        if fuzzy_canonical and value is not None:
            ctype = _infer_candidate_type(value)
            self.candidates.append(
                TunedVariableCandidate(
                    name=name,
                    candidate_type=ctype,
                    confidence=DetectionConfidence.MEDIUM,
                    location=_make_location(stmt_node),
                    current_value=value,
                    suggested_range=_suggest_range(fuzzy_canonical, ctype, value),
                    detection_source="ast",
                    reasoning=f"Variable '{name}' fuzzy-matches canonical parameter '{fuzzy_canonical}'",
                    canonical_name=fuzzy_canonical,
                )
            )
            return

        # Pattern 3: Value-type heuristic (string looks like a model name)
        if isinstance(value, str) and _looks_like_model_string(value):
            self.candidates.append(
                TunedVariableCandidate(
                    name=name,
                    candidate_type=CandidateType.CATEGORICAL,
                    confidence=DetectionConfidence.MEDIUM,
                    location=_make_location(stmt_node),
                    current_value=value,
                    suggested_range=SuggestedRange(
                        range_type="Choices", kwargs={"values": [value]}
                    ),
                    detection_source="ast",
                    reasoning=f"Variable '{name}' assigned a value that looks like a model name: '{value}'",
                    canonical_name="model",
                )
            )

    def _fuzzy_match(self, name: str) -> str | None:
        """Attempt a fuzzy match of name against known parameters.

        Requires that the shorter string covers at least 60% of the longer
        string to avoid spurious matches like ``name`` → ``model_name``.
        """
        lower = name.lower()
        for known, canonical in _REVERSE_MAPPING.items():
            if known == lower:
                return canonical
            if known in lower or lower in known:
                shorter = min(len(known), len(lower))
                longer = max(len(known), len(lower))
                if shorter >= 3 and shorter / longer >= 0.6:
                    return canonical
        return None

    def visit_Dict(self, node: ast.Dict) -> None:
        """Detect tunable keys in dict literals (e.g., {"temperature": 0.7})."""
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None or value_node is None:
                continue
            if not isinstance(key_node, ast.Constant) or not isinstance(
                key_node.value, str
            ):
                continue

            key_name = key_node.value
            if key_name not in _KNOWN_PARAMS:
                continue

            if _is_parameter_range_call(value_node):
                continue

            value = _extract_literal_value(value_node)
            if value is None:
                continue

            canonical = _REVERSE_MAPPING.get(key_name)
            ctype = _infer_candidate_type(value)
            self.candidates.append(
                TunedVariableCandidate(
                    name=key_name,
                    candidate_type=ctype,
                    confidence=DetectionConfidence.MEDIUM,
                    location=_make_location(key_node),
                    current_value=value,
                    suggested_range=_suggest_range(canonical, ctype, value),
                    detection_source="ast",
                    reasoning=f"Dict key '{key_name}' matches known LLM parameter",
                    canonical_name=canonical,
                )
            )

        self.generic_visit(node)


class ASTDetectionStrategy:
    """AST-based detection of tuned variable candidates.

    Uses pattern matching against known LLM parameter names and
    value-type heuristics to identify tunable variables. Stateless
    and thread-safe.
    """

    def detect(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[TunedVariableCandidate]:
        """Detect candidates by walking the AST of the target function.

        Args:
            source: Python source code.
            function_name: Target function to analyze.
            context: Optional dict with ``existing_tvars`` key listing
                already-configured variable names to skip.

        Returns:
            List of detected candidates.
        """
        existing_tvars: frozenset[str] = frozenset()
        if context and "existing_tvars" in context:
            existing_tvars = frozenset(context["existing_tvars"])

        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("Failed to parse source for AST detection")
            return []

        candidates: list[TunedVariableCandidate] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    collector = _AssignmentCollector(existing_tvars)
                    collector.visit(node)
                    candidates.extend(collector.candidates)

        return candidates


# ---------------------------------------------------------------------------
# LLM Detection Strategy
# ---------------------------------------------------------------------------

_LLM_DETECTION_PROMPT = """\
Analyze the following Python function and identify variables that could be \
tuned (optimized) to improve the function's behavior. Focus on:

1. LLM API parameters (temperature, model, max_tokens, top_p, etc.)
2. Algorithmic parameters (thresholds, weights, counts, sizes)
3. Configuration choices (strategy selection, mode toggles)

For each tunable variable, return a JSON array where each element has:
- "name": variable name as it appears in the code
- "type": one of "numeric_continuous", "numeric_integer", "categorical", "boolean"
- "current_value": the current value if detectable (null otherwise)
- "reasoning": one sentence explaining why this variable is tunable

Return ONLY the JSON array, no other text.

Function to analyze:
```python
{source}
```
"""


def _extract_json_from_markdown(text: str) -> str:
    """Strip markdown code fences and return the inner JSON text."""
    if "```" not in text:
        return text
    for part in text.split("```"):
        stripped = part.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("["):
            return stripped
    return text


def _parse_candidate_item(item: Any) -> TunedVariableCandidate | None:
    """Parse a single dict from the LLM response into a candidate."""
    if not isinstance(item, dict) or "name" not in item:
        return None

    name = str(item["name"])
    type_str = item.get("type", "categorical")
    try:
        ctype = CandidateType(type_str)
    except ValueError:
        ctype = CandidateType.CATEGORICAL

    canonical = _REVERSE_MAPPING.get(name)
    return TunedVariableCandidate(
        name=name,
        candidate_type=ctype,
        confidence=DetectionConfidence.LOW,
        location=SourceLocation(line=0, col_offset=0),
        current_value=item.get("current_value"),
        suggested_range=_suggest_range(canonical, ctype, item.get("current_value")),
        detection_source="llm",
        reasoning=item.get("reasoning", ""),
        canonical_name=canonical,
    )


class LLMDetectionStrategy:
    """LLM-based detection of tuned variable candidates.

    Uses an injected LLM callable to semantically analyze function code
    and identify variables that could be tuned. Gracefully degrades to
    returning no candidates if the LLM is unavailable.

    Args:
        llm_callable: A callable that takes a prompt string and returns
            a response string. If None, this strategy is a no-op.
    """

    def __init__(self, llm_callable: Callable[[str], str] | None = None) -> None:
        self._llm = llm_callable

    def detect(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[TunedVariableCandidate]:
        """Detect candidates using LLM analysis.

        Args:
            source: Python source code.
            function_name: Target function to analyze.
            context: Optional context dict.

        Returns:
            List of detected candidates, or empty list if LLM unavailable.
        """
        if self._llm is None:
            return []

        # Extract just the target function source for the prompt
        func_source = self._extract_function_source(source, function_name)
        if not func_source:
            return []

        prompt = _LLM_DETECTION_PROMPT.format(source=func_source)

        try:
            response = self._llm(prompt)
        except Exception:
            logger.warning("LLM detection failed", exc_info=True)
            return []

        return self._parse_response(response)

    def _extract_function_source(self, source: str, function_name: str) -> str | None:
        """Extract the source of a specific function from full source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return ast.get_source_segment(source, node)
        return None

    def _parse_response(self, response: str) -> list[TunedVariableCandidate]:
        """Parse LLM JSON response into candidates."""
        text = _extract_json_from_markdown(response.strip())

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM detection response as JSON")
            return []

        if not isinstance(items, list):
            return []

        candidates: list[TunedVariableCandidate] = []
        for item in items:
            candidate = _parse_candidate_item(item)
            if candidate is not None:
                candidates.append(candidate)

        return candidates
