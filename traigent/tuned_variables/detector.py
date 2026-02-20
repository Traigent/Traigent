"""Tuned variable detector orchestrator.

Combines multiple detection strategies (AST-based, LLM-based) to identify
variables in user function source code that could be optimized by Traigent.

Thread-safe: the detector holds only immutable configuration set at
construction time. All result objects are frozen dataclasses.

Example::

    from traigent.tuned_variables.detector import TunedVariableDetector

    detector = TunedVariableDetector()
    result = detector.detect_from_source(source_code, "my_function")
    for candidate in result.high_confidence:
        print(f"{candidate.name}: {candidate.suggested_range}")
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from traigent.tuned_variables.dataflow_strategy import DataFlowDetectionStrategy
from traigent.tuned_variables.detection_strategies import (
    ASTDetectionStrategy,
    DetectionStrategy,
)
from traigent.tuned_variables.detection_types import (
    DetectionConfidence,
    DetectionResult,
    TunedVariableCandidate,
)

logger = logging.getLogger(__name__)


class TunedVariableDetector:
    """Detect tuned variable candidates in user function source code.

    Orchestrates multiple detection strategies and merges their results
    with deduplication and confidence upgrading.

    Args:
        strategies: List of detection strategies. If None, defaults to
            ``[ASTDetectionStrategy(), DataFlowDetectionStrategy()]``.
            Strategies must be stateless or thread-safe.
    """

    def __init__(
        self,
        strategies: list[DetectionStrategy] | None = None,
    ) -> None:
        self._strategies: list[DetectionStrategy] = strategies or [
            ASTDetectionStrategy(),
            DataFlowDetectionStrategy(),
        ]

    def detect_from_source(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Detect tuned variable candidates from a source code string.

        Does NOT import or execute the source code. Uses AST-only analysis
        plus any configured LLM strategies.

        Args:
            source: Python source code.
            function_name: Target function name to analyze.
            context: Optional context dict. Recognized keys:
                - ``existing_tvars``: set of already-tuned variable names to skip.

        Returns:
            DetectionResult with all merged candidates.
        """
        all_candidate_lists: list[list[TunedVariableCandidate]] = []
        strategy_names: list[str] = []

        for strategy in self._strategies:
            try:
                candidates = strategy.detect(source, function_name, context=context)
                all_candidate_lists.append(candidates)
                strategy_names.append(type(strategy).__name__)
            except Exception:
                logger.warning(
                    "Detection strategy %s failed",
                    type(strategy).__name__,
                    exc_info=True,
                )

        merged = self._merge_candidates(all_candidate_lists)
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        return DetectionResult(
            function_name=function_name,
            candidates=tuple(merged),
            detection_strategies_used=tuple(strategy_names),
            source_hash=source_hash,
        )

    def detect_from_callable(
        self,
        func: Callable[..., Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Detect tuned variable candidates from a callable.

        Uses ``inspect.getsource()`` to obtain source, then delegates
        to ``detect_from_source()``.

        Args:
            func: The callable to analyze.
            context: Optional context dict.

        Returns:
            DetectionResult with all merged candidates.
        """
        func_name = getattr(func, "__name__", "<unknown>")

        try:
            source = textwrap.dedent(inspect.getsource(func))
        except OSError:
            logger.warning("Cannot retrieve source for %s", func_name)
            return DetectionResult(
                function_name=func_name,
                warnings=("Could not retrieve source code for analysis.",),
            )

        return self.detect_from_source(source, func_name, context=context)

    def detect_from_file(
        self,
        file_path: str | Path,
        function_name: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        """Detect tuned variable candidates from a Python file.

        If ``function_name`` is None, analyzes all top-level functions.

        Args:
            file_path: Path to Python file.
            function_name: Optional specific function to analyze.
            context: Optional context dict.

        Returns:
            List of DetectionResult, one per analyzed function.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            return []

        source = path.read_text(encoding="utf-8")

        if function_name:
            return [self.detect_from_source(source, function_name, context=context)]

        # Discover all top-level functions
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("Failed to parse %s", path)
            return []

        results: list[DetectionResult] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result = self.detect_from_source(source, node.name, context=context)
                if result.count > 0:
                    results.append(result)

        return results

    def _merge_candidates(
        self,
        all_candidate_lists: list[list[TunedVariableCandidate]],
    ) -> list[TunedVariableCandidate]:
        """Merge candidates from multiple strategies.

        Deduplication by name (since detection is scoped to a single
        function). When multiple strategies detect the same variable,
        confidence is upgraded and reasoning is combined.
        """
        if len(all_candidate_lists) <= 1:
            return all_candidate_lists[0] if all_candidate_lists else []

        # Index by name for deduplication (scoped to single function)
        seen: dict[str, TunedVariableCandidate] = {}

        for candidates in all_candidate_lists:
            for c in candidates:
                key = c.name
                if key not in seen:
                    seen[key] = c
                else:
                    # Merge: upgrade confidence and combine reasoning
                    existing = seen[key]
                    merged_confidence = _upgrade_confidence(
                        existing.confidence, c.confidence
                    )
                    merged_reasoning = existing.reasoning
                    if c.reasoning and c.reasoning not in merged_reasoning:
                        merged_reasoning = f"{merged_reasoning}; {c.reasoning}"

                    seen[key] = TunedVariableCandidate(
                        name=existing.name,
                        candidate_type=existing.candidate_type,
                        confidence=merged_confidence,
                        location=existing.location,
                        current_value=existing.current_value or c.current_value,
                        suggested_range=existing.suggested_range or c.suggested_range,
                        detection_source="combined",
                        reasoning=merged_reasoning,
                        canonical_name=existing.canonical_name or c.canonical_name,
                    )

        return list(seen.values())


def _upgrade_confidence(
    a: DetectionConfidence, b: DetectionConfidence
) -> DetectionConfidence:
    """Return the higher of two confidence levels, with a boost for overlap.

    When two strategies agree, a LOW candidate is upgraded to MEDIUM.
    """
    _order = {
        DetectionConfidence.LOW: 0,
        DetectionConfidence.MEDIUM: 1,
        DetectionConfidence.HIGH: 2,
    }
    max_level = max(_order[a], _order[b])
    # Boost: if both detected it, upgrade LOW -> MEDIUM
    if max_level == 0:
        return DetectionConfidence.MEDIUM
    if max_level == 1:
        return DetectionConfidence.MEDIUM
    return DetectionConfidence.HIGH
