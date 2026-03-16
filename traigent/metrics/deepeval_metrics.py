"""DeepEval metric integration helpers.

This module bridges metrics from the `deepeval` evaluation framework so they can
be used as Traigent ``metric_functions``. DeepEval is distributed under the
Apache 2.0 license (see https://github.com/confident-ai/deepeval). We rely on
the official metric implementations, only adding glue code to convert Traigent
evaluation data into DeepEval's ``LLMTestCase`` format.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
    DEEPEVAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - executed only when deepeval missing
    DEEPEVAL_AVAILABLE = False
    DEEPEVAL_IMPORT_ERROR = exc
    LLMTestCase = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# String shortcut registry
# ---------------------------------------------------------------------------

DEEPEVAL_METRIC_SHORTCUTS: dict[str, str] = {
    "relevancy": "AnswerRelevancyMetric",
    "answer_relevancy": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    "hallucination": "HallucinationMetric",
    "toxicity": "ToxicityMetric",
    "bias": "BiasMetric",
    "contextual_relevancy": "ContextualRelevancyMetric",
    "contextual_precision": "ContextualPrecisionMetric",
    "contextual_recall": "ContextualRecallMetric",
    "summarization": "SummarizationMetric",
}

_SHORTCUT_TO_CLASS: dict[str, type] | None = None

# Input key candidates (tried in order) when extracting LLMTestCase.input
_INPUT_KEY_CANDIDATES = ("question", "input", "prompt", "query")


def _ensure_deepeval_available() -> None:
    """Raise ``ImportError`` with a helpful message when deepeval is missing."""
    if not DEEPEVAL_AVAILABLE:
        msg = (
            "deepeval is required for DeepEvalScorer. "
            "Install with: pip install 'traigent[deepeval]'"
        )
        if DEEPEVAL_IMPORT_ERROR is not None:
            msg += f"\nOriginal error: {DEEPEVAL_IMPORT_ERROR}"
        raise ImportError(msg)


def _get_shortcut_classes() -> dict[str, type]:
    """Lazily build a mapping from shortcut class name → actual class."""
    global _SHORTCUT_TO_CLASS
    if _SHORTCUT_TO_CLASS is not None:
        return _SHORTCUT_TO_CLASS

    _ensure_deepeval_available()
    import deepeval.metrics as dm

    _SHORTCUT_TO_CLASS = {}
    for class_name in set(DEEPEVAL_METRIC_SHORTCUTS.values()):
        cls = getattr(dm, class_name, None)
        if cls is not None:
            _SHORTCUT_TO_CLASS[class_name] = cls
    return _SHORTCUT_TO_CLASS


def _extract_input_text(input_data: Any) -> str:
    """Extract a string suitable for ``LLMTestCase.input`` from *input_data*."""
    if input_data is None:
        return ""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in _INPUT_KEY_CANDIDATES:
            if key in input_data and input_data[key] is not None:
                return str(input_data[key])
        return str(input_data)
    return str(input_data)


def _coerce_to_str_list(value: Any) -> list[str] | None:
    """Coerce *value* into ``list[str]`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    # Mappings would iterate over keys only — treat as a single stringified item.
    if isinstance(value, Mapping):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(x) for x in value]
    return [str(value)]


def _safe_copy(instance: Any) -> Any:
    """Return a copy of *instance*, falling back gracefully."""
    try:
        return copy.deepcopy(instance)
    except Exception:
        pass
    try:
        return copy.copy(instance)
    except Exception:
        logger.warning(
            "DeepEvalScorer: could not copy metric instance %r; "
            "parallel trial execution may produce race conditions.",
            type(instance).__name__,
        )
        return instance


class DeepEvalScorer:
    """Bridge DeepEval metrics into Traigent's ``metric_functions`` system.

    Args:
        metrics: List of metric shortcut strings (e.g. ``"relevancy"``) or
            pre-configured DeepEval metric instances.
        model: Judge model name applied to all string-shortcut metrics.
            Ignored for pre-configured instances.
        threshold: Default threshold for string-shortcut metrics.
        context_key: Key in ``metadata`` or ``input_data`` holding context
            for RAG metrics.
        retrieval_context_key: Key for retrieval context in ``metadata`` or
            ``input_data``.
    """

    def __init__(
        self,
        metrics: list[str | Any],
        *,
        model: str | None = None,
        threshold: float = 0.5,
        context_key: str = "context",
        retrieval_context_key: str = "retrieval_context",
    ) -> None:
        _ensure_deepeval_available()
        if not metrics:
            raise ValueError("metrics must be a non-empty list")

        self._model = model
        self._threshold = threshold
        self._context_key = context_key
        self._retrieval_context_key = retrieval_context_key

        # Resolve all metrics eagerly so errors surface at construction time.
        self._resolved: list[tuple[str, Any]] = [
            self._resolve_metric(m) for m in metrics
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_metric_functions(self) -> dict[str, Callable[..., float]]:
        """Return a dict compatible with ``metric_functions=`` parameter.

        Each key is a metric name and each value is a callable with the
        signature ``(output, expected, input_data, metadata) -> float``.
        """
        result: dict[str, Callable[..., float]] = {}
        seen: dict[str, int] = {}

        for name, instance in self._resolved:
            if name in seen:
                seen[name] += 1
                unique_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
                unique_name = name
            result[unique_name] = self._make_metric_fn(unique_name, instance)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metric(self, metric: str | Any) -> tuple[str, Any]:
        """Resolve a string shortcut or validate a pre-configured instance."""
        if isinstance(metric, str):
            class_name = DEEPEVAL_METRIC_SHORTCUTS.get(metric.lower())
            if class_name is None:
                raise ValueError(
                    f"Unknown DeepEval metric shortcut: {metric!r}. "
                    f"Available shortcuts: {sorted(DEEPEVAL_METRIC_SHORTCUTS.keys())}"
                )
            classes = _get_shortcut_classes()
            metric_class = classes.get(class_name)
            if metric_class is None:  # pragma: no cover
                raise ImportError(
                    f"DeepEval metric class {class_name!r} not found. "
                    "Please check your deepeval installation."
                )
            kwargs: dict[str, Any] = {"threshold": self._threshold}
            if self._model is not None:
                kwargs["model"] = self._model
            return (metric.lower(), metric_class(**kwargs))

        # Pre-configured instance — prefer .name attribute for readability.
        name = getattr(metric, "name", None)
        if name is not None:
            name = str(name)
        if not name:
            # Derive from class name: AnswerRelevancyMetric → answer_relevancy
            cls_name = type(metric).__name__
            name = cls_name.replace("Metric", "")
            # CamelCase → snake_case
            chars: list[str] = []
            for i, ch in enumerate(name):
                if ch.isupper() and i > 0:
                    chars.append("_")
                chars.append(ch.lower())
            name = "".join(chars)
        return (name, metric)

    def _make_metric_fn(self, name: str, instance: Any) -> Callable[..., float]:
        """Create a closure matching Traigent's flexible metric signature."""
        context_key = self._context_key
        retrieval_context_key = self._retrieval_context_key

        def metric_fn(
            output: Any,
            expected: Any = None,
            input_data: Any = None,
            metadata: dict[str, Any] | None = None,
        ) -> float:
            metadata = metadata or {}

            # Build LLMTestCase kwargs defensively.
            tc_kwargs: dict[str, Any] = {
                "input": _extract_input_text(input_data),
                "actual_output": str(output) if output is not None else "",
            }

            if expected is not None:
                tc_kwargs["expected_output"] = str(expected)

            # Extract context for RAG metrics.
            # Use key-presence check (not `or`) so explicit empty list is respected.
            if context_key in metadata:
                ctx = metadata[context_key]
            elif isinstance(input_data, dict) and context_key in input_data:
                ctx = input_data[context_key]
            else:
                ctx = None
            ctx_list = _coerce_to_str_list(ctx)
            if ctx_list is not None:
                tc_kwargs["context"] = ctx_list

            if retrieval_context_key in metadata:
                ret_ctx = metadata[retrieval_context_key]
            elif isinstance(input_data, dict) and retrieval_context_key in input_data:
                ret_ctx = input_data[retrieval_context_key]
            else:
                ret_ctx = None
            ret_ctx_list = _coerce_to_str_list(ret_ctx)
            if ret_ctx_list is not None:
                tc_kwargs["retrieval_context"] = ret_ctx_list

            test_case = LLMTestCase(**tc_kwargs)

            # Thread-safe: copy the metric instance so .measure()/.score
            # mutations don't race across parallel trials.
            metric_copy = _safe_copy(instance)
            metric_copy.measure(test_case)
            return float(metric_copy.score)

        # Preserve a helpful name for debugging / inspect.
        metric_fn.__name__ = f"deepeval_{name}"
        metric_fn.__qualname__ = f"DeepEvalScorer.{name}"
        return metric_fn


__all__ = [
    "DEEPEVAL_AVAILABLE",
    "DEEPEVAL_IMPORT_ERROR",
    "DEEPEVAL_METRIC_SHORTCUTS",
    "DeepEvalScorer",
]
