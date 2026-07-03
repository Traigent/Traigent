"""DeepEval metric integration helpers.

This module bridges metrics from the `deepeval` evaluation framework so they can
be used as Traigent ``metric_functions``. DeepEval is distributed under the
Apache 2.0 license (see https://github.com/confident-ai/deepeval). We rely on
the official metric implementations, only adding glue code to convert Traigent
evaluation data into DeepEval's ``LLMTestCase`` format.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_MISSING = object()


@dataclass(frozen=True)
class _TracerProviderSnapshot:
    trace_module: Any
    provider: Any
    raw_provider: Any
    once_guard: Any
    once_done: Any


# Aikido Intel flags deepeval telemetry as exfiltrating host OpenTelemetry
# spans to deepeval/New Relic; upstream issue confident-ai/deepeval#2497 is
# still open. These must be set before any import path can execute deepeval.
_DEEPEVAL_TELEMETRY_OPT_OUT_ENV: dict[str, str] = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",  # deepeval analytics/PostHog opt-out
    "ERROR_REPORTING": "NO",  # deepeval Sentry/error-reporting opt-out
}


def _set_deepeval_telemetry_opt_out_env() -> None:
    """Disable deepeval telemetry knobs unless the host configured them."""
    for name, value in _DEEPEVAL_TELEMETRY_OPT_OUT_ENV.items():
        os.environ.setdefault(name, value)


def _capture_otel_tracer_provider() -> _TracerProviderSnapshot | None:
    """Capture the global OpenTelemetry tracer provider state if available."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
    except Exception:
        return None

    once_guard = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", _MISSING)
    once_done = getattr(once_guard, "_done", _MISSING)
    return _TracerProviderSnapshot(
        trace_module=trace,
        provider=provider,
        raw_provider=getattr(trace, "_TRACER_PROVIDER", _MISSING),
        once_guard=once_guard,
        once_done=once_done,
    )


def _restore_otel_tracer_provider(
    snapshot: _TracerProviderSnapshot | None,
) -> None:
    """Restore OpenTelemetry state if deepeval changed the global provider."""
    if snapshot is None:
        return

    trace = snapshot.trace_module
    try:
        current_provider = trace.get_tracer_provider()
    except Exception:
        return
    if current_provider is snapshot.provider:
        return

    restored_once_guard = False
    if snapshot.raw_provider is not _MISSING:
        try:
            trace._TRACER_PROVIDER = snapshot.raw_provider
        except Exception:
            pass

    if snapshot.once_guard is not _MISSING and snapshot.once_done is not _MISSING:
        try:
            snapshot.once_guard._done = snapshot.once_done
            restored_once_guard = True
        except Exception:
            pass

    try:
        restored = trace.get_tracer_provider() is snapshot.provider
    except Exception:
        restored = False

    if not restored:
        try:
            trace.set_tracer_provider(snapshot.provider)
            restored = trace.get_tracer_provider() is snapshot.provider
        except Exception:
            restored = False

    if restored:
        logger.warning(
            "Restored host OpenTelemetry TracerProvider after deepeval import "
            "changed the global provider; see Aikido Intel and deepeval#2497."
        )
    else:
        logger.warning(
            "deepeval import changed the global OpenTelemetry TracerProvider, "
            "but Traigent could not restore it; see Aikido Intel and "
            "deepeval#2497."
        )

    if restored and not restored_once_guard:
        logger.warning(
            "OpenTelemetry TracerProvider was restored, but the provider "
            "one-shot guard state could not be restored."
        )


def _import_deepeval_test_case() -> Any:
    _set_deepeval_telemetry_opt_out_env()
    tracer_snapshot = _capture_otel_tracer_provider()
    try:
        from deepeval.test_case import LLMTestCase
    finally:
        _restore_otel_tracer_provider(tracer_snapshot)
    return LLMTestCase


try:  # pragma: no cover - import guard
    LLMTestCase = _import_deepeval_test_case()

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
    _set_deepeval_telemetry_opt_out_env()
    tracer_snapshot = _capture_otel_tracer_provider()
    try:
        import deepeval.metrics as dm
    finally:
        _restore_otel_tracer_provider(tracer_snapshot)

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
    """Return a copy of *instance*, falling back gracefully.

    Raises ``TypeError`` if the instance cannot be copied at all, to avoid
    returning a shared mutable reference that would cause race conditions
    during parallel trial execution.
    """
    try:
        return copy.deepcopy(instance)
    except Exception:
        pass
    try:
        return copy.copy(instance)
    except Exception as exc:
        raise TypeError(
            f"DeepEvalScorer: metric instance {type(instance).__name__!r} "
            f"cannot be copied. Parallel trial execution requires copyable "
            f"metric instances to avoid race conditions."
        ) from exc


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
