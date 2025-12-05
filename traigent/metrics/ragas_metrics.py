"""RAGAS metric integration helpers.

This module wraps popular metrics from the `ragas` evaluation framework so they can
be consumed by Traigent evaluators. The RAGAS project is distributed under the
Apache 2.0 license, which permits redistribution and modification with
attribution (see https://github.com/explodinggradients/ragas). We rely on the
official implementations, only adding glue code to convert Traigent
`ExampleResult` objects into the structures expected by RAGAS.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Compatibility FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - import guard
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        AnswerRelevancy,
        AnswerSimilarity,
        Faithfulness,
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
        NonLLMStringSimilarity,
    )
    from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM

    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - executed only when ragas missing
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = exc
    Metric = MetricWithEmbeddings = MetricWithLLM = object
    ragas_evaluate = EvaluationDataset = SingleTurnSample = None

from traigent.api.types import ExampleResult

POPULAR_RAGAS_METRICS: tuple[str, ...] = (
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_similarity",
)

DEFAULT_COLUMN_CANDIDATES: Mapping[str, Sequence[str]] = {
    "user_input": ("user_input", "question", "prompt", "input", "query"),
    "retrieved_contexts": (
        "retrieved_contexts",
        "contexts",
        "context",
        "retrieved_documents",
        "documents",
        "chunks",
    ),
    "reference_contexts": (
        "reference_contexts",
        "ground_truth_contexts",
        "gold_contexts",
        "reference_documents",
    ),
}

_CANONICAL_COLUMNS = {
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "reference",
}


@dataclass(slots=True)
class RagasConfig:
    """Configuration controlling how RAGAS metrics extract inputs.

    Attributes:
        column_map: Optional mapping from canonical RAGAS column names (``"user_input"``,
            ``"retrieved_contexts"``, ``"reference_contexts"``, ``"reference"``) to dataset keys.
            Values must be strings naming the corresponding metadata entry. ``None`` means
            canonical names are used as-is.
        llm: Optional RAGAS-compatible LLM wrapper used by metrics that require model judgements.
        embeddings: Optional embedding model used by similarity-based metrics.
    """

    column_map: Mapping[str, str] | None = None
    llm: Any | None = None
    embeddings: Any | None = None


class RagasConfigurationError(RuntimeError):
    """Raised when ragas metrics are requested but cannot be computed."""


_GLOBAL_RAGAS_CONFIG = RagasConfig()

_RAGAS_METRIC_FACTORIES: dict[str, Callable[[RagasConfig], Metric]] = {}

_UNSET = object()


def configure_ragas_defaults(
    *,
    column_map: Mapping[str, str] | None | object = _UNSET,
    llm: Any | object = _UNSET,
    embeddings: Any | object = _UNSET,
) -> None:
    """Set default RAGAS configuration used by evaluators.

    Each argument is optional. Pass an explicit ``None`` to clear a value, omit the
    argument to leave the existing value unchanged.

    Args:
        column_map: Mapping from canonical RAGAS column names (``"user_input"``, ``"retrieved_contexts"``,
            ``"reference_contexts"``, ``"reference"``) to the keys stored in your dataset/evaluation
            metadata. Values must be strings naming the corresponding column. Example:
            ``{"retrieved_contexts": "gold_contexts"}``.
        llm: RAGAS-compatible LLM wrapper (e.g. ``ragas.llms.LangchainLLM``) required for faithfulness
            or answer relevancy metrics. Pass ``None`` to clear.
        embeddings: Optional embedding model used by similarity metrics. Pass ``None`` to clear.

    Raises:
        RagasConfigurationError: If ``column_map`` is not a mapping of strings to strings.

    Example:
        >>> from ragas.llms import LangchainLLM
        >>> configure_ragas_defaults(
        ...     column_map={"retrieved_contexts": "gold_contexts"},
        ...     llm=LangchainLLM(...),
        ... )
    """

    global _GLOBAL_RAGAS_CONFIG

    current = _GLOBAL_RAGAS_CONFIG

    new_column_map: Mapping[str, str] | None
    if column_map is _UNSET:
        new_column_map = current.column_map
    elif column_map is None:
        new_column_map = None
    elif isinstance(column_map, Mapping):
        try:
            new_column_map = {str(key): str(value) for key, value in column_map.items()}
        except Exception as exc:
            raise RagasConfigurationError(
                "column_map must map strings to strings"
            ) from exc
        invalid_keys = set(new_column_map.keys()) - _CANONICAL_COLUMNS
        if invalid_keys:
            raise RagasConfigurationError(
                f"Invalid column_map keys: {sorted(invalid_keys)}. "
                f"Allowed keys are {sorted(_CANONICAL_COLUMNS)}."
            )
    else:
        raise RagasConfigurationError(
            "column_map must be a mapping of strings to strings or None."
        )

    new_llm = current.llm if llm is _UNSET else llm
    new_embeddings = current.embeddings if embeddings is _UNSET else embeddings

    _GLOBAL_RAGAS_CONFIG = RagasConfig(
        column_map=new_column_map,
        llm=new_llm,
        embeddings=new_embeddings,
    )


def _ensure_ragas_available() -> None:
    if not RAGAS_AVAILABLE:
        message = "ragas is not installed. Install it with `pip install ragas` to enable ragas metrics."
        if RAGAS_IMPORT_ERROR:
            message = f"{message} (import failed with: {RAGAS_IMPORT_ERROR})"
        raise RagasConfigurationError(message)


def _rename_metric(metric: Metric, name: str) -> Metric:
    if getattr(metric, "name", name) != name:
        metric.name = name
    return metric


def _context_precision_factory(_: RagasConfig) -> Metric:
    return _rename_metric(NonLLMContextPrecisionWithReference(), "context_precision")


def _context_recall_factory(_: RagasConfig) -> Metric:
    return _rename_metric(NonLLMContextRecall(), "context_recall")


def _answer_similarity_factory(config: RagasConfig) -> Metric:
    if config.embeddings is not None:
        return _rename_metric(
            AnswerSimilarity(embeddings=config.embeddings), "answer_similarity"
        )
    return _rename_metric(NonLLMStringSimilarity(), "answer_similarity")


def _faithfulness_factory(config: RagasConfig) -> Metric:
    if config.llm is None:
        raise RagasConfigurationError(
            "The 'faithfulness' metric requires `ragas_llm` to be configured."
        )
    return _rename_metric(Faithfulness(llm=config.llm), "faithfulness")


def _answer_relevancy_factory(config: RagasConfig) -> Metric:
    if config.llm is None:
        raise RagasConfigurationError(
            "The 'answer_relevancy' metric requires `ragas_llm` to be configured."
        )
    return _rename_metric(AnswerRelevancy(llm=config.llm), "answer_relevancy")


_RAGAS_METRIC_FACTORIES.update(
    {
        "context_precision": _context_precision_factory,
        "context_recall": _context_recall_factory,
        "answer_similarity": _answer_similarity_factory,
        "faithfulness": _faithfulness_factory,
        "answer_relevancy": _answer_relevancy_factory,
    }
)


def _build_metric(name: str, *, config: RagasConfig) -> Metric:
    """Instantiate a RAGAS metric using the internal factory registry."""

    try:
        return _RAGAS_METRIC_FACTORIES[name](config)
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown ragas metric '{name}'") from exc


def _extract_candidate(
    sources: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any | None:
    for source in sources:
        if not source:
            continue
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _normalise_contexts(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, Mapping):
                if "text" in item and item["text"]:
                    items.append(str(item["text"]))
                elif "content" in item and item["content"]:
                    items.append(str(item["content"]))
                else:
                    items.append(str(item))
            else:
                items.append(str(item))
        return items or None
    if isinstance(value, Mapping):
        return _normalise_contexts(list(value.values()))
    return [str(value)]


def _determine_column_keys(
    column_map: Mapping[str, str] | None,
    key: str,
) -> Sequence[str]:
    if column_map and key in column_map:
        return (column_map[key],)
    return DEFAULT_COLUMN_CANDIDATES.get(key, ())


def _prepare_samples(
    example_results: Sequence[ExampleResult],
    *,
    config: RagasConfig,
    required_columns: set[str],
    dataset_examples: Sequence[Any] | None = None,
    outputs: Sequence[Any] | None = None,
) -> list[SingleTurnSample]:
    samples: list[SingleTurnSample] = []

    for index, result in enumerate(example_results):
        metadata_sources: list[Mapping[str, Any]] = []
        if result.metadata:
            metadata_sources.append(result.metadata)
        if dataset_examples is not None and index < len(dataset_examples):
            example_metadata = getattr(dataset_examples[index], "metadata", None)
            if example_metadata:
                metadata_sources.append(example_metadata)
        input_sources: list[Mapping[str, Any]] = []
        if isinstance(result.input_data, Mapping):
            input_sources.append(result.input_data)
        if dataset_examples is not None and index < len(dataset_examples):
            dataset_input = getattr(dataset_examples[index], "input_data", None)
            if isinstance(dataset_input, Mapping):
                input_sources.append(dataset_input)

        response = result.actual_output
        if response is None and outputs is not None and index < len(outputs):
            response = outputs[index]
        if response is None:
            continue

        reference = result.expected_output
        if reference is None:
            continue

        user_input_keys = _determine_column_keys(config.column_map, "user_input")
        user_input = _extract_candidate(input_sources, user_input_keys)
        if user_input is None and hasattr(result, "input_data"):
            user_input = str(result.input_data)
        if user_input is None:
            user_input = str(reference)

        contexts_keys = _determine_column_keys(config.column_map, "retrieved_contexts")
        contexts_raw = _extract_candidate(metadata_sources, contexts_keys)
        contexts = _normalise_contexts(contexts_raw)

        reference_contexts_keys = _determine_column_keys(
            config.column_map, "reference_contexts"
        )
        reference_contexts_raw = _extract_candidate(
            metadata_sources, reference_contexts_keys
        )
        reference_contexts = _normalise_contexts(reference_contexts_raw)

        fields = {
            "user_input": user_input,
            "response": str(response),
            "reference": str(reference) if reference is not None else None,
            "retrieved_contexts": contexts,
            "reference_contexts": reference_contexts,
        }

        if any(
            col in required_columns
            and (fields.get(col) is None or fields.get(col) == [])
            for col in required_columns
        ):
            continue

        samples.append(
            SingleTurnSample(
                user_input=fields.get("user_input"),
                response=fields.get("response"),
                reference=fields.get("reference"),
                retrieved_contexts=fields.get("retrieved_contexts"),
                reference_contexts=fields.get("reference_contexts"),
            )
        )

    return samples


def compute_ragas_metrics(
    example_results: Sequence[ExampleResult],
    metric_names: Sequence[str],
    *,
    dataset_examples: Sequence[Any] | None = None,
    outputs: Sequence[Any] | None = None,
    config: RagasConfig | None = None,
) -> dict[str, float]:
    """Compute the requested ragas metrics.

    Args:
        example_results: Collection of evaluation results from Traigent.
        metric_names: Ragas metric identifiers to compute.
        dataset_examples: Optional backing dataset examples (used when
            additional metadata lives on the dataset rather than the
            ExampleResult instances).
        outputs: Optional list of raw outputs aligned with example_results,
            used when `ExampleResult.actual_output` is unavailable.
        config: Configuration holding custom column mapping, LLM, and embeddings.
            If omitted, the defaults supplied via :func:`configure_ragas_defaults`
            are used.

    Returns:
        Dictionary mapping metric name to aggregated score.
    """
    if not metric_names:
        return {}

    _ensure_ragas_available()

    if config is None:
        config = RagasConfig(
            column_map=(
                dict(_GLOBAL_RAGAS_CONFIG.column_map)
                if _GLOBAL_RAGAS_CONFIG.column_map
                else None
            ),
            llm=_GLOBAL_RAGAS_CONFIG.llm,
            embeddings=_GLOBAL_RAGAS_CONFIG.embeddings,
        )

    example_results = list(example_results)
    if (not example_results) and dataset_examples is not None:
        example_results = []
        for idx, example in enumerate(dataset_examples):
            expected = getattr(example, "expected_output", None)
            actual = None
            if outputs is not None and idx < len(outputs):
                actual = outputs[idx]
            example_results.append(
                ExampleResult(
                    example_id=f"example_{idx}",
                    input_data=getattr(example, "input_data", {}) or {},
                    expected_output=expected,
                    actual_output=actual,
                    metrics={},
                    execution_time=0.0,
                    success=(actual is not None),
                    error_message=None if (actual is not None) else "output missing",
                    metadata=getattr(example, "metadata", {}) or {},
                )
            )

    metrics: list[Metric] = []
    required_columns: set[str] = set()

    for name in metric_names:
        metric = _build_metric(name, config=config)
        metrics.append(metric)
        for columns in metric.required_columns.values():
            required_columns.update(columns)

    samples = _prepare_samples(
        example_results,
        config=config,
        required_columns=required_columns,
        dataset_examples=dataset_examples,
        outputs=outputs,
    )

    if not samples:
        raise RagasConfigurationError(
            "Unable to compute ragas metrics because no evaluation sample contains "
            "the required fields (user_input/response/reference/contexts)."
        )

    dataset = EvaluationDataset(samples)

    # RAGAS uses asyncio under the hood; allow nested loops so evaluators can call
    # this helper from within event loops safely.
    result = ragas_evaluate(
        dataset,
        metrics=metrics,
        llm=config.llm,
        embeddings=config.embeddings,
        show_progress=False,
        allow_nest_asyncio=True,
    )

    aggregated: dict[str, float] = {}
    for metric in metrics:
        values = result[metric.name]
        aggregated[metric.name] = float(np.nanmean(values)) if len(values) else 0.0

    return aggregated


__all__ = [
    "POPULAR_RAGAS_METRICS",
    "RAGAS_AVAILABLE",
    "RagasConfigurationError",
    "RagasConfig",
    "compute_ragas_metrics",
]
