"""Base classes for evaluation strategies."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from collections.abc import Mapping
from collections.abc import Mapping as CollectionsMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from traigent.api.types import ExampleResult
from traigent.evaluators.dataset_registry import (
    DatasetRegistryEntry,
    resolve_dataset_reference,
)
from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils.error_handler import APIKeyError
from traigent.utils.error_handler import TraigentError as FriendlyTraigentError
from traigent.utils.exceptions import ConfigurationError, EvaluationError
from traigent.utils.exceptions import TraigentError as CoreTraigentError
from traigent.utils.exceptions import TrialPrunedError, ValidationError
from traigent.utils.langchain_interceptor import get_captured_response_by_key
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease

# Tracing utilities are imported lazily to avoid circular imports
# The actual import happens in _get_tracing_functions()
_tracing_functions_cache: dict[str, Any] | None = None


def _get_tracing_functions() -> tuple[Any, Any, bool]:
    """Lazily import tracing functions to avoid circular imports.

    Returns:
        Tuple of (example_evaluation_span, record_example_result, is_available)
    """
    global _tracing_functions_cache
    if _tracing_functions_cache is not None:
        return (
            _tracing_functions_cache.get("example_evaluation_span"),
            _tracing_functions_cache.get("record_example_result"),
            _tracing_functions_cache.get("available", False),
        )

    try:
        from traigent.core.tracing import example_evaluation_span as ees
        from traigent.core.tracing import record_example_result as rer

        _tracing_functions_cache = {
            "example_evaluation_span": ees,
            "record_example_result": rer,
            "available": True,
        }
        return ees, rer, True
    except Exception:
        _tracing_functions_cache = {"available": False}
        return None, None, False


# Legacy compatibility - these will be None, use _get_tracing_functions() instead
TRACING_AVAILABLE = False  # Will be checked dynamically
example_evaluation_span = None  # type: ignore
record_example_result = None  # type: ignore

logger = get_logger(__name__)

try:  # pragma: no cover - import guard for optional dependency
    from traigent.metrics.ragas_metrics import (
        POPULAR_RAGAS_METRICS,
        RAGAS_AVAILABLE,
        RagasConfig,
        RagasConfigurationError,
        compute_ragas_metrics,
    )
except Exception:  # pragma: no cover - executed only when module missing
    POPULAR_RAGAS_METRICS = ()
    RAGAS_AVAILABLE = False

    @dataclass(slots=True)
    class _FallbackRagasConfig:
        column_map: Mapping[str, str] | None = None
        llm: Any | None = None
        embeddings: Any | None = None

    RagasConfig = _FallbackRagasConfig  # type: ignore[misc, assignment]
    RagasConfigurationError = RuntimeError  # type: ignore[misc, assignment]
    compute_ragas_metrics = None  # type: ignore[assignment]


DATASET_ROOT_ENV = "TRAIGENT_DATASET_ROOT"


def _get_dataset_root() -> Path:
    """Return the trusted root directory for evaluation datasets."""

    env_value = os.getenv(DATASET_ROOT_ENV)
    if env_value:
        candidate = Path(env_value).expanduser()
        try:
            return candidate.resolve(strict=True)
        except FileNotFoundError as exc:  # pragma: no cover - defensive guard
            raise ValidationError(
                f"Configured dataset root does not exist: {candidate}"
            ) from exc
    return Path.cwd().resolve()


def _resolve_dataset_source(
    source: str,
) -> tuple[Path, DatasetRegistryEntry | None]:
    """Resolve a dataset reference to an absolute path.

    Security: All dataset paths must reside under the configured dataset root.
    When TRAIGENT_DATASET_ROOT is not set, the current working directory is
    treated as the trusted root to prevent accidental traversal.
    """

    dataset_root = _get_dataset_root()
    resolved_reference, registry_entry = resolve_dataset_reference(source)
    path_obj = Path(resolved_reference)
    is_absolute_path = path_obj.is_absolute()
    candidate = path_obj if is_absolute_path else dataset_root / path_obj

    try:
        resolved_path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"Dataset file not found: {source}") from exc
    except RuntimeError as exc:  # pragma: no cover - symlink loops
        raise ValidationError(f"Invalid dataset path: {source}") from exc

    # Enforce dataset root constraint for both relative and absolute paths.
    # This prevents accidental access to files outside the intended dataset root
    # even when TRAIGENT_DATASET_ROOT is not explicitly set.
    try:
        resolved_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValidationError(
            f"Dataset path must reside under {dataset_root}: {resolved_path}"
        ) from exc

    if not resolved_path.is_file():
        raise ValidationError(f"Dataset path must be a file: {source}")

    return resolved_path, registry_entry


@contextmanager
def _maybe_restore_trial_context(trial_ctx: dict[str, Any] | None) -> Any:
    """Reapply an existing trial context when executing in a new scope."""

    if trial_ctx is None:
        yield
        return

    from traigent.config.context import set_trial_context, trial_context

    token = set_trial_context(trial_ctx)
    try:
        yield
    finally:
        trial_context.reset(token)


def _is_empty_expected_output(value: Any) -> bool:
    """Check if an expected output value is effectively empty.

    Returns True if the value is None, an empty string, or a string containing
    only whitespace. These cases cannot be used for meaningful accuracy computation.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _build_dataset(
    cls: type[Dataset],
    *,
    examples: list[EvaluationExample],
    resolved_path: Path,
    source: str,
    registry_entry: DatasetRegistryEntry | None = None,
    name_hint: str | None = None,
    description_hint: str | None = None,
    metadata_hint: dict[str, Any] | None = None,
) -> Dataset:
    if not examples:
        raise ValidationError(f"No valid examples found in {source}")

    # Check for missing/empty expected outputs and warn users
    missing_count = sum(
        1 for ex in examples if _is_empty_expected_output(ex.expected_output)
    )
    if missing_count > 0:
        if missing_count == len(examples):
            logger.warning(
                "Dataset '%s' has no expected outputs (output field missing or empty "
                "in all %d examples). Accuracy metrics will not be meaningful. "
                "Consider adding expected outputs or using metrics that don't require them.",
                source,
                len(examples),
            )
        else:
            logger.warning(
                "Dataset '%s' has %d/%d examples with missing or empty expected outputs. "
                "Accuracy metrics will only be computed for examples with valid outputs.",
                source,
                missing_count,
                len(examples),
            )

    metadata: dict[str, Any] | None = None
    if metadata_hint:
        metadata = dict(metadata_hint)
    if registry_entry and registry_entry.metadata:
        metadata = {**(metadata or {}), **registry_entry.metadata}

    # Store source path for JS runtime and other consumers
    if metadata is None:
        metadata = {}
    metadata["source_path"] = str(resolved_path)

    # Compute dataset hash for cache invalidation (file size + mtime_ns for efficiency)
    # Uses nanosecond precision mtime to detect rapid changes within the same second
    try:
        stat_info = resolved_path.stat()
        metadata["dataset_hash"] = f"{stat_info.st_size}_{stat_info.st_mtime_ns}"
    except OSError:
        # If we can't stat the file, skip the hash
        pass

    metadata_out = metadata or None

    description = (
        registry_entry.description
        if registry_entry and registry_entry.description
        else description_hint or f"Dataset loaded from {source}"
    )
    name = registry_entry.name if registry_entry else name_hint or resolved_path.stem

    return cls(
        examples=examples,
        name=name,
        description=description,
        metadata=metadata_out,
    )


def _parse_jsonl_examples(resolved_path: Path, source: str) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []

    try:
        with resolved_path.open(encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f"Invalid JSON on line {line_num} in {source}: {exc}"
                    ) from exc

                if "input" not in data:
                    raise ValidationError(
                        f"Missing 'input' field on line {line_num} in {source}"
                    )

                example = EvaluationExample(
                    input_data=data["input"],
                    expected_output=data.get("output"),
                    metadata={
                        k: v for k, v in data.items() if k not in {"input", "output"}
                    },
                )
                examples.append(example)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Error loading dataset from {source}: {exc}") from exc

    return examples


def _parse_json_dataset(
    resolved_path: Path, source: str
) -> tuple[list[EvaluationExample], str | None, str | None, dict[str, Any] | None]:
    try:
        with resolved_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {source}: {exc}") from exc
    except Exception as exc:
        raise ValidationError(f"Error loading JSON from {source}: {exc}") from exc

    name_hint: str | None = None
    description_hint: str | None = None
    metadata_hint: dict[str, Any] | None = None

    examples: list[EvaluationExample] = []
    iterable: Iterable[tuple[int, Any]]

    if isinstance(data, list):
        iterable = enumerate(data)
    elif isinstance(data, dict):
        if "examples" in data:
            examples_field = data["examples"]
            if not isinstance(examples_field, list):
                raise ValidationError(f"'examples' field must be a list in {source}")
            iterable = enumerate(examples_field)
            raw_metadata = data.get("metadata")
            if raw_metadata is not None:
                if not isinstance(raw_metadata, dict):
                    raise ValidationError(
                        f"'metadata' field must be an object in {source}"
                    )
                metadata_hint = dict(raw_metadata)
            name_hint = data.get("name")
            if name_hint is not None and not isinstance(name_hint, str):
                raise ValidationError(f"'name' field must be a string in {source}")
            description_hint = data.get("description")
            if description_hint is not None and not isinstance(description_hint, str):
                raise ValidationError(
                    f"'description' field must be a string in {source}"
                )
        elif "input" in data:
            iterable = [(0, data)]
        else:
            raise ValidationError(f"Invalid JSON structure in {source}")
    else:
        raise ValidationError(f"JSON must be array or object in {source}")

    for index, item in iterable:
        if not isinstance(item, dict):
            raise ValidationError(f"Example {index} must be an object in {source}")
        if "input" not in item:
            raise ValidationError(
                f"Missing 'input' field in example {index} in {source}"
            )
        example = EvaluationExample(
            input_data=item["input"],
            expected_output=item.get("output"),
            metadata={k: v for k, v in item.items() if k not in {"input", "output"}},
        )
        examples.append(example)

    return examples, name_hint, description_hint, metadata_hint


@dataclass
class EvaluationExample:
    """Single example from evaluation dataset."""

    input_data: dict[str, Any]
    expected_output: Any | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata: dict[str, Any] = {}


@dataclass
class Dataset:
    """Dataset for evaluation."""

    examples: list[EvaluationExample]
    name: str = "dataset"  # Default name instead of empty string
    description: str = "Traigent evaluation dataset"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata: dict[str, Any] = {}

        # Validate examples
        if not isinstance(self.examples, list):
            raise TypeError("Examples must be a list")

        for i, example in enumerate(self.examples):
            if not isinstance(example, EvaluationExample):
                raise TypeError(f"Example {i} must be an EvaluationExample instance")

    @classmethod
    def from_jsonl(cls, file_path: str) -> Dataset:
        """Load dataset from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Dataset instance

        Raises:
            ValidationError: If file format is invalid
        """
        resolved_path, registry_entry = _resolve_dataset_source(file_path)

        # Additional security: Check file extension
        if resolved_path.suffix.lower() not in [".jsonl", ".json"]:
            raise ValidationError(
                f"Dataset must be JSONL or JSON file, got: {resolved_path.suffix}"
            )

        if resolved_path.suffix.lower() == ".json":
            examples, name_hint, description_hint, metadata_hint = _parse_json_dataset(
                resolved_path, file_path
            )
            return _build_dataset(
                cls,
                examples=examples,
                resolved_path=resolved_path,
                source=file_path,
                registry_entry=registry_entry,
                name_hint=name_hint,
                description_hint=description_hint,
                metadata_hint=metadata_hint,
            )

        examples = _parse_jsonl_examples(resolved_path, file_path)

        return _build_dataset(
            cls,
            examples=examples,
            resolved_path=resolved_path,
            source=file_path,
            registry_entry=registry_entry,
        )

    def __len__(self) -> int:
        """Get number of examples in dataset."""
        return len(self.examples)

    def __getitem__(self, index: int) -> EvaluationExample:
        """Get example by index."""
        return self.examples[index]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    @property
    def size(self) -> int:
        """Get number of examples in dataset."""
        return len(self.examples)

    def add_example(self, example: EvaluationExample) -> None:
        """Add an example to the dataset."""
        if not isinstance(example, EvaluationExample):
            raise TypeError("Example must be an EvaluationExample instance")
        self.examples.append(example)


@dataclass
class EvaluationResult:
    """Result of evaluating a function with a configuration."""

    config: dict[str, Any]
    example_results: list[Any] = field(default_factory=list)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    total_examples: int = 0
    successful_examples: int = 0
    duration: float = 0.0
    summary_stats: dict[str, Any] | None = None

    # Sample budget tracking
    sample_budget_exhausted: bool = False
    examples_consumed: int = 0

    # Legacy fields for backward compatibility
    metrics: dict[str, float] | None = None
    outputs: list[Any] | None = None
    errors: list[str | None] | None = None

    def __post_init__(self) -> None:
        # Backward compatibility mapping
        if self.metrics is None:
            self.metrics = self.aggregated_metrics
        if self.outputs is None:
            self.outputs = [
                result.actual_output if hasattr(result, "actual_output") else None
                for result in self.example_results
            ]
        if self.errors is None:
            self.errors = [
                result.error_message if hasattr(result, "error_message") else None
                for result in self.example_results
            ]

    @property
    def success_rate(self) -> float:
        """Get fraction of successful evaluations."""
        if self.total_examples == 0:
            return 1.0
        return self.successful_examples / self.total_examples

    @property
    def has_errors(self) -> bool:
        """Check if any evaluations failed."""
        return self.successful_examples < self.total_examples

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        from traigent.utils.persistence import _safe_json_value

        return {
            "config": _safe_json_value(self.config),
            "example_results": [
                r.to_dict() if hasattr(r, "to_dict") else _safe_json_value(r)
                for r in self.example_results
            ],
            "aggregated_metrics": _safe_json_value(self.aggregated_metrics),
            "total_examples": self.total_examples,
            "successful_examples": self.successful_examples,
            "duration": self.duration,
            "summary_stats": _safe_json_value(self.summary_stats),
            "sample_budget_exhausted": self.sample_budget_exhausted,
            "examples_consumed": self.examples_consumed,
            "metrics": _safe_json_value(self.metrics),
            "outputs": _safe_json_value(self.outputs),
            "errors": self.errors,
            "success_rate": self.success_rate,
            "has_errors": self.has_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Reconstruct EvaluationResult from a dictionary."""
        from traigent.api.types import ExampleResult

        # Rehydrate example_results if they are dicts
        example_results = []
        for r in data.get("example_results", []):
            if isinstance(r, dict) and "example_id" in r:
                example_results.append(ExampleResult.from_dict(r))
            else:
                example_results.append(r)

        return cls(
            config=data.get("config", {}),
            example_results=example_results,
            aggregated_metrics=data.get("aggregated_metrics", {}),
            total_examples=data.get("total_examples", 0),
            successful_examples=data.get("successful_examples", 0),
            duration=data.get("duration", 0.0),
            summary_stats=data.get("summary_stats"),
            sample_budget_exhausted=data.get("sample_budget_exhausted", False),
            examples_consumed=data.get("examples_consumed", 0),
            metrics=data.get("metrics"),
            outputs=data.get("outputs"),
            errors=data.get("errors"),
        )


class BaseEvaluator(ABC):
    """Base class for all evaluation strategies.

    This class defines the interface for evaluating functions with different
    configurations. It supports different evaluation metrics and error handling.

    New features:
    - Common function execution logic
    - Metric registry for custom metrics
    - Support for custom evaluation functions
    - Batch evaluation with concurrency
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        timeout: float | None = None,
        max_workers: int = 1,
        custom_eval_func: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize evaluator.

        Args:
            metrics: List of metric names to compute
            timeout: Timeout for individual evaluations (seconds)
            max_workers: Maximum number of concurrent evaluations
            custom_eval_func: Optional custom evaluation function
            **kwargs: Additional configuration
        """
        self.metrics = metrics or ["accuracy"]
        self.timeout = timeout
        self.max_workers = max_workers
        self.custom_eval_func = custom_eval_func
        self.config = kwargs

        # Initialize metric registry with defaults
        self._metric_registry: dict[str, Any] = {
            "accuracy": self._compute_accuracy,
            "success_rate": self._compute_success_rate,
            "error_rate": self._compute_error_rate,
            "avg_output_length": self._compute_avg_output_length,
            "cost": self._compute_cost,
            "latency": self._compute_latency,
        }

        self._ragas_metric_names: set[str] = set(POPULAR_RAGAS_METRICS)

    @abstractmethod
    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate function with given configuration on dataset.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset
            sample_lease: Optional sample budget lease controlling intra-trial consumption
            progress_callback: Optional callback invoked after each example with
                progress information. Receives ``(example_index, payload)``.

        Returns:
            EvaluationResult with metrics and outputs

        Raises:
            EvaluationError: If evaluation fails
        """
        pass

    def register_metric(self, name: str, func: Callable[..., Any]) -> None:
        """Register a custom metric function.

        Args:
            name: Name of the metric
            func: Function that computes the metric
                 Signature: func(outputs, expected, errors, **context) -> float
        """
        self._metric_registry[name] = func
        logger.debug(f"Registered custom metric: {name}")

    def _get_ragas_config(self) -> RagasConfig:
        column_map = None
        if self.config.get("ragas_column_map"):
            mapping = self.config["ragas_column_map"]
            if isinstance(mapping, Mapping):
                column_map = mapping
        return RagasConfig(
            column_map=column_map,
            llm=self.config.get("ragas_llm"),
            embeddings=self.config.get("ragas_embeddings"),
        )

    def override_metric(self, name: str, func: Callable[..., Any]) -> None:
        """Override an existing metric implementation.

        Args:
            name: Name of the metric to override
            func: New function that computes the metric
        """
        if name in self._metric_registry:
            logger.debug(f"Overriding metric: {name}")
        self._metric_registry[name] = func

    def compute_metrics(
        self,
        outputs: list[Any],
        expected_outputs: list[Any],
        errors: list[str | None],
        **context: Any,
    ) -> dict[str, float]:
        """Compute evaluation metrics using the metric registry.

        Args:
            outputs: Actual outputs from function
            expected_outputs: Expected outputs
            errors: Error messages (None for successful evaluations)
            **context: Additional context for metric computation.
                metrics_override: Optional list of metric names to compute
                    instead of self.metrics. Avoids mutating shared state
                    for thread-safety.

        Returns:
            Dictionary of metric name to value
        """
        metrics: dict[str, float] = {}
        metric_names = context.pop("metrics_override", None) or self.metrics

        ragas_metric_names = [
            name for name in metric_names if name in self._ragas_metric_names
        ]
        ragas_results: dict[str, float] = {}

        if ragas_metric_names and compute_ragas_metrics is not None:
            example_results = context.get("example_results") or []
            dataset_obj = context.get("dataset")
            dataset_examples: list[Any] | None = None
            if dataset_obj is not None and hasattr(dataset_obj, "examples"):
                dataset_examples = list(dataset_obj.examples)
            elif context.get("dataset_examples") is not None:
                dataset_examples = list(context["dataset_examples"])

            try:
                ragas_results = compute_ragas_metrics(
                    example_results=example_results,
                    metric_names=ragas_metric_names,
                    dataset_examples=dataset_examples,
                    outputs=outputs,
                    config=self._get_ragas_config(),
                )
            except RagasConfigurationError as exc:
                logger.warning("RAGAS metrics unavailable: %s", exc)
                ragas_results = dict.fromkeys(ragas_metric_names, 0.0)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to compute ragas metrics: %s", exc)
                ragas_results = dict.fromkeys(ragas_metric_names, 0.0)

        for metric_name in metric_names:
            if metric_name in ragas_results:
                metrics[metric_name] = ragas_results[metric_name]
                continue

            if metric_name in self._metric_registry:
                metric_func = self._metric_registry[metric_name]
                try:
                    # Call metric function with all available data
                    metrics[metric_name] = metric_func(
                        outputs, expected_outputs, errors, **context
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return metrics

    # Default metric implementations
    def _compute_accuracy(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **kwargs,
    ) -> float:
        """Default accuracy metric (exact match).

        Note: Empty strings and whitespace-only strings in expected outputs
        are treated as missing (equivalent to None) and excluded from accuracy
        computation. This prevents misleading metrics when datasets lack proper
        expected outputs.
        """
        if not expected:
            return 0.0

        correct = 0
        total = 0

        for output, exp, error in zip(outputs, expected, errors, strict=False):
            # Skip if error occurred or expected output is missing/empty
            if error is None and not _is_empty_expected_output(exp):
                if output == exp:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    async def _apply_mock_delay_if_enabled(self) -> None:
        """Apply mock delay if TRAIGENT_MOCK_LLM and TRAIGENT_MOCK_DELAY_MS are set.

        This simulates realistic LLM latency in mock LLM mode to make parallel execution
        visible in traces. Uses asyncio.sleep to not block the event loop.
        """
        if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() not in ("true", "1", "yes"):
            return

        delay_str = os.environ.get("TRAIGENT_MOCK_DELAY_MS", "")
        if not delay_str:
            return

        try:
            # Strip common suffixes like "ms" for user-friendliness
            delay_str_clean = delay_str.lower().rstrip("ms").strip()
            delay_ms = max(0, int(delay_str_clean))
        except ValueError:
            logger.warning(f"Invalid TRAIGENT_MOCK_DELAY_MS value '{delay_str}'")
            return

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

    @staticmethod
    def _should_expand_input_mapping(
        func: Callable[..., Any], payload: CollectionsMapping[str, Any]
    ) -> bool:
        """Decide whether to expand input mapping into keyword arguments."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True

        parameters = list(signature.parameters.values())
        if not parameters:
            return False

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return True

        filtered = [
            param
            for param in parameters
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

        if not filtered:
            return True

        if len(filtered) == 1:
            param = filtered[0]
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                return False
            return param.name in payload

        return True

    def _compute_success_rate(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **kwargs,
    ) -> float:
        """Default success rate metric."""
        if not errors:
            return 1.0

        success_count = sum(1 for error in errors if error is None)
        return success_count / len(errors)

    def _compute_error_rate(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **kwargs,
    ) -> float:
        """Default error rate metric."""
        if not errors:
            return 0.0

        error_count = sum(1 for error in errors if error is not None)
        return error_count / len(errors)

    def _compute_avg_output_length(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **kwargs,
    ) -> float:
        """Default average output length metric."""
        valid_outputs = [
            out
            for out, err in zip(outputs, errors, strict=False)
            if err is None and out is not None
        ]

        if not valid_outputs:
            return 0.0

        lengths = []
        for output in valid_outputs:
            if isinstance(output, str):
                lengths.append(len(output))
            elif hasattr(output, "__len__"):
                lengths.append(len(output))

        return sum(lengths) / len(lengths) if lengths else 0.0

    def _compute_cost(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **context,
    ) -> float:
        """Default cost metric - extracts total cost from evaluation context."""
        # Extract cost information from context (set by metrics tracker)
        if "example_metrics" in context:
            example_metrics = context["example_metrics"]
            if example_metrics:
                # Calculate average cost from successful examples
                total_cost = 0.0
                count = 0
                for _i, (error, metrics) in enumerate(
                    zip(errors, example_metrics, strict=False)
                ):
                    if error is None and metrics and hasattr(metrics, "cost"):
                        total_cost += metrics.cost.total_cost
                        count += 1
                return total_cost / count if count > 0 else 0.0

        # Fallback: look for cost in context directly
        if "cost" in context:
            return float(context["cost"])
        if "total_cost" in context:
            return float(context["total_cost"])

        return 0.0

    def _compute_latency(
        self,
        outputs: list[Any],
        expected: list[Any],
        errors: list[str | None],
        **context,
    ) -> float:
        """Compute average latency (response time) in seconds.

        Returns the average execution time across all successful examples.
        """
        if "example_results" in context:
            example_results = context["example_results"]
            if example_results:
                response_times = [
                    r.execution_time
                    for r in example_results
                    if hasattr(r, "execution_time") and r.execution_time > 0
                ]
                if response_times:
                    return float(sum(response_times) / len(response_times))

        # Fallback: look for latency in context directly
        if "latency" in context:
            return float(context["latency"])
        if "avg_response_time" in context:
            return float(context["avg_response_time"])

        return 0.0

    # Common evaluation methods to reduce duplication in subclasses

    def _prepare_call_arguments(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Determine call arguments based on injection mode.

        Args:
            func: Function to call
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            Tuple of (positional_args, keyword_args)
        """
        injection_mode = getattr(func, "_traigent_injection_mode", "context")

        if injection_mode == "parameter" and isinstance(input_data, CollectionsMapping):
            return (), {**input_data, **config}
        if injection_mode == "parameter":
            return (input_data,), dict(config)
        if isinstance(
            input_data, CollectionsMapping
        ) and self._should_expand_input_mapping(func, input_data):
            return (), dict(input_data)
        return (input_data,), {}

    async def _execute_async_with_timeout(
        self,
        func: Callable[..., Any],
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> Any:
        """Execute async function with optional timeout.

        Args:
            func: Async function to execute
            call_args: Positional arguments
            call_kwargs: Keyword arguments

        Returns:
            Function output
        """
        if self.timeout:
            return await asyncio.wait_for(
                func(*call_args, **call_kwargs), timeout=self.timeout
            )
        return await func(*call_args, **call_kwargs)

    async def _execute_sync_in_thread(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
        executor: Any | None,
        trial_ctx: Any | None,
    ) -> tuple[Any, str | None]:
        """Execute sync function in thread pool with context propagation.

        Args:
            func: Sync function to execute
            config: Configuration for context
            call_args: Positional arguments
            call_kwargs: Keyword arguments
            executor: Optional thread pool executor
            trial_ctx: Trial context to propagate to thread

        Returns:
            Tuple of (output, error_message)
        """
        from traigent.config.context import ConfigurationContext, set_trial_context
        from traigent.config.context import trial_context as trial_context_var

        def call_with_config() -> Any:
            # Re-establish context in thread (contextvars don't propagate)
            trial_token = None
            if trial_ctx is not None:
                trial_token = set_trial_context(trial_ctx)
            try:
                with ConfigurationContext(config):
                    return func(*call_args, **call_kwargs)
            finally:
                if trial_token is not None:
                    trial_context_var.reset(trial_token)

        loop = asyncio.get_event_loop()
        temporary_executor = None
        submit_executor = executor

        if submit_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            temporary_executor = ThreadPoolExecutor(max_workers=1)
            submit_executor = temporary_executor

        try:
            future = loop.run_in_executor(submit_executor, call_with_config)
            if self.timeout:
                done, _ = await asyncio.wait(
                    {future},
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    future.cancel()
                    error_msg = f"Function call timed out after {self.timeout}s"
                    logger.warning(error_msg)
                    return None, error_msg
                return next(iter(done)).result(), None
            return await future, None
        finally:
            if temporary_executor is not None:
                temporary_executor.shutdown(wait=False, cancel_futures=True)

    def _check_api_key_error(self, error: Exception) -> None:
        """Check if exception is an API key error and raise APIKeyError if so.

        Args:
            error: The exception to check

        Raises:
            APIKeyError: If the error appears to be API key related
        """
        api_key_tokens = ("api key", "api_key", "authentication", "openai_api_key")
        if any(token in str(error).lower() for token in api_key_tokens):
            raise APIKeyError(
                f"API key error detected. Set the required API key environment "
                f"variable or use TRAIGENT_MOCK_LLM=true for testing. "
                f"Original error: {error}"
            ) from error

    async def _execute_function(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
        executor: Any | None = None,
    ) -> tuple[Any, str | None]:
        """Execute function with configuration context and timeout.

        This method encapsulates the common pattern of:
        1. Setting up configuration context
        2. Handling async vs sync functions
        3. Applying timeouts
        4. Error handling

        Args:
            func: Function to execute
            config: Configuration parameters
            input_data: Input data for the function
            executor: Optional thread pool executor for sync functions

        Returns:
            Tuple of (output, error_message)
        """
        try:
            from traigent.config.context import ConfigurationContext, get_trial_context

            trial_ctx = get_trial_context()
            call_args, call_kwargs = self._prepare_call_arguments(
                func, config, input_data
            )

            with ConfigurationContext(config):
                if asyncio.iscoroutinefunction(func):
                    output = await self._execute_async_with_timeout(
                        func, call_args, call_kwargs
                    )
                else:
                    output, error = await self._execute_sync_in_thread(
                        func, config, call_args, call_kwargs, executor, trial_ctx
                    )
                    if error:
                        return None, error

            await self._apply_mock_delay_if_enabled()
            return output, None

        except ConfigurationError:
            raise
        except TimeoutError:
            error_msg = f"Function call timed out after {self.timeout}s"
            logger.warning(error_msg)
            return None, error_msg
        except (FriendlyTraigentError, CoreTraigentError):
            raise
        except Exception as e:
            self._check_api_key_error(e)
            error_msg = f"Function call failed: {e}"
            logger.warning(error_msg)
            return None, error_msg

    async def _evaluate_single_non_detailed(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        example: EvaluationExample,
        index: int,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
    ) -> tuple[Any, str | None]:
        """Evaluate a single example in non-detailed mode with tracing."""
        example_id = (
            example.metadata.get("example_id", f"example_{index}")
            if example.metadata
            else f"example_{index}"
        )
        start_time = time.time()

        with self._example_trace_context(example_id, index, example) as span:
            output, error = await self._execute_function(
                func, config, example.input_data, executor=None
            )
            execution_time = time.time() - start_time

            if span is not None:
                _, record_fn, available = _get_tracing_functions()
                if available and record_fn is not None:
                    record_fn(
                        span,
                        success=(error is None),
                        actual_output=output,
                        error=error,
                        execution_time=execution_time,
                    )

        if progress_callback:
            progress_callback(
                index, {"success": error is None, "output": output, "error": error}
            )

        return output, error

    async def _evaluate_batch_sequential(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        sample_lease: SampleBudgetLease | None,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
    ) -> tuple[list[Any], list[str | None], list[ExampleResult | None], int, bool]:
        """Evaluate dataset sequentially (max_workers=1)."""
        outputs: list[Any] = []
        errors: list[str | None] = []
        example_results: list[ExampleResult | None] = []
        consumed = 0
        exhausted = False

        try:
            for i, example in enumerate(dataset.examples):
                if sample_lease and not sample_lease.try_take(1):
                    exhausted = True
                    break

                if detailed:
                    result = await self._evaluate_single_detailed(
                        func,
                        config,
                        example,
                        i,
                        executor=None,
                        progress_callback=progress_callback,
                    )
                    example_results.append(result)
                    outputs.append(result.actual_output)
                    errors.append(result.error_message)
                else:
                    output, error = await self._evaluate_single_non_detailed(
                        func, config, example, i, progress_callback
                    )
                    outputs.append(output)
                    errors.append(error)

                consumed += 1
        except TrialPrunedError as e:
            # Attach partial results that were collected before pruning
            if detailed and example_results:
                e.example_results = [r for r in example_results if r is not None]
            raise

        return outputs, errors, example_results, consumed, exhausted

    def _create_failed_example_result(
        self,
        example: EvaluationExample,
        index: int,
        error: Exception,
    ) -> ExampleResult:
        """Create ExampleResult for a failed concurrent evaluation."""
        return ExampleResult(
            example_id=f"example_{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics={},
            execution_time=0.0,
            success=False,
            error_message=str(error),
            metadata=example.metadata.copy() if example.metadata else {},
        )

    def _process_concurrent_exception(
        self,
        result: Exception,
        index: int,
        dataset: Dataset,
        sample_lease: SampleBudgetLease | None,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        outputs: list[Any],
        errors: list[str | None],
        example_results: list[ExampleResult | None],
    ) -> bool:
        """Process exception from concurrent evaluation. Returns True to skip."""
        if isinstance(result, asyncio.CancelledError):
            if sample_lease:
                sample_lease.rollback(1)
            if progress_callback:
                progress_callback(
                    index, {"success": False, "error": "sample_budget_cancelled"}
                )
            return True  # Skip this result

        if isinstance(result, (APIKeyError, FriendlyTraigentError, CoreTraigentError)):
            raise result

        if detailed:
            example = dataset.examples[index]
            failed_result = self._create_failed_example_result(example, index, result)
            example_results.append(failed_result)
        outputs.append(None)
        errors.append(str(result))
        if progress_callback:
            progress_callback(index, {"success": False, "error": str(result)})
        return False

    def _process_concurrent_success(
        self,
        result: Any,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        index: int,
        outputs: list[Any],
        errors: list[str | None],
        example_results: list[ExampleResult | None],
    ) -> None:
        """Process successful result from concurrent evaluation."""
        if detailed:
            example_results.append(result)
            outputs.append(result.actual_output)
            errors.append(result.error_message)
        else:
            output, error = result
            outputs.append(output)
            errors.append(error)
            if progress_callback:
                progress_callback(
                    index, {"success": error is None, "output": output, "error": error}
                )

    async def _handle_task_result(
        self,
        task: asyncio.Task[Any],
        index: int,
        examples: list[EvaluationExample],
        detailed: bool,
        sample_lease: SampleBudgetLease | None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        outputs_by_index: dict[int, Any],
        errors_by_index: dict[int, str | None],
        example_results_by_index: dict[int, ExampleResult | None],
        pending_tasks: dict[asyncio.Task[Any], int],
    ) -> None:
        """Handle a completed task result, storing success or error appropriately."""
        try:
            result = task.result()
        except TrialPrunedError as e:
            await self._cancel_pending_tasks(pending_tasks, sample_lease)
            # Attach partial example results to the exception before re-raising
            if detailed and example_results_by_index:
                e.example_results = self._collect_partial_results(
                    example_results_by_index
                )
            raise
        except asyncio.CancelledError:
            # S7497: CancelledError must be re-raised after cleanup
            if sample_lease:
                sample_lease.rollback(1)
            if progress_callback:
                await self._safe_progress_callback(
                    progress_callback,
                    index,
                    {"success": False, "error": "sample_budget_cancelled"},
                    pending_tasks,
                    sample_lease,
                )
            raise  # Re-raise CancelledError to properly propagate cancellation
        except (FriendlyTraigentError, CoreTraigentError):
            await self._cancel_pending_tasks(pending_tasks, sample_lease)
            raise
        except Exception as exc:
            self._store_error_result(
                index,
                exc,
                examples,
                detailed,
                outputs_by_index,
                errors_by_index,
                example_results_by_index,
            )
            if progress_callback:
                await self._safe_progress_callback(
                    progress_callback,
                    index,
                    {"success": False, "error": str(exc)},
                    pending_tasks,
                    sample_lease,
                )
            return  # Error stored, continue to next task

        # Success case
        self._store_success_result(
            index,
            result,
            detailed,
            progress_callback,
            outputs_by_index,
            errors_by_index,
            example_results_by_index,
            pending_tasks,
            sample_lease,
        )

    def _store_error_result(
        self,
        index: int,
        exc: Exception,
        examples: list[EvaluationExample],
        detailed: bool,
        outputs_by_index: dict[int, Any],
        errors_by_index: dict[int, str | None],
        example_results_by_index: dict[int, ExampleResult | None],
    ) -> None:
        """Store an error result in the result dictionaries."""
        if detailed:
            example = examples[index]
            example_results_by_index[index] = self._create_failed_example_result(
                example, index, exc
            )
        outputs_by_index[index] = None
        errors_by_index[index] = str(exc)

    def _store_success_result(
        self,
        index: int,
        result: Any,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        outputs_by_index: dict[int, Any],
        errors_by_index: dict[int, str | None],
        example_results_by_index: dict[int, ExampleResult | None],
        pending_tasks: dict[asyncio.Task[Any], int],
        sample_lease: SampleBudgetLease | None,
    ) -> None:
        """Store a successful result in the result dictionaries."""
        if detailed:
            example_results_by_index[index] = result
            outputs_by_index[index] = result.actual_output
            errors_by_index[index] = result.error_message
        else:
            output, error = result
            outputs_by_index[index] = output
            errors_by_index[index] = error
            if progress_callback:
                # Note: This is sync in the original, we handle exceptions elsewhere
                progress_callback(
                    index,
                    {"success": error is None, "output": output, "error": error},
                )

    async def _safe_progress_callback(
        self,
        progress_callback: Callable[[int, dict[str, Any]], Any],
        index: int,
        data: dict[str, Any],
        pending_tasks: dict[asyncio.Task[Any], int],
        sample_lease: SampleBudgetLease | None,
    ) -> None:
        """Call progress callback with exception handling."""
        try:
            progress_callback(index, data)
        except TrialPrunedError:
            await self._cancel_pending_tasks(pending_tasks, sample_lease)
            raise
        except Exception:
            await self._cancel_pending_tasks(pending_tasks, sample_lease)
            raise

    async def _cancel_pending_tasks(
        self,
        pending_tasks: dict[asyncio.Task[Any], int],
        sample_lease: SampleBudgetLease | None,
    ) -> None:
        """Cancel all pending tasks and rollback sample budget."""
        if not pending_tasks:
            return

        tasks_list = list(pending_tasks.keys())
        indices_list = list(pending_tasks.values())

        for task in tasks_list:
            task.cancel()

        results = await asyncio.gather(*tasks_list, return_exceptions=True)
        if sample_lease:
            for _index, result in zip(indices_list, results, strict=False):
                if isinstance(result, asyncio.CancelledError):
                    sample_lease.rollback(1)

    def _collect_partial_results(
        self,
        example_results_by_index: dict[int, ExampleResult | None],
    ) -> list[ExampleResult]:
        """Collect partial example results for pruned trials.

        Returns a list of non-None ExampleResult objects sorted by index.
        This is used to capture partial results when a trial is pruned early.
        """
        ordered_indices = sorted(example_results_by_index.keys())
        return [
            result
            for i in ordered_indices
            if (result := example_results_by_index[i]) is not None
        ]

    def _collect_ordered_results(
        self,
        outputs_by_index: dict[int, Any],
        errors_by_index: dict[int, str | None],
        example_results_by_index: dict[int, ExampleResult | None],
        detailed: bool,
    ) -> tuple[list[Any], list[str | None], list[ExampleResult | None]]:
        """Collect results in order by index."""
        ordered_indices = sorted(outputs_by_index.keys())
        outputs = [outputs_by_index[i] for i in ordered_indices]
        errors = [errors_by_index[i] for i in ordered_indices]
        if detailed:
            example_results = [example_results_by_index.get(i) for i in ordered_indices]
        else:
            example_results = []
        return outputs, errors, example_results

    async def _evaluate_batch(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        detailed: bool = False,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> tuple[list[Any], list[str | None], list[ExampleResult | None], int, bool]:
        """Evaluate function on entire dataset with optional detailed tracking.

        This method implements the common pattern of:
        1. Sequential vs concurrent evaluation based on max_workers
        2. Optional detailed result tracking
        3. Error handling and result collection

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset
            detailed: Whether to create detailed ExampleResult objects

        Returns:
            Tuple of (outputs, errors, example_results)
            If detailed=False, example_results will be empty list
        """
        if sample_lease and sample_lease.remaining() <= 0:
            return [], [], [], 0, True

        if self.max_workers == 1:
            return await self._evaluate_batch_sequential(
                func, config, dataset, sample_lease, detailed, progress_callback
            )

        return await self._evaluate_batch_concurrent(
            func, config, dataset, sample_lease, detailed, progress_callback
        )

    async def _evaluate_batch_concurrent(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        sample_lease: SampleBudgetLease | None,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
    ) -> tuple[list[Any], list[str | None], list[ExampleResult | None], int, bool]:
        """Execute concurrent batch evaluation."""
        outputs_by_index: dict[int, Any] = {}
        errors_by_index: dict[int, str | None] = {}
        example_results_by_index: dict[int, ExampleResult | None] = {}
        examples = list(dataset.examples)
        semaphore = asyncio.Semaphore(self.max_workers)

        try:
            consumed, exhausted = await self._run_concurrent_tasks(
                func,
                config,
                examples,
                sample_lease,
                detailed,
                progress_callback,
                outputs_by_index,
                errors_by_index,
                example_results_by_index,
                semaphore,
            )
        except TrialPrunedError as e:
            # Attach partial results that were collected before pruning
            if detailed and example_results_by_index:
                e.example_results = self._collect_partial_results(
                    example_results_by_index
                )
            raise

        outputs, errors, example_results = self._collect_ordered_results(
            outputs_by_index, errors_by_index, example_results_by_index, detailed
        )

        if exhausted and progress_callback:
            progress_callback(
                consumed, {"success": None, "stop_reason": "sample_budget_exhausted"}
            )

        return outputs, errors, example_results, consumed, exhausted

    async def _run_concurrent_tasks(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        examples: list[EvaluationExample],
        sample_lease: SampleBudgetLease | None,
        detailed: bool,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        outputs_by_index: dict[int, Any],
        errors_by_index: dict[int, str | None],
        example_results_by_index: dict[int, ExampleResult | None],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, bool]:
        """Run concurrent task scheduling and processing loop."""
        pending_tasks: dict[asyncio.Task[Any], int] = {}
        next_index = 0
        consumed = 0
        exhausted = False

        async def evaluate_with_semaphore(example: EvaluationExample, idx: int) -> Any:
            async with semaphore:
                if detailed:
                    return await self._evaluate_single_detailed(
                        func,
                        config,
                        example,
                        idx,
                        executor=None,
                        progress_callback=progress_callback,
                    )
                return await self._execute_function(
                    func, config, example.input_data, executor=None
                )

        def schedule_more() -> bool:
            """Schedule more tasks. Returns True if budget exhausted."""
            nonlocal next_index
            while next_index < len(examples) and len(pending_tasks) < self.max_workers:
                if sample_lease and not sample_lease.try_take(1):
                    return True
                example = examples[next_index]
                task = asyncio.create_task(evaluate_with_semaphore(example, next_index))
                pending_tasks[task] = next_index
                next_index += 1
            return False

        exhausted = schedule_more()

        while pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                index = pending_tasks.pop(task)
                await self._handle_task_result(
                    task,
                    index,
                    examples,
                    detailed,
                    sample_lease,
                    progress_callback,
                    outputs_by_index,
                    errors_by_index,
                    example_results_by_index,
                    pending_tasks,
                )
                consumed += 1

            if not exhausted:
                exhausted = schedule_more()

        return consumed, exhausted

    @staticmethod
    def _get_capture_key_context() -> Callable[[str], Any]:
        """Get capture_key context manager, with fallback noop if unavailable."""
        try:
            from traigent.utils.langchain_interceptor import capture_key

            return capture_key  # type: ignore[no-any-return]
        except Exception:
            from contextlib import contextmanager

            def noop_capture_key(_: str) -> Any:
                @contextmanager
                def _noop() -> Any:
                    yield

                return _noop()

            return noop_capture_key

    def _extract_original_prompt(self, input_data: Any) -> list[dict[str, Any]]:
        """Extract original prompt in messages format from input_data."""
        if isinstance(input_data, dict):
            if "messages" in input_data:
                return cast(list[dict[str, Any]], input_data["messages"])
            elif "text" in input_data:
                return [{"role": "user", "content": input_data["text"]}]
            else:
                return [{"role": "user", "content": str(input_data)}]
        return [{"role": "user", "content": str(input_data)}]

    def _extract_response_metrics(
        self,
        example_id: str,
        example: EvaluationExample,
        result: ExampleResult,
        config: dict[str, Any],
    ) -> tuple[dict[str, float], Any]:
        """Extract response metrics and return (metrics_payload_update, response_metrics)."""
        model_name = config.get("model") if isinstance(config, dict) else None
        response_metrics = None

        try:
            captured = get_captured_response_by_key(example_id)
            if captured is None:
                return {}, None

            original_prompt: Any | None = self._extract_original_prompt(
                example.input_data
            )
            response_text: str | None = (
                result.actual_output if isinstance(result.actual_output, str) else None
            )
            prompt_length: int | None = None
            response_length: int | None = None

            if getattr(self, "privacy_enabled", False):
                original_prompt = None
                response_text = None
                prompt_repr = (
                    str(example.input_data)
                    if not isinstance(example.input_data, dict)
                    else str(example.input_data.get("text") or example.input_data)
                )
                prompt_length = len(prompt_repr)
                response_length = (
                    len(str(result.actual_output))
                    if result.actual_output is not None
                    else 0
                )

            response_metrics = extract_llm_metrics(
                captured,
                model_name=model_name,
                original_prompt=original_prompt,
                response_text=response_text,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to extract response metrics for example %s: %s", example_id, exc
            )
            return {}, None

        if response_metrics is None:
            return {}, None

        result.metrics.update(
            {
                "input_tokens": response_metrics.tokens.input_tokens,
                "output_tokens": response_metrics.tokens.output_tokens,
                "total_tokens": response_metrics.tokens.total_tokens,
                "input_cost": response_metrics.cost.input_cost,
                "output_cost": response_metrics.cost.output_cost,
                "total_cost": response_metrics.cost.total_cost,
            }
        )
        return {"total_cost": response_metrics.cost.total_cost}, response_metrics

    async def _run_custom_evaluator(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        example: EvaluationExample,
        example_id: str,
    ) -> ExampleResult:
        """Run custom evaluator function with proper context."""
        from traigent.config.context import ConfigurationContext, get_trial_context

        if self.custom_eval_func is None:
            raise RuntimeError("custom_eval_func must be set before evaluation")

        trial_ctx = get_trial_context()
        capture_key = self._get_capture_key_context()
        with _maybe_restore_trial_context(trial_ctx):
            with ConfigurationContext(config):
                with capture_key(example_id):
                    if asyncio.iscoroutinefunction(self.custom_eval_func):
                        result = await self.custom_eval_func(func, config, example)
                    else:
                        result = self.custom_eval_func(func, config, example)

        if not isinstance(result, ExampleResult):
            raise ValueError(
                f"Custom evaluator must return ExampleResult, got {type(result)}"
            )
        return result

    def _build_progress_payload(
        self,
        example_id: str,
        example: EvaluationExample,
        result: ExampleResult,
        config: dict[str, Any],
        error: str | None,
    ) -> dict[str, Any]:
        """Build progress callback payload with metrics."""
        metrics_payload: dict[str, float] = {}

        if error is None and example.expected_output is not None:
            try:
                metrics_payload["accuracy"] = (
                    1.0 if result.actual_output == example.expected_output else 0.0
                )
            except Exception as exc:
                logger.warning(
                    "Failed to compute accuracy metric for example %s: %s",
                    example_id,
                    exc,
                )

        metrics_update, _ = self._extract_response_metrics(
            example_id, example, result, config
        )
        metrics_payload.update(metrics_update)

        return {
            "success": result.success,
            "error": result.error_message,
            "metrics": metrics_payload,
            "output": result.actual_output,
        }

    def _record_example_trace(
        self,
        span: Any,
        result: ExampleResult,
    ) -> None:
        """Record example result to tracing span if tracing is available.

        Args:
            span: The tracing span (may be None)
            result: The example result to record
        """
        if span is None:
            return

        _, record_fn, available = _get_tracing_functions()
        if not available or record_fn is None:
            return

        record_fn(
            span,
            success=result.success,
            actual_output=result.actual_output,
            metrics=result.metrics,
            error=result.error_message,
            execution_time=result.execution_time,
        )

    @contextmanager
    def _example_trace_context(
        self,
        example_id: str,
        example_index: int,
        example: EvaluationExample,
    ) -> Any:
        """Create example tracing context if tracing is available.

        Args:
            example_id: Unique example identifier
            example_index: Index of example in dataset
            example: The evaluation example

        Yields:
            Tracing span or None if tracing not available
        """
        span_fn, _, available = _get_tracing_functions()
        if not available or span_fn is None:
            yield None
            return

        # Get input data for tracing (respect privacy settings)
        input_data = (
            None if getattr(self, "privacy_enabled", False) else (example.input_data)
        )

        with span_fn(
            example_id=example_id,
            example_index=example_index,
            input_data=input_data,
            expected_output=example.expected_output,
        ) as span:
            yield span

    def _create_failure_result(
        self,
        example_id: str,
        example: EvaluationExample,
        execution_time: float,
        error: Exception,
    ) -> ExampleResult:
        """Create an ExampleResult for a failed evaluation."""
        return ExampleResult(
            example_id=example_id,
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics=dict.fromkeys(self.metrics, 0.0),
            execution_time=execution_time,
            success=False,
            error_message=str(error),
            metadata=example.metadata.copy() if example.metadata else {},
        )

    async def _try_custom_evaluator(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        example: EvaluationExample,
        example_id: str,
        start_time: float,
        span: Any,
        example_index: int,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
    ) -> ExampleResult:
        """Try running the custom evaluator, handling exceptions appropriately."""
        # Exceptions that should be re-raised without handling
        passthrough_exceptions = (
            TrialPrunedError,
            asyncio.CancelledError,
            APIKeyError,
            FriendlyTraigentError,
            CoreTraigentError,
            ConfigurationError,
        )
        try:
            result = await self._run_custom_evaluator(func, config, example, example_id)
            self._record_example_trace(span, result)
            if progress_callback:
                progress_callback(
                    example_index,
                    {
                        "success": result.success,
                        "error": result.error_message,
                        "metrics": result.metrics,
                        "output": result.actual_output,
                    },
                )
            return result
        except passthrough_exceptions:
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"Custom evaluation failed for example {example_id}: {e}")
            result = self._create_failure_result(example_id, example, execution_time, e)
            self._record_example_trace(span, result)
            return result

    async def _evaluate_single_detailed(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        example: EvaluationExample,
        example_index: int,
        executor: Any | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> ExampleResult:
        """Evaluate function on single example with detailed tracking.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            example: EvaluationExample to evaluate
            example_index: Index of example in dataset

        Returns:
            ExampleResult with detailed information
        """
        example_id = (
            example.metadata.get("example_id", f"example_{example_index}")
            if example.metadata
            else f"example_{example_index}"
        )
        start_time = time.time()

        if getattr(self, "privacy_enabled", False):
            logger.debug(f"Evaluating example {example_id}: [redacted]")
        else:
            logger.debug(f"Evaluating example {example_id}: {example.input_data}")

        # Wrap example evaluation in tracing span
        with self._example_trace_context(example_id, example_index, example) as span:
            # Use custom evaluator if available
            if self.custom_eval_func:
                return await self._try_custom_evaluator(
                    func,
                    config,
                    example,
                    example_id,
                    start_time,
                    span,
                    example_index,
                    progress_callback,
                )

            # Default evaluation with correlation key
            capture_key = self._get_capture_key_context()
            with capture_key(example_id):
                output, error = await self._execute_function(
                    func, config, example.input_data, executor
                )
            execution_time = time.time() - start_time

            result = ExampleResult(
                example_id=example_id,
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=output,
                metrics={},
                execution_time=execution_time,
                success=(error is None),
                error_message=error,
                metadata=example.metadata.copy() if example.metadata else {},
            )

            # Record to tracing span
            self._record_example_trace(span, result)

            if progress_callback:
                progress_payload = self._build_progress_payload(
                    example_id, example, result, config, error
                )
                progress_callback(example_index, progress_payload)

            if error is None:
                logger.debug(
                    f"Example {example_id} completed successfully in "
                    f"{execution_time:.2f}s"
                )
            return result

    def validate_function(self, func: Callable[..., Any]) -> None:
        """Validate that function is callable and has expected signature.

        Args:
            func: Function to validate

        Raises:
            EvaluationError: If function is invalid
        """
        if not callable(func):
            raise EvaluationError("Function must be callable") from None

        # Additional validation could be added here
        # (e.g., check function signature, annotations)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate

        Raises:
            EvaluationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise EvaluationError("Configuration must be a dictionary")

        # Additional validation could be added here
        # (e.g., check required parameters, value ranges)


def load_dataset_from_file(file_path: str) -> Dataset:
    """Load dataset from file (supports JSONL and JSON formats).

    Args:
        file_path: Path to dataset file

    Returns:
        Dataset instance

    Raises:
        ValidationError: If file format is invalid or file not found
    """
    resolved_path, registry_entry = _resolve_dataset_source(file_path)
    suffix = resolved_path.suffix.lower()

    if suffix == ".jsonl":
        examples = _parse_jsonl_examples(resolved_path, file_path)
        return _build_dataset(
            Dataset,
            examples=examples,
            resolved_path=resolved_path,
            source=file_path,
            registry_entry=registry_entry,
        )

    if suffix == ".json":
        examples, name_hint, description_hint, metadata_hint = _parse_json_dataset(
            resolved_path, file_path
        )
        return _build_dataset(
            Dataset,
            examples=examples,
            resolved_path=resolved_path,
            source=file_path,
            registry_entry=registry_entry,
            name_hint=name_hint,
            description_hint=description_hint,
            metadata_hint=metadata_hint,
        )

    raise ValidationError(
        f"Unsupported file format for {file_path}. Supported formats: .jsonl, .json"
    )


def _load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dataset instance

    Raises:
        ValidationError: If file format is invalid
    """
    resolved_path, registry_entry = _resolve_dataset_source(file_path)
    if resolved_path.suffix.lower() != ".json":
        raise ValidationError(f"Dataset must be JSON file, got: {resolved_path.suffix}")

    examples, name_hint, description_hint, metadata_hint = _parse_json_dataset(
        resolved_path, file_path
    )

    return _build_dataset(
        Dataset,
        examples=examples,
        resolved_path=resolved_path,
        source=file_path,
        registry_entry=registry_entry,
        name_hint=name_hint,
        description_hint=description_hint,
        metadata_hint=metadata_hint,
    )


class SimpleScoringEvaluator(BaseEvaluator):
    """Evaluator that wraps simple scoring functions.

    This evaluator allows users to provide simple scoring functions instead of
    full evaluator implementations. The scoring function should accept:
    - output: The actual output from the function
    - expected: The expected output (optional)
    - llm_metrics: Automatically captured LLM metrics (optional)

    And return either:
    - A single score (float)
    - A dictionary of metric names to scores
    """

    def __init__(
        self,
        scoring_function: Callable[..., Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        capture_llm_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize simple scoring evaluator.

        Args:
            scoring_function: Function that takes (output, expected, llm_metrics) and returns score(s)
            metric_functions: Dictionary of metric name to scoring function
            metrics: List of metric names to compute
            timeout: Timeout for individual evaluations (seconds)
            capture_llm_metrics: Whether to automatically capture LLM metrics
            **kwargs: Additional configuration
        """
        # Determine metrics from provided functions
        if metric_functions:
            metrics = list(metric_functions.keys())
        elif metrics is None and scoring_function:
            metrics = ["score"]  # Default metric name for single scoring function

        super().__init__(metrics, timeout, **kwargs)
        self.scoring_function = scoring_function
        self.metric_functions = metric_functions or {}
        self.capture_llm_metrics = capture_llm_metrics

        # Import metrics tracking modules
        try:
            from traigent.evaluators.metrics_tracker import (
                MetricsTracker,
                extract_llm_metrics,
            )
            from traigent.utils.langchain_interceptor import (
                clear_captured_responses,
                get_all_captured_responses,
                patch_langchain_for_metadata_capture,
            )

            self._metrics_tracker = MetricsTracker()
            self._extract_llm_metrics = extract_llm_metrics
            self._clear_captured_responses = clear_captured_responses
            self._get_all_captured_responses = get_all_captured_responses
            # Ensure LangChain patch is applied
            patch_langchain_for_metadata_capture()
            self._metrics_available = True
        except ImportError:
            logger.warning(
                "Metrics tracking modules not available, LLM metrics capture disabled"
            )
            self._metrics_available = False
            self.capture_llm_metrics = False

    def _extract_response_text_from_output(self, output: Any) -> str | None:
        """Extract text content from various output types.

        Args:
            output: The output from function execution

        Returns:
            Extracted text string or None
        """
        if isinstance(output, str):
            return output
        if hasattr(output, "text"):
            return cast(str | None, output.text)
        if hasattr(output, "content"):
            if isinstance(output.content, str):
                return output.content
            if isinstance(output.content, list) and output.content:
                if hasattr(output.content[0], "text"):
                    return cast(str | None, output.content[0].text)
        return None

    def _build_llm_metrics_dict(
        self, metrics_obj: Any, model_name: str | None
    ) -> dict[str, Any]:
        """Build LLM metrics dictionary from metrics object.

        Args:
            metrics_obj: The extracted LLM metrics object
            model_name: Model name from config

        Returns:
            Dictionary of LLM metrics
        """
        return {
            "total_tokens": getattr(metrics_obj.tokens, "total_tokens", 0),
            "prompt_tokens": getattr(metrics_obj.tokens, "prompt_tokens", 0),
            "completion_tokens": getattr(metrics_obj.tokens, "completion_tokens", 0),
            "total_cost": getattr(metrics_obj.cost, "total_cost", 0.0),
            "input_cost": getattr(metrics_obj.cost, "input_cost", 0.0),
            "output_cost": getattr(metrics_obj.cost, "output_cost", 0.0),
            "response_time_ms": getattr(metrics_obj.response, "response_time_ms", 0),
            "tokens_per_second": getattr(metrics_obj.response, "tokens_per_second", 0),
            "model": model_name or "unknown",
            "_full_metrics": metrics_obj,
        }

    def _build_metric_kwargs(
        self,
        params: list[str],
        output: Any,
        example: Any,
        config: dict[str, Any],
        dataset: Dataset,
        example_index: int,
        llm_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build kwargs dictionary for metric function calls.

        Args:
            params: Parameter names from function signature
            output: Function output
            example: Current example being evaluated
            config: Configuration dictionary
            dataset: Full dataset
            example_index: Index of current example
            llm_metrics: Captured LLM metrics (if any)

        Returns:
            Dictionary of kwargs to pass to metric function
        """
        kwargs = {}
        if "output" in params or "actual" in params:
            kwargs["output"] = output
        if "expected" in params:
            kwargs["expected"] = example.expected_output
        if "llm_metrics" in params and llm_metrics:
            kwargs["llm_metrics"] = llm_metrics
        if "example" in params:
            kwargs["example"] = example
        if "input_data" in params:
            kwargs["input_data"] = example.input_data
        if "metadata" in params:
            kwargs["metadata"] = example.metadata or {}
        if "config" in params:
            kwargs["config"] = config
        if "dataset" in params:
            kwargs["dataset"] = dataset
        if "example_index" in params:
            kwargs["example_index"] = example_index
        return kwargs

    def _call_metric_functions(
        self,
        output: Any,
        example: Any,
        config: dict[str, Any],
        dataset: Dataset,
        example_index: int,
        llm_metrics: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Call all metric functions and collect results.

        Args:
            output: Function output
            example: Current example being evaluated
            config: Configuration dictionary
            dataset: Full dataset
            example_index: Index of current example
            llm_metrics: Captured LLM metrics (if any)

        Returns:
            Dictionary of metric name to score
        """
        import inspect

        example_metrics = {}
        for metric_name, metric_func in self.metric_functions.items():
            try:
                sig = inspect.signature(metric_func)
                params = list(sig.parameters.keys())
                kwargs = self._build_metric_kwargs(
                    params, output, example, config, dataset, example_index, llm_metrics
                )
                score = metric_func(**kwargs)
                example_metrics[metric_name] = score
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Metric function {metric_name} failed: {e}")
                example_metrics[metric_name] = 0.0
        return example_metrics

    def _call_scoring_function(
        self, output: Any, example: Any, llm_metrics: dict[str, Any] | None
    ) -> dict[str, float]:
        """Call single scoring function and return metrics.

        Args:
            output: Function output
            example: Current example being evaluated
            llm_metrics: Captured LLM metrics (if any)

        Returns:
            Dictionary of metric name to score
        """
        import inspect

        example_metrics = {}
        try:
            if self.scoring_function is None:
                return {}
            sig = inspect.signature(self.scoring_function)
            params = list(sig.parameters.keys())
            kwargs = {}
            if "output" in params or "actual" in params:
                kwargs["output"] = output
            if "expected" in params:
                kwargs["expected"] = example.expected_output
            if "llm_metrics" in params and llm_metrics:
                kwargs["llm_metrics"] = llm_metrics

            result = self.scoring_function(**kwargs)
            if isinstance(result, dict):
                example_metrics.update(result)
            else:
                example_metrics["score"] = float(result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Scoring function failed: {e}")
            example_metrics["score"] = 0.0
        return example_metrics

    def _add_llm_metrics_to_example(
        self, example_metrics: dict[str, Any], llm_metrics: dict[str, Any]
    ) -> None:
        """Add captured LLM metrics to example metrics dictionary.

        Args:
            example_metrics: Dictionary to add metrics to (modified in place)
            llm_metrics: Captured LLM metrics
        """
        import os

        example_metrics["prompt_tokens"] = llm_metrics.get("prompt_tokens", 0)
        example_metrics["completion_tokens"] = llm_metrics.get("completion_tokens", 0)
        example_metrics["total_tokens"] = llm_metrics.get("total_tokens", 0)

        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )
        missing_default = None if strict_nulls else 0.0

        example_metrics["input_cost"] = llm_metrics.get("input_cost", missing_default)
        example_metrics["output_cost"] = llm_metrics.get("output_cost", missing_default)
        example_metrics["total_cost"] = llm_metrics.get("total_cost", missing_default)
        example_metrics["response_time_ms"] = llm_metrics.get(
            "response_time_ms", missing_default
        )

    @staticmethod
    def _compute_percentile(values: list[float], percentile: float) -> float:
        """Compute percentile for a list of floats with linear interpolation.

        Args:
            values: List of numeric values
            percentile: Percentile to compute (0.0 to 1.0)

        Returns:
            Computed percentile value
        """
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower_idx = math.floor(index)
        upper_idx = math.ceil(index)
        if lower_idx == upper_idx:
            return float(ordered[int(index)])
        lower_val = ordered[lower_idx]
        upper_val = ordered[upper_idx]
        fraction = index - lower_idx
        return float(lower_val + (upper_val - lower_val) * fraction)

    def _aggregate_custom_metrics(
        self, all_metrics: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Aggregate custom metrics across all examples.

        Args:
            all_metrics: List of metric dictionaries from each example

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}
        for metric in self.metrics:
            metric_values = [m.get(metric, 0.0) for m in all_metrics if m]
            if metric_values:
                if "p95" in metric.lower():
                    aggregated[metric] = self._compute_percentile(metric_values, 0.95)
                else:
                    aggregated[metric] = sum(metric_values) / len(metric_values)
            else:
                aggregated[metric] = 0.0
        return aggregated

    def _aggregate_llm_metrics(
        self,
        all_metrics: list[dict[str, Any]],
        example_results: list[Any],
    ) -> dict[str, float]:
        """Aggregate LLM metrics across all examples.

        Args:
            all_metrics: List of metric dictionaries from each example
            example_results: List of ExampleResult objects

        Returns:
            Dictionary of aggregated LLM metrics
        """
        aggregated = {}

        # Aggregate token metrics (total, not average)
        token_metrics = ["prompt_tokens", "completion_tokens", "total_tokens"]
        for metric in token_metrics:
            metric_values = [
                m.get(metric, 0.0) for m in all_metrics if m and metric in m
            ]
            aggregated[metric] = sum(metric_values) if metric_values else 0.0

        # Aggregate cost metrics (total, not average)
        cost_metrics = ["input_cost", "output_cost", "total_cost"]
        for metric in cost_metrics:
            metric_values = [
                m.get(metric, 0.0) for m in all_metrics if m and metric in m
            ]
            aggregated[metric] = sum(metric_values) if metric_values else 0.0

        # Calculate average response time
        response_times = [
            r.execution_time
            for r in example_results
            if hasattr(r, "execution_time") and r.execution_time > 0
        ]
        if response_times:
            aggregated["avg_response_time"] = sum(response_times) / len(response_times)

        return aggregated

    def _aggregate_ragas_metrics(
        self,
        example_results: list[Any],
        aggregated_metrics: dict[str, Any],
    ) -> None:
        """Aggregate RAGAS metrics if applicable.

        Args:
            example_results: List of ExampleResult objects
            aggregated_metrics: Dictionary to add RAGAS metrics to (modified in place)
        """
        ragas_metric_names = [
            name
            for name in self.metrics
            if name in getattr(self, "_ragas_metric_names", set())
        ]
        if not ragas_metric_names or compute_ragas_metrics is None:
            return

        try:
            ragas_results = compute_ragas_metrics(
                example_results,
                ragas_metric_names,
                config=self._get_ragas_config(),
            )
            aggregated_metrics.update(ragas_results)
        except RagasConfigurationError as exc:
            logger.warning("RAGAS metrics unavailable: %s", exc)
            for name in ragas_metric_names:
                aggregated_metrics.setdefault(name, 0.0)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to compute ragas metrics: %s", exc)
            for name in ragas_metric_names:
                aggregated_metrics.setdefault(name, 0.0)

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate function using scoring functions.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset

        Returns:
            EvaluationResult with metrics and outputs

        Raises:
            EvaluationError: If evaluation fails
        """
        logger.info(
            f"Starting simple scoring evaluation with {len(dataset.examples)} examples, config: {config}"
        )

        start_time = time.time()

        # Initialize metrics tracker if capturing LLM metrics
        if self.capture_llm_metrics and self._metrics_available:
            self._metrics_tracker.start_tracking()

        from traigent.config.context import ConfigurationContext, get_trial_context

        trial_ctx = get_trial_context()

        # Evaluate function on all examples
        example_results = []
        all_metrics = []
        outputs = []
        errors: list[str | None] = []
        budget_exhausted = False

        for i, example in enumerate(dataset.examples):
            if sample_lease and not sample_lease.try_take(1):
                budget_exhausted = True
                logger.info(
                    "Sample budget exhausted before processing example index %s",
                    i,
                )
                break
            try:
                # Clear any previous captured responses before evaluation
                if self.capture_llm_metrics and self._metrics_available:
                    self._clear_captured_responses()

                # Track timing for this example
                example_start_time = time.time()

                # Execute function with configuration context
                with _maybe_restore_trial_context(trial_ctx):
                    with ConfigurationContext(config):
                        if asyncio.iscoroutinefunction(func):
                            output = await func(**example.input_data)
                        else:
                            output = func(**example.input_data)

                per_example_duration = time.time() - example_start_time

                # Capture LLM metrics after function execution
                llm_metrics = self._capture_llm_metrics_for_example(output, config, i)

                # Call scoring functions to compute metrics
                example_metrics = self._compute_example_metrics(
                    output, example, config, dataset, i, llm_metrics
                )

                # Add LLM metrics if captured
                if llm_metrics:
                    self._add_llm_metrics_to_example(example_metrics, llm_metrics)

                # Calculate execution time
                execution_time = per_example_duration
                if llm_metrics and llm_metrics.get("response_time_ms", 0) > 0:
                    execution_time = llm_metrics["response_time_ms"] / 1000.0

                # Create ExampleResult
                example_result = ExampleResult(
                    example_id=f"example_{i}",
                    input_data=example.input_data,
                    expected_output=example.expected_output,
                    actual_output=output,
                    metrics=example_metrics,
                    execution_time=execution_time,
                    success=True,
                    error_message=None,
                    metadata=example.metadata.copy() if example.metadata else {},
                )

                example_results.append(example_result)
                if progress_callback:
                    progress_callback(
                        i,
                        {
                            "success": True,
                            "metrics": example_metrics,
                            "output": output,
                            "llm_metrics": llm_metrics,
                        },
                    )
                all_metrics.append(example_metrics)
                outputs.append(output)
                errors.append(None)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Evaluation failed for example {i}: {e}")
                failed_result = self._create_failed_example_result(example, i, e)
                example_results.append(failed_result)
                if progress_callback:
                    progress_callback(
                        i,
                        {
                            "success": False,
                            "error": str(e),
                            "metrics": failed_result.metrics,
                            "output": None,
                        },
                    )
                all_metrics.append(failed_result.metrics)
                outputs.append(None)
                errors.append(str(e))

        if budget_exhausted and progress_callback:
            progress_callback(
                len(example_results),
                {"success": None, "stop_reason": "sample_budget_exhausted"},
            )

        duration = time.time() - start_time

        # End metrics tracking if it was started
        if self.capture_llm_metrics and self._metrics_available:
            self._metrics_tracker.end_tracking()

        # Aggregate metrics across all examples
        aggregated_metrics = self._aggregate_custom_metrics(all_metrics)

        # Then aggregate LLM metrics if captured
        if self.capture_llm_metrics and self._metrics_available:
            llm_agg = self._aggregate_llm_metrics(all_metrics, example_results)
            aggregated_metrics.update(llm_agg)

        # Aggregate RAGAS metrics if applicable
        self._aggregate_ragas_metrics(example_results, aggregated_metrics)

        aggregated_metrics.setdefault("examples_attempted", len(example_results))

        # Log results
        success_count = sum(1 for result in example_results if result.is_successful)
        logger.info(
            f"Simple scoring evaluation completed: {success_count}/{len(example_results)} successful, "
            f"duration: {duration:.2f}s, metrics: {aggregated_metrics}"
        )
        if budget_exhausted:
            logger.info(
                "Sample budget exhausted after %s examples (dataset size=%s)",
                len(example_results),
                len(dataset.examples),
            )

        # Return evaluation result
        result = EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated_metrics,
            total_examples=len(example_results),
            successful_examples=success_count,
            duration=duration,
            metrics=aggregated_metrics,
            outputs=outputs,
            errors=errors,
        )
        result.sample_budget_exhausted = budget_exhausted
        result.examples_consumed = len(example_results)
        return result

    def _capture_llm_metrics_for_example(
        self, output: Any, config: dict[str, Any], example_index: int
    ) -> dict[str, Any] | None:
        """Capture LLM metrics after function execution.

        Args:
            output: Function output
            config: Configuration dictionary
            example_index: Index of current example

        Returns:
            Dictionary of LLM metrics or None
        """
        if not (self.capture_llm_metrics and self._metrics_available):
            return None

        captured_responses = self._get_all_captured_responses()
        if not captured_responses:
            return None

        response = captured_responses[0]
        if not response:
            return None

        model_name = config.get("model")
        response_text = self._extract_response_text_from_output(output)

        metrics_obj = self._extract_llm_metrics(
            response=response,
            model_name=model_name,
            original_prompt=None,
            response_text=response_text,
        )

        llm_metrics = self._build_llm_metrics_dict(metrics_obj, model_name)

        logger.debug(
            f"Captured LLM metrics for example {example_index}: "
            f"tokens={llm_metrics['total_tokens']}, "
            f"cost=${llm_metrics['total_cost']:.8f}"
        )

        return llm_metrics

    def _compute_example_metrics(
        self,
        output: Any,
        example: Any,
        config: dict[str, Any],
        dataset: Dataset,
        example_index: int,
        llm_metrics: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Compute metrics for a single example.

        Args:
            output: Function output
            example: Current example being evaluated
            config: Configuration dictionary
            dataset: Full dataset
            example_index: Index of current example
            llm_metrics: Captured LLM metrics (if any)

        Returns:
            Dictionary of metric name to score
        """
        if self.metric_functions:
            return self._call_metric_functions(
                output, example, config, dataset, example_index, llm_metrics
            )
        elif self.scoring_function:
            return self._call_scoring_function(output, example, llm_metrics)
        return {}

    def _create_failed_example_result(
        self, example: EvaluationExample, index: int, error: Exception
    ) -> ExampleResult:
        """Create an ExampleResult for a failed evaluation.

        Args:
            example: The example that failed
            index: Index of the example
            error: The exception that occurred

        Returns:
            ExampleResult with failure information
        """
        return ExampleResult(
            example_id=f"example_{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics=dict.fromkeys(self.metrics, 0.0),
            execution_time=0.0,
            success=False,
            error_message=str(error),
            metadata=example.metadata.copy() if example.metadata else {},
        )
