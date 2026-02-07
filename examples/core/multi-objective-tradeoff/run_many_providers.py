#!/usr/bin/env python3
"""Math expression evaluation that can target multiple model providers via LiteLLM."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)
try:
    import litellm
    from litellm import completion
    from litellm.cost_calculator import cost_per_token

    # Enable debug mode for LiteLLM if DEBUG environment variable is set
    if os.getenv("LITELLM_DEBUG", "").lower() in {"1", "true", "yes", "y"}:
        litellm._turn_on_debug()

    # Allow LiteLLM to drop unsupported parameters for models
    litellm.drop_params = True
except ImportError as exc:  # pragma: no cover - dependency hint for example usage
    raise ImportError(
        "examples/core/multi-objective-tradeoff/run_many_providers.py requires 'litellm'. "
        "Install it with `pip install litellm`."
    ) from exc

try:  # Optional observability stack
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - tracing is optional
    trace = None
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    ReadableSpan = None  # type: ignore[assignment]
    SpanExportResult = None  # type: ignore[assignment]
    SpanExporter = object  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

from traigent.api.types import OptimizationResult  # noqa: E402
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema  # noqa: E402
from traigent.utils.langchain_interceptor import (  # noqa: E402
    capture_langchain_response,
)

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "multi-objective-tradeoff"
)
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"

# Configuration constants
TEMPERATURE_CHOICES = [0.0, 0.5]
GLOBAL_PARALLEL_CONFIG = None  # Will be set during traigent.configure()

SUPPORTED_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "PERPLEXITYAI_API_KEY",
    "REPLICATE_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
)


def _prompt() -> str:
    return PROMPT_PATH.read_text().strip()


_PROMPT = _prompt()


if _OTEL_AVAILABLE:

    class _JsonlSpanExporter(SpanExporter):
        """Minimal JSONL exporter that writes spans to a local file."""

        def __init__(self, file_path: Path) -> None:
            self._file_path = file_path.expanduser().resolve()
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()

        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            records: list[str] = []
            for span in spans:
                try:
                    record = _serialize_span(span)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Failed to serialize span %s: %s",
                        getattr(span, "name", "<unknown>"),
                        exc,
                    )
                    continue
                records.append(json.dumps(record, ensure_ascii=False))

            if not records:
                return SpanExportResult.SUCCESS

            try:
                with self._lock, self._file_path.open("a", encoding="utf-8") as handle:
                    for line in records:
                        handle.write(line)
                        handle.write("\n")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to write OpenTelemetry spans: %s", exc)
                return SpanExportResult.FAILURE

            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            # Nothing to flush; files are handled synchronously
            return None

    def _serialize_span(span: ReadableSpan) -> dict[str, Any]:
        """Convert a span into a JSON-serializable dictionary."""

        start = datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=UTC)
        end = datetime.fromtimestamp(span.end_time / 1_000_000_000, tz=UTC)

        attributes = dict(span.attributes.items()) if span.attributes else {}

        events: list[dict[str, Any]] = []
        for event in span.events:
            event_time = datetime.fromtimestamp(event.timestamp / 1_000_000_000, tz=UTC)
            events.append(
                {
                    "name": event.name,
                    "time": event_time.isoformat(),
                    "attributes": dict(event.attributes or {}),
                }
            )

        return {
            "name": span.name,
            "trace_id": f"{span.context.trace_id:032x}",
            "span_id": f"{span.context.span_id:016x}",
            "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_ms": (span.end_time - span.start_time) / 1_000_000,
            "status": getattr(span.status.status_code, "name", "UNSET"),
            "attributes": attributes,
            "events": events,
            "resource": dict(span.resource.attributes if span.resource else {}),
        }


_TRACING_READY = False
_TRACER = None


def _setup_tracing() -> None:
    """Initialise OpenTelemetry tracing if the optional dependencies are present."""

    global _TRACING_READY, _TRACER

    if _TRACING_READY or not _OTEL_AVAILABLE:
        return

    try:
        log_path = os.getenv("TRAIGENT_TRACE_LOG")
        trace_path = (
            Path(log_path).expanduser()
            if log_path
            else BASE / "telemetry" / "traces.jsonl"
        )

        resource = Resource.create(
            {
                "service.name": "traigent-multi-provider",
                "service.namespace": "examples",
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = _JsonlSpanExporter(trace_path)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        atexit.register(provider.shutdown)
        _TRACER = trace.get_tracer("traigent.examples.multi_provider")
        _TRACING_READY = True
        logger.info("OpenTelemetry tracing enabled → %s", trace_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialise OpenTelemetry tracing: %s", exc)
        _TRACING_READY = False
        _TRACER = None


def _start_span(name: str):
    if _TRACING_READY and _TRACER is not None:
        return _TRACER.start_as_current_span(name)
    return nullcontext()


_setup_tracing()


def _truncate(text: str | None, width: int = 48) -> str:
    """Trim long strings for debug dumps."""

    if not text:
        return ""
    shortened = text.strip().replace("\n", " ")
    if len(shortened) <= width:
        return shortened
    return f"{shortened[: width - 3]}..."


OBJECTIVE_SCHEMA = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)
        # ObjectiveDefinition(name="cost", orientation="minimize", weight=0.3),
    ]
)


def _extract_expression(question: str) -> str:
    """Remove leading instruction text and trailing punctuation to get the math expression."""

    stripped = question.strip()
    stripped = stripped.rstrip(".?!")

    prefixes = [
        "compute ",
        "what is ",
        "evaluate ",
        "simplify ",
        "calculate ",
        "find the value of ",
    ]

    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _evaluate_expression(question: str) -> str:
    """Evaluate the mathematical expression embedded in the question."""

    expr = _extract_expression(question)
    expr = expr.replace("^", "**")

    safe_globals = {"__builtins__": {}, "pow": pow, "bin": bin, "int": int}

    try:
        value = eval(
            expr, safe_globals, {}
        )  # noqa: S307 - controlled input for examples
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Failed to evaluate expression '{expr}': {exc}") from exc

    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float):
        value = int(round(value))
    elif not isinstance(value, int):
        value = int(value)

    return str(int(value))


def _normalize_model_output(raw: str) -> str:
    """Best-effort extraction of the first integer from the model response."""

    text = raw.strip()

    if not text:
        return ""

    match = re.search(r"answer[^\d-]*(?:is|:)?\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    equals_hits = re.findall(r"=\s*(-?\d+)", text)
    if equals_hits:
        return equals_hits[-1]

    # Strip common enumerated list prefixes so we do not pick step numbers.
    text_without_lists = re.sub(r"(?m)^\s*\(?\d+[\).]\s+", "", text)

    all_ints = re.findall(r"-?\d+", text_without_lists)
    if all_ints:
        return all_ints[-1]

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:64]


def _extract_choice_text(choice: Any) -> str:
    """Coerce LiteLLM choice payloads into a plain string."""

    if choice is None:
        return ""

    if hasattr(choice, "model_dump"):
        choice = choice.model_dump()

    if not isinstance(choice, dict):
        return str(choice)

    message = choice.get("message")
    if hasattr(message, "model_dump"):
        message = message.model_dump()

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif hasattr(item, "get"):
                    text = item.get("text") if callable(getattr(item, "get", None)) else None  # type: ignore[arg-type]
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)
        text_field = message.get("text")
        if isinstance(text_field, str):
            return text_field

    text = choice.get("text")
    if isinstance(text, list):
        return "\n".join(str(item) for item in text if item is not None)
    if isinstance(text, str):
        return text

    return ""


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print aggregated and raw optimization data."""

    primary = result.objectives[0] if result.objectives else None

    aggregated = result.to_aggregated_dataframe(primary_objective=primary)
    if not aggregated.empty:
        display_cols = [
            "model",
            "temperature",
            "max_tokens",
            "samples_count",
            "accuracy",
            "cost",
            "duration",
            "avg_response_time",
        ]
        cols = [c for c in display_cols if c in aggregated.columns]
        table = aggregated[cols] if cols else aggregated
        if "avg_response_time" in table.columns:
            table.loc[:, "avg_response_time"] = (
                table["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table.columns:
            table.loc[:, "duration"] = table["duration"].astype(float).round(3)
        print("\nAggregated configurations and performance:")
        print(table.to_string(index=False))

    stats = result.experiment_stats
    counts = stats.trial_counts
    print("\nExperiment totals:")
    print(f"  Total duration: {stats.total_duration:.3f}s")
    print(f"  Total cost: {stats.total_cost:.6f}")
    print(f"  Unique configurations: {stats.unique_configurations}")
    print(
        "  Trials — completed: {completed}, failed: {failed}, cancelled: {cancelled}, pending: {pending}, running: {running}".format(
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            cancelled=counts.get("cancelled", 0),
            pending=counts.get("pending", 0),
            running=counts.get("running", 0),
        )
    )
    if counts.get("exceptions"):
        print(f"  Exceptions captured: {counts['exceptions']}")
    if stats.average_trial_duration is not None:
        print(f"  Avg trial duration: {stats.average_trial_duration:.3f}s")
    if stats.success_rate is not None:
        print(f"  Success rate: {stats.success_rate:.3f}")
    if stats.error_message:
        print(f"  Stats warning: {stats.error_message}")

    raw = result.to_dataframe()
    if not raw.empty:
        preferred_raw = [
            "trial_id",
            "status",
            "model",
            "temperature",
            "max_tokens",
            "accuracy",
            "cost",
            "duration",
            "avg_response_time",
        ]
        cols_raw = [c for c in preferred_raw if c in raw.columns]
        table_raw = raw[cols_raw] if cols_raw else raw
        if "avg_response_time" in table_raw.columns:
            table_raw.loc[:, "avg_response_time"] = (
                table_raw["avg_response_time"].astype(float).round(3)
            )
        if "duration" in table_raw.columns:
            table_raw.loc[:, "duration"] = table_raw["duration"].astype(float).round(3)
        print("\nRaw (per-sample) trials:")
        print(table_raw.to_string(index=False))


def _dump_example_results(
    result: OptimizationResult, show_full_output: bool = False
) -> None:
    """Dump per-example metrics so latency and cost can be audited."""

    if not result.trials:
        print("\nNo trials captured.")
        return

    print("\nDetailed example metrics:")

    for trial in result.trials:
        config = trial.config or {}
        model = config.get("model", "<unknown-model>")
        print(
            f"\nTrial {trial.trial_id} – model={model}, temperature={config.get('temperature')}, max_tokens={config.get('max_tokens')}"
        )

        example_results = (trial.metadata or {}).get("example_results") or []
        if not example_results:
            print("  (no example_results captured)")
            continue

        for idx, example in enumerate(example_results, start=1):
            if hasattr(example, "metrics"):
                metrics = example.metrics or {}
                execution_time = getattr(example, "execution_time", None)
                success = getattr(example, "success", None)
                actual_output = getattr(example, "actual_output", None)
            else:
                metrics = example.get("metrics", {})
                execution_time = example.get("execution_time")
                success = example.get("success")
                actual_output = example.get("actual_output")

            accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
            total_cost = (
                metrics.get("total_cost") if isinstance(metrics, dict) else None
            )

            function_time = None
            model_response_time = None
            if isinstance(metrics, dict):
                function_time = metrics.get("function_duration")
                model_response_time = metrics.get("model_response_time")
                if model_response_time is None:
                    maybe_resp = metrics.get("response_time")
                    if maybe_resp is not None:
                        model_response_time = maybe_resp
                    else:
                        resp_ms = metrics.get("response_time_ms")
                        if resp_ms is not None:
                            model_response_time = float(resp_ms) / 1000.0
            if function_time is None:
                function_time = execution_time
            if model_response_time is None:
                model_response_time = function_time

            input_tokens = (
                metrics.get("input_tokens") if isinstance(metrics, dict) else None
            )
            output_tokens = (
                metrics.get("output_tokens") if isinstance(metrics, dict) else None
            )

            function_display = (
                f"{function_time:.3f}s"
                if isinstance(function_time, (int, float))
                else "?"
            )
            model_display = (
                f"{model_response_time:.3f}s"
                if isinstance(model_response_time, (int, float))
                else "?"
            )
            accuracy_display = (
                f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "?"
            )
            cost_display = (
                f"${total_cost:.6f}" if isinstance(total_cost, (int, float)) else "?"
            )

            sample = str(actual_output) if actual_output is not None else ""
            if not show_full_output:
                sample = _truncate(sample)

            print(
                f"  Example {idx:02d}: success={success} accuracy={accuracy_display} "
                f"function_time={function_display} model_time={model_display} "
                f"input_tokens={input_tokens} output_tokens={output_tokens} cost={cost_display} output='{sample}'"
            )


def _ensure_api_key() -> None:
    if any(os.getenv(var) for var in SUPPORTED_PROVIDER_KEYS):
        return
    joined = ", ".join(SUPPORTED_PROVIDER_KEYS)
    raise RuntimeError(
        "No provider API key detected. Set the environment variable for your provider ("
        f"one of: {joined})."
    )


def _resolve_litellm_model(model: str) -> str:
    if "/" in model:
        return model
    lowered = model.lower()
    if lowered.startswith("claude"):
        return f"anthropic/{model}"
    if (
        lowered.startswith("gpt-")
        or lowered.startswith("o1-")
        or lowered.startswith("o3-")
    ):
        return f"openai/{model}"
    return model


def _invoke_model(
    prompt: str,
    litellm_model: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    provider_name = litellm_model.split("/", 1)[0] if "/" in litellm_model else "custom"
    prompt_preview = prompt[:256]

    logger.info(
        "Invoking provider='%s' model='%s' temperature=%.2f max_tokens=%s",
        provider_name,
        litellm_model,
        float(temperature),
        int(max_tokens),
    )

    with _start_span("llm.completion") as span:
        if span is not None:
            span.set_attribute("llm.provider", provider_name)
            span.set_attribute("llm.model", litellm_model)
            span.set_attribute("llm.temperature", float(temperature))
            span.set_attribute("llm.max_tokens", int(max_tokens))
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.prompt_preview", prompt_preview)

        start_time = time.perf_counter()
        try:
            result = completion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=None,
                timeout=None,
            )
        except Exception as exc:
            if span is not None and Status is not None and StatusCode is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        response_time_ms = (time.perf_counter() - start_time) * 1000.0

        if hasattr(result, "model_dump"):
            payload: dict[str, Any] = result.model_dump()
        elif isinstance(result, dict):
            payload = result
        else:
            payload = {}

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LiteLLM returned no choices")

        raw_text = _extract_choice_text(choices[0])

        usage_obj = payload.get("usage")
        if usage_obj is None and hasattr(result, "usage"):
            usage_obj = result.usage

        if hasattr(usage_obj, "model_dump"):
            usage_data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage_data = usage_obj
        else:
            usage_data = {}

        prompt_tokens = int(
            usage_data.get("prompt_tokens")
            or usage_data.get("input_tokens")
            or usage_data.get("cache_creation_input_tokens")
            or 0
        )
        completion_tokens = int(
            usage_data.get("completion_tokens") or usage_data.get("output_tokens") or 0
        )
        total_tokens = int(
            usage_data.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        try:
            input_cost, output_cost = cost_per_token(
                model=litellm_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_object=getattr(result, "usage", None),
            )
        except Exception:
            input_cost, output_cost = 0.0, 0.0
        total_cost = float(input_cost or 0.0) + float(output_cost or 0.0)
        cost_metrics = SimpleNamespace(
            input_cost=float(input_cost or 0.0),
            output_cost=float(output_cost or 0.0),
            total_cost=total_cost,
        )

        result.response_time_ms = response_time_ms
        result.cost = cost_metrics
        if not hasattr(result, "metrics"):
            result.metrics = SimpleNamespace(response_time_ms=response_time_ms)
        else:
            metrics_attr = result.metrics
            metrics_attr.response_time_ms = response_time_ms

        response_metadata = getattr(result, "response_metadata", {}) or {}
        response_metadata.update(
            {
                "model": payload.get("model") or litellm_model,
                "response_time_ms": response_time_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        result.response_metadata = response_metadata

        if span is not None:
            span.set_attribute("llm.response_time_ms", response_time_ms)
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            span.set_attribute("llm.total_tokens", total_tokens)
            span.set_attribute("llm.total_cost", float(total_cost))
            span.set_attribute("llm.response_preview", raw_text[:256])
            span.set_attribute(
                "llm.normalized_output",
                _normalize_model_output(raw_text) if raw_text else "",
            )

    capture_langchain_response(result)

    normalized = _normalize_model_output(raw_text)
    normalized_output = normalized if normalized else raw_text.strip()

    return normalized_output


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=OBJECTIVE_SCHEMA,
    configuration_space={
        "model": [
            # "anthropic/claude-3-7",
            # "anthropic/claude-3-5-haiku-20241022",
            # "anthropic/claude-3-haiku-20240307",
            # "anthropic/claude-sonnet-4",
            # "anthropic/claude-sonnet-4-5",
            # "openai/gpt-3.5-turbo",
            "openai/gpt-4o-mini",
            # "openai/gpt-5o-mini",
            "openai/gpt-4.1-nano",
            # "openai/gpt-5-nano-2025-08-07",
        ],
        "temperature": TEMPERATURE_CHOICES,
        "max_tokens": [64],
    },
    execution_mode="edge_analytics",
    injection_mode="seamless",
    parallel_config=GLOBAL_PARALLEL_CONFIG,
    algorithm="bayesian",
    max_trials=20,
)
def answer(
    question: str,
    model: str = "openai/gpt-4o-nano",
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> str:
    if MOCK:
        return _evaluate_expression(question)

    _ensure_api_key()

    prompt = f"Expression: {question}\n\n{_PROMPT}"
    litellm_model = _resolve_litellm_model(model)
    return _invoke_model(
        prompt,
        litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Run the multi-provider tradeoff demo"
        )
        parser.add_argument(
            "--verbose-results",
            action="store_true",
            help="Print detailed per-example outputs for each trial",
        )
        parser.add_argument(
            "--max-trials",
            type=int,
            default=None,
            help="Override the number of optimization trials to run",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="anthropic/claude-3-7-sonnet-latest",
            help="Model identifier (include provider prefix for non-default providers)",
        )
        args = parser.parse_args()

        print("Crunching math expressions across multiple providers…")

        async def main() -> None:
            trials = (
                args.max_trials
                if args.max_trials is not None
                else (10 if not MOCK else 4)
            )
            r = await answer.optimize(max_trials=trials, model=args.model)
            print({"best_config": r.best_config, "best_score": r.best_score})
            _print_results(r)
            if args.verbose_results:
                _dump_example_results(r, show_full_output=True)

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
