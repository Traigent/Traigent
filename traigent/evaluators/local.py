"""Local evaluation strategy."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import copy
import inspect
import math
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from traigent.core.execution_budget import ExecutionBudget
    from traigent.core.meta_types import TraigentMetadata

from traigent.config.types import resolve_execution_mode
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationResult,
    _accuracy_matches_after_unwrap,
    _accuracy_value_for_example,
    coercion_log_scope,
    _example_correlation_key,
    _is_empty_expected_output,
)
from traigent.evaluators.metrics_tracker import (
    EMPTY_OUTPUT_RATE_WARNING_THRESHOLD,
    ExampleMetrics,
    MetricsCalculator,
    MetricsTracker,
    ResponseHandlerFactory,
    compute_empty_output_rate,
    compute_truncated_output_rate,
    enforce_user_metric_ceiling,
    enforce_unknown_cost_invariant,
    extract_llm_metrics,
)
from traigent.utils.exceptions import EvaluationError
from traigent.utils.langchain_interceptor import (
    capture_scope,
    get_captured_response_by_key,
    drain_captured_responses,
    seal_captured_responses,
    finish_captured_responses,
    snapshot_captured_responses,
    patch_langchain_for_metadata_capture,
)
from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture
from traigent.utils.logging import get_logger
from traigent.utils.response_model import infer_response_model

logger = get_logger(__name__)
_METADATA_PATCHES_ATTEMPTED = False

_PREVIEW_COST_KEYS = (
    "cost",
    "input_cost",
    "output_cost",
    "total_cost",
    "cost_unmeasured",
    "cost_unpriced",
    "cost_lower_bound",
)

_OUTPUT_METRIC_PARAM_NAMES = {
    "actual",
    "actual_output",
    "output",
    "prediction",
    "predicted",
    "result",
}
_EXPECTED_METRIC_PARAM_NAMES = {
    "expected",
    "expected_output",
    "ground_truth",
    "reference",
    "target",
}
_LLM_METRIC_PARAM_NAMES = {"llm_metrics", "metrics"}


@dataclass(frozen=True)
class MetricBinding:
    """Resolved shape of a metric-callback invocation, computed without calling it.

    ``resolve_metric_call_binding`` returns one of these describing the first
    argument candidate that :func:`inspect.Signature.bind` accepts (or a
    terminal ``bind_ok=False`` binding). :meth:`LocalEvaluator._invoke_metric_function`
    consumes it and only then performs the real call, so the metric function is
    never executed while the shape is being determined.

    Attributes:
        args: Positional arguments the winning candidate would pass.
        kwargs: Keyword arguments the winning candidate would pass.
        binding_mode: How arguments were matched -- one of ``"var_positional"``,
            ``"keyword"``, ``"positional_3"``, ``"positional_2"``,
            ``"positional_1"``, ``"positional_0"``, or ``"unbound"``.
        matched_parameters: Metric parameter names filled by a recognized
            keyword (empty for positional / var-positional binding).
        unmatched_parameters: Bindable metric parameter names NOT filled by a
            recognized keyword. When no candidate binds at all
            (``bind_ok=False``), this leads with any *required*
            positional-only parameter names -- they can never be filled by a
            recognized keyword (``build_metric_keyword_arguments`` never
            offers one to a positional-only slot) and are the actual reason
            binding failed, even though ``inspect.Signature.bind`` still
            reports the trailing keyword-or-positional parameters as
            "missing" -- followed by the other bindable (POSITIONAL_OR_KEYWORD
            / KEYWORD_ONLY) names.
        bind_ok: Whether any candidate bound successfully.
        bind_exception: The terminal ``TypeError`` from ``Signature.bind`` when
            nothing bound; ``None`` on success. Held as the original exception
            so callers can re-raise it unchanged.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    binding_mode: str
    matched_parameters: tuple[str, ...]
    unmatched_parameters: tuple[str, ...]
    bind_ok: bool
    bind_exception: TypeError | None = None

    @property
    def bind_error(self) -> str | None:
        """Human-readable bind-error message (``None`` when ``bind_ok``)."""
        return None if self.bind_exception is None else str(self.bind_exception)


def build_metric_keyword_arguments(
    parameters: list[inspect.Parameter],
    output: Any,
    example_obj: Any,
    config: dict[str, Any],
    llm_payload: dict[str, Any],
    example_index: int,
) -> dict[str, Any]:
    """Build keyword arguments for a metric callback from recognized param names.

    Pure function (no evaluator state) shared by
    :meth:`LocalEvaluator._build_metric_keyword_arguments` and
    :func:`resolve_metric_call_binding`, so the keyword-matching rules cannot
    drift between the runtime invocation path and the no-execution contract
    inspector.
    """
    keyword_values: dict[str, Any] = {
        "example": example_obj,
        "input_data": example_obj.input_data,
        "metadata": example_obj.metadata or {},
        "config": config,
        "example_index": example_index,
    }
    for name in _OUTPUT_METRIC_PARAM_NAMES:
        keyword_values[name] = output
    for name in _EXPECTED_METRIC_PARAM_NAMES:
        keyword_values[name] = example_obj.expected_output
    for name in _LLM_METRIC_PARAM_NAMES:
        keyword_values[name] = llm_payload

    kwargs: dict[str, Any] = {}
    accepts_arbitrary_kwargs = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_arbitrary_kwargs = True
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        if parameter.name in keyword_values:
            kwargs[parameter.name] = keyword_values[parameter.name]

    if accepts_arbitrary_kwargs:
        kwargs.setdefault("output", output)
        kwargs.setdefault("expected", example_obj.expected_output)
        kwargs.setdefault("llm_metrics", llm_payload)

    return kwargs


def resolve_metric_call_binding(
    metric_func: Callable[..., Any],
    output: Any,
    example_obj: Any,
    config: dict[str, Any],
    llm_payload: dict[str, Any],
    example_index: int,
) -> MetricBinding:
    """Resolve how ``metric_func`` would be called WITHOUT invoking it.

    Mirrors the candidate ordering used by
    :meth:`LocalEvaluator._invoke_metric_function` (var-positional fast path,
    then a recognized-keyword candidate when any parameter is recognized, then
    positional fallbacks of decreasing arity) and returns the first candidate
    that ``inspect.Signature.bind`` accepts -- or a terminal ``bind_ok=False``
    :class:`MetricBinding` carrying the last bind error. The metric function is
    never executed.

    Raises:
        TypeError, ValueError: If ``inspect.signature(metric_func)`` cannot be
            computed (e.g. some builtins). Callers that must not fail closed
            should guard this call.
    """
    signature = inspect.signature(metric_func)
    parameters = list(signature.parameters.values())
    positional_args = (output, example_obj.expected_output, llm_payload)

    call_candidates: list[tuple[tuple[Any, ...], dict[str, Any], str]] = []

    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        call_candidates.append((positional_args, {}, "var_positional"))

    kwargs = build_metric_keyword_arguments(
        parameters, output, example_obj, config, llm_payload, example_index
    )
    if kwargs:
        call_candidates.append(((), kwargs, "keyword"))

    call_candidates.extend(
        [
            (positional_args, {}, "positional_3"),
            (positional_args[:2], {}, "positional_2"),
            (positional_args[:1], {}, "positional_1"),
            ((), {}, "positional_0"),
        ]
    )

    bindable_names = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    bind_error: TypeError | None = None
    for args, candidate_kwargs, mode in call_candidates:
        try:
            signature.bind(*args, **candidate_kwargs)
        except TypeError as exc:
            bind_error = exc
            continue
        matched = tuple(sorted(candidate_kwargs))
        unmatched = tuple(
            sorted(name for name in bindable_names if name not in candidate_kwargs)
        )
        return MetricBinding(
            args=args,
            kwargs=candidate_kwargs,
            binding_mode=mode,
            matched_parameters=matched,
            unmatched_parameters=unmatched,
            bind_ok=True,
            bind_exception=None,
        )

    # A required POSITIONAL_ONLY parameter (e.g. `def scorer(x, /, output,
    # ...)`) can never receive a recognized keyword -- `bindable_names` above
    # deliberately excludes it, since `build_metric_keyword_arguments` never
    # assigns one -- but it is consuming the leading slot(s) of every
    # positional candidate, which is *why* the trailing keyword-or-positional
    # names end up unfilled. Report it first: it is the actual culprit, not
    # the recognized names that follow it in the signature.
    required_positional_only = [
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
    ]
    return MetricBinding(
        args=(),
        kwargs={},
        binding_mode="unbound",
        matched_parameters=(),
        unmatched_parameters=tuple(required_positional_only + sorted(bindable_names)),
        bind_ok=False,
        bind_exception=bind_error,
    )


@dataclass
class PromptInfo:
    """Extracted prompt information for metrics calculation."""

    original_prompt: list[dict[str, str]] | None
    prompt_length: int | None
    response_length: int | None


class _CaptureOwnership:
    """Captured responses grouped by the example that made the calls (#2444).

    Ownership is recorded when a response is captured (the capturing context
    carries its example index), so attribution never depends on how many calls
    each example made or on the order concurrent examples finished in.

    Consumption is tracked per individual captured call (by position in the
    capture list), never per example: ``taken`` holds the calls whose spend a
    metrics row already includes, and everything else -- unowned calls, calls of
    an example that produced no row, siblings of a returned response -- is
    reported exactly once through :meth:`leftover`.
    """

    def __init__(
        self,
        responses: list[Any],
        owners: list[int | None],
        requested: list[str | None] | None = None,
    ) -> None:
        self.responses = responses
        self.requested: list[str | None] = (
            list(requested)
            if requested is not None and len(requested) == len(responses)
            else [None] * len(responses)
        )
        self.by_example: dict[int, list[int]] = {}
        self.unowned: list[int] = []
        for position, owner in enumerate(owners):
            if owner is None:
                self.unowned.append(position)
            else:
                self.by_example.setdefault(owner, []).append(position)
        self.taken: set[int] = set()
        # Identity of every captured response and of every returned object
        # already priced, across ALL owners: a returned object that is
        # already accounted is never charged a second time (#2444).
        self.captured_ids: set[int] = {id(r) for r in responses}
        self.returned_seen: set[int] = set()
        # Model that priced an example's SINGLE call (response model first);
        # absent for multi-call rows, so metadata reconciliation never reprices
        # them as one configured model.
        self.reconcile_model: dict[int, str | None] = {}

    @classmethod
    def build(
        cls,
        responses: list[Any],
        owners: list[int | None] | None,
        requested: list[str | None] | None = None,
    ) -> _CaptureOwnership | None:
        """Ownership view of a drained capture batch.

        ``None`` (legacy path) only outside a capture scope, where no owners
        exist.  Inside a scope, owners that do not align one-to-one with the
        responses fail CLOSED: every response is treated as unowned, so the
        whole batch is reported as unknown spend with its exact priceable
        floor rather than being guessed onto examples.
        """
        if owners is None:
            return None
        if len(owners) != len(responses):
            owners = [None] * len(responses)
        return cls(responses, owners, requested)

    def owned_untaken(self, example_index: int) -> list[int]:
        return [
            p for p in self.by_example.get(example_index, []) if p not in self.taken
        ]

    def find(self, obj: Any, example_index: int) -> int | None:
        """Position of the untaken capture that IS ``obj`` (identity).

        Looks at the example's own calls first, then at unowned ones.
        """
        if obj is None:
            return None
        for position in self.owned_untaken(example_index) + [
            p for p in self.unowned if p not in self.taken
        ]:
            if self.responses[position] is obj:
                return position
        return None

    def example_responses(self, example_index: int) -> list[Any]:
        return [self.responses[p] for p in self.by_example.get(example_index, [])]

    def add_late(self, response: Any, requested: str | None) -> None:
        """A call that surfaced after the main drain and belongs to no judge."""
        self.responses.append(response)
        self.requested.append(requested)
        self.unowned.append(len(self.responses) - 1)
        self.captured_ids.add(id(response))

    def absorb_late(
        self,
        drained: tuple[list[Any], list[int | None] | None, list[str | None] | None],
    ) -> None:
        """Route everything a final drain returned into :meth:`leftover`.

        Whatever its recorded owner, a call that surfaced after the per-example
        accounting finished belongs to no row: unknown spend with its floor.
        """
        responses, _owners, requested = drained
        for position, response in enumerate(responses):
            self.add_late(
                response,
                requested[position]
                if requested is not None and position < len(requested)
                else None,
            )

    def leftover(self) -> list[tuple[Any, str | None]]:
        """Captured calls no row has included, with their requested model."""
        return [
            (r, self.requested[p])
            for p, r in enumerate(self.responses)
            if p not in self.taken
        ]


# Progress has its own cursor over evaluation-scoped identity history. Preview
# extraction may advance this cursor, but never consumes final accounting's
# returned_seen/taken sets. Context isolation also covers a shared evaluator
# serving concurrent evaluate() calls.
_progress_capture_history: ContextVar[_CaptureOwnership | None] = ContextVar(
    "traigent_progress_capture_history", default=None
)


class _AggregatedResponses:
    """Wrapper that aggregates metrics from multiple LLM responses.

    Used when a single example makes multiple LLM calls (e.g., main response + judge).
    Provides usage_metadata and response_metadata that sum the individual responses.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        totals = self._sum_token_usage()
        self._build_metadata(totals)

    def _extract_tokens_from_response(self, resp: Any) -> tuple[int, int, int]:
        """Extract (input, output, total) tokens from a single response."""
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            return (
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0),
            )

        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_obj, "total_tokens", None)
            if isinstance(input_tokens, (int, float)) and isinstance(
                output_tokens, (int, float)
            ):
                if isinstance(total_tokens, (int, float)):
                    return (int(input_tokens), int(output_tokens), int(total_tokens))
                return (
                    int(input_tokens),
                    int(output_tokens),
                    int(input_tokens + output_tokens),
                )

        resp_meta = getattr(resp, "response_metadata", None)
        if not resp_meta or not isinstance(resp_meta, dict):
            return (0, 0, 0)

        token_usage = resp_meta.get("token_usage", {})
        if not isinstance(token_usage, dict):
            return (0, 0, 0)

        return (
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
            token_usage.get("total_tokens", 0),
        )

    def _sum_token_usage(self) -> tuple[int, int, int]:
        """Sum up token usage from all responses."""
        total_input = 0
        total_output = 0
        total_tokens = 0

        for resp in self._responses:
            inp, out, tot = self._extract_tokens_from_response(resp)
            total_input += inp
            total_output += out
            total_tokens += tot

        # Ensure total is at least sum of parts
        if total_tokens == 0 and (total_input > 0 or total_output > 0):
            total_tokens = total_input + total_output

        return (total_input, total_output, total_tokens)

    def _build_metadata(self, totals: tuple[int, int, int]) -> None:
        """Build usage_metadata and response_metadata from totals."""
        total_input, total_output, total_tokens = totals

        self.usage_metadata = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
        }

        self.response_metadata: dict[str, Any] = {
            "token_usage": {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_tokens,
            }
        }

        # Copy model name from first response if available
        if self._responses:
            first_meta = getattr(self._responses[0], "response_metadata", None)
            if first_meta and isinstance(first_meta, dict):
                for key in ("model", "model_name"):
                    if key in first_meta:
                        self.response_metadata[key] = first_meta[key]

        logger.debug(
            f"Aggregated {len(self._responses)} LLM responses: "
            f"input={total_input}, output={total_output}, total={total_tokens}"
        )


def _ensure_metadata_capture_patches() -> None:
    """Patch LangChain and LiteLLM metadata capture lazily.

    Importing the SDK should not emit optional-integration warnings just because
    LocalEvaluator is imported as part of broader package initialization.
    """
    global _METADATA_PATCHES_ATTEMPTED
    if _METADATA_PATCHES_ATTEMPTED:
        return
    patch_langchain_for_metadata_capture()
    patch_litellm_for_metadata_capture()
    _METADATA_PATCHES_ATTEMPTED = True


if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease


class LocalEvaluator(BaseEvaluator):
    """Local evaluation strategy.

    Evaluates functions locally in the current process. Supports both
    synchronous and asynchronous functions with timeout handling.

    Example::

        evaluator = LocalEvaluator(["accuracy"], timeout=30.0)
        result = await evaluator.evaluate(my_function, config, dataset)
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        max_workers: int = 1,
        detailed: bool = False,
        execution_mode: str | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize local evaluator.

        Args:
            metrics: List of metric names to compute
            timeout: Timeout for individual evaluations (seconds)
            max_workers: Maximum number of concurrent evaluations
            detailed: Whether to preserve detailed example results
            execution_mode: Internal execution-mode value for determining
                submission format: ``"local"`` (public mode ``local``),
                ``"hybrid"`` (the internal value for public mode ``cloud``) or
                ``"hybrid_api"`` (public mode ``hybrid_api``).
            **kwargs: Additional configuration
        """
        if "privacy_enabled" in kwargs:
            import warnings

            kwargs.pop("privacy_enabled")
            warnings.warn(
                "LocalEvaluator privacy_enabled is deprecated and ignored; local "
                "evaluator traces and results are content-free by default.",
                DeprecationWarning,
                stacklevel=2,
            )

        _ensure_metadata_capture_patches()

        if metrics is None and metric_functions:
            metrics = list(metric_functions.keys())

        super().__init__(metrics, timeout, max_workers, **kwargs)
        self.detailed = detailed
        self.execution_mode_enum = (
            resolve_execution_mode(execution_mode)
            if execution_mode is not None
            else None
        )
        self.execution_mode = (
            self.execution_mode_enum.value if self.execution_mode_enum else None
        )
        self.metric_functions = metric_functions or {}
        # One-shot guard for the dual-scorer run notice (issue #1845). The
        # evaluator instance is shared across a run's trials, so a flag here
        # emits the notice once per run rather than once per example/trial.
        self._dual_scorer_notice_logged = False
        # One-shot guard for the high-empty-output-rate run warning (issue
        # #1851). Same rationale as the dual-scorer flag: the evaluator instance
        # is shared across the run's trials, so this emits ONE warning per run
        # (naming the first offending config) rather than one per trial/example.
        self._empty_output_warning_logged = False
        # One-shot guard for the truncated-finish_reason run warning (issue
        # #1809). Same rationale as the empty-output flag.
        self._truncated_output_warning_logged = False

    def _extract_prompt_info(
        self,
        example_input: Any,
        output: Any,
        config: dict[str, Any] | None = None,
    ) -> PromptInfo:
        """Extract prompt information for metrics calculation.

        Keeps prompt content out of metric extraction while retaining lengths
        for downstream cost estimation.

        Args:
            example_input: Input data from dataset example
            output: Output from function call
            config: Configuration dictionary (uses ``config["prompt"]`` for
                template-aware length calculation when available)

        Returns:
            PromptInfo with content-free prompt lengths.
        """
        prompt_length: int | None = None
        response_length: int | None = None
        original_prompt: list[dict[str, str]] | None = None

        prompt_length = self._calculate_input_length(example_input, config=config)
        response_length = self._extract_response_length(output)

        return PromptInfo(
            original_prompt=original_prompt,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def _calculate_prompt_template_length(
        self,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> int | None:
        """Calculate template-aware input length when prompt template is available.

        Uses ``config["prompt"]`` only. If formatting fails, falls back to additive
        approximation ``len(template) + len(str(example_input))``.
        """
        if not isinstance(config, dict):
            return None

        prompt_template = config.get("prompt")
        if not isinstance(prompt_template, str) or not prompt_template:
            return None

        if isinstance(example_input, dict):
            try:
                return len(prompt_template.format(**example_input))
            except Exception as exc:
                logger.debug(
                    "Prompt template format failed, using additive length fallback: %s",
                    exc,
                )
                return len(prompt_template) + len(str(example_input))

        return len(prompt_template) + len(str(example_input))

    def _calculate_input_length(
        self,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Calculate the character length of input data."""
        template_length = self._calculate_prompt_template_length(
            example_input, config=config
        )
        if template_length is not None:
            return template_length

        if isinstance(example_input, dict):
            if "text" in example_input:
                return len(example_input["text"])
            elif "messages" in example_input:
                # For messages, sum up all content lengths
                return sum(
                    len(msg.get("content", ""))
                    for msg in example_input["messages"]
                    if isinstance(msg, dict)
                )
            else:
                return len(str(example_input))
        return len(str(example_input))

    def _reconstruct_prompt(self, example_input: Any) -> list[dict[str, str]]:
        """Reconstruct prompt messages from input data."""
        if isinstance(example_input, dict):
            if "text" in example_input:
                return [{"role": "user", "content": example_input["text"]}]
            elif "messages" in example_input:
                return cast(list[dict[str, str]], example_input["messages"])
            else:
                return [{"role": "user", "content": str(example_input)}]
        return [{"role": "user", "content": str(example_input)}]

    @staticmethod
    def _infer_model_name_from_output(
        output: Any,
        index: int,
        captured_responses: list[Any],
    ) -> str | None:
        """Try to infer the model name from the output or captured LangChain responses.

        This is a fallback for when the optimization config does not contain a
        ``model`` key (e.g. the user is only tuning prompts, not models).
        Without a model name, cost calculation is skipped entirely, which
        causes the $0 cost bug reported in TraigentFrontend#325.
        """
        # 1. Check if output is a dict with a model field
        if isinstance(output, dict):
            for key in ("model", "model_name", "model_id"):
                val = output.get(key)
                if isinstance(val, str) and val:
                    return val

        # 2. Check output object attributes (e.g. OpenAI ChatCompletion)
        for attr in ("model", "model_name"):
            val = getattr(output, attr, None)
            if isinstance(val, str) and val:
                return val

        # 3. Check captured LangChain response (llm_output.model_name)
        if index < len(captured_responses):
            resp = captured_responses[index]
            llm_output = getattr(resp, "llm_output", None)
            if isinstance(llm_output, dict):
                val = llm_output.get("model_name") or llm_output.get("model")
                if isinstance(val, str) and val:
                    return val

        return None

    @staticmethod
    def _model_inference_pool(
        index: int,
        all_captured_responses: list[Any],
        captures: _CaptureOwnership | None,
    ) -> tuple[int, list[Any]]:
        """(index, responses) ``_infer_model_name_from_output`` should read.

        With capture ownership there is NO sibling pool: a call's reported model
        is never provenance for another call, so only the output's own identity
        applies.  The positional guess over all captures is the legacy path
        (no capture scope) only.
        """
        if captures is None:
            return index, all_captured_responses
        # Ownership mode: a sibling call's reported model is never provenance
        # for another call (#2444); only the output's own identity applies.
        return 0, []

    def _extract_response_text(self, output: Any) -> str | None:
        """Extract text content from various output types.

        Args:
            output: Output from function call (dict, str, or LLM response object)

        Returns:
            Extracted text or None if not extractable
        """
        if isinstance(output, dict):
            # ``with_usage()`` may carry a non-string output under "text"
            # (#2522); that is not response text and has no text length.
            text = output.get("text")
            return text if isinstance(text, str) else None
        elif isinstance(output, str):
            return output
        elif hasattr(output, "content") and output.content:
            first = output.content[0]
            if hasattr(first, "text"):
                return cast(str, first.text)
        elif hasattr(output, "text"):
            return cast(str, output.text)
        return None

    def _extract_response_length(self, output: Any) -> int | None:
        """Extract response length for privacy mode."""
        response_text = self._extract_response_text(output)
        if response_text is None:
            return None
        return len(response_text)

    def _get_metrics_source(
        self,
        output: Any,
        example_index: int,
        dataset: Dataset,
        all_captured_responses: list[Any],
    ) -> Any:
        """Determine the best source for extracting LLM metrics.

        Priority:
        1. If output is a dict with 'raw_response', prefer it (SDK object)
        2. Else use captured LangChain response by correlation key
        3. Else use captured response by index (with multi-call handling)
        4. Else fall back to output itself

        When there are multiple LLM calls per example (e.g., main response + judge),
        this method handles the mismatch by computing aggregate metrics.

        Args:
            output: The function output
            example_index: Current example index
            dataset: Evaluation dataset
            all_captured_responses: List of captured LangChain responses

        Returns:
            The best source object for metrics extraction
        """
        # Prefer raw_response from dict output
        if isinstance(output, dict) and output.get("raw_response") is not None:
            return output["raw_response"]

        # Try correlation key (example_id)
        try:
            ex = dataset.examples[example_index]
            ex_key = _example_correlation_key(ex, example_index)
        except Exception:
            ex_key = f"example_{example_index}"

        by_key = get_captured_response_by_key(ex_key)
        if by_key is not None:
            return by_key

        # Handle case where there are more captured responses than outputs
        # (multiple LLM calls per example, e.g., main response + judge scoring)
        num_outputs = len(dataset.examples)
        num_responses = len(all_captured_responses)

        if num_responses > num_outputs and num_outputs > 0:
            # Calculate how many responses per output (e.g., 2 for main + judge)
            responses_per_output = num_responses // num_outputs
            if responses_per_output >= 1:
                # Get the slice of responses for this example
                start_idx = example_index * responses_per_output
                end_idx = start_idx + responses_per_output
                # Return aggregated metrics wrapper if we have multiple
                if end_idx <= num_responses and responses_per_output > 1:
                    return _AggregatedResponses(
                        all_captured_responses[start_idx:end_idx]
                    )
                elif start_idx < num_responses:
                    return all_captured_responses[start_idx]

        # Fall back to index-based or output itself
        if example_index < len(all_captured_responses):
            return all_captured_responses[example_index]

        return output

    def _estimate_string_tokens(
        self,
        example_metric: ExampleMetrics,
        output: str,
        example_input: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Estimate token counts for string outputs.

        Updates example_metric in place with estimated token counts.
        Uses approximation of 1 token per 4 characters.

        These counts are a guess derived from character length, not a
        measurement -- there was no captured LLM usage for this example
        (that is the only reason this method runs at all, see the caller).
        ``example_metric.tokens.estimated`` is set so downstream consumers
        (trial-level aggregation, cost pricing) never mistake a fabricated
        count for real usage (Traigent#2263). Privacy mode's own
        length-derived cost estimation is a legitimate, intentional use of
        this same approximation -- the flag does not disable it, it only
        stops the estimate from being silently indistinguishable from a
        real one once it lands on a trial.

        Input tokens come from the dataset input or the configured prompt
        template only; a system message written inside the user's function is
        not visible here, so the estimate can be well below the provider's
        count (#2517). Owner ruling: the estimate is kept and always carried
        with the ``estimated`` label; it is never priced as cost.

        Args:
            example_metric: Metrics object to update
            output: String output from function
            example_input: Original input for estimating input tokens
            config: Configuration dictionary (uses ``config["prompt"]`` when
                available for template-aware input token estimation)
        """
        # Estimate output tokens
        example_metric.tokens.output_tokens = max(1, len(output) // 4)
        example_metric.tokens.estimated = True

        # Estimate input tokens from local lengths only. Privacy mode may not
        # retain raw prompts, but it still needs length-derived cost metrics.
        input_length = self._calculate_prompt_template_length(
            example_input, config=config
        )
        if input_length is None:
            input_length = self._calculate_input_length(example_input, config=config)
        example_metric.tokens.input_tokens = max(1, input_length // 4)

        # Update total
        example_metric.tokens.total_tokens = (
            example_metric.tokens.input_tokens + example_metric.tokens.output_tokens
        )

    def _calculate_example_accuracy(
        self,
        actual_value: Any,
        expected_output: Any,
    ) -> float | None:
        """Calculate accuracy for a single example.

        Args:
            actual_value: Actual output value
            expected_output: Expected output value

        Returns:
            Accuracy value (1.0 or 0.0) or None if not calculable
        """
        return _accuracy_value_for_example(actual_value, expected_output)

    def _transfer_token_metrics_to_example_result(
        self,
        example_result: Any,
        example_metric: ExampleMetrics,
    ) -> None:
        """Transfer token and cost metrics to example result.

        Args:
            example_result: ExampleResult object to update
            example_metric: Source metrics
        """
        if example_result is None:
            return

        example_result.metrics["input_tokens"] = example_metric.tokens.input_tokens
        example_result.metrics["output_tokens"] = example_metric.tokens.output_tokens
        example_result.metrics["total_tokens"] = example_metric.tokens.total_tokens
        if not example_metric.cost_measured:
            # Unknown cost (no usage, or usage no price covers): leave the cost
            # keys out. Unknown is not $0 and a downstream sum over example
            # costs must not see a zero (#2517); carry the flag and the exact
            # known floor instead (#2444).
            enforce_unknown_cost_invariant(
                example_result.metrics,
                unknown=True,
                floor=self._example_cost_floor(example_metric),
            )
            return
        example_result.metrics["input_cost"] = example_metric.cost.input_cost
        example_result.metrics["output_cost"] = example_metric.cost.output_cost
        example_result.metrics["total_cost"] = example_metric.cost.total_cost

    def _objective_returned_none_error(
        self,
        metric_name: str,
        example_id: str,
        example_index: int,
        config: dict[str, Any],
    ) -> EvaluationError:
        """Build the fail-closed error for an objective metric returning None.

        Companion to the exception-path guard in ``_invoke_metric_function``
        (which raises when an objective metric *raises*). A ``None`` return is
        not an exception, so without this an optimization objective's ``None``
        would be substituted with a fabricated ``0.0`` and silently corrupt the
        search. Callers raise the returned error only for objective metrics
        (``metric_name in self.metrics``); non-objective metrics retain the
        legacy ``0.0``/skip behaviour.
        """
        return EvaluationError(
            f"Objective metric '{metric_name}' returned None for example "
            f"{example_id}. Refusing to substitute a fabricated 0.0 score for "
            "an optimization objective.",
            config=config,
            details={
                "metric_name": metric_name,
                "example_id": example_id,
                "example_index": example_index,
                "is_objective": True,
                # Mirror the exception-path record shape (error_type) so
                # downstream consumers parse both failure modes uniformly.
                "error_type": "NoneReturn",
                "failure_mode": "returned_none",
            },
        )

    def _handle_metric_function_exception(
        self,
        exc: Exception,
        metric_name: str,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> float:
        example_id = (
            example_obj.metadata.get("example_id", f"example_{example_index}")
            if example_obj.metadata
            else f"example_{example_index}"
        )
        logger.warning(
            "Metric function %s failed for example %s: %s",
            metric_name,
            example_id,
            exc,
        )
        # ``self.metrics`` is this evaluator's configured objective/metric
        # list (every real construction path -- optimization_pipeline's
        # _create_local_evaluator, cloud/client.py, cli/optimization_validator
        # -- builds it directly from the optimization's ``objectives``).
        # A metric named here is an optimization objective; a fabricated
        # 0.0 for it would silently pin/corrupt the search, so fail the
        # trial closed instead of scoring it.
        is_objective = metric_name in self.metrics
        error_record: dict[str, Any] = {
            "metric_name": metric_name,
            "example_id": example_id,
            "example_index": example_index,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "is_objective": is_objective,
        }
        if is_objective:
            raise EvaluationError(
                f"Objective metric '{metric_name}' raised "
                f"{type(exc).__name__} for example {example_id}: {exc}. "
                "Refusing to substitute a fabricated 0.0 score for an "
                "optimization objective.",
                config=config,
                original_error=exc,
                details=error_record,
            ) from exc
        if metric_errors is not None:
            metric_errors.append(error_record)
        return 0.0

    async def _resolve_metric_function_value(
        self,
        value: Any,
        metric_name: str,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            return await value
        except Exception as exc:
            return self._handle_metric_function_exception(
                exc,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )

    async def _apply_custom_metric_functions(
        self,
        example_metric: ExampleMetrics,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
        captures: _CaptureOwnership | None = None,
    ) -> list[str]:
        """Apply user-defined metric functions for a single example.

        Updates example_metric.custom_metrics in place.

        Args:
            example_metric: Metrics object to update
            output: Function output
            example_obj: Dataset example object
            config: Configuration parameters
            example_index: Current example index
            metric_errors: Optional accumulator for structured degradation
                records from non-objective metric failures; forwarded to
                ``_invoke_metric_function``. An objective failure instead
                raises ``EvaluationError`` and propagates out of this call.

        Returns:
            The custom-metric keys actually written to
            ``example_metric.custom_metrics`` by this call. For a scalar-
            returning function this is the function's own name; for a
            mapping-returning function it is the produced sub-keys (never the
            function name). Callers use this to transfer the produced metrics
            downstream without assuming a key exists for every function name
            (a mapping-returning function never populates ``custom_metrics``
            under its own name).
        """
        cost_known = example_metric.cost_measured
        llm_payload = {
            "total_tokens": example_metric.tokens.total_tokens,
            "prompt_tokens": example_metric.tokens.input_tokens,
            "completion_tokens": example_metric.tokens.output_tokens,
            # Unknown cost is None for a metric function, never a placeholder
            # or partial figure that reads as spend (#2444).
            "total_cost": example_metric.cost.total_cost if cost_known else None,
            "input_cost": example_metric.cost.input_cost if cost_known else None,
            "output_cost": example_metric.cost.output_cost if cost_known else None,
            "response_time_ms": example_metric.response.response_time_ms,
            "cost_unmeasured": 0.0 if cost_known else 1.0,
            "cost_lower_bound": self._example_cost_floor(example_metric)
            if not cost_known
            else example_metric.cost.total_cost,
        }

        example_id = (
            example_obj.metadata.get("example_id", f"example_{example_index}")
            if example_obj.metadata
            else f"example_{example_index}"
        )
        produced_keys: list[str] = []
        for metric_name, metric_func in self.metric_functions.items():
            raw_value = self._invoke_metric_function(
                metric_func,
                metric_name,
                output,
                example_obj,
                config,
                llm_payload,
                example_index,
                metric_errors=metric_errors,
            )
            value = await self._resolve_metric_function_value(
                raw_value,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )
            # A ``None`` return is NOT an exception, so it bypasses the
            # objective fail-closed guard in ``_invoke_metric_function``.
            # Mirror that guard here: an objective metric that returns
            # ``None`` (scalar) or omits its value inside a returned mapping
            # would otherwise be coerced to a fabricated ``0.0`` / silently
            # dropped, pinning and corrupting the search. Non-objective
            # metrics keep the legacy ``0.0``/skip behaviour.
            if isinstance(value, Mapping):
                sub_items = list(value.items())
                _sub_index = -1
                try:
                    for _sub_index, (result_name, result_value) in enumerate(sub_items):
                        # A mapping sub-value can itself be awaitable (e.g.
                        # ``{"quality": async_score(...)}``); resolve it the same way
                        # as a top-level async metric so it never reaches float() as
                        # a raw coroutine (and its exceptions get the objective /
                        # degradation-record handling, keyed by the sub-metric name).
                        result_value = await self._resolve_metric_function_value(
                            result_value,
                            str(result_name),
                            example_obj,
                            config,
                            example_index,
                            metric_errors=metric_errors,
                        )
                        if result_value is None:
                            if str(result_name) in self.metrics:
                                raise self._objective_returned_none_error(
                                    str(result_name),
                                    example_id,
                                    example_index,
                                    config,
                                )
                            if metric_errors is not None:
                                metric_errors.append(
                                    {
                                        "metric_name": str(result_name),
                                        "example_id": example_id,
                                        "example_index": example_index,
                                        "error_type": "NoneReturn",
                                        "failure_mode": "returned_none",
                                        "is_objective": False,
                                    }
                                )
                            continue
                        key = str(result_name)
                        example_metric.custom_metrics[key] = float(result_value)
                        produced_keys.append(key)
                except BaseException:
                    # An objective sub-value that raised (or returned None,
                    # which raises the guard above) aborts this loop early.
                    # LATER sub-values that are still-unresolved coroutines
                    # would otherwise never be awaited or closed by anything
                    # else, leaking an unawaited-coroutine warning (and
                    # skipping the coroutine's own cleanup). Close them before
                    # propagating.
                    for _, pending_value in sub_items[_sub_index + 1 :]:
                        if inspect.iscoroutine(pending_value):
                            pending_value.close()
                    raise
                continue
            if value is None:
                if metric_name in self.metrics:
                    raise self._objective_returned_none_error(
                        metric_name,
                        example_id,
                        example_index,
                        config,
                    )
                if metric_errors is not None:
                    metric_errors.append(
                        {
                            "metric_name": metric_name,
                            "example_id": example_id,
                            "example_index": example_index,
                            "error_type": "NoneReturn",
                            "failure_mode": "returned_none",
                            "is_objective": False,
                        }
                    )
                example_metric.custom_metrics[metric_name] = 0.0
            else:
                example_metric.custom_metrics[metric_name] = float(value)
            produced_keys.append(metric_name)
        return produced_keys

    def _build_metric_keyword_arguments(
        self,
        parameters: list[inspect.Parameter],
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        llm_payload: dict[str, Any],
        example_index: int,
    ) -> dict[str, Any]:
        return build_metric_keyword_arguments(
            parameters, output, example_obj, config, llm_payload, example_index
        )

    def _invoke_metric_function(
        self,
        metric_func: Callable[..., Any],
        metric_name: str,
        output: Any,
        example_obj: Any,
        config: dict[str, Any],
        llm_payload: dict[str, Any],
        example_index: int,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Invoke a single metric function with appropriate arguments.

        Args:
            metric_func: The metric function to call
            metric_name: Name of the metric (for logging)
            output: Function output
            example_obj: Dataset example
            config: Configuration parameters
            llm_payload: LLM metrics payload
            example_index: Current example index
            metric_errors: Optional accumulator for structured degradation
                records (``{metric_name, example_id, example_index,
                error_type, error_message, is_objective}``) for
                NON-objective metric failures. Callers that don't need to
                surface these (e.g. the best-effort progress-preview lane
                in ``BaseEvaluator``) may omit it; the failure is still
                logged either way.

        Returns:
            Metric value on success, an awaitable metric value for async metric
            functions, or ``0.0`` for a failed *informational* metric -- never
            a bare, indistinguishable 0.0. A structured error record is
            appended to ``metric_errors`` (when provided) so a caller can tell
            a real 0.0 apart from a degraded one.

        Raises:
            EvaluationError: If ``metric_name`` is one of this evaluator's
                configured objectives (``self.metrics``) and the metric
                function raises. Substituting a fabricated 0.0 for an
                optimization objective would silently corrupt the search,
                so the trial fails closed instead.
        """
        try:
            binding = resolve_metric_call_binding(
                metric_func, output, example_obj, config, llm_payload, example_index
            )
            if binding.bind_ok:
                return metric_func(*binding.args, **binding.kwargs)
            if binding.bind_exception is not None:
                raise binding.bind_exception
            return metric_func(output, example_obj.expected_output)
        except Exception as exc:
            return self._handle_metric_function_exception(
                exc,
                metric_name,
                example_obj,
                config,
                example_index,
                metric_errors=metric_errors,
            )

    def _build_local_progress_payload(
        self,
        example_metric: ExampleMetrics,
        example_result: Any,
        output: Any,
    ) -> dict[str, Any]:
        """Build payload for progress callback in local evaluation mode.

        Note: This method has a different signature from the base class
        _build_progress_payload to support LocalEvaluator's specific needs.

        Args:
            example_metric: Current example metrics
            example_result: Example result (if in detailed mode)
            output: Function output

        Returns:
            Payload dictionary for progress callback
        """
        payload_metrics = dict(example_metric.custom_metrics)
        total_cost_value = example_metric.cost.total_cost

        if not example_metric.cost_measured:
            # Unknown cost: no $0 in the progress payload (#2517).
            for cost_key in ("input_cost", "output_cost", "total_cost", "cost"):
                payload_metrics.pop(cost_key, None)
            payload_metrics["cost_unmeasured"] = 1.0
            floor = LocalEvaluator._example_cost_floor(example_metric)
            if floor > 0:
                payload_metrics["cost_lower_bound"] = floor
        elif total_cost_value is not None:
            payload_metrics.setdefault("total_cost", float(total_cost_value))
            payload_metrics.setdefault("cost", float(total_cost_value))

        if self.detailed and example_result is not None:
            for key, value in example_result.metrics.items():
                if key in {"cost", "total_cost"} and key in payload_metrics:
                    continue
                if not example_metric.cost_measured and key in {
                    "input_cost",
                    "output_cost",
                    "total_cost",
                    "cost",
                }:
                    continue
                payload_metrics.setdefault(key, value)

        if not example_metric.cost_measured:
            enforce_unknown_cost_invariant(
                payload_metrics,
                unknown=True,
                floor=LocalEvaluator._example_cost_floor(example_metric),
            )

        return {
            "success": example_metric.success,
            "error": example_metric.error,
            "metrics": payload_metrics,
            "output": output,
        }

    def _extract_response_metrics(
        self,
        example_id: str,
        example: Any,
        result: Any,
        config: dict[str, Any],
        *,
        dataset: Dataset | None = None,
        example_index: int = 0,
    ) -> tuple[dict[str, float], Any]:
        """Remember which row cost values the tentative progress pass changed."""
        before = {
            k: result.metrics[k] for k in _PREVIEW_COST_KEYS if k in result.metrics
        }
        extracted = super()._extract_response_metrics(
            example_id,
            example,
            result,
            config,
            dataset=dataset,
            example_index=example_index,
        )
        after = {
            k: result.metrics[k] for k in _PREVIEW_COST_KEYS if k in result.metrics
        }
        result._sdk_preview_cost_evidence = (before, after)
        return extracted

    def _extract_progress_call_metrics(
        self,
        example_id: str,
        example: Any,
        result: Any,
        config: dict[str, Any],
        dataset: Dataset | None,
        example_index: int,
    ) -> ExampleMetrics | None:
        """Preview the same per-call accounting the final pass will consume.

        Snapshot under the capture lock, retain every sibling and its owner,
        and consume only this temporary ownership view, never the live bucket.
        """
        responses, owners, requested = snapshot_captured_responses()
        captures = _CaptureOwnership.build(responses, owners, requested)
        if captures is None or dataset is None:
            return super()._extract_progress_call_metrics(
                example_id, example, result, config, dataset, example_index
            )
        history = _progress_capture_history.get()
        if history is not None:
            captures.returned_seen = history.returned_seen.copy()
            captures.captured_ids.update(history.captured_ids)
        metrics = self._extract_llm_metrics_for_output(
            result.actual_output, example_index, config, dataset, responses, captures
        )
        if history is not None:
            history.returned_seen.update(captures.returned_seen)
            history.captured_ids.update(captures.captured_ids)
        return metrics

    def _extract_llm_metrics_for_output(
        self,
        output: Any,
        index: int,
        config: dict[str, Any],
        dataset: Dataset,
        all_captured_responses: list[Any],
        captures: _CaptureOwnership | None = None,
    ) -> ExampleMetrics:
        """Extract LLM metrics for a single output.

        Args:
            output: Function output
            index: Example index
            config: Configuration parameters
            dataset: Evaluation dataset
            all_captured_responses: Captured LangChain responses

        Returns:
            ExampleMetrics with extracted or estimated metrics
        """
        example_metric = ExampleMetrics()

        if output is None:
            # The decorated function raised before producing any output at
            # all (or a custom evaluator explicitly returned no output) --
            # nothing was ever extracted, so every field below stays a
            # placeholder default. Mark `measured=False` so aggregation
            # (`MetricsTracker.aggregate_metrics`/`format_for_backend`/
            # `format_as_summary_stats`, `BaseEvaluator._compute_cost`)
            # excludes this row's zeros from MEAN/median/std cost, token, and
            # response-time statistics instead of averaging them in as if
            # they were a real (measured) zero -- see
            # `ExampleMetrics.measured` for the full rationale (Traigent#2160
            # sol re-review, third DO_NOT_MERGE round).
            example_metric.measured = False
            return example_metric

        model_name = config.get("model") or config.get("model_name")
        configured_model = model_name if isinstance(model_name, str) else None

        # Fallback: try to extract model name from the output or captured responses
        # when the optimization config does not include "model" as a parameter.
        if not model_name:
            model_name = self._infer_model_name_from_output(
                output,
                *self._model_inference_pool(index, all_captured_responses, captures),
            )

        logger.debug(
            f"EVALUATOR DEBUG: i={index}, output type={type(output).__name__}, "
            f"model_name='{model_name}'"
        )

        original_prompt = None
        prompt_length = None
        response_length = None

        # Extract prompt info using helper method
        if index < len(dataset.examples):
            example_input = dataset.examples[index].input_data
            prompt_info = self._extract_prompt_info(
                example_input, output, config=config
            )
            original_prompt = prompt_info.original_prompt
            prompt_length = prompt_info.prompt_length
            response_length = prompt_info.response_length

        # Determine response length without passing content to metric extraction.
        response_text = self._extract_response_text(output)
        if response_text is not None and response_length is None:
            response_length = len(response_text)

        if captures is not None:
            extracted_metrics = self._extract_owned_call_metrics(
                output,
                index,
                dataset,
                captures,
                model_name=model_name,
                configured_model=configured_model,
                original_prompt=original_prompt,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        else:
            # Get best metrics source using helper
            metrics_source = self._get_metrics_source(
                output, index, dataset, all_captured_responses
            )

            # Pass lengths to extract_llm_metrics for privacy mode
            extracted_metrics = extract_llm_metrics(
                response=metrics_source,
                model_name=model_name,
                original_prompt=original_prompt,
                response_text=None,
                prompt_length=prompt_length,
                response_length=response_length,
            )

        logger.debug(
            f"Extracted metrics for model {model_name}: "
            f"tokens={extracted_metrics.tokens.total_tokens}, "
            + (
                f"cost=${extracted_metrics.cost.total_cost:.8f}"
                if extracted_metrics.cost_measured
                else "cost=unknown"
            )
        )

        example_metric = extracted_metrics

        # If we didn't get token metrics but output is a string, estimate tokens
        if extracted_metrics.tokens.total_tokens == 0 and isinstance(output, str):
            example_input_for_tokens = (
                dataset.examples[index].input_data
                if index < len(dataset.examples)
                else None
            )
            self._estimate_string_tokens(
                example_metric, output, example_input_for_tokens, config=config
            )

        return example_metric

    @staticmethod
    def _looks_like_usage_response(obj: Any) -> bool:
        """True when the tracker's own handler chain finds usage on ``obj``.

        Derived from ``ResponseHandlerFactory`` rather than a second hand-kept
        list of shapes, so every usage shape ``extract_llm_metrics`` supports
        (OpenAI/Anthropic/LangChain objects, usage dicts, generic
        ``metadata["usage"|"token_usage"]``) takes the same per-call path.
        """
        if obj is None or isinstance(obj, (str, bytes, list, tuple, int, float)):
            return False
        try:
            extracted = ResponseHandlerFactory.create_handler_chain().handle(obj)
            return bool(
                extracted is not None
                and (
                    extracted.tokens.input_tokens > 0
                    or extracted.tokens.output_tokens > 0
                    or extracted.cost.total_cost > 0
                )
            )
        except Exception:  # noqa: BLE001 - recognition must never raise
            return False

    @staticmethod
    def _example_cost_floor(metric: ExampleMetrics) -> float:
        """Exact known spend of a row whose total is unknown (a floor only)."""
        priced = 0.0 if metric.cost.unmeasured else float(metric.cost.total_cost)
        return priced + float(metric.cost.known_cost)

    @staticmethod
    def _call_cost_floor(call: ExampleMetrics) -> float:
        """Exact known spend of one extracted call (priced part, never a guess)."""
        if call.cost.unmeasured:
            return float(call.cost.known_cost)
        return float(call.cost.total_cost) + float(call.cost.known_cost)

    @staticmethod
    def _resolve_call_model(
        response: Any,
        requested_model: str | None,
        agent_model: str | None,
        is_agent_call: bool,
    ) -> str | None:
        """Model identity of one call: response, then its request, then config."""
        return (
            infer_response_model(response)
            or requested_model
            or (agent_model if is_agent_call else None)
        )

    @staticmethod
    def _price_one_call(
        response: Any,
        *,
        agent_model: str | None,
        is_agent_call: bool,
        requested_model: str | None = None,
        **prompt_kwargs: Any,
    ) -> ExampleMetrics:
        """Extract ONE captured call on its own (own model, charge, usage).

        Model identity, strongest first: the model the response reports, then
        the model THAT call requested (recorded at interception), then -- only
        for the call established as the agent's -- the configured model.  A call
        with none of these has an unknown cost: never $0 and never the agent's
        price (#2444).
        """
        model = LocalEvaluator._resolve_call_model(
            response, requested_model, agent_model, is_agent_call
        )
        metrics = extract_llm_metrics(
            response=response,
            model_name=model,
            response_text=None,
            **(prompt_kwargs if is_agent_call else {}),
        )
        if (
            model is None
            and metrics.tokens.total_tokens > 0
            and not metrics.cost.cost_explicit
            and not metrics.cost.unmeasured
            and not metrics.cost.total_cost
        ):
            metrics.cost.unpriced = True
        return metrics

    @staticmethod
    def _combine_call_metrics(calls: list[ExampleMetrics]) -> ExampleMetrics:
        """Sum separately-extracted calls (first is the primary) into one row.

        Costs and tokens add; unknown-cost state is preserved rather than
        merged away: any call without usage makes the row unmeasured and any
        unpriceable call makes it unpriced, with the priced part kept as the
        known-spend floor.
        """
        if len(calls) == 1:
            return calls[0]
        out = copy.deepcopy(calls[0])
        out.tokens.input_tokens = sum(c.tokens.input_tokens for c in calls)
        out.tokens.output_tokens = sum(c.tokens.output_tokens for c in calls)
        out.tokens.total_tokens = sum(c.tokens.total_tokens for c in calls)
        out.tokens.estimated = any(c.tokens.estimated for c in calls)
        out.cost.input_cost = sum(c.cost.input_cost for c in calls)
        out.cost.output_cost = sum(c.cost.output_cost for c in calls)
        out.cost.total_cost = sum(c.cost.total_cost for c in calls)
        out.cost.unpriced = any(c.cost.unpriced for c in calls)
        out.cost.unmeasured = any(c.cost.unmeasured for c in calls)
        out.cost.cost_estimated = any(c.cost.cost_estimated for c in calls)
        out.cost.cost_explicit = all(c.cost.cost_explicit for c in calls)
        out.cost.known_cost = sum(c.cost.known_cost for c in calls)
        if out.cost.unmeasured:
            out.cost.known_cost += sum(
                c.cost.total_cost for c in calls if not c.cost.unmeasured
            )
        out.measured = any(c.measured for c in calls)
        out.call_breakdown = [b for c in calls for b in c.call_breakdown]
        return out

    def _extract_owned_call_metrics(
        self,
        output: Any,
        index: int,
        dataset: Dataset,
        captures: _CaptureOwnership,
        *,
        model_name: str | None,
        configured_model: str | None,
        original_prompt: Any,
        prompt_length: int | None,
        response_length: int | None,
    ) -> ExampleMetrics:
        """Price exactly the calls this example made, each on its own (#2444).

        A returned ``raw_response`` / response object consumes only the
        captured call that IS that object (identity); the example's other
        calls still count, each at its own model.  Consumption is recorded per
        call so nothing is dropped or counted twice.
        """
        if isinstance(output, dict) and output.get("raw_response") is not None:
            returned: Any = output["raw_response"]
        elif self._looks_like_usage_response(output):
            returned = output
        else:
            returned = None
        prompt_kwargs = {
            "original_prompt": original_prompt,
            "prompt_length": prompt_length,
            "response_length": response_length,
        }
        returned_pos = captures.find(
            returned if returned is not None else output, index
        )
        # Already accounted elsewhere (another owner's capture, a consumed
        # entry, or a returned object priced for an earlier example): this
        # example made no new call with it, so it is never charged again.
        already_accounted = (
            returned is not None
            and returned_pos is None
            and (
                id(returned) in captures.captured_ids
                or id(returned) in captures.returned_seen
            )
        )
        if already_accounted:
            returned = None
        owned = [p for p in captures.owned_untaken(index) if p != returned_pos]
        uncaptured = returned if returned_pos is None else None
        primary: Any = None
        primary_requested: str | None = None
        if returned_pos is not None:
            captures.taken.add(returned_pos)
            primary = captures.responses[returned_pos]
            primary_requested = captures.requested[returned_pos]
        elif returned is not None and not owned:
            primary = returned  # no other call of this example: it IS the agent's
        sole_call = primary is None and uncaptured is None and len(owned) == 1

        if uncaptured is not None:
            captures.returned_seen.add(id(uncaptured))
        resolved_models: list[str | None] = []
        calls: list[ExampleMetrics] = []
        if primary is not None:
            resolved_models.append(
                self._resolve_call_model(
                    primary, primary_requested, configured_model, True
                )
            )
            calls.append(
                self._price_one_call(
                    primary,
                    agent_model=configured_model,
                    is_agent_call=True,
                    requested_model=primary_requested,
                    **prompt_kwargs,
                )
            )
        for position in owned:
            captures.taken.add(position)
            resolved_models.append(
                self._resolve_call_model(
                    captures.responses[position],
                    captures.requested[position],
                    configured_model,
                    sole_call,
                )
            )
            calls.append(
                self._price_one_call(
                    captures.responses[position],
                    agent_model=configured_model,
                    is_agent_call=sole_call,
                    requested_model=captures.requested[position],
                    **prompt_kwargs,
                )
            )
        if uncaptured is not None and owned:
            # A returned response whose relationship to this example's captured
            # calls cannot be established (the same call, or another one): its
            # usage is never ignored and never added blindly.  The row is
            # unknown, with the largest spend both readings agree on.
            returned_call = self._price_one_call(
                uncaptured,
                agent_model=configured_model,
                is_agent_call=False,
                **{},
            )
            combined = self._combine_call_metrics(calls)
            floor = max(
                sum(self._call_cost_floor(c) for c in calls),
                self._call_cost_floor(returned_call),
            )
            combined.cost.unmeasured = True
            combined.cost.known_cost = floor
            return combined
        captures.reconcile_model[index] = (
            resolved_models[0] if len(calls) == 1 and uncaptured is None else None
        )
        if not calls and already_accounted:
            # Its only usage is a response some other row already accounts for:
            # no new spend evidence here, so this row's cost is unknown.
            unaccounted = ExampleMetrics()
            unaccounted.cost.unmeasured = True
            return unaccounted
        if not calls:
            # No call of this example was captured: the output itself is the
            # only source (never another example's call).
            return extract_llm_metrics(
                response=output,
                model_name=model_name,
                original_prompt=original_prompt,
                response_text=None,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        return self._combine_call_metrics(calls)

    def _update_example_metric_from_result(
        self,
        example_metric: ExampleMetrics,
        example_result: Any,
        index: int,
    ) -> None:
        """Update example metric with data from example result.

        Args:
            example_metric: Metrics to update (modified in place)
            example_result: Source example result
            index: Example index
        """
        if example_result is None:
            return

        # Transfer execution time to response_time_ms (convert seconds to milliseconds)
        if hasattr(example_result, "execution_time"):
            example_metric.response.response_time_ms = (
                example_result.execution_time * 1000
            )

        # Transfer computed metrics (like accuracy) to custom_metrics
        if hasattr(example_result, "metrics") and example_result.metrics:
            # Final dataset-order ownership can differ from completion-order
            # previews. Undo only preview changes still present on this row;
            # preserve authoritative evidence supplied before or after them.
            preview = getattr(example_result, "_sdk_preview_cost_evidence", None)
            if preview is not None:
                before, after = preview
                missing = object()
                for key in before.keys() | after.keys():
                    previous = before.get(key, missing)
                    written = after.get(key, missing)
                    if (
                        previous == written
                        or example_result.metrics.get(key, missing) != written
                    ):
                        continue
                    if key in before:
                        example_result.metrics[key] = previous
                    else:
                        example_result.metrics.pop(key, None)
                del example_result._sdk_preview_cost_evidence
            example_metric.custom_metrics.update(example_result.metrics)
            if example_result.metrics.get("cost_unmeasured"):
                # The evaluator's floor may overlap captured spend: their
                # relationship is unknown, so take max, never add them.
                example_metric.cost.known_cost = max(
                    self._example_cost_floor(example_metric),
                    float(example_result.metrics.get("cost_lower_bound") or 0.0),
                )
                example_metric.cost.unmeasured = True

        # Transfer token/cost metrics to example_results if in detailed mode
        if self.detailed:
            self._transfer_token_metrics_to_example_result(
                example_result, example_metric
            )

    def _add_standard_metrics_to_custom(self, example_metric: ExampleMetrics) -> None:
        """Add standard token/cost metrics to custom_metrics.

        Args:
            example_metric: Metrics object to update (modified in place)
        """
        example_metric.custom_metrics["input_tokens"] = (
            example_metric.tokens.input_tokens
        )
        example_metric.custom_metrics["output_tokens"] = (
            example_metric.tokens.output_tokens
        )
        example_metric.custom_metrics["total_tokens"] = (
            example_metric.tokens.total_tokens
        )
        if not example_metric.cost_measured:
            return
        example_metric.custom_metrics["input_cost"] = example_metric.cost.input_cost
        example_metric.custom_metrics["output_cost"] = example_metric.cost.output_cost
        example_metric.custom_metrics["total_cost"] = example_metric.cost.total_cost

    def _inject_usage_from_meta(
        self, usage: dict[str, Any], metrics: ExampleMetrics
    ) -> None:
        """Inject usage data (tokens, response time) from __traigent_meta__.

        Args:
            usage: Usage dict from __traigent_meta__ with optional keys:
                input_tokens, output_tokens, response_time_ms.
            metrics: ExampleMetrics to update.
        """
        # Inject token counts (key-presence checks allow explicit zeros)
        for key, attr in [
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ]:
            if key not in usage:
                continue
            try:
                val = usage[key]
                if val < 0:
                    logger.warning(f"Negative {key} clamped: {val} → 0")
                setattr(metrics.tokens, attr, max(0, val))
            except Exception as e:
                logger.error(f"Failed to inject {key}: {e}", extra={"usage": usage})

        # Recompute total_tokens if any token counts were injected
        if "input_tokens" in usage or "output_tokens" in usage:
            try:
                metrics.tokens.total_tokens = (
                    metrics.tokens.input_tokens + metrics.tokens.output_tokens
                )
            except Exception as e:
                logger.error(f"Failed to compute total_tokens: {e}")

        # Inject response_time_ms if provided
        if "response_time_ms" not in usage:
            return
        try:
            response_time = usage["response_time_ms"]
            if response_time < 0:
                logger.warning(
                    f"Negative response_time_ms clamped: {response_time} → 0.0"
                )
            metrics.response.response_time_ms = max(0.0, float(response_time))
        except Exception as e:
            logger.error(f"Failed to inject response_time_ms: {e}")

    def _inject_call_breakdown_from_meta(
        self, calls: list[Any], metrics: ExampleMetrics
    ) -> None:
        """Inject the per-call/per-model cost breakdown from ``__traigent_meta__``.

        Multi-model/multi-step agents (cascade, router) can report a
        ``"calls"`` list -- one entry per LLM call, attributing tokens/cost to
        the model that made that specific call -- alongside the existing
        blended ``total_cost``/``usage`` (Traigent#1598). Each entry is
        defensively normalized the same way ``_inject_usage_from_meta`` does:
        negative values are clamped and a malformed entry is skipped (logged),
        never allowed to raise and drop the whole example's measurement.

        Args:
            calls: Raw ``calls`` list from ``__traigent_meta__``.
            metrics: ExampleMetrics to update (``call_breakdown`` field).
        """
        normalized: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                logger.error(f"Skipping non-dict call breakdown entry: {call!r}")
                continue
            try:
                model = call.get("model")
                if not isinstance(model, str) or not model:
                    logger.error(
                        f"Skipping call breakdown entry with no model: {call!r}"
                    )
                    continue
                input_tokens = int(call.get("input_tokens", 0) or 0)
                output_tokens = int(call.get("output_tokens", 0) or 0)
                cost = float(call.get("cost", 0.0) or 0.0)
                if input_tokens < 0 or output_tokens < 0 or cost < 0:
                    logger.warning(
                        f"Negative value(s) clamped to 0 in call breakdown entry "
                        f"for model {model!r}"
                    )
                normalized.append(
                    {
                        "model": model,
                        "input_tokens": max(0, input_tokens),
                        "output_tokens": max(0, output_tokens),
                        "cost": max(0.0, cost),
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed to inject call breakdown entry: {e}", extra={"call": call}
                )

        metrics.call_breakdown = normalized

    def _extract_and_inject_traigent_meta(
        self,
        output: Any,
        metrics: ExampleMetrics,
        model_name: str | None = None,
    ) -> TraigentMetadata | None:
        """Extract __traigent_meta__ from output and inject into metrics.

        This method runs AFTER cost calculation to allow user-provided costs
        to override SDK-calculated values when plausible.

        Args:
            output: Raw function output (may be dict with __traigent_meta__).
            metrics: ExampleMetrics to update with user-provided values.
            model_name: Optional model name for best-effort token-cost plausibility
                checks. Without a model or known pricing, reported BYOK costs are
                accepted as-is.

        Returns:
            The meta dict if found, None otherwise.
        """
        from traigent.core.meta_types import (
            TraigentMetadata,
            is_traigent_metadata,
            is_valid_call_breakdown,
        )

        if not isinstance(output, dict):
            return None

        meta = output.get("__traigent_meta__")
        if meta is None:
            return None

        # Sever a malformed `calls` before validating the envelope. `calls` is
        # ATTRIBUTION (which model spent it); `total_cost` is the authoritative
        # amount. Letting a bad attribution entry invalidate the whole envelope
        # threw away a valid total_cost and the run under-reported spend -- which
        # is fail-OPEN for a budget, whatever the old comment called it. Drop the
        # attribution, keep the money.
        if isinstance(meta, dict) and "calls" in meta:
            if not is_valid_call_breakdown(meta["calls"]):
                logger.error(
                    "Invalid __traigent_meta__['calls'] attribution; dropping the "
                    "per-model breakdown and keeping the reported total_cost.",
                    extra={"calls": meta["calls"]},
                )
                meta = {k: v for k, v in meta.items() if k != "calls"}

        if not is_traigent_metadata(meta):
            logger.error(
                "Invalid __traigent_meta__ structure",
                extra={
                    "meta": meta,
                    "expected_keys": ["total_cost (required)", "usage (optional)"],
                    "validation": "Type guard failed",
                },
            )
            return None

        meta = cast(TraigentMetadata, meta)
        logger.debug(f"Validated __traigent_meta__ with keys: {meta.keys()}")

        # Inject usage data (tokens, response time)
        if "usage" in meta:
            self._inject_usage_from_meta(cast(dict, meta["usage"]), metrics)

        # Inject per-call/per-model cost breakdown for multi-model agents (#1598)
        if "calls" in meta:
            self._inject_call_breakdown_from_meta(cast(list, meta["calls"]), metrics)

        # A validated ``__traigent_meta__`` carries a user-reported cost (and
        # optionally usage): that is a real measurement, reported by the agent
        # rather than extracted from a response object, so it counts as
        # captured usage for the cost-objective guard.
        from traigent.utils.cost_calculator import record_captured_usage

        record_captured_usage()

        # Inject cost
        try:
            total_cost = meta["total_cost"]
            if total_cost < 0:
                logger.warning(f"Negative total_cost clamped: {total_cost} → 0.0")
            metrics.cost.total_cost = max(0.0, float(total_cost))
            # The agent reported its own spend: a measurement, even if $0 (#2517).
            metrics.cost.unmeasured = False
            # The authoritative total covers every call of this row, so any
            # floor/unpriced state accumulated for them is now included in it:
            # keeping it would count the same spend twice in the tracker.
            metrics.cost.known_cost = 0.0
            metrics.cost.unpriced = False
            if model_name:
                from traigent.evaluators.metrics_tracker import (
                    _reconcile_reported_cost_with_tokens,
                )

                _reconcile_reported_cost_with_tokens(metrics, model_name)
        except Exception as e:
            logger.error(f"Failed to inject total_cost: {e}")

        # Recompute derived metrics
        try:
            if metrics.tokens.total_tokens == 0:
                metrics.response.tokens_per_second = 0.0
            else:
                MetricsCalculator.calculate_tokens_per_second(metrics)
        except Exception as e:
            logger.error(f"Failed to recompute tokens_per_second: {e}")

        logger.debug(
            f"Injected cost=${metrics.cost.total_cost:.4f}, "
            f"tokens={metrics.tokens.total_tokens}"
        )

        return meta

    async def _process_single_output(
        self,
        output: Any,
        index: int,
        config: dict[str, Any],
        dataset: Dataset,
        errors: list[str | None],
        example_results: list[Any] | None,
        all_captured_responses: list[Any],
        progress_callback: Callable[[int, dict[str, Any]], Any] | None,
        *,
        metric_errors: list[dict[str, Any]] | None = None,
        captures: _CaptureOwnership | None = None,
    ) -> ExampleMetrics:
        """Process metrics for a single output.

        Args:
            output: Function output
            index: Example index
            config: Configuration parameters
            dataset: Evaluation dataset
            errors: List of error messages
            example_results: List of example results (may be None)
            all_captured_responses: Captured responses
            progress_callback: Optional progress callback
            metric_errors: Optional accumulator for structured degradation
                records from non-objective custom metric failures for this
                evaluation run; forwarded to ``_apply_custom_metric_functions``.

        Returns:
            ExampleMetrics for this output
        """
        # Per-example user metrics: prefer the carrier the boundary already
        # attached to the ExampleResult (detailed mode). When the carrier is
        # present the output was ALREADY unpacked upstream, so re-unpacking here
        # would double-unpack a nested matching tuple — e.g. an output of
        # ``("inner", {"x": 1.0})`` would be split again into ``"inner"`` and a
        # spurious ``{"x": 1.0}``. We therefore unpack HERE only when the
        # carrier is absent (the genuinely non-detailed lane, where ``outputs``
        # still holds the raw tuple). Any non-matching shape is left untouched.
        carrier = None
        if (
            example_results
            and index < len(example_results)
            and example_results[index] is not None
            and getattr(example_results[index], "user_metrics", None) is not None
        ):
            carrier = example_results[index]

        if carrier is not None:
            user_metrics = carrier.user_metrics
        else:
            output, user_metrics = self._unpack_user_metrics(output)

        # Extract LLM metrics
        example_metric = self._extract_llm_metrics_for_output(
            output, index, config, dataset, all_captured_responses, captures
        )

        if captures is not None:
            # Reconcile reported cost only against the model that actually
            # priced this example's single call (response model first); a
            # multi-call row has no single model, so it is never repriced as one
            # configured model (#2444).
            model_name = captures.reconcile_model.get(index)
        else:
            model_name = config.get("model") or config.get("model_name")
            if not model_name:
                model_name = self._infer_model_name_from_output(
                    output,
                    *self._model_inference_pool(
                        index, all_captured_responses, captures
                    ),
                )

        # META-EXTRACTION-POINT: Extract and inject __traigent_meta__ if present.
        # This happens after extract_llm_metrics so honest reported costs can
        # override SDK calculation while implausible token-priced under-reports
        # are clamped back to the canonical token-derived estimate.
        self._extract_and_inject_traigent_meta(
            output, example_metric, model_name=model_name
        )

        # Set success status
        example_metric.success = errors[index] is None
        example_metric.error = errors[index]

        # Calculate accuracy
        expected_output = (
            dataset.examples[index].expected_output
            if index < len(dataset.examples)
            else None
        )
        # Thread expected-output eligibility to tracker formatters. The
        # built-in exact-match denominator excludes blank expected values, but
        # keeps all real attempts (including errors); custom accuracy metric
        # functions receive their own explicit provenance at aggregation time.
        example_metric.accuracy_eligible = not _is_empty_expected_output(
            expected_output
        )
        # Compare the output as-is: the comparator unwraps a real response
        # wrapper itself and leaves a structured answer intact (#2523).
        accuracy_value = self._calculate_example_accuracy(output, expected_output)
        if accuracy_value is not None:
            example_metric.custom_metrics.setdefault("accuracy", accuracy_value)
        elif example_metric.accuracy_eligible:
            # Gold exists but nothing could be compared: unmeasured, never the
            # tracker's success-implies-1.0 fallback (#2495).
            example_metric.accuracy_unavailable = True

        # Merge per-example user metrics into custom_metrics, never overriding
        # an evaluator-computed key (e.g. accuracy stays the computed value).
        self._merge_user_metrics(
            example_metric.custom_metrics, user_metrics, context="local lane"
        )

        # Transfer metrics from example_results if available
        if example_results and index < len(example_results):
            self._update_example_metric_from_result(
                example_metric, example_results[index], index
            )

        # Add standard metrics to custom_metrics
        self._add_standard_metrics_to_custom(example_metric)

        # Apply custom metric functions
        if self.metric_functions and index < len(dataset.examples):
            example_obj = dataset.examples[index]
            produced_metric_keys = await self._apply_custom_metric_functions(
                example_metric,
                output,
                example_obj,
                config,
                index,
                metric_errors=metric_errors,
                captures=captures,
            )

            # Fold LLM spend made *inside* a metric function / judge (e.g. an
            # LLM-as-judge calling litellm.completion) into this example's
            # cost/tokens. Without this, judge calls are captured by the same
            # interceptor the agent call uses but are never read: this
            # example's cost was already settled above by
            # `_extract_llm_metrics_for_output`, before metric functions ran
            # (Traigent#2297). Left uncorrected, `total_cost` and the
            # permit-based cost enforcement in `core/cost_enforcement.py`
            # never see judge spend.
            target_result = (
                example_results[index]
                if (
                    self.detailed
                    and example_results
                    and index < len(example_results)
                    and example_results[index] is not None
                )
                else None
            )
            self._fold_metric_function_llm_cost(
                example_metric, example_result=target_result, captures=captures
            )

            # Transfer custom metrics to example_results if in detailed mode.
            # Iterate the keys actually produced (which include mapping
            # sub-keys and exclude any function name a mapping-returning
            # function never populated) rather than the function names -- the
            # latter KeyErrors for mapping-returning metric functions.
            if (
                self.detailed
                and example_results
                and index < len(example_results)
                and example_results[index] is not None
            ):
                # A metric function is authoritative for the key(s) it
                # produces and overrides a built-in computed value of the same
                # name -- this is the contract the custom ``scoring_function``
                # path relies on (it defines the objective, e.g. ``accuracy``).
                # Mapping sub-keys inherit the same override semantics as
                # scalar metric-function values, for consistency.
                target_metrics = example_results[index].metrics
                for metric_key in produced_metric_keys:
                    if metric_key not in example_metric.custom_metrics:
                        # Withdrawn by the judge fold: the example's cost is
                        # unknown, so a cost key is not transferred (#2517).
                        continue
                    target_metrics[metric_key] = example_metric.custom_metrics[
                        metric_key
                    ]

        # Emit boundary for this row: an unknown-cost row leaves with no numeric
        # cost in any channel (row metrics, custom metrics), only flag + floor.
        self._enforce_row_cost_invariant(
            example_metric,
            example_results[index]
            if example_results and index < len(example_results)
            else None,
        )

        # Send progress callback
        if progress_callback:
            example_result_for_progress = (
                example_results[index]
                if example_results and index < len(example_results)
                else None
            )
            payload = self._build_local_progress_payload(
                example_metric, example_result_for_progress, output
            )
            progress_callback(index, payload)

        return example_metric

    def _fold_metric_function_llm_cost(
        self,
        example_metric: ExampleMetrics,
        *,
        example_result: Any = None,
        captures: _CaptureOwnership | None = None,
    ) -> None:
        """Attribute LLM spend made by a metric function to this example.

        A metric function (or an LLM-as-judge inside one) may call an
        intercepted provider (litellm.completion/acompletion). That call is
        captured by the same interceptor buffer the agent's own call uses.
        Within a single trial, evaluated sequentially, the buffer holds
        *only* what was captured since this example's metric functions
        started: `evaluate()` drains the buffer into
        `all_captured_responses` and clears it before the per-example loop
        begins, and every previous example already cleared its own share
        here. So in that sequential-within-a-trial case, whatever is in the
        buffer now is this example's judge/evaluator spend, and nothing
        else's (Traigent#2297).

        Isolation: each trial runs in its own capture scope and every capture
        records the example that made it, so a concurrent trial's calls are never
        in this buffer.  The fold takes the buffer with ONE atomic drain; a call
        owned by an example (e.g. a worker that outlived its timeout) is not this
        judge's and is routed to the trial-level unknown-spend floor instead.

        Folds that spend into `example_metric.cost`/`.tokens` so it enters
        the trial's actual cost and the permit-based cost-enforcement ledger
        (`core/cost_enforcement.py`), and records it separately under
        `custom_metrics["evaluation_cost"]` -- a one-line metadata entry, not
        a new public field -- so a judge's share is reported alongside the
        agent's cost rather than silently merged into it.

        ``evaluation_cost`` shares the user channel: an existing user value
        wins the name collision. SDK insertions carry ``sdk_evaluation_cost``
        provenance on the row, so unknown-cost sanitization withdraws only SDK
        values. If a trial contains both producers, tracker aggregation uses
        only the user values for that name. Like other non-reserved metrics,
        this optional breakdown may be dropped by the user-metric key ceiling;
        folded spend remains in the standard cost accounting.

        Does not touch the pre-run cost estimator (`check_and_approve`);
        that estimator has no view of metric-function calls at all and is
        tracked separately (see the issue's linked follow-up on
        calls-per-example).

        Pricing deliberately does NOT fall back to this example's agent-call
        model: a judge commonly calls a different, cheaper model than the
        agent under test (as in the issue's own repro), and each captured
        response already carries its own ``model`` field, so
        ``extract_llm_metrics`` infers per-response pricing directly
        (mirroring the response-inference fallback ``_process_single_output``
        already uses for the agent's own call).

        Args:
            example_metric: This example's metrics (mutated in place).
            example_result: The detailed-mode ``ExampleResult`` for this
                example, if any. ``_update_example_metric_from_result``
                already transferred the pre-fold cost/tokens into its
                ``metrics`` dict earlier in ``_process_single_output``; when
                given, this refreshes that copy so it doesn't go stale.
        """
        # One atomic take of the buffer (a late call from a timed-out worker
        # can never fall between a snapshot and a clear).
        drained, drained_owners, drained_requested = drain_captured_responses()
        if not drained:
            return
        if drained_owners is None or len(drained_owners) != len(drained):
            drained_owners = [None] * len(drained)
        if drained_requested is None or len(drained_requested) != len(drained):
            drained_requested = [None] * len(drained)

        eval_input_cost = 0.0
        eval_output_cost = 0.0
        eval_total_cost = 0.0
        eval_input_tokens = 0
        eval_output_tokens = 0
        eval_tokens = 0
        judge_unmeasured = False
        for response, owner, requested in zip(
            drained, drained_owners, drained_requested, strict=True
        ):
            if owner is not None:
                # Made by an example's own evaluation (e.g. a worker that outlived
                # its timeout), not by this judge: never charge it here. It stays
                # unknown spend with its exact priceable floor, reported once at
                # trial level (#2444).
                if captures is not None:
                    captures.add_late(response, requested)
                continue
            metrics = self._price_one_call(
                response,
                agent_model=None,
                is_agent_call=False,
                requested_model=requested,
            )
            # A judge call with unknown cost (no usage, or no identifiable
            # model/price) is an unknown spend, not a free one: it makes this
            # example's cost unmeasured (#2517).
            judge_unmeasured = judge_unmeasured or not (
                metrics.measured
                and not metrics.cost.unmeasured
                and not metrics.cost.unpriced
                and (
                    metrics.tokens.total_tokens
                    or metrics.cost.total_cost
                    or metrics.cost.cost_explicit
                )
            )
            eval_input_cost += metrics.cost.input_cost
            eval_output_cost += metrics.cost.output_cost
            eval_total_cost += metrics.cost.total_cost
            eval_input_tokens += metrics.tokens.input_tokens
            eval_output_tokens += metrics.tokens.output_tokens
            eval_tokens += metrics.tokens.total_tokens
            # Forward-compat with #2308: an unpriced judge call must not be
            # silently reported as verified-free once folded into this
            # example's totals.
            example_metric.cost.unpriced = (
                example_metric.cost.unpriced or metrics.cost.unpriced
            )

        if judge_unmeasured:
            # Independent of the agent's own measured state: an unknown judge
            # spend makes the example's cost unknown whatever else was seen.
            self._mark_example_cost_unmeasured_by_judge(example_metric, example_result)

        if not eval_total_cost and eval_tokens == 0:
            # Nothing priced and nothing tokenized: there is no judge spend to
            # attribute, so ABANDON the fold and leave every field of
            # ``example_metric`` exactly as it was.
            return

        # Only now -- past the abandon-guard -- is this row genuinely measured.
        # A captured judge/evaluator response is a real measurement even when
        # the agent's own call never produced output (e.g. every example errors
        # before the agent responds, so `_extract_llm_metrics_for_output` set
        # `measured=False`). The judge still ran and its spend is real, so this
        # example must not be excluded from measured-only aggregation
        # (`MetricsTracker.aggregate_metrics`/`format_for_backend`) the way a
        # genuinely never-measured example is (Traigent#2297 review).
        #
        # This assignment MUST stay below the guard. Above it, a captured judge
        # response that priced to zero cost AND zero tokens flipped a genuinely
        # unmeasured row to `measured=True` carrying all-zero cost/token
        # metrics, which then re-entered the measured-only MEAN denominators at
        # `metrics_tracker.py`'s `measured_metrics` filter and dragged every
        # mean down -- precisely the defect `ExampleMetrics.measured`'s own
        # docblock warns about.
        example_metric.measured = True

        agent_cost_unmeasured = example_metric.cost.unmeasured
        if agent_cost_unmeasured:
            # The agent's own cost is unknown, so the judge's spend is a known
            # floor, not part of a total: adding it to ``total_cost`` would
            # present a partial total as complete, and dropping it would hide
            # real spend. Aggregation reports it as ``cost_lower_bound``.
            example_metric.cost.known_cost += eval_total_cost
        else:
            example_metric.cost.input_cost += eval_input_cost
            example_metric.cost.output_cost += eval_output_cost
            example_metric.cost.total_cost += eval_total_cost
        example_metric.tokens.input_tokens += eval_input_tokens
        example_metric.tokens.output_tokens += eval_output_tokens
        example_metric.tokens.total_tokens += eval_tokens

        # Keep the standard cost/token custom_metrics (written earlier in
        # `_process_single_output`, before metric functions ran) consistent
        # with the now-updated totals, and surface the judge's own share.
        # This name remains on the user channel. A pre-existing user value
        # wins; otherwise record SDK provenance for row/aggregate withdrawal.
        if "evaluation_cost" not in example_metric.custom_metrics:
            example_metric.custom_metrics["evaluation_cost"] = eval_total_cost
            example_metric.sdk_evaluation_cost = True
        example_metric.custom_metrics["input_tokens"] = (
            example_metric.tokens.input_tokens
        )
        example_metric.custom_metrics["output_tokens"] = (
            example_metric.tokens.output_tokens
        )
        example_metric.custom_metrics["total_tokens"] = (
            example_metric.tokens.total_tokens
        )
        if example_result is not None:
            example_result.metrics["input_tokens"] = example_metric.tokens.input_tokens
            example_result.metrics["output_tokens"] = (
                example_metric.tokens.output_tokens
            )
            example_result.metrics["total_tokens"] = example_metric.tokens.total_tokens
            if example_metric.sdk_evaluation_cost:
                example_result.metrics["evaluation_cost"] = eval_total_cost

        # One notion of "known cost" on every channel: a measured row shows its
        # refreshed numeric cost; an unknown row (unmeasured OR unpriced) has
        # every numeric cost withdrawn and carries the flag and exact floor.
        if example_metric.cost_measured:
            example_metric.custom_metrics["input_cost"] = example_metric.cost.input_cost
            example_metric.custom_metrics["output_cost"] = (
                example_metric.cost.output_cost
            )
            example_metric.custom_metrics["total_cost"] = example_metric.cost.total_cost
            if example_result is not None:
                example_result.metrics["input_cost"] = example_metric.cost.input_cost
                example_result.metrics["output_cost"] = example_metric.cost.output_cost
                example_result.metrics["total_cost"] = example_metric.cost.total_cost
        else:
            self._withdraw_numeric_cost(example_metric, example_result)

    @classmethod
    def _withdraw_numeric_cost(
        cls, example_metric: ExampleMetrics, example_result: Any
    ) -> None:
        """Remove numeric costs from an unknown-cost row; emit flag + floor."""
        floor = cls._example_cost_floor(example_metric)
        enforce_unknown_cost_invariant(
            example_metric.custom_metrics,
            unknown=True,
            floor=floor,
            sdk_evaluation_cost=example_metric.sdk_evaluation_cost,
        )
        if example_result is not None:
            enforce_unknown_cost_invariant(
                example_result.metrics,
                unknown=True,
                floor=floor,
                sdk_evaluation_cost=example_metric.sdk_evaluation_cost,
            )

    @classmethod
    def _enforce_row_cost_invariant(
        cls, example_metric: ExampleMetrics, example_result: Any
    ) -> None:
        """Apply the unknown-cost emit rule to one finished row's channels."""
        if not example_metric.cost_measured:
            cls._withdraw_numeric_cost(example_metric, example_result)

    @staticmethod
    def _mark_example_cost_unmeasured_by_judge(
        example_metric: ExampleMetrics, example_result: Any
    ) -> None:
        """An unmeasured judge call makes the whole example's cost unknown.

        The agent's own measured spend stays as a known floor
        (``cost.known_cost``), never as a total; the cost keys already copied
        into the row and custom metrics are withdrawn so nothing downstream
        reads the partial sum as the example's cost.
        """
        if not example_metric.cost.unmeasured:
            example_metric.cost.known_cost += example_metric.cost.total_cost
            example_metric.cost.unmeasured = True
        LocalEvaluator._withdraw_numeric_cost(example_metric, example_result)

    def _log_dual_scorer_notice_once(self) -> None:
        """Emit a single run-level notice when both scorers appear (issue #1845).

        Fires at most once per evaluator instance (== once per run), never per
        example/trial, so a customer sees one clear line rather than N.
        """
        if self._dual_scorer_notice_logged:
            return
        self._dual_scorer_notice_logged = True
        logger.info(
            "A custom scoring_function defines the 'accuracy' objective; "
            "metrics['score'] carries the optimization signal (the objective "
            "value, == metrics['accuracy']), and the SDK's built-in "
            "exact-match scorer is recorded separately as "
            "metrics['exact_match_default'] for diagnostics."
        )

    def _maybe_warn_high_empty_output_rate(
        self, rate: float, config: dict[str, Any]
    ) -> None:
        """Emit a single run-level warning for a high empty-output rate (#1851).

        Fires at most once per evaluator instance (== once per run), never per
        example/trial, mirroring the dual-scorer notice (#1845). An empty or
        whitespace-only output at any meaningful rate signals truncation,
        output-parsing failure, or refusals, so the accuracy comparison is
        measuring an artifact, not the config knobs. Every trial still records
        its own ``empty_output_rate`` metric; this warning names the first
        offending config so the run does not finish silently.
        """
        if rate <= EMPTY_OUTPUT_RATE_WARNING_THRESHOLD:
            return
        if self._empty_output_warning_logged:
            return
        self._empty_output_warning_logged = True
        logger.warning(
            "%.1f%% of outputs for config %r are empty or whitespace-only "
            "(threshold %.0f%%) — likely token truncation, output-parsing "
            "failure, or refusals; accuracy comparisons are unreliable until "
            "resolved. Each trial's empty_output_rate metric records the "
            "per-config rate.",
            rate * 100.0,
            config,
            EMPTY_OUTPUT_RATE_WARNING_THRESHOLD * 100.0,
        )

    def _maybe_warn_truncated_output(self, rate: float, config: dict[str, Any]) -> None:
        """Emit a single run-level warning when outputs were truncated (#1809).

        Fires at most once per evaluator instance (== once per run, naming the
        first offending config), mirroring the empty-output-rate warning
        (#1851). Any nonzero rate warns -- unlike the empty-output threshold,
        a single truncated example already means at least one score in the
        comparison is a measurement artifact, not a real signal about the
        config: reasoning models (gemini-2.5/3.x, gpt-5, o-series) spend
        hidden reasoning tokens against ``max_tokens`` before any answer text,
        so a cap sized for a non-reasoning model truncates the answer
        mid-output and that example scores as an artifact, not on the config's
        real capability. Every trial still records its own
        ``truncated_output_rate`` metric; this warning only decides whether to
        also log.
        """
        if rate <= 0.0:
            return
        if self._truncated_output_warning_logged:
            return
        self._truncated_output_warning_logged = True
        logger.warning(
            "%.1f%% of measured outputs for config %r were truncated "
            "(provider finish_reason indicates the length/token cap was hit, "
            "not a natural stop) — reasoning models spend hidden thinking "
            "tokens against max_tokens before any answer text, so a cap sized "
            "for a non-reasoning model can silently truncate the answer; "
            "accuracy comparisons are unreliable until max_tokens is raised. "
            "Each trial's truncated_output_rate metric records the "
            "per-config rate.",
            rate * 100.0,
            config,
        )

    def _compute_accuracy_aggregated(
        self,
        outputs: list[Any],
        dataset: Dataset,
        errors: list[str | None] | None = None,
    ) -> tuple[float | None, int]:
        """Compute aggregated accuracy across outputs.

        This is the accuracy path that actually reaches users of the default
        (local) evaluator: its result overwrites ``aggregated_metrics["accuracy"]``
        set by ``BaseEvaluator._compute_accuracy`` below. It must therefore share
        that method's denominator semantics rather than silently disagreeing with
        it (Traigent#1963) -- an errored example still counts against the
        denominator; only the missing-expected-output exclusion drops an example
        entirely. That exclusion uses ``_is_empty_expected_output`` (None, or an
        empty/whitespace-only string), the SAME predicate ``BaseEvaluator.
        _compute_accuracy`` uses -- a plain ``expected is None`` check here would
        leave an empty-string expected output counted as a real (near-certain)
        miss on this path while ``_compute_accuracy`` excludes it entirely,
        disagreeing about which rows even enter the denominator.

        Args:
            outputs: List of outputs
            dataset: Evaluation dataset
            errors: Per-example error messages aligned with ``outputs`` (None for
                a successful call). Omitted (``None``) treats every example as
                successful, which keeps existing direct callers that pre-date
                error-awareness unchanged.

        Returns:
            Tuple of (accuracy value or None, total count)
        """
        total = 0
        correct = 0
        error_list = errors if errors is not None else [None] * len(outputs)

        for raw_output, example, error in zip(
            outputs, dataset.examples, error_list, strict=False
        ):
            expected = example.expected_output
            if _is_empty_expected_output(expected):
                continue

            # An errored example is a real attempt with a real (missing)
            # answer, not a non-event -- it counts against the denominator
            # even though it can never count as correct.
            total += 1
            if error is not None:
                continue

            # Unwrap the same way the per-example path already does (dict
            # {"text": ...}, strict (output, metrics) tuple) before comparing
            # -- a raw tuple/dict never equals a scalar expected value, which
            # silently understated this aggregate on mixed-shape runs
            # (Traigent#1771).
            if raw_output is not None and _accuracy_matches_after_unwrap(
                raw_output, expected
            ):
                correct += 1

        if total > 0:
            return correct / total, total
        return None, total

    def _merge_comprehensive_metrics(
        self,
        aggregated_metrics: dict[str, float],
        comprehensive_metrics: dict[str, Any],
        *,
        preserve_authoritative_accuracy: bool,
    ) -> None:
        """Merge comprehensive metrics into aggregated metrics.

        Args:
            aggregated_metrics: Target metrics dict (modified in place)
            comprehensive_metrics: Source comprehensive metrics
        """
        logger.debug(
            f"LOCAL EVALUATOR DEBUG: comprehensive_metrics['cost'] = "
            f"{comprehensive_metrics.get('cost', 'MISSING')}"
        )

        # If cost is in objectives but was computed as 0, use comprehensive value
        if "cost" in self.metrics and comprehensive_metrics.get("cost") is not None:
            logger.debug(
                f"LOCAL EVALUATOR DEBUG: aggregated cost="
                f"{aggregated_metrics.get('cost', 'MISSING')}, "
                f"comprehensive cost={comprehensive_metrics['cost']}"
            )
            aggregated_cost = float(aggregated_metrics.get("cost", 0.0) or 0.0)
            comprehensive_cost = float(comprehensive_metrics["cost"])
            if math.isclose(aggregated_cost, 0.0, abs_tol=1e-9) and not math.isclose(
                comprehensive_cost, 0.0, abs_tol=1e-9
            ):
                logger.info(
                    f"🔍 LOCAL EVALUATOR: Overriding cost metric: "
                    f"{aggregated_cost} -> {comprehensive_cost}"
                )

        for key, value in comprehensive_metrics.items():
            if value is None:
                continue
            if (
                key == "score"
                and preserve_authoritative_accuracy
                and "accuracy" in aggregated_metrics
                and key not in aggregated_metrics
            ):
                # A built-in accuracy can be a real 0.0 while its sibling
                # score key is absent (for example every expected output is
                # blank). Keep the public score alias aligned with that
                # authoritative result instead of accepting the tracker's
                # unrelated fallback score.
                aggregated_metrics[key] = aggregated_metrics["accuracy"]
                continue
            if (
                key in {"accuracy", "score"}
                and key in aggregated_metrics
                and (
                    preserve_authoritative_accuracy
                    or aggregated_metrics[key] not in (None, 0.0)
                )
            ):
                # ``compute_metrics`` / ``_compute_accuracy_aggregated`` is
                # authoritative for these objectives. In particular, 0.0 is
                # an honest score, not a missing value for the tracker to
                # replace with per-example metadata. A custom scorer is the
                # exception: in non-detailed mode its aggregate is supplied
                # by the tracker, so retain the established fallback for that
                # explicit custom-objective contract.
                continue
            aggregated_metrics[key] = value

    @staticmethod
    def _compute_percentile(values: list[float], percentile: float) -> float:
        """Compute percentile for a list of floats using linear interpolation."""
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

    def _compute_aggregated_custom_metrics(
        self,
        example_results: list[Any],
    ) -> dict[str, float]:
        """Compute aggregated metrics from custom metric functions.

        Args:
            example_results: List of example results with metrics

        Returns:
            Dictionary of aggregated metric values
        """
        aggregated: dict[str, float] = {}

        for metric_name in self.metric_functions:
            values: list[float] = []
            # A mapping-returning metric function populates its SUB-keys, never
            # its own name, so ``metric_name`` may be absent from every result.
            # Track whether it was seen so we don't fabricate a bogus aggregate
            # ``0.0`` for a function name that produced no such key.
            saw_metric_name = False
            for result in example_results:
                if result is None:
                    continue
                if metric_name not in result.metrics:
                    continue
                saw_metric_name = True
                raw_value = result.metrics[metric_name]
                if raw_value is None:
                    continue
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    continue

            if values:
                if "p95" in metric_name.lower():
                    aggregated[metric_name] = self._compute_percentile(values, 0.95)
                else:
                    aggregated[metric_name] = sum(values) / len(values)
            elif saw_metric_name or not self.detailed:
                # Keep the legacy 0.0 sentinel when the name was present but
                # produced no usable numeric value, OR in non-detailed mode
                # (no per-example rows to inspect, so scalar metric functions
                # must retain their legacy 0.0 aggregate). In detailed mode a
                # name never seen at all -- e.g. a mapping function's own name,
                # whose sub-keys are aggregated elsewhere -- is NOT fabricated
                # as a bogus 0.0.
                aggregated[metric_name] = 0.0

        return aggregated

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        budget: ExecutionBudget | None = None,
    ) -> EvaluationResult:
        """Evaluate function with given configuration on dataset.

        Runs inside its own LLM-response capture scope so that two evaluations
        running concurrently cannot drain each other's captured responses and
        charge one trial for another's judge spend (Traigent#2387). The trial
        lifecycle opens an outer scope per trial; this inner one keeps a DIRECT
        concurrent caller of ``evaluate()`` correct too, and nesting is safe.
        """
        with coercion_log_scope():
            async with capture_scope():
                history_token = _progress_capture_history.set(_CaptureOwnership([], []))
                try:
                    return await self._evaluate_within_capture_scope(
                        func,
                        config,
                        dataset,
                        sample_lease=sample_lease,
                        progress_callback=progress_callback,
                        budget=budget,
                    )
                finally:
                    _progress_capture_history.reset(history_token)

    async def _evaluate_within_capture_scope(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        budget: ExecutionBudget | None = None,
    ) -> EvaluationResult:
        """Evaluate function with given configuration on dataset.

        Args:
            func: Function to evaluate
            config: Configuration parameters
            dataset: Evaluation dataset

        Returns:
            EvaluationResult with metrics and outputs

        Raises:
            EvaluationError: If evaluation fails
        """
        self.validate_function(func)
        self.validate_config(config)

        execution_budget_lease, blocked_result = (
            self._prepare_execution_budget_evaluation(
                budget, config, sample_lease=sample_lease
            )
        )
        if blocked_result is not None:
            return blocked_result
        if sample_lease is None:
            sample_lease = execution_budget_lease

        if not dataset.examples:
            raise EvaluationError("Dataset cannot be empty")

        self._warn_context_mode_metadata_defaulted_params(func, dataset)

        logger.info(
            f"Starting {'detailed ' if self.detailed else ''}evaluation with {len(dataset.examples)} examples, "
            f"config: {config}"
        )

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
        metrics_tracker.start_tracking()

        start_time = time.time()

        # Use common batch evaluation method from BaseEvaluator
        (
            outputs,
            errors,
            example_results,
            consumed_examples,
            budget_exhausted,
        ) = await self._await_with_sample_lease_cleanup(
            self._evaluate_batch(
                func,
                config,
                dataset,
                sample_lease=sample_lease,
                detailed=self.detailed,
                progress_callback=progress_callback,
                execution_budget_lease=execution_budget_lease,
            ),
            sample_lease,
            execution_budget_lease,
            cleanup_on_cancel=False,
        )

        # Collect expected outputs
        if consumed_examples:
            capped = min(consumed_examples, len(dataset.examples))
            expected_outputs = [
                dataset.examples[i].expected_output for i in range(capped)
            ]
        else:
            expected_outputs = []

        duration = time.time() - start_time
        metrics_tracker.end_tracking()

        # If detailed mode, compute individual example metrics first
        if self.detailed and example_results:
            for _i, ex_result in enumerate(example_results):
                if (
                    ex_result is not None
                    and ex_result.success
                    and ex_result.error_message is None
                ):
                    # Compute metrics for this specific example
                    example_metrics = self._compute_example_metrics(
                        ex_result.actual_output, ex_result.expected_output
                    )
                    # Update the ExampleResult with computed metrics
                    ex_result.metrics.update(example_metrics)

        # Get all captured LangChain responses before clearing
        # One atomic take of responses + owners + requested models (a late call
        # from a timed-out worker can never be cleared unaccounted or misalign).
        all_captured_responses, drained_owners, drained_requested = (
            drain_captured_responses()
        )
        captures = _CaptureOwnership.build(
            all_captured_responses, drained_owners, drained_requested
        )
        logger.debug(
            f"Got {len(all_captured_responses)} captured LangChain responses for {len(outputs)} outputs"
        )

        # Structured degradation records for non-objective metric failures,
        # accumulated across this evaluate() call and surfaced on the
        # returned EvaluationResult (never silently coalesced into 0.0).
        # Kept as a local rather than instance state: LocalEvaluator instances
        # are shared across concurrent trials (parallel_trials), so instance
        # state here would race between trials.
        metric_errors: list[dict[str, Any]] = []

        # Track metrics for each example output using helper method
        for i, output in enumerate(outputs):
            example_metric = await self._await_with_sample_lease_cleanup(
                self._process_single_output(
                    output=output,
                    index=i,
                    config=config,
                    dataset=dataset,
                    errors=errors,
                    example_results=example_results,
                    all_captured_responses=all_captured_responses,
                    progress_callback=progress_callback,
                    metric_errors=metric_errors,
                    captures=captures,
                ),
                sample_lease,
                execution_budget_lease,
            )
            metrics_tracker.add_example_metrics(example_metric)

        # Compute aggregated metrics (include built-ins even with custom functions)
        if self.metric_functions:
            # Filter to registry/RAGAS metrics, but exclude those already provided
            # by custom metric_functions (e.g., skip built-in accuracy if
            # scoring_function provides it - avoids redundant computation).
            # Note: scoring_function maps to "accuracy" key only; cost/latency
            # built-ins remain unless explicitly overridden via metric_functions.
            custom_metric_keys = set(self.metric_functions.keys())
            base_metric_names = [
                name
                for name in self.metrics
                if (name in self._metric_registry or name in self._ragas_metric_names)
                and name not in custom_metric_keys
            ]
            if base_metric_names:
                aggregated_metrics = self.compute_metrics(
                    outputs,
                    expected_outputs,
                    errors,
                    dataset=dataset,
                    example_results=example_results,
                    example_metrics=metrics_tracker.example_metrics,
                    metrics_override=base_metric_names,
                    metric_errors=metric_errors,
                )
            else:
                aggregated_metrics = {}
        else:
            aggregated_metrics = self.compute_metrics(
                outputs,
                expected_outputs,
                errors,
                dataset=dataset,
                example_results=example_results,
                example_metrics=metrics_tracker.example_metrics,
                metric_errors=metric_errors,
            )

        # A custom scoring_function / metric_function OWNS the "accuracy"
        # objective when "accuracy" is both an objective and a custom metric
        # function. In that case the SDK's built-in exact-match scorer is a
        # DIAGNOSTIC, not the optimization signal, so it must not occupy
        # metrics["score"] (issue #1845).
        accuracy_is_custom_objective = (
            "accuracy" in self.metrics and "accuracy" in self.metric_functions
        )
        accuracy_is_user_metric = "accuracy" in self.metric_functions

        if "accuracy" in self.metrics:
            accuracy_value, _ = self._compute_accuracy_aggregated(
                outputs, dataset, errors
            )
            if accuracy_value is not None:
                if accuracy_is_custom_objective:
                    # Keep the default exact-match value as a labelled
                    # diagnostic; the custom aggregate below owns "accuracy"
                    # and "score" (set to the objective value).
                    aggregated_metrics["exact_match_default"] = accuracy_value
                    self._log_dual_scorer_notice_once()
                else:
                    # No custom scorer: the default exact-match IS the objective,
                    # so it is correctly both "accuracy" and "score".
                    aggregated_metrics["accuracy"] = accuracy_value
                    aggregated_metrics.setdefault("score", accuracy_value)

        # Aggregate custom metric functions using helper
        if self.metric_functions:
            custom_aggregated = self._compute_aggregated_custom_metrics(example_results)
            aggregated_metrics.update(custom_aggregated)
            # metrics["score"] is the optimization signal. When a custom scorer
            # owns the objective, that signal is the objective value (the custom
            # scorer mean == metrics["accuracy"]), NOT the default exact-match
            # scorer (which now lives under "exact_match_default"). Issue #1845.
            if accuracy_is_custom_objective and "accuracy" in aggregated_metrics:
                aggregated_metrics["score"] = aggregated_metrics["accuracy"]

        # Seal normal accounting before draining. Late captures have a separate
        # buffer folded at the final handoff below. Workers still live here
        # leave eventual spend unknown even without a response yet (#2444).
        if seal_captured_responses():
            metrics_tracker.unattributed_spend = True
            for metric in metrics_tracker.example_metrics:
                if not metric.measured:
                    metric.cost.unmeasured = True
        if captures is not None:
            captures.absorb_late(drain_captured_responses())
            # Spend no example row consumed (a call with no owner, or an
            # example that produced no row): its cost cannot be attributed, so
            # the trial total becomes unmeasured and the exact priced part is
            # kept as the known-spend floor -- never dropped, never guessed.
            leftover = captures.leftover()
            if leftover:
                # Each leftover is priced by its OWN response model; the agent
                # model is never assumed for a call not established as the
                # agent's.  A call that cannot be priced adds nothing to the
                # floor (its cost is unknown, and the trial is unmeasured).
                floor = 0.0
                for r, requested in leftover:
                    priced = self._price_one_call(
                        r,
                        agent_model=None,
                        is_agent_call=False,
                        requested_model=requested,
                    )
                    floor += self._call_cost_floor(priced)
                metrics_tracker.unattributed_spend = True
                metrics_tracker.unattributed_known_cost = floor

        # Add comprehensive metrics from tracker using helper. Thread the
        # evaluator's runtime-only computable names (registry + RAGAS) so a
        # user tuple key cannot overwrite an evaluator-computed value during the
        # tracker's user-metric aggregation pass.
        comprehensive_metrics = metrics_tracker.format_for_backend(
            extra_reserved=self._evaluator_computable_metric_names(),
            user_supplied_accuracy=accuracy_is_user_metric,
        )
        self._merge_comprehensive_metrics(
            aggregated_metrics,
            comprehensive_metrics,
            preserve_authoritative_accuracy=not accuracy_is_custom_objective,
        )

        aggregated_metrics.setdefault("examples_attempted", len(outputs))

        # Emit boundary for the trial: when the authoritative tracker declares
        # the cost unknown, WITHDRAW every earlier numeric cost value (e.g. a
        # base-evaluator ``cost`` computed from a placeholder zero) and keep only
        # the flag plus the exact known floor (#2444).
        enforce_unknown_cost_invariant(
            aggregated_metrics,
            sdk_evaluation_cost=metrics_tracker.sdk_evaluation_cost_only,
        )

        # Empty/whitespace-output guard (issue #1851): compute the fraction of
        # this config's outputs that are empty on EVERY trial and expose it as a
        # reserved metric so portals/harnesses can gate on it, then surface ONE
        # run-level warning naming the offending config above the threshold. This
        # works for every calling pattern (decorator/wrapper/managed) because it
        # reads only the returned outputs — the metadata-free complement to the
        # finish_reason guard (#1809).
        empty_output_rate = compute_empty_output_rate(outputs)
        aggregated_metrics["empty_output_rate"] = empty_output_rate
        self._maybe_warn_high_empty_output_rate(empty_output_rate, config)

        # Truncated-output guard (issue #1809): compute the fraction of this
        # config's examples whose provider response carried a truncated
        # finish/stop reason (reads the per-example finish_reason the response
        # handler chain already recorded on ``metrics_tracker.example_metrics``
        # while extracting tokens/cost -- no extra provider call), expose it
        # as a reserved metric, then surface ONE run-level warning naming the
        # first offending config. The complement to the metadata-free
        # empty-output-rate guard above: this fires even when the truncated
        # text is non-empty (a reasoning model's cut-off answer is rarely
        # blank, just wrong).
        truncated_output_rate = compute_truncated_output_rate(
            metrics_tracker.example_metrics
        )
        aggregated_metrics["truncated_output_rate"] = truncated_output_rate
        self._maybe_warn_truncated_output(truncated_output_rate, config)

        # Authoritative cap on the FINAL trial metrics: user keys (arriving via
        # the per-example custom_metrics -> comprehensive_metrics merge above)
        # must not push the union past the MeasuresDict ceiling. Only user keys
        # are dropped; reserved evaluator keys are never sacrificed.
        enforce_user_metric_ceiling(
            aggregated_metrics,
            context="local trial aggregation",
            extra_reserved=self._evaluator_computable_metric_names(),
        )

        # Generate summary_stats (needed for insights)
        summary_stats = None
        if self.execution_mode_enum:
            summary_stats = metrics_tracker.format_as_summary_stats(
                user_supplied_accuracy=accuracy_is_user_metric
            )
            if isinstance(summary_stats.get("metrics"), dict):
                enforce_unknown_cost_invariant(
                    summary_stats["metrics"],
                    nullify=True,
                    sdk_evaluation_cost=metrics_tracker.sdk_evaluation_cost_only,
                )
            logger.debug(f"Generated summary_stats for {self.execution_mode} mode")

        # Calculate success statistics
        if self.detailed and example_results:
            success_count = sum(
                1 for ex_result in example_results if ex_result and ex_result.success
            )
            total_count = len(example_results)
        else:
            success_count = sum(1 for error in errors if error is None)
            total_count = len(errors)

        # Log results
        logger.info(
            f"{'Detailed ' if self.detailed else ''}evaluation completed: {success_count}/{total_count} successful, "
            f"duration: {duration:.2f}s, metrics: {aggregated_metrics}"
        )
        if budget_exhausted:
            logger.info(
                "Sample budget exhausted after %s examples (dataset size=%s)",
                consumed_examples,
                len(dataset.examples),
            )

        # Per-trial, per-model cost breakdown for multi-model/multi-step agents
        # (Traigent#1598) -- aggregated from every example's
        # __traigent_meta__["calls"] reported this trial. Empty when no
        # example reported a breakdown.
        model_costs = metrics_tracker.aggregate_call_breakdown()

        result = EvaluationResult(
            config=config,
            example_results=example_results if self.detailed else [],
            aggregated_metrics=aggregated_metrics,
            total_examples=total_count,
            successful_examples=success_count,
            duration=duration,
            # Legacy compatibility fields
            metrics=aggregated_metrics,
            outputs=outputs,
            errors=errors,
            metric_errors=metric_errors,
            model_costs=model_costs,
        )

        # Attach summary_stats if generated
        if summary_stats:
            summary_stats.setdefault("metadata", {})
            summary_stats["metadata"]["sample_budget_exhausted"] = budget_exhausted
            summary_stats["metadata"]["examples_consumed"] = total_count
            result.summary_stats = summary_stats

        result.sample_budget_exhausted = budget_exhausted
        result.examples_consumed = consumed_examples

        # Last handoff before return: include known spend produced while
        # summary/result generation ran. After this boundary a still-live
        # timed-out worker cannot update the returned result; the trial remains
        # unmeasured, never a claim that its remaining work costs $0.
        try:
            floor = float(
                comprehensive_metrics.get("cost_lower_bound")
                or comprehensive_metrics.get("cost")
                or 0.0
            )
            while late := finish_captured_responses():
                for response, requested in late:
                    floor += self._call_cost_floor(
                        self._price_one_call(
                            response,
                            agent_model=None,
                            is_agent_call=False,
                            requested_model=requested,
                        )
                    )
                    # Commit each priced contribution before the next call can
                    # raise: the finally-block budget debit reads this result.
                    enforce_unknown_cost_invariant(
                        result.aggregated_metrics,
                        unknown=True,
                        floor=floor,
                        sdk_evaluation_cost=metrics_tracker.sdk_evaluation_cost_only,
                    )
                    if result.summary_stats and isinstance(
                        result.summary_stats.get("metrics"), dict
                    ):
                        enforce_unknown_cost_invariant(
                            result.summary_stats["metrics"],
                            unknown=True,
                            floor=floor,
                            nullify=True,
                            sdk_evaluation_cost=metrics_tracker.sdk_evaluation_cost_only,
                        )
        finally:
            # Debit once and close the lease even if strict late pricing raises.
            self._finalize_execution_budget_evaluation(
                budget, result, execution_budget_lease
            )
        return result

    def _compute_real_accuracy(self, actual_output: Any, expected_output: Any) -> float:
        """Compute real accuracy by comparing actual vs expected output.

        Args:
            actual_output: Output from function
            expected_output: Expected output

        Returns:
            Accuracy value (1.0 for match, 0.0 otherwise)
        """
        # Check for missing expected output
        if expected_output is None:
            return 0.0
        if isinstance(expected_output, str) and not expected_output.strip():
            return 0.0

        # Same comparator as the aggregate and the tracker path: a real
        # response wrapper is unwrapped, a structured answer is compared as a
        # structure rather than reduced to its (absent) "text" field (#2523).
        if _accuracy_matches_after_unwrap(actual_output, expected_output):
            return 1.0

        return 0.0

    def _compute_example_metrics(
        self, actual_output: Any, expected_output: Any
    ) -> dict[str, float]:
        """Compute metrics for a single example.

        Args:
            actual_output: Output from function
            expected_output: Expected output

        Returns:
            Dictionary of metric names to values for this example
        """
        metrics = {}

        # SECURITY: Always compute real accuracy. The previous
        # TRAIGENT_MOCK_LLM-gated branch fabricated accuracy via
        # ``random.uniform()`` plus output-string heuristics, which would
        # silently replace real evaluation results with fake scores
        # whenever the env var was set in a production environment.
        if "accuracy" in self.metrics:
            metrics["accuracy"] = self._compute_real_accuracy(
                actual_output, expected_output
            )

        # Success (whether function completed without error)
        if "success_rate" in self.metrics:
            metrics["success"] = 1.0 if actual_output is not None else 0.0

        return metrics

    def compute_metrics(
        self,
        outputs: list[Any],
        expected_outputs: list[Any],
        errors: list[str | None],
        **context: Any,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            outputs: Actual outputs from function
            expected_outputs: Expected outputs
            errors: Error messages (None for successful evaluations)

        Returns:
            Dictionary of metric name to value
        """
        # Use the base class implementation which now includes all default metrics
        return cast(
            dict[str, float],
            super().compute_metrics(outputs, expected_outputs, errors, **context),
        )
