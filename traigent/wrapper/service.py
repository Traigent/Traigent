"""TraigentService - Wrapper for external agentic services.

Provides a minimal framework for exposing agentic services via the
Traigent hybrid API protocol. Services can use decorators to define
their configuration space, execute logic, and evaluation logic.

Example:
    from traigent.wrapper import TraigentService

    app = TraigentService(tunable_id="my_agent")

    @app.tvars
    def config_space():
        return {
            "model": {"type": "enum", "values": ["gpt-4o", "claude-3"]},
            "temperature": {"type": "float", "range": [0.0, 2.0]},
        }

    @app.execute
    async def run_agent(example_id: str, data: dict, config: dict) -> dict:
        # Your agent logic here
        return {"output": result, "cost_usd": 0.002, "latency_ms": 150}

    @app.evaluate
    async def score(output: dict, target: dict, kwargs: dict) -> dict:
        return {"accuracy": 0.95, "safety": 1.0}

    if __name__ == "__main__":
        app.run(port=8080)
"""

# Traceability: HYBRID-MODE-OPTIMIZATION CLIENT-WRAPPER-SDK

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from traigent.hybrid.protocol import (
    EstimatedTokensPerExample,
    EvaluationKwargDefinition,
    ScalarValue,
)
from traigent.utils.logging import get_logger
from traigent.wrapper.errors import (
    BadRequestError,
    HybridAPIError,
    IdempotencyConflictError,
    InvalidInputsError,
    ServiceUnavailableError,
    UnknownTunableIdError,
)

logger = get_logger(__name__)


# OpenAPI contract: tunable and objective names must be valid Python identifiers.
#: Most distinct request_ids that may be running at once; further new requests
#: get a retryable 503 instead of growing the in-flight map. Completed results
#: live in the separate FIFO cache bounded by ``_idempotency_cache_max_size``.
MAX_INFLIGHT_REQUESTS = 512

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_]\w*\Z")
_ESTIMATED_INPUT_TOKENS_ENV = "TRAIGENT_ESTIMATED_INPUT_TOKENS_PER_EXAMPLE"
_ESTIMATED_OUTPUT_TOKENS_ENV = "TRAIGENT_ESTIMATED_OUTPUT_TOKENS_PER_EXAMPLE"


def _estimated_tokens_from_env() -> EstimatedTokensPerExample | None:
    """Return optional per-example token estimate from environment variables."""
    input_tokens = os.environ.get(_ESTIMATED_INPUT_TOKENS_ENV)
    output_tokens = os.environ.get(_ESTIMATED_OUTPUT_TOKENS_ENV)
    if input_tokens is None and output_tokens is None:
        return None
    if input_tokens is None or output_tokens is None:
        logger.warning(
            "Both %s and %s must be set together; ignoring partial token estimate configuration.",
            _ESTIMATED_INPUT_TOKENS_ENV,
            _ESTIMATED_OUTPUT_TOKENS_ENV,
        )
        return None
    return EstimatedTokensPerExample.from_dict(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )


def _resolve_estimated_tokens_per_example(
    value: EstimatedTokensPerExample | dict[str, Any] | None,
) -> EstimatedTokensPerExample | None:
    """Normalize optional per-example token estimate from config or env."""
    if value is None:
        return _estimated_tokens_from_env()
    if isinstance(value, EstimatedTokensPerExample):
        return value
    if isinstance(value, dict):
        return EstimatedTokensPerExample.from_dict(value)
    raise TypeError(
        "estimated_tokens_per_example must be an EstimatedTokensPerExample instance, a dict, or None"
    )


@dataclass
class ServiceConfig:
    """Configuration for TraigentService.

    Attributes:
        tunable_id: Unique identifier for this tunable.
        version: API version string.
        supports_keep_alive: Whether to enable keep-alive support.
        supports_streaming: Whether to enable streaming support.
        max_batch_size: Maximum inputs per execute request.
        schema_version: TVL schema version.
        constraints: Optional constraints (legacy textual or typed TVL 0.9 format).
        objectives: Optional objective definitions.
        exploration: Optional exploration configuration.
        promotion_policy: Optional promotion policy definition.
        defaults: Optional default configuration values.
        measures: Optional declared measure names.
        evaluation_kwargs: Optional supported evaluate kwargs.
        estimated_tokens_per_example: Optional per-example token estimate used
            for hybrid pre-run approval checks.
    """

    tunable_id: str = "default"
    version: str = "1.0"
    supports_keep_alive: bool = False
    supports_streaming: bool = False
    max_batch_size: int = 100
    schema_version: str = "0.9"
    constraints: dict[str, Any] | list[Any] | None = None
    objectives: list[dict[str, Any]] | None = None
    exploration: dict[str, Any] | None = None
    promotion_policy: dict[str, Any] | None = None
    defaults: dict[str, Any] | None = None
    measures: list[str] | None = None
    evaluation_kwargs: list[EvaluationKwargDefinition] | list[dict[str, Any]] | None = (
        None
    )
    estimated_tokens_per_example: EstimatedTokensPerExample | None = None


# Bounds on the keep-alive session table (#3128).
MAX_SESSION_ID_LENGTH = 256
MAX_SESSIONS = 10_000
SESSION_TTL_SECONDS = 3600.0


@dataclass
class Session:
    """Active session state.

    Attributes:
        session_id: Unique session identifier.
        created_at: Timestamp when session was created.
        last_activity: Timestamp of last activity.
        state: Custom state stored by the service.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()


@dataclass(frozen=True)
class EvaluationContext:
    """Optional wrapper-only context for evaluate handlers."""

    execution_id: str | None = None
    execution_config: dict[str, Any] | None = None
    execute_session_id: str | None = None
    evaluate_session_id: str | None = None
    tunable_id: str | None = None


@dataclass(frozen=True)
class _ExecutionContextEntry:
    """Cached execution context for follow-up evaluation calls."""

    execution_id: str
    execution_config: dict[str, Any]
    execute_session_id: str | None
    tunable_id: str


class TraigentService:
    """Wrapper framework for Traigent-compatible agentic services.

    Provides decorators for defining configuration space, execution
    logic, and evaluation logic. Can serve via HTTP or be used
    directly as a library.

    Example:
        app = TraigentService(tunable_id="qa_agent")

        @app.tvars
        def config_space():
            return {"model": {"type": "enum", "values": ["gpt-5.6-sol"]}}

        @app.execute
        async def run(example_id, data, config):
            return {"output": "result"}

        app.run(port=8080)
    """

    def __init__(
        self,
        tunable_id: str = "default",
        version: str = "1.0",
        supports_keep_alive: bool = False,
        supports_streaming: bool = False,
        max_batch_size: int = 100,
        constraints: dict[str, Any] | list[Any] | None = None,
        objectives: list[dict[str, Any]] | None = None,
        exploration: dict[str, Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
        measures: list[str] | None = None,
        evaluation_kwargs: (
            list[EvaluationKwargDefinition] | list[dict[str, Any]] | None
        ) = None,
        estimated_tokens_per_example: (
            EstimatedTokensPerExample | dict[str, Any] | None
        ) = None,
    ) -> None:
        """Initialize TraigentService.

        Args:
            tunable_id: Unique identifier for this tunable.
            version: API version string.
            supports_keep_alive: Whether to enable keep-alive support.
            supports_streaming: Whether to enable streaming support.
            max_batch_size: Maximum inputs per execute request.
            constraints: Optional constraints config.
            objectives: Optional objective definitions.
            exploration: Optional exploration configuration.
            promotion_policy: Optional promotion policy.
            defaults: Optional default tunable values.
            measures: Optional declared measure names.
            evaluation_kwargs: Optional supported evaluate kwargs.
            estimated_tokens_per_example: Optional per-example token estimate
                for cost approval. Falls back to environment variables when not
                provided.
        """
        self.config = ServiceConfig(
            tunable_id=tunable_id,
            version=version,
            supports_keep_alive=supports_keep_alive,
            supports_streaming=supports_streaming,
            max_batch_size=max_batch_size,
            constraints=constraints,
            objectives=objectives,
            exploration=exploration,
            promotion_policy=promotion_policy,
            defaults=defaults,
            measures=measures,
            evaluation_kwargs=evaluation_kwargs,
            estimated_tokens_per_example=_resolve_estimated_tokens_per_example(
                estimated_tokens_per_example
            ),
        )

        # Registered handlers
        self._tvars_handler: Callable[[], dict[str, Any]] | None = None
        self._execute_handler: Callable[..., Any] | None = None
        self._evaluate_handler: Callable[..., Any] | None = None
        self._constraints_handler: Callable[[], dict[str, Any] | list[Any]] | None = (
            None
        )
        self._objectives_handler: Callable[[], list[dict[str, Any]]] | None = None
        self._exploration_handler: Callable[[], dict[str, Any]] | None = None
        self._promotion_policy_handler: Callable[[], dict[str, Any]] | None = None
        self._defaults_handler: Callable[[], dict[str, Any]] | None = None
        self._measures_handler: Callable[[], list[str]] | None = None
        self._evaluation_kwargs_handler: Callable[[], list[Any]] | None = None
        self._evaluate_handler_arity: int | None = None

        # Session management
        self._sessions: dict[str, Session] = {}
        self._started_at: float = time.time()

        # In-process idempotency caches keyed by request_id.
        # Value is (payload fingerprint excluding request_id, response payload).
        # Bounded to prevent unbounded memory growth in long-lived servers.
        self._idempotency_cache_max_size = 1000
        self._execute_idempotency_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._evaluate_idempotency_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        # request_id -> (payload fingerprint, future) for requests still running.
        self._inflight_requests: dict[
            tuple[str, str], tuple[str, asyncio.Future[dict[str, Any]]]
        ] = {}
        self._execution_context_cache: dict[str, _ExecutionContextEntry] = {}

        # Cached config space
        self._cached_tvars: dict[str, Any] | None = None

    def _fingerprint_request(self, request: dict[str, Any]) -> str:
        """Build stable request fingerprint for idempotency checks.

        Excludes request_id because request_id is used as the dedupe key itself.
        """
        payload = {k: v for k, v in request.items() if k != "request_id"}
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except TypeError:
            # Fallback for non-JSON-native values in custom wrappers.
            return repr(payload)

    def tvars[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register TVAR configuration function.

        The decorated function should return a dictionary of TVAR
        definitions following TVL 0.9 specification.

        Example:
            @app.tvars
            def config_space():
                return {
                    "model": {"type": "enum", "values": ["gpt-5.6-sol"]},
                    "temperature": {"type": "float", "range": [0.0, 2.0]},
                }
        """
        self._tvars_handler = func
        self._cached_tvars = None  # Invalidate cache
        return func

    def tunables[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register tunable configuration function.

        Client-friendly alias for tvars(). The decorated function should
        return a dictionary of tunable definitions.

        Example:
            @app.tunables
            def config_space():
                return {
                    "model": {"type": "enum", "values": ["gpt-5.6-sol", "claude-3"]},
                    "temperature": {"type": "float", "range": [0.0, 1.0]},
                }
        """
        return self.tvars(func)

    def execute[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register execution handler.

        The decorated function receives:
            - example_id: str - Identifier for this example
            - data: dict - Input data
            - config: dict - Configuration parameters

        Should return a dict with:
            - output: Any - The result
            - cost_usd: float (optional) - Cost for this execution
            - latency_ms: float (optional) - Processing latency
            - metrics: dict (optional) - Quality metrics (combined mode)

        Example:
            @app.execute
            async def run_agent(example_id: str, data: dict, config: dict) -> dict:
                result = await my_llm_call(data["query"], **config)
                return {"output": result, "cost_usd": 0.002}
        """
        self._execute_handler = func
        return func

    def evaluate[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register evaluation handler.

        The decorated function receives:
            - output: Any - The agent output
            - target: Any - The expected output
            - kwargs: dict - Evaluator-specific request parameters
            - context: EvaluationContext (optional) - Execute-time context

        Should return a dict of metric names to values.

        Example:
            @app.evaluate
            async def score(output: dict, target: dict, kwargs: dict) -> dict:
                accuracy = compute_accuracy(output, target)
                return {"accuracy": accuracy, "safety": 1.0}
        """
        self._evaluate_handler_arity = self._validate_evaluate_signature(func)
        self._evaluate_handler = func
        return func

    def constraints[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register constraints declaration function."""
        self._constraints_handler = func  # type: ignore[assignment]
        return func

    def objectives[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register objectives declaration function."""
        self._objectives_handler = func  # type: ignore[assignment]
        return func

    def exploration[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register exploration declaration function."""
        self._exploration_handler = func  # type: ignore[assignment]
        return func

    def promotion_policy[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register promotion policy declaration function."""
        self._promotion_policy_handler = func  # type: ignore[assignment]
        return func

    def defaults[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register default-config declaration function."""
        self._defaults_handler = func  # type: ignore[assignment]
        return func

    def measures[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register declared measure names function."""
        self._measures_handler = func  # type: ignore[assignment]
        return func

    def evaluation_kwargs[F: Callable[..., Any]](self, func: F) -> F:
        """Decorator to register supported evaluate kwargs."""
        self._evaluation_kwargs_handler = func  # type: ignore[assignment]
        return func

    @staticmethod
    def _validate_evaluate_signature(func: Callable[..., Any]) -> int:
        """Validate supported evaluate callback signatures."""
        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())
        for parameter in parameters:
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise ValueError(
                    "evaluate handlers may only declare positional parameters"
                )
        if len(parameters) not in (2, 3, 4):
            raise ValueError(
                "evaluate handlers must have signature "
                "(output, target), (output, target, kwargs), or "
                "(output, target, kwargs, context)"
            )
        return len(parameters)

    def _resolve_declared_section(
        self,
        handler: Callable[[], Any] | None,
        configured: Any,
    ) -> Any:
        """Resolve section value from decorator handler or constructor config."""
        if handler is None:
            return configured
        if inspect.iscoroutinefunction(handler):
            raise ValueError("declaration handlers must be synchronous functions")
        value = handler()
        if inspect.isawaitable(value):
            # Defensive: close accidental coroutine objects to avoid runtime warnings.
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise ValueError("declaration handlers must be synchronous functions")
        return value

    @staticmethod
    def _is_scalar_value(value: Any) -> bool:
        """Return whether a value is OpenAPI-scalar compatible."""
        return isinstance(value, (str, int, float, bool)) and not isinstance(
            value, complex
        )

    def _normalize_evaluation_kwarg_definition(
        self, spec: Any
    ) -> EvaluationKwargDefinition:
        """Normalize one declared evaluation kwarg definition."""
        if isinstance(spec, EvaluationKwargDefinition):
            definition = spec
        elif isinstance(spec, dict):
            definition = EvaluationKwargDefinition.from_dict(spec)
        else:
            raise ValueError(
                "evaluation_kwargs entries must be dicts or EvaluationKwargDefinition instances"
            )

        if not _IDENTIFIER_RE.match(definition.name):
            raise ValueError(
                f"Invalid evaluation kwarg name '{definition.name}': "
                "must be a valid Python identifier"
            )
        if definition.description is not None and not isinstance(
            definition.description, str
        ):
            raise ValueError("evaluation kwarg description must be a string")
        if definition.domain is not None and not isinstance(definition.domain, dict):
            raise ValueError("evaluation kwarg domain must be a dict")
        if definition.default is not None and not self._is_scalar_value(
            definition.default
        ):
            raise ValueError("evaluation kwarg default must be a scalar value")

        domain = dict(definition.domain or {})
        self._validate_evaluation_kwarg_domain(definition.type, domain)

        if definition.default is not None:
            self._validate_evaluation_kwarg_value(definition, definition.default)

        return EvaluationKwargDefinition(
            name=definition.name,
            type=definition.type,
            description=definition.description,
            domain=domain or None,
            default=definition.default,
        )

    def _validate_evaluation_kwarg_domain(
        self, kwarg_type: str, domain: dict[str, Any]
    ) -> None:
        """Validate the declared domain of one evaluation kwarg definition."""
        allowed_domain_keys = {"values", "range", "resolution"}
        unknown_domain_keys = set(domain) - allowed_domain_keys
        if unknown_domain_keys:
            raise ValueError(
                f"evaluation kwarg domain contains unsupported keys: {sorted(unknown_domain_keys)}"
            )

        if "values" in domain:
            self._validate_evaluation_kwarg_domain_values(domain["values"])
        if "range" in domain:
            self._validate_evaluation_kwarg_domain_range(domain["range"])
        if "resolution" in domain and not self._is_plain_number(domain["resolution"]):
            raise ValueError("evaluation kwarg domain.resolution must be numeric")
        if kwarg_type == "enum" and "values" not in domain:
            raise ValueError("enum evaluation kwargs must declare domain.values")

    @staticmethod
    def _is_plain_number(value: Any) -> bool:
        """Return whether a value is an int or float but not a bool."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _validate_evaluation_kwarg_domain_values(self, values: Any) -> None:
        if not isinstance(values, list) or not values:
            raise ValueError("evaluation kwarg domain.values must be a non-empty list")
        if not all(self._is_scalar_value(item) for item in values):
            raise ValueError(
                "evaluation kwarg domain.values entries must all be scalar values"
            )

    def _validate_evaluation_kwarg_domain_range(self, range_spec: Any) -> None:
        if (
            not isinstance(range_spec, list)
            or len(range_spec) != 2
            or not all(self._is_plain_number(item) for item in range_spec)
        ):
            raise ValueError(
                "evaluation kwarg domain.range must be a two-item numeric list"
            )

    def _resolve_declared_evaluation_kwargs(
        self,
    ) -> list[EvaluationKwargDefinition] | None:
        """Resolve configured or decorated evaluation kwargs definitions."""
        raw_value = self._resolve_declared_section(
            self._evaluation_kwargs_handler, self.config.evaluation_kwargs
        )
        if raw_value is None:
            return None
        if not isinstance(raw_value, list):
            raise ValueError("evaluation_kwargs must be a list")

        normalized = [
            self._normalize_evaluation_kwarg_definition(item) for item in raw_value
        ]
        return normalized or None

    def _validate_evaluation_kwarg_value(
        self,
        definition: EvaluationKwargDefinition,
        value: ScalarValue,
    ) -> None:
        """Validate one incoming evaluation kwarg value."""
        if not self._is_scalar_value(value):
            raise ValueError("evaluation kwarg values must be scalar")

        domain = definition.domain or {}
        values = domain.get("values")
        TraigentService._validate_evaluation_kwarg_type(definition, value, values)

        if values is not None and definition.type != "enum" and value not in values:
            raise ValueError("must be one of the declared values")

        range_spec = domain.get("range")
        if (
            range_spec is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            if value < range_spec[0] or value > range_spec[1]:
                raise ValueError(
                    f"must be within range [{range_spec[0]}, {range_spec[1]}]"
                )

    @staticmethod
    def _validate_evaluation_kwarg_type(
        definition: EvaluationKwargDefinition, value: ScalarValue, values: Any
    ) -> None:
        if definition.type == "bool":
            if not isinstance(value, bool):
                raise ValueError("must be a bool")
        elif definition.type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("must be an int")
        elif definition.type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("must be numeric")
        else:
            TraigentService._validate_evaluation_kwarg_text_type(
                definition, value, values
            )

    @staticmethod
    def _validate_evaluation_kwarg_text_type(
        definition: EvaluationKwargDefinition, value: ScalarValue, values: Any
    ) -> None:
        if definition.type == "str":
            if not isinstance(value, str):
                raise ValueError("must be a string")
        elif definition.type == "enum":
            if values is None or value not in values:
                raise ValueError("must be one of the declared enum values")

    def _validate_effective_evaluation_kwargs(
        self,
        effective_kwargs: dict[str, ScalarValue],
        declared_kwargs: list[EvaluationKwargDefinition] | None,
    ) -> None:
        """Validate incoming evaluation kwargs against declared definitions."""
        if not effective_kwargs:
            return
        if declared_kwargs is None:
            raise BadRequestError(
                "This service does not accept evaluation kwargs",
                error_code="INVALID_EVALUATION_KWARGS",
                details={
                    "invalid_keys": sorted(effective_kwargs),
                    "invalid_values": {},
                    "allowed_keys": [],
                    "allowed_values": {},
                },
            )

        definitions_by_name = {
            definition.name: definition for definition in declared_kwargs
        }
        invalid_keys = sorted(
            key for key in effective_kwargs if key not in definitions_by_name
        )
        invalid_values: dict[str, ScalarValue] = {}
        for key, value in effective_kwargs.items():
            definition = definitions_by_name.get(key)
            if definition is None:
                continue
            try:
                self._validate_evaluation_kwarg_value(definition, value)
            except ValueError:
                invalid_values[key] = value

        if invalid_keys or invalid_values:
            allowed_values = {
                definition.name: definition.domain["values"]
                for definition in declared_kwargs
                if definition.domain is not None and "values" in definition.domain
            }
            raise BadRequestError(
                "Invalid evaluation kwargs provided for this service",
                error_code="INVALID_EVALUATION_KWARGS",
                details={
                    "invalid_keys": invalid_keys,
                    "invalid_values": invalid_values,
                    "allowed_keys": sorted(definitions_by_name),
                    "allowed_values": allowed_values,
                },
            )

    @staticmethod
    def _validate_declared_sections(sections: dict[str, Any]) -> None:
        """Validate types of resolved declarative config-space sections."""
        _type_rules: list[tuple[str, type | tuple[type, ...], str]] = [
            ("constraints", (dict, list), "constraints must be a dict or list"),
            ("objectives", list, "objectives must be a list"),
            ("exploration", dict, "exploration must be a dict"),
            ("promotion_policy", dict, "promotion_policy must be a dict"),
            ("defaults", dict, "defaults must be a dict"),
        ]
        for key, expected, msg in _type_rules:
            value = sections.get(key)
            if value is not None and not isinstance(value, expected):
                raise ValueError(msg)
        measures = sections.get("measures")
        if measures is not None and (
            not isinstance(measures, list)
            or not all(isinstance(m, str) for m in measures)
        ):
            raise ValueError("measures must be a list of strings")

    def get_config_space(self, tunable_id: str | None = None) -> dict[str, Any]:
        """Get TVAR definitions.

        Args:
            tunable_id: Optional tunable the caller asked for. A tunable this
                service does not serve is rejected with ``UNKNOWN_TUNABLE_ID``
                rather than answered with the default space.

        Returns:
            Dictionary with schema_version, tunable_id, and tunables.
        """
        if tunable_id is not None and tunable_id != self.config.tunable_id:
            raise UnknownTunableIdError(
                "Unknown tunable_id. Use GET /traigent/v1/capabilities to list "
                "valid tunable_ids."
            )
        if self._cached_tvars is None and self._tvars_handler is not None:
            tvars_dict = self._tvars_handler()
            if not isinstance(tvars_dict, dict):
                raise ValueError("tunables handler must return a dict")
            self._cached_tvars = self._normalize_tvars(tvars_dict)

        tvars = self._cached_tvars or {}
        tunables = [{"name": name, **spec} for name, spec in tvars.items()]

        sections = {
            "constraints": self._resolve_declared_section(
                self._constraints_handler, self.config.constraints
            ),
            "objectives": self._resolve_declared_section(
                self._objectives_handler, self.config.objectives
            ),
            "exploration": self._resolve_declared_section(
                self._exploration_handler, self.config.exploration
            ),
            "promotion_policy": self._resolve_declared_section(
                self._promotion_policy_handler, self.config.promotion_policy
            ),
            "defaults": self._resolve_declared_section(
                self._defaults_handler, self.config.defaults
            ),
            "measures": self._resolve_declared_section(
                self._measures_handler, self.config.measures
            ),
        }
        evaluation_kwargs = self._resolve_declared_evaluation_kwargs()
        self._validate_declared_sections(sections)

        response: dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "tunable_id": self.config.tunable_id,
            "tunables": tunables,
            "tvars": tunables,  # Backward compatibility alias
            "constraints": sections["constraints"] or {},
        }
        for key in (
            "objectives",
            "exploration",
            "promotion_policy",
            "defaults",
            "measures",
        ):
            if sections[key] is not None:
                response[key] = sections[key]
        if evaluation_kwargs is not None:
            response["evaluation_kwargs"] = [
                definition.to_dict() for definition in evaluation_kwargs
            ]
        if self.config.estimated_tokens_per_example is not None:
            response["estimated_tokens_per_example"] = (
                self.config.estimated_tokens_per_example.to_dict()
            )
        return response

    @staticmethod
    def _normalize_single_tvar(spec: Any) -> dict[str, Any]:
        """Normalize a single TVAR spec to TVL format."""
        if isinstance(spec, list):
            return {"type": "enum", "domain": {"values": spec}}
        if not isinstance(spec, dict):
            return {"type": "str", "domain": {"values": [spec]}, "default": spec}

        normalized_spec = dict(spec)
        domain = normalized_spec.get("domain", {})
        if not isinstance(domain, dict):
            domain = {}

        # Accept wrapper-friendly top-level values/range/resolution and
        # normalize to contract-compliant nested domain shape.
        for key in ("values", "range", "resolution"):
            if key in normalized_spec and key not in domain:
                domain[key] = normalized_spec.pop(key)

        normalized_spec["domain"] = domain
        return normalized_spec

    def _normalize_tvars(self, tvars: dict[str, Any]) -> dict[str, Any]:
        """Normalize TVAR definitions to TVL format."""
        normalized: dict[str, dict[str, Any]] = {}
        for name, spec in tvars.items():
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(
                    f"Invalid TVAR name '{name}': must be a valid Python identifier "
                    f"(pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$)"
                )
            normalized[name] = self._normalize_single_tvar(spec)
        return normalized

    def get_capabilities(self) -> dict[str, Any]:
        """Get service capabilities.

        Returns:
            Dictionary with version and feature flags.
        """
        return {
            "version": self.config.version,
            "supports_evaluate": self._evaluate_handler is not None,
            "supports_evaluation_kwargs": bool(
                self._resolve_declared_evaluation_kwargs()
            ),
            "supports_keep_alive": self.config.supports_keep_alive,
            "supports_streaming": self.config.supports_streaming,
            "max_batch_size": self.config.max_batch_size,
            "max_payload_bytes": None,
            # Explicitly advertise supported capability IDs for multi-capability clients.
            "tunable_ids": [self.config.tunable_id],
        }

    def get_health(self) -> dict[str, Any]:
        """Get service health status.

        Returns:
            Dictionary with status and details.
        """
        uptime_seconds = max(0.0, time.time() - self._started_at)
        return {
            "status": "healthy",
            "version": self.config.version,
            "uptime_seconds": uptime_seconds,
            "details": {
                "tunable_id": self.config.tunable_id,
                "active_sessions": len(self._sessions),
            },
        }

    def _check_idempotency_cache(
        self,
        cache: dict[str, tuple[str, dict[str, Any]]],
        request_id: str,
        fingerprint: str,
        operation: str,
    ) -> dict[str, Any] | None:
        """Return cached response if request_id was seen before, or None."""
        cached = cache.get(request_id)
        if cached is None:
            return None
        cached_fingerprint, cached_response = cached
        if cached_fingerprint != fingerprint:
            raise IdempotencyConflictError(
                f"request_id reuse with different payload in {operation} request"
            )
        return copy.deepcopy(cached_response)

    def _store_in_idempotency_cache(
        self,
        cache: dict[str, tuple[str, dict[str, Any]]],
        request_id: str,
        fingerprint: str,
        response: dict[str, Any],
    ) -> None:
        """Store a response in an idempotency cache with FIFO eviction."""
        if len(cache) >= self._idempotency_cache_max_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[request_id] = (fingerprint, copy.deepcopy(response))

    def _store_execution_context(
        self,
        execution_id: str,
        config: dict[str, Any],
        session_id: str | None,
    ) -> None:
        """Store execution context for optional later evaluation lookup."""
        if len(self._execution_context_cache) >= self._idempotency_cache_max_size:
            oldest_key = next(iter(self._execution_context_cache))
            del self._execution_context_cache[oldest_key]
        self._execution_context_cache[execution_id] = _ExecutionContextEntry(
            execution_id=execution_id,
            execution_config=copy.deepcopy(config),
            execute_session_id=session_id,
            tunable_id=self.config.tunable_id,
        )

    def _lookup_execution_context(
        self, execution_id: str | None, evaluate_session_id: str | None
    ) -> EvaluationContext:
        """Best-effort lookup of cached execution context."""
        if not execution_id:
            return EvaluationContext(evaluate_session_id=evaluate_session_id)

        entry = self._execution_context_cache.get(execution_id)
        if entry is None:
            logger.warning(
                "Evaluate request referenced unknown or evicted execution_id %s; continuing without execution_config",
                execution_id,
            )
            return EvaluationContext(
                execution_id=execution_id,
                execution_config=None,
                evaluate_session_id=evaluate_session_id,
                tunable_id=self.config.tunable_id,
            )
        return EvaluationContext(
            execution_id=entry.execution_id,
            execution_config=copy.deepcopy(entry.execution_config),
            execute_session_id=entry.execute_session_id,
            evaluate_session_id=evaluate_session_id,
            tunable_id=entry.tunable_id,
        )

    @staticmethod
    def _determine_batch_status(total: int, failed: int) -> str:
        """Return 'completed', 'partial', or 'failed' for a batch result."""
        if failed == 0:
            return "completed"
        if failed < total:
            return "partial"
        return "failed"

    @staticmethod
    def _parse_example_input(inp: Any) -> tuple[str, Any]:
        """Extract (input id, data) from a single execute input.

        The contract key is ``input_id``; ``example_id`` is the deprecated alias.
        """
        if isinstance(inp, dict):
            raw_id = inp.get("input_id") or inp.get("example_id")
            example_id = str(raw_id) if raw_id else str(uuid.uuid4())
            return example_id, inp.get("data", inp)
        return str(uuid.uuid4()), inp

    @staticmethod
    def _reported_cost(value: Any) -> float | None:
        """Return a handler's reported cost, or ``None`` when it reported none.

        "The handler did not say" is not a measured $0 (Traigent#2517): only a
        finite, non-negative number is a reported cost. ``None``, a missing
        key, booleans, non-numbers, NaN/Inf and negatives are all "not
        reported". A reported ``0`` is a measured zero and is kept.
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            cost = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        return cost

    @staticmethod
    def _unpack_execute_result(result: Any) -> dict[str, Any]:
        """Normalize a single execute-handler result into a flat dict.

        ``cost_usd`` is ``None`` when the handler did not report one.
        """
        if not isinstance(result, dict):
            return {
                "output": result,
                "cost_usd": None,
                "latency_ms": None,
                "metrics": {},
            }
        metrics = (
            result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        )
        return {
            "output": result.get("output", result),
            "cost_usd": TraigentService._reported_cost(result.get("cost_usd")),
            "latency_ms": result.get("latency_ms"),
            "metrics": metrics,
        }

    @staticmethod
    def _build_operational_metrics(
        *, priced_cost: float, cost_complete: bool, latency_ms: float
    ) -> dict[str, Any]:
        """Aggregate operational metrics for an execute response.

        ``total_cost_usd`` is a required number in the hybrid-mode contract, so
        it is always present. It is a *total* only when every executed example
        reported a cost. Otherwise it is the priced lower bound and the
        response says so with ``cost_unmeasured`` (a documented-extensible
        custom operational metric): a consumer that honours the flag treats the
        number as "at least", never as the spend. The contract has no
        first-class way to say "unknown total"; until it does, the flag is
        the signal.
        """
        metrics: dict[str, Any] = {
            "total_cost_usd": priced_cost,
            "cost_usd": priced_cost,
            "latency_ms": latency_ms,
        }
        if not cost_complete:
            metrics["cost_unmeasured"] = 1.0
        return metrics

    @staticmethod
    def _execute_output_item(
        example_id: str, unpacked: dict[str, Any]
    ) -> dict[str, Any]:
        output_item: dict[str, Any] = {
            "input_id": example_id,
            # Deprecated alias, kept for the SDK's own legacy HTTP client.
            "example_id": example_id,
            "output": unpacked["output"],
        }
        # ``cost_usd`` is optional per output item: omit it rather than
        # send a fabricated 0.0 for a handler that reported nothing.
        if unpacked["cost_usd"] is not None:
            output_item["cost_usd"] = unpacked["cost_usd"]
        if isinstance(unpacked["latency_ms"], (int, float)):
            output_item["latency_ms"] = float(unpacked["latency_ms"])
        if unpacked["metrics"]:
            output_item["metrics"] = unpacked["metrics"]
        return output_item

    @staticmethod
    def _batch_error(
        status: str, failed_example_ids: list[str], mode: str
    ) -> dict[str, Any]:
        if mode == "EXECUTION":
            return {
                "code": "EXECUTION_PARTIAL_FAILURE"
                if status == "partial"
                else "EXECUTION_FAILED",
                "message": "One or more examples failed during execution",
                "failed_examples": failed_example_ids,
            }
        return {
            "code": "EVALUATION_PARTIAL_FAILURE"
            if status == "partial"
            else "EVALUATION_FAILED",
            "message": "One or more items failed during evaluation",
            "failed_examples": failed_example_ids,
        }

    @staticmethod
    def _select_execute_inputs(request: dict[str, Any]) -> tuple[str, Any]:
        """Pick the input array: ``inputs`` wins over the deprecated ``examples``."""
        if "inputs" in request:
            # Present wins, even when null or wrong-shaped: that is INVALID_INPUTS.
            return "inputs", request["inputs"]
        if request.get("examples") is not None:
            return "examples", request["examples"]
        return "inputs (or the deprecated alias examples)", None

    def _validate_execute_batch(
        self, examples: Any, tunable_id: Any, field_name: str = "examples"
    ) -> None:
        if not isinstance(examples, list) or len(examples) == 0:
            raise InvalidInputsError(f"{field_name} must be a non-empty list")

        if tunable_id != self.config.tunable_id:
            raise ValueError(
                "tunable_id mismatch: the request's tunable_id does not match "
                f"this service ('{self.config.tunable_id}')"
            )

        max_batch = self.config.max_batch_size
        if max_batch and len(examples) > max_batch:
            raise ValueError(
                f"Batch size {len(examples)} exceeds max_batch_size {max_batch}"
            )

    @staticmethod
    def _accumulate_execute_cost(
        unpacked: dict[str, Any], priced_cost: float, cost_complete: bool
    ) -> tuple[float, bool]:
        if unpacked["cost_usd"] is None:
            cost_complete = False
        else:
            priced_cost += unpacked["cost_usd"]
        return priced_cost, cost_complete

    def _attach_execute_metadata(
        self,
        response: dict[str, Any],
        all_quality_metrics: list[dict[str, float]],
        session_id: Any,
    ) -> None:
        if all_quality_metrics:
            response["quality_metrics"] = self._aggregate_metrics(all_quality_metrics)

        if session_id:
            response["session_id"] = session_id

    async def _run_idempotent(
        self,
        operation: str,
        request: dict[str, Any],
        impl: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run ``impl`` once per ``request_id``, including concurrent requests.

        The request_id is reserved (with its payload fingerprint) before the
        handler runs. A concurrent request with the same id and a different
        payload gets 409 immediately; one with the same payload waits for the
        first outcome and never runs the handler a second time. If the first run
        raises, waiters get the same exception; if it is cancelled they get a
        retryable 503. The reservation is always removed afterwards, so only a
        new request arriving later may run again.
        """
        if "request_id" not in request:
            return await impl(request)
        request_id = str(request["request_id"])
        fingerprint = self._fingerprint_request(request)
        key = (operation, request_id)
        reserved = self._inflight_requests.get(key)
        if reserved is not None:
            reserved_fingerprint, existing = reserved
            if reserved_fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    f"request_id reuse with different payload in {operation} request"
                )
            # Same payload: share the first run's outcome, success or failure.
            # A waiter never re-runs the batch (that could repeat cost or side
            # effects); it re-raises the first run's exception.
            return copy.deepcopy(await asyncio.shield(existing))
        if len(self._inflight_requests) >= MAX_INFLIGHT_REQUESTS:
            raise ServiceUnavailableError(
                "Too many in-flight requests; retry shortly",
                retry_after=1,
                error_code="TOO_MANY_IN_FLIGHT_REQUESTS",
            )
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._inflight_requests[key] = (fingerprint, future)
        try:
            outcome = await impl(request)
        except asyncio.CancelledError:
            future.set_exception(
                ServiceUnavailableError(
                    "The original request was cancelled before it finished; "
                    "retry the request",
                    retry_after=1,
                    error_code="REQUEST_CANCELLED",
                )
            )
            future.exception()  # mark retrieved when nobody is waiting
            raise
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()
            raise
        finally:
            self._inflight_requests.pop(key, None)
        future.set_result(copy.deepcopy(outcome))
        return outcome

    async def handle_execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle execute request (see ``_handle_execute_impl``)."""
        if self._execute_handler is None:
            raise ValueError("No execute handler registered")
        return await self._run_idempotent("execute", request, self._handle_execute_impl)

    async def handle_evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle evaluate request (see ``_handle_evaluate_impl``)."""
        if self._evaluate_handler is None:
            raise ValueError("No evaluate handler registered")
        return await self._run_idempotent(
            "evaluate", request, self._handle_evaluate_impl
        )

    async def _handle_execute_impl(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle execute request.

        Args:
            request: Execute request with tunable_id, config, and examples.

        Returns:
            Execute response with outputs and metrics.

        Raises:
            ValueError: If no execute handler registered.
        """
        if self._execute_handler is None:
            raise ValueError("No execute handler registered")

        request_id = str(request.get("request_id", str(uuid.uuid4())))
        request_fingerprint = self._fingerprint_request(request)
        cached = self._check_idempotency_cache(
            self._execute_idempotency_cache, request_id, request_fingerprint, "execute"
        )
        if cached is not None:
            return cached

        tunable_id = request.get("tunable_id", self.config.tunable_id)
        config = request.get("config", {})
        input_field, examples = self._select_execute_inputs(request)
        session_id = self._valid_session_id_or_none(request.get("session_id"))

        TraigentService._validate_execute_batch(self, examples, tunable_id, input_field)

        if session_id:
            self._touch_or_create_session(session_id)

        start_time = time.time()
        outputs: list[dict[str, Any]] = []
        # Sum of the costs handlers actually reported, and whether every
        # executed example reported one. An example that did not (or that
        # raised before it could) leaves the batch's spend unknown: the sum
        # is then only a lower bound, never claimed as the total (#2517).
        priced_cost = 0.0
        cost_complete = True
        all_quality_metrics: list[dict[str, float]] = []
        failed_example_ids: list[str] = []

        for inp in cast(list[Any], examples):
            example_id, data = self._parse_example_input(inp)
            try:
                result = self._execute_handler(example_id, data, config)
                if inspect.isawaitable(result):
                    result = await result

                unpacked = self._unpack_execute_result(result)
                priced_cost, cost_complete = TraigentService._accumulate_execute_cost(
                    unpacked, priced_cost, cost_complete
                )
                if unpacked["metrics"]:
                    all_quality_metrics.append(unpacked["metrics"])

                output_item = TraigentService._execute_output_item(example_id, unpacked)
                outputs.append(output_item)

            except HybridAPIError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Execute failed for {example_id}: {e}")
                failed_example_ids.append(example_id)
                # A handler that raised may have spent before it did: a
                # failure alone does not establish zero cost.
                cost_complete = False
                outputs.append(
                    {"input_id": example_id, "example_id": example_id, "error": str(e)}
                )

        elapsed_ms = (time.time() - start_time) * 1000
        status = self._determine_batch_status(len(outputs), len(failed_example_ids))

        execution_id = str(uuid.uuid4())
        response: dict[str, Any] = {
            "request_id": request_id,
            "execution_id": execution_id,
            "status": status,
            "outputs": outputs,
            "operational_metrics": self._build_operational_metrics(
                priced_cost=priced_cost,
                cost_complete=cost_complete,
                latency_ms=elapsed_ms,
            ),
        }

        if failed_example_ids:
            response["error"] = TraigentService._batch_error(
                status, failed_example_ids, "EXECUTION"
            )

        TraigentService._attach_execute_metadata(
            self, response, all_quality_metrics, session_id
        )

        self._store_execution_context(execution_id, config, session_id)

        self._store_in_idempotency_cache(
            self._execute_idempotency_cache, request_id, request_fingerprint, response
        )
        return response

    @staticmethod
    def _unwrap_value_envelope(value: Any) -> Any:
        """Unwrap exactly ``{"value": x}`` (one key) back to ``x``.

        The hybrid client wraps scalar targets and outputs as ``{"value": x}``
        so the evaluate payload matches the object-typed wire contract (#2904).
        Handlers keep receiving the original scalar. Any other object, including
        one with ``value`` plus further keys, passes through unchanged.
        """
        if isinstance(value, dict) and len(value) == 1 and "value" in value:
            return value["value"]
        return value

    @staticmethod
    def _resolve_evaluation_fields(evaluation: dict[str, Any]) -> tuple[str, Any, Any]:
        """Extract (example_id, output, target) from an evaluation item.

        Contract: a ``target`` or ``output`` that is exactly ``{"value": x}``
        is unwrapped to ``x`` before it reaches the evaluation handler, so
        customer handlers see the original scalar (#2904).
        """
        raw_id = evaluation.get("input_id") or evaluation.get("example_id")
        example_id = str(raw_id) if raw_id else str(uuid.uuid4())

        output = evaluation.get("output")
        if output is None and "output_id" in evaluation:
            output = {"output_id": evaluation["output_id"]}
        target = evaluation.get("target")
        if target is None and "target_id" in evaluation:
            target = {"target_id": evaluation["target_id"]}

        return (
            example_id,
            TraigentService._unwrap_value_envelope(output),
            TraigentService._unwrap_value_envelope(target),
        )

    def _prepare_evaluation_kwargs(
        self, raw_kwargs: Any, raw_config_alias: Any
    ) -> dict[str, ScalarValue]:
        declared_evaluation_kwargs = self._resolve_declared_evaluation_kwargs()
        effective_kwargs_raw = (
            raw_kwargs if raw_kwargs is not None else raw_config_alias or {}
        )
        if not isinstance(effective_kwargs_raw, dict):
            raise BadRequestError(
                "evaluate kwargs must be an object",
                error_code="INVALID_EVALUATION_KWARGS",
            )
        if not all(
            self._is_scalar_value(value) for value in effective_kwargs_raw.values()
        ):
            raise BadRequestError(
                "evaluation kwargs must contain only scalar values",
                error_code="INVALID_EVALUATION_KWARGS",
            )
        effective_kwargs: dict[str, ScalarValue] = {
            str(key): value for key, value in effective_kwargs_raw.items()
        }
        self._validate_effective_evaluation_kwargs(
            effective_kwargs, declared_evaluation_kwargs
        )
        if declared_evaluation_kwargs:
            for defn in declared_evaluation_kwargs:
                if defn.default is not None and defn.name not in effective_kwargs:
                    effective_kwargs[defn.name] = defn.default
        return effective_kwargs

    def _validate_evaluation_batch(self, evaluations: Any, tunable_id: Any) -> None:
        if not isinstance(evaluations, list):
            raise ValueError("evaluations must be a list")

        if tunable_id != self.config.tunable_id:
            raise ValueError(
                "tunable_id mismatch: the request's tunable_id does not match "
                f"this service ('{self.config.tunable_id}')"
            )

    def _invoke_evaluate_handler(
        self,
        output: Any,
        target: Any,
        effective_kwargs: dict[str, ScalarValue],
        evaluation_context: Any,
    ) -> Any:
        if self._evaluate_handler_arity == 2:
            return cast(Callable[..., Any], self._evaluate_handler)(output, target)
        elif self._evaluate_handler_arity == 3:
            return cast(Callable[..., Any], self._evaluate_handler)(
                output, target, effective_kwargs
            )
        else:
            return cast(Callable[..., Any], self._evaluate_handler)(
                output, target, effective_kwargs, evaluation_context
            )

    async def _handle_evaluate_impl(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle evaluate request.

        Args:
            request: Evaluate request with evaluations.

        Returns:
            Evaluate response with results and aggregate metrics.

        Raises:
            ValueError: If no evaluate handler registered.
        """
        if self._evaluate_handler is None:
            raise ValueError("No evaluate handler registered")

        request_id = str(request.get("request_id", str(uuid.uuid4())))
        request_fingerprint = self._fingerprint_request(request)
        cached = self._check_idempotency_cache(
            self._evaluate_idempotency_cache,
            request_id,
            request_fingerprint,
            "evaluate",
        )
        if cached is not None:
            return cached

        tunable_id = request.get("tunable_id", self.config.tunable_id)
        execution_id = request.get("execution_id")
        evaluations = request.get("evaluations", [])
        raw_kwargs = request.get("kwargs")
        raw_config_alias = request.get("config")
        session_id = self._valid_session_id_or_none(request.get("session_id"))

        TraigentService._validate_evaluation_batch(self, evaluations, tunable_id)

        if session_id:
            self._touch_or_create_session(session_id)

        effective_kwargs = TraigentService._prepare_evaluation_kwargs(
            self, raw_kwargs, raw_config_alias
        )
        evaluation_context = self._lookup_execution_context(execution_id, session_id)

        results: list[dict[str, Any]] = []
        all_metrics: list[dict[str, float]] = []
        failed_example_ids: list[str] = []

        for evaluation in evaluations:
            example_id, output, target = self._resolve_evaluation_fields(evaluation)
            try:
                result = TraigentService._invoke_evaluate_handler(
                    self, output, target, effective_kwargs, evaluation_context
                )
                if inspect.isawaitable(result):
                    result = await result

                metrics = result if isinstance(result, dict) else {"score": result}
                all_metrics.append(metrics)
                results.append(
                    {
                        "input_id": example_id,
                        # Deprecated alias, kept for the legacy HTTP client.
                        "example_id": example_id,
                        "metrics": metrics,
                    }
                )

            except HybridAPIError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Evaluate failed for {example_id}: {e}")
                failed_example_ids.append(example_id)
                results.append(
                    {
                        "input_id": example_id,
                        "example_id": example_id,
                        "metrics": {},
                        "error": str(e),
                    }
                )

        status = self._determine_batch_status(len(results), len(failed_example_ids))

        response: dict[str, Any] = {
            "request_id": request_id,
            "status": status,
            "results": results,
            "aggregate_metrics": self._compute_aggregate_metrics(all_metrics),
        }

        if failed_example_ids:
            response["error"] = TraigentService._batch_error(
                status, failed_example_ids, "EVALUATION"
            )

        self._store_in_idempotency_cache(
            self._evaluate_idempotency_cache, request_id, request_fingerprint, response
        )
        return response

    @staticmethod
    def _collect_numeric_values(
        metrics_list: list[dict[str, float]],
    ) -> dict[str, list[float]]:
        """Collect finite numeric values grouped by metric name.

        Filters out bools, NaN, and Inf.
        """
        by_name: dict[str, list[float]] = {}
        for metrics in metrics_list:
            for name, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
                by_name.setdefault(name, []).append(float(value))
        return by_name

    def _aggregate_metrics(
        self, metrics_list: list[dict[str, float]]
    ) -> dict[str, float]:
        """Aggregate per-example metrics to single values."""
        if not metrics_list:
            return {}
        return {
            name: sum(vals) / len(vals)
            for name, vals in self._collect_numeric_values(metrics_list).items()
        }

    def _compute_aggregate_metrics(
        self, metrics_list: list[dict[str, float]]
    ) -> dict[str, dict[str, float | int]]:
        """Compute wire aggregate statistics (mean, std, n) for each metric.

        The published hybrid schemas (docs/hybrid-mode-schemas.json,
        docs/hybrid-mode-openapi.yaml) require ``std`` to be a number, so a
        single observation (n == 1) is emitted as 0.0 on the wire. Sample std
        is undefined there; the client-side describe statistics
        (``hybrid_api.py``) keep it as None internally. A nullable ``std`` is a
        later cross-repo schema change.
        """
        if not metrics_list:
            return {}

        import statistics

        aggregates: dict[str, dict[str, float | int]] = {}
        for name, values in self._collect_numeric_values(metrics_list).items():
            n = len(values)
            mean = sum(values) / n
            std = statistics.stdev(values) if n > 1 else 0.0
            aggregates[name] = {"mean": mean, "std": std, "n": n}
        return aggregates

    def handle_keep_alive(self, session_id: str) -> bool:
        """Handle keep-alive request.

        Args:
            session_id: Session to keep alive.

        Returns:
            True if session is alive, False if not found.
        """
        if not self.config.supports_keep_alive:
            return False

        return self._touch_or_create_session(session_id)

    @staticmethod
    def _is_valid_session_id(session_id: Any) -> bool:
        """A non-empty string within the contract's length limit."""
        return (
            isinstance(session_id, str) and 0 < len(session_id) <= MAX_SESSION_ID_LENGTH
        )

    @staticmethod
    def _valid_session_id_or_none(session_id: Any) -> str | None:
        """An invalid session id is treated as absent: never echoed or cached."""
        return session_id if TraigentService._is_valid_session_id(session_id) else None

    def _touch_or_create_session(self, session_id: str) -> bool:
        """Touch an existing session or create it when keep-alive is enabled."""
        if not self._is_valid_session_id(session_id):
            return False

        session = self._sessions.get(session_id)
        if session is not None:
            if time.time() - session.last_activity > SESSION_TTL_SECONDS:
                # Expired: evict, then treat as absent rather than reviving it.
                self._sessions.pop(session_id, None)
            else:
                session.touch()
                return True

        if not self.config.supports_keep_alive:
            return False

        self._add_session(session_id)
        return True

    def _add_session(self, session_id: str) -> None:
        """Insert a session, evicting expired then least-recently-active ones."""
        sessions = self._sessions
        if len(sessions) >= MAX_SESSIONS:
            cutoff = time.time() - SESSION_TTL_SECONDS
            for sid in [k for k, v in sessions.items() if v.last_activity < cutoff]:
                sessions.pop(sid, None)
        while len(sessions) >= MAX_SESSIONS:
            oldest = min(sessions, key=lambda k: sessions[k].last_activity)
            sessions.pop(oldest, None)
        sessions[session_id] = Session(session_id=session_id)

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session.

        Returns:
            New session ID.
        """
        session_id = session_id or str(uuid.uuid4())
        if not self._is_valid_session_id(session_id):
            raise ValueError(
                f"session_id must be a string of 1-{MAX_SESSION_ID_LENGTH} characters"
            )
        self._add_session(session_id)
        return session_id

    def run(
        self,
        host: str = "0.0.0.0",  # nosec B104 - intentional for server binding
        port: int = 8080,
        server: Literal["uvicorn", "hypercorn"] = "uvicorn",
    ) -> None:
        """Start HTTP server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            server: ASGI server to use (uvicorn or hypercorn).
        """
        from traigent.wrapper.server import create_app, run_server

        app = create_app(self)
        run_server(app, host=host, port=port, server=server)
