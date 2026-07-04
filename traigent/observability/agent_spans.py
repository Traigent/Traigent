"""Public helpers for SDK workflow trace spans emitted inside optimization trials."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from traigent.config.context import get_workflow_trace_context
from traigent.integrations.observability.workflow_traces import (
    SpanPayload,
    SpanStatus,
    SpanType,
)
from traigent.security.redaction import is_sensitive_key_name
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:/@+-]{1,128}$")
_SPAN_TYPE_ALIASES = {
    "agent": SpanType.NODE.value,
}
_VALID_SPAN_TYPES = {span_type.value for span_type in SpanType}


def _is_safe_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _coerce_non_negative_int(value: int | None, *, field_name: str) -> int:
    if value is None:
        return 0
    if not _is_safe_number(value):
        logger.debug("Ignoring non-numeric %s for agent span", field_name)
        return 0
    coerced = int(value)
    if coerced < 0:
        logger.debug("Ignoring negative %s for agent span", field_name)
        return 0
    return coerced


def _coerce_non_negative_float(value: float | None, *, field_name: str) -> float:
    if value is None:
        return 0.0
    if not _is_safe_number(value):
        logger.debug("Ignoring non-numeric %s for agent span", field_name)
        return 0.0
    coerced = float(value)
    if coerced < 0:
        logger.debug("Ignoring negative %s for agent span", field_name)
        return 0.0
    return coerced


def _normalize_node_id(node_id: str) -> str | None:
    if not isinstance(node_id, str):
        logger.debug("Skipping agent span: node_id must be a string")
        return None

    normalized = node_id.strip()
    if not normalized:
        logger.debug("Skipping agent span: node_id is empty")
        return None
    if normalized in {"__start__", "__end__"}:
        logger.debug("Skipping agent span for reserved workflow node %s", normalized)
        return None
    return normalized


def _normalize_span_type(span_type: str) -> str:
    raw_span_type = str(span_type).strip().lower()
    raw_span_type = _SPAN_TYPE_ALIASES.get(raw_span_type, raw_span_type)
    if raw_span_type in _VALID_SPAN_TYPES:
        return raw_span_type

    logger.debug("Unknown agent span_type %r; defaulting to node", span_type)
    return SpanType.NODE.value


def _is_sensitive_metadata_key(key: str) -> bool:
    """Delegates to the canonical `traigent.security.redaction` keyword set."""
    return is_sensitive_key_name(key)


def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, float | int]:
    sanitized: dict[str, float | int] = {}
    if not metadata:
        return sanitized

    for raw_key, value in metadata.items():
        key = str(raw_key)
        if _is_sensitive_metadata_key(key):
            logger.debug("Dropping sensitive agent span metadata key %s", key)
            continue
        if not _is_safe_number(value):
            logger.debug("Dropping non-numeric agent span metadata key %s", key)
            continue
        sanitized[key] = int(value) if isinstance(value, int) else float(value)

    return sanitized


def _sanitize_model(model: str | None) -> str | None:
    if model is None:
        return None
    if not isinstance(model, str):
        logger.debug("Dropping non-string model for agent span")
        return None

    candidate = model.strip()
    if not candidate or not _MODEL_ID_PATTERN.fullmatch(candidate):
        logger.debug("Dropping unsafe model identifier for agent span")
        return None
    return candidate


def add_agent_span(
    node_id: str,
    *,
    span_type: str = "agent",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    model: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Add an agent workflow span for the active optimization trial.

    The helper is safe to call from user code: when no optimization trial or
    workflow trace transport is active, it logs at DEBUG and returns.
    """

    trace_context = get_workflow_trace_context()
    if not trace_context:
        logger.debug("Skipping agent span: no active optimization trial context")
        return

    workflow_trace_manager = trace_context.get("workflow_trace_manager")
    if workflow_trace_manager is None:
        logger.debug("Skipping agent span: no workflow trace manager in trial context")
        return

    if not bool(getattr(workflow_trace_manager, "is_enabled", False)):
        logger.debug("Skipping agent span: workflow trace collection is disabled")
        return

    normalized_node_id = _normalize_node_id(node_id)
    if normalized_node_id is None:
        return

    configuration_run_id = trace_context.get("configuration_run_id")
    trace_id = trace_context.get("workflow_trace_id")
    if not configuration_run_id or not trace_id:
        logger.debug("Skipping agent span: trial context has no trace/run linkage")
        return

    safe_metadata: dict[str, Any] = _sanitize_metadata(metadata)
    safe_latency_ms = _coerce_non_negative_float(latency_ms, field_name="latency_ms")
    if safe_latency_ms:
        safe_metadata["latency_ms"] = safe_latency_ms
    safe_model = _sanitize_model(model)
    if safe_model is not None:
        safe_metadata["model"] = safe_model

    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(milliseconds=safe_latency_ms)

    span = SpanPayload(
        span_id=uuid.uuid4().hex[:16],
        trace_id=str(trace_id),
        configuration_run_id=str(configuration_run_id),
        span_name=normalized_node_id,
        span_type=_normalize_span_type(span_type),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        node_id=normalized_node_id,
        status=SpanStatus.COMPLETED.value,
        input_tokens=_coerce_non_negative_int(input_tokens, field_name="input_tokens"),
        output_tokens=_coerce_non_negative_int(
            output_tokens, field_name="output_tokens"
        ),
        cost_usd=_coerce_non_negative_float(cost_usd, field_name="cost_usd"),
        metadata=safe_metadata,
    )

    try:
        workflow_trace_manager.register_node(
            normalized_node_id,
            node_type="agent",
        )
        workflow_trace_manager.collect_span(span)
    except Exception:
        logger.warning("Failed to collect agent workflow span", exc_info=True)
