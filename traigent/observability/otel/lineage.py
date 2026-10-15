"""Per-span context stamping: caller-supplied attributes and optimizer lineage.

Only identifiers are stamped.  Optimizer payloads (configs, metadata, scores)
never go onto spans: optimization data and content-bearing observability stay
separate.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span

from traigent.config.context import get_trial_context, get_workflow_trace_context
from traigent.observability.config import most_restrictive_content_mode
from traigent.observability.otel import contract as C


@dataclass(frozen=True)
class _CallerAttributes:
    session_id: str | None = None
    user_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    prompt_name: str | None = None
    prompt_version: int | None = None
    content_mode: str | None = None


_caller: contextvars.ContextVar[_CallerAttributes | None] = contextvars.ContextVar(
    "traigent_otel_caller_attributes", default=None
)


def merged_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    prompt_reference: Mapping[str, Any] | None = None,
    content_mode: str | None = None,
) -> _CallerAttributes:
    """The caller attributes that WOULD be active after nesting these values.

    Pure: it reads the current context but does not change it.  ``content_mode``
    only ever tightens the mode already in effect (most restrictive wins).
    """
    base = _caller.get() or _CallerAttributes()
    ref = prompt_reference or {}
    version = ref.get("version")
    effective = base.content_mode
    if content_mode is not None:
        effective = (
            content_mode
            if effective is None
            else most_restrictive_content_mode(effective, content_mode)
        )
    return _CallerAttributes(
        session_id=session_id if session_id is not None else base.session_id,
        user_id=user_id if user_id is not None else base.user_id,
        tags=tuple(tags) if tags is not None else base.tags,
        metadata=(
            tuple(dict(metadata).items()) if metadata is not None else base.metadata
        ),
        prompt_name=ref.get("name", base.prompt_name),
        prompt_version=version if isinstance(version, int) else base.prompt_version,
        content_mode=effective,
    )


@contextlib.contextmanager
def activate(state: _CallerAttributes) -> Iterator[None]:
    """Make ``state`` current for the ``with`` body ONLY (token set and reset here).

    Never hold this across a ``yield``: a generator wrapper re-enters it around
    every resume so a suspended stream owns no context state.
    """
    token = _caller.set(state)
    try:
        yield
    finally:
        _caller.reset(token)


@contextlib.contextmanager
def attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    prompt_reference: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Attach attributes to every span started inside the ``with`` block.

    Nesting merges (inner wins per field).  ``metadata`` is content: it is
    exported only where the content mode allows.  The previous value is always
    restored, including on exceptions.

    The merge with the ambient state happens on ENTRY, not when the object is
    created: a scope prepared early and entered later (under another session or
    a tighter ``content_mode`` override) sees the state in effect where it is
    entered, so it can neither restore a captured session nor loosen a mode.
    """
    with activate(
        merged_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
            prompt_reference=prompt_reference,
        )
    ):
        yield


def set_current(state: _CallerAttributes) -> None:
    """Set ``state`` current WITHOUT a reset token.

    Only for a context that is discarded as a whole (a stream's private
    ``contextvars.Context``), where there is nothing to restore.
    """
    _caller.set(state)


def _str_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def current_lineage() -> dict[str, str]:
    """Active optimization-trial ids visible from THIS context (else empty)."""
    out: dict[str, str] = {}
    trial = get_trial_context()
    if isinstance(trial, dict):
        trial_id = _str_id(trial.get("trial_id"))
        if trial_id:
            out[C.ATTR_TRIAL_ID] = trial_id
    workflow = get_workflow_trace_context()
    if isinstance(workflow, dict):
        opt_run = _str_id(workflow.get("workflow_trace_id"))
        if opt_run:
            out[C.ATTR_OPTIMIZATION_SESSION_ID] = opt_run
    return out


def stamp_span(span: Span, *, metadata_mode: str = "metadata") -> None:
    """Called from ``on_start`` for EVERY span, including third-party ones.

    ``metadata_mode`` is the exporter's configured mode.  A tighter mode that an
    enclosing ``observe(content_mode=...)`` put in context governs this span
    too: nothing is stamped that the tighter mode forbids, and the span carries
    the declaration so every exporter of the provider can honour it.
    """
    if not span.is_recording():
        return
    for key, value in current_lineage().items():
        span.set_attribute(key, value)
    caller = _caller.get()
    if caller is None:
        return
    if caller.content_mode is not None:
        effective = most_restrictive_content_mode(metadata_mode, caller.content_mode)
        if effective != metadata_mode:
            span.set_attribute(C.CONTENT_MODE_ATTRIBUTE, effective)
        metadata_mode = effective
    if caller.session_id:
        span.set_attribute(C.ATTR_SESSION_ID, caller.session_id)
    if caller.user_id:
        span.set_attribute(C.ATTR_USER_ID, caller.user_id)
    if caller.prompt_name:
        span.set_attribute(C.ATTR_PROMPT_NAME, str(caller.prompt_name))
    if caller.prompt_version is not None:
        span.set_attribute(C.ATTR_PROMPT_VERSION, caller.prompt_version)
    if metadata_mode == "metadata":
        return  # content: never placed on the span in metadata mode
    if caller.tags:
        span.set_attribute(
            C.ATTR_TAGS,
            list(caller.tags)
            if metadata_mode == "record"
            else [C.REDACTED_PLACEHOLDER],
        )
    for key, value in caller.metadata:
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(
                f"{C.CONTENT_METADATA_PREFIX}{key}",
                value if metadata_mode == "record" else C.REDACTED_PLACEHOLDER,
            )
