"""Egress-authoritative content policy for OTel spans.

``ContentPolicy.sanitize`` rebuilds every span from scratch: only fields that
are explicitly allowed for the effective content mode are copied into the new
``ReadableSpan``.  Nothing is "removed from" the original, so a field the
policy does not know about cannot survive by omission.

Boundary (documented, tested): this protects only spans that leave through
Traigent's exporter.  It cannot protect another exporter attached to the same
provider; see ``traigent.observability.otel.instrument`` for how instrumentors
are gated for that case.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, SpanContext, SpanKind, TraceState
from opentelemetry.trace.status import Status, StatusCode

from traigent.observability.config import most_restrictive_content_mode
from traigent.observability.otel import contract as C
from traigent.security.redaction import (
    is_credential_key_name,
    redact_sensitive_data,
    redact_sensitive_text,
)

_SCALARS = (str, bool, int, float)


def _has_control(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _strip_trace_state(ctx: SpanContext | None) -> SpanContext | None:
    """Rebuild a span/link context with an EMPTY ``tracestate``.

    W3C tracestate is an arbitrary vendor key/value channel that would
    otherwise be serialized verbatim.  Policy: no mode forwards it (no
    allowlisted vendor key is required today), so it is dropped everywhere.
    """
    if ctx is None:
        return None
    return SpanContext(
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        is_remote=ctx.is_remote,
        trace_flags=ctx.trace_flags,
        trace_state=TraceState(),
    )


def looks_sensitive(text: str) -> bool:
    """True when the shared secret scrubbers would redact something in ``text``."""
    return bool(redact_sensitive_text(text) != text)


def coerce_clean(value: Any, spec: C.AttrSpec) -> Any | None:
    """``coerce_value`` plus: a credential-shaped string value is dropped.

    A string that merely fits its bound is still a channel (a session id
    holding an API key), in every mode.  Mirrors the JS ``coerceClean``.
    """
    coerced = coerce_value(value, spec)
    if coerced is None:
        return None
    if isinstance(coerced, str):
        return None if looks_sensitive(coerced) else coerced
    if isinstance(coerced, tuple) and any(
        isinstance(v, str) and looks_sensitive(v) for v in coerced
    ):
        return None
    return coerced


def coerce_value(value: Any, spec: C.AttrSpec) -> Any | None:
    """Return ``value`` if it satisfies ``spec`` exactly, else ``None`` (drop)."""
    kind = spec.kind
    if kind in ("str", "enum"):
        if not isinstance(value, str) or _has_control(value):
            return None
        if len(value) > spec.max_len or not value:
            return None
        if kind == "enum" and (spec.choices is None or value not in spec.choices):
            return None
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if spec.lo is not None and value < spec.lo:
            return None
        if spec.hi is not None and value > spec.hi:
            return None
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        if spec.lo is not None and number < spec.lo:
            return None
        if spec.hi is not None and number > spec.hi:
            return None
        return number
    if kind == "bool":
        return value if isinstance(value, bool) else None
    if kind == "str_seq":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return None
        if len(value) > spec.max_items:
            return None
        out = []
        for item in value:
            if not isinstance(item, str) or _has_control(item):
                return None
            if not item or len(item) > spec.max_len:
                return None
            out.append(item)
        return tuple(out)
    return None


def _record_text(text: str, limit: int) -> str:
    """Record-mode free text (names, status): secret-scrubbed, then capped."""
    scrubbed: str = redact_sensitive_text(text)
    return scrubbed[:limit]


def _record_value(value: Any) -> Any | None:
    """Record-mode value: existing secret scrubbers plus a per-attribute cap."""
    if isinstance(value, str):
        scrubbed: Any = redact_sensitive_text(value)
    elif isinstance(value, _SCALARS):
        scrubbed = value
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            scrubbed = tuple(redact_sensitive_data(list(value)))
        except Exception:
            return None
    else:
        return None
    try:
        size = len(json.dumps(scrubbed, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return None
    if size > C.MAX_RECORD_ATTR_BYTES:
        if isinstance(scrubbed, str):
            return scrubbed.encode("utf-8")[: C.MAX_RECORD_ATTR_BYTES].decode(
                "utf-8", "ignore"
            )
        return None
    return scrubbed


class ContentPolicy:
    """Rebuild spans according to the content mode (most restrictive wins)."""

    def __init__(
        self,
        mode: str = "metadata",
        allowed_span_names: Collection[str] | None = None,
    ) -> None:
        if mode not in C.CONTENT_MODES:
            raise ValueError(f"content mode must be one of {C.CONTENT_MODES}")
        self.mode = mode
        self.allowed_span_names: frozenset[str] | None = None
        if allowed_span_names is not None:
            if isinstance(allowed_span_names, (str, bytes)):
                raise ValueError("allowed_span_names must be a collection of names")
            names = frozenset(allowed_span_names)
            for name in names:
                if not isinstance(name, str) or not C.is_token_charset(name):
                    raise ValueError(
                        "allowed_span_names entries must be static strings of "
                        "letters, digits and _ . - / : (max 80 characters)"
                    )
            self.allowed_span_names = names

    # -- effective mode -------------------------------------------------
    def effective_mode(self, span_attrs: Mapping[str, Any]) -> str:
        """Resolve per the contract's ``content_mode.resolution_vectors``.

        ABSENT declaration inherits the configured mode.  A PRESENT declaration
        that is not exactly one of the contract values (wrong case, empty,
        non-string, unknown) is INVALID and resolves to ``metadata`` whatever
        the configured mode is.  A valid one never loosens: most restrictive
        of configured and declared.
        """
        if C.CONTENT_MODE_ATTRIBUTE not in span_attrs:
            return self.mode
        declared = span_attrs[C.CONTENT_MODE_ATTRIBUTE]
        if isinstance(declared, str) and declared in C.CONTENT_MODES:
            return most_restrictive_content_mode(self.mode, declared)
        return "metadata"

    # -- attributes -----------------------------------------------------
    def _attrs(
        self, attrs: Mapping[str, Any] | None, mode: str
    ) -> tuple[dict[str, Any], int]:
        out: dict[str, Any] = {}
        dropped = 0
        for key, value in (attrs or {}).items():
            if not isinstance(key, str):
                dropped += 1
                continue
            spec = C.ATTRIBUTE_ALLOWLIST.get(key)
            if spec is not None:
                coerced = coerce_clean(value, spec)
                if coerced is None:
                    dropped += 1
                else:
                    out[key] = coerced
                continue
            is_content = key in C.CONTENT_ATTRIBUTE_KEYS or (
                key.startswith(C.CONTENT_METADATA_PREFIX) and C.is_safe_key(key)
            )
            if mode == "redacted" and is_content:
                out[key] = C.REDACTED_PLACEHOLDER
                continue
            if (
                mode == "record"
                and C.is_safe_key(key)
                and not is_credential_key_name(key)
            ):
                recorded = _record_value(value)
                if recorded is not None:
                    out[key] = recorded
                    continue
            dropped += 1
        return out, dropped

    # -- resource / scope -----------------------------------------------
    def _resource(self, resource: Resource | None, mode: str) -> Resource:
        kept: dict[str, Any] = {}
        for key, value in (resource.attributes if resource else {}).items():
            spec = C.RESOURCE_ALLOWLIST.get(key)
            if spec is None or key == C.CONTENT_MODE_ATTRIBUTE:
                continue
            coerced = coerce_clean(value, spec)
            if coerced is not None:
                kept[key] = coerced
        # The declaration is ours: always set, never copied from user input.
        kept[C.CONTENT_MODE_ATTRIBUTE] = self.mode
        return Resource(kept, schema_url="")

    @staticmethod
    def _scope(scope: InstrumentationScope | None) -> InstrumentationScope:
        name = getattr(scope, "name", None)
        version = getattr(scope, "version", None)
        if (
            not isinstance(name, str)
            or not C.is_safe_scope_name(name)
            or looks_sensitive(name)
        ):
            name = "unknown"
        if (
            not isinstance(version, str)
            or not C.is_safe_scope_version(version)
            or looks_sensitive(version)
        ):
            version = None
        return InstrumentationScope(name=name, version=version)

    # -- names ----------------------------------------------------------
    def _name(
        self, raw: Any, attrs: Mapping[str, Any], scope_name: str, mode: str
    ) -> str:
        if mode == "record" and isinstance(raw, str) and raw:
            return _record_text(raw, C.MAX_RECORD_NAME_LEN)
        if isinstance(raw, str):
            # No scope is exempt: ``observe(name)`` shares the SDK's own scope
            # but its name is user-supplied, so it is not a content-free channel.
            if self.allowed_span_names is not None:
                if raw in self.allowed_span_names:
                    return raw
            elif C.is_safe_token(raw):
                return raw
        op = attrs.get("gen_ai.operation.name")
        model = attrs.get("gen_ai.request.model")
        if isinstance(op, str):
            derived = f"{op} {model}" if isinstance(model, str) else op
            if len(derived) <= C.MAX_SPAN_NAME_LEN + 128:
                return derived
        kind = attrs.get("openinference.span.kind")
        if isinstance(kind, str):
            return kind.lower()
        otype = attrs.get(C.ATTR_OBSERVATION_TYPE)
        if isinstance(otype, str) and otype in C.OBSERVATION_TYPES:
            return otype
        return "span"

    # -- events / links / status ----------------------------------------
    def _events(
        self, events: Sequence[Event], mode: str
    ) -> tuple[tuple[Event, ...], int]:
        out: list[Event] = []
        dropped = 0
        for event in events or ():
            if mode == "record":
                attrs, d = self._attrs(event.attributes, mode)
                name = (
                    _record_text(event.name, C.MAX_RECORD_NAME_LEN)
                    if isinstance(event.name, str) and event.name
                    else "event"
                )
                out.append(Event(name, attrs, event.timestamp))
                dropped += d
                continue
            allowed = C.ALLOWED_EVENTS.get(event.name)
            if allowed is None:
                dropped += 1
                continue
            attrs = {}
            for key in allowed:
                raw = (event.attributes or {}).get(key)
                if (
                    isinstance(raw, str)
                    and raw
                    and len(raw) <= 128
                    and not _has_control(raw)
                ):
                    attrs[key] = raw
            out.append(Event(event.name, attrs, event.timestamp))
        return tuple(out), dropped

    def _links(self, links: Sequence[Link], mode: str) -> tuple[tuple[Link, ...], int]:
        out: list[Link] = []
        dropped = 0
        for link in links or ():
            attrs: dict[str, Any] = {}
            if mode == "record":
                attrs, d = self._attrs(link.attributes, mode)
                dropped += d
            else:
                dropped += len(link.attributes or {})
            out.append(Link(_strip_trace_state(link.context), attrs))
        return tuple(out), dropped

    @staticmethod
    def _status(status: Status, attrs: Mapping[str, Any], mode: str) -> Status:
        if status.status_code is StatusCode.ERROR:
            if mode == "record":
                text = status.description
                return Status(
                    StatusCode.ERROR,
                    _record_text(text, C.MAX_RECORD_STATUS_LEN) if text else None,
                )
            etype = attrs.get("error.type")
            return Status(StatusCode.ERROR, etype if isinstance(etype, str) else None)
        return Status(status.status_code)

    # -- entry point ----------------------------------------------------
    def sanitize(self, span: ReadableSpan) -> ReadableSpan:
        raw_attrs = dict(span.attributes or {})
        mode = self.effective_mode(raw_attrs)
        attrs, dropped = self._attrs(raw_attrs, mode)
        attrs.pop(C.CONTENT_MODE_ATTRIBUTE, None)  # only re-added if it tightens
        # error.type falls back to the exception event's type for status text
        events, ev_dropped = self._events(span.events, mode)
        if "error.type" not in attrs:
            for event in events:
                etype = (event.attributes or {}).get("exception.type")
                if isinstance(etype, str) and event.name == "exception":
                    attrs.setdefault("error.type", etype)
                    break
        links, link_dropped = self._links(span.links, mode)
        dropped += ev_dropped + link_dropped
        scope = self._scope(span.instrumentation_scope)
        name = self._name(span.name, attrs, scope.name, mode)
        if mode != "record" and name != span.name:
            dropped += 1  # a replaced (non-identifier) name is a counted drop
        if dropped:
            attrs[C.ATTR_DROPPED_ATTRS] = dropped
        # the effective mode is per span: declare it only if it tightens
        if mode != self.mode:
            attrs[C.CONTENT_MODE_ATTRIBUTE] = mode
        return ReadableSpan(
            name=name,
            context=_strip_trace_state(span.context),
            parent=_strip_trace_state(span.parent),
            resource=self._resource(span.resource, mode),
            attributes=attrs,
            events=events,
            links=links,
            kind=span.kind if isinstance(span.kind, SpanKind) else SpanKind.INTERNAL,
            status=self._status(span.status, attrs, mode),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=scope,
        )
