"""Configuration for the Traigent observability client."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env, scope_api_path
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env
from traigent.utils.env_config import (
    is_backend_offline,
    is_truthy,
    resolve_environment_label,
)

MAX_BATCH_SIZE = 10_000
MAX_QUEUE_SIZE = 1_000_000
MAX_BATCH_BYTES = 10 * 1024 * 1024
MAX_BUFFER_AGE_SECONDS = 3600.0
MAX_TIMEOUT_SECONDS = 600.0
OBSERVABILITY_CONTENT_MODES = frozenset({"metadata", "redacted", "record"})

_EXECUTION_CONTEXT_ENV_VARS = {
    "agent_id": "TRAIGENT_AGENT_ID",
    "agent_version": "TRAIGENT_AGENT_VERSION",
    "release_id": "TRAIGENT_RELEASE_ID",
    "deployment_id": "TRAIGENT_DEPLOYMENT_ID",
    "code_revision": "TRAIGENT_CODE_REVISION",
    "configuration_id": "TRAIGENT_CONFIGURATION_ID",
    "configuration_version": "TRAIGENT_CONFIGURATION_VERSION",
    "prompt_id": "TRAIGENT_PROMPT_ID",
    "prompt_version": "TRAIGENT_PROMPT_VERSION",
    "toolset_id": "TRAIGENT_TOOLSET_ID",
    "toolset_version": "TRAIGENT_TOOLSET_VERSION",
    "evaluator_id": "TRAIGENT_EVALUATOR_ID",
    "evaluator_version": "TRAIGENT_EVALUATOR_VERSION",
    "dataset_id": "TRAIGENT_DATASET_ID",
    "dataset_version": "TRAIGENT_DATASET_VERSION",
    "experiment_run_id": "TRAIGENT_EXPERIMENT_RUN_ID",
    "configuration_run_id": "TRAIGENT_CONFIGURATION_RUN_ID",
    "optimization_run_id": "TRAIGENT_OPTIMIZATION_RUN_ID",
    "intervention_id": "TRAIGENT_INTERVENTION_ID",
}


def nonblank_credential(value: str | None) -> bool:
    """A credential value counts only when it carries non-whitespace content.

    Used by both header construction and the missing-credential guard so a
    whitespace-only api_key/jwt can neither authenticate a request nor
    overwrite working auth supplied via ``extra_headers``.
    """
    return bool(value and value.strip())


def _read_observability_offline_mode() -> bool:
    return is_backend_offline() or is_truthy(os.getenv("TRAIGENT_DISABLE_TELEMETRY"))


_CONTENT_MODE_RANK: dict[str, int] = {"metadata": 0, "redacted": 1, "record": 2}

# Dataclass field sentinel: distinguishes "the constructor's `content_mode`
# argument was not passed" (fall back to env/legacy-env resolution) from an
# explicit value, including an explicit "" (which must raise -- see
# `_validate_content_mode_value`). Kept as a real member of the field's `str`
# type (rather than `None`) so the field stays typed `str` for every
# downstream reader of `ObservabilityConfig.content_mode`.
_CONTENT_MODE_UNSET = "__unset__"


def most_restrictive_content_mode(*modes: str) -> str:
    """Return the most restrictive of one or more ALREADY-VALID content modes.

    Restrictiveness order (most to least): `metadata` > `redacted` > `record`
    (see `_CONTENT_MODE_RANK`). Callers must validate each mode first --
    this raises `KeyError` on anything outside `OBSERVABILITY_CONTENT_MODES`.
    """
    return min(modes, key=lambda mode: _CONTENT_MODE_RANK[mode])


def _validate_content_mode_value(value: str, *, source: str) -> str:
    """Normalize and validate a single candidate content-mode value.

    Raises `ValueError` for anything outside `OBSERVABILITY_CONTENT_MODES`,
    including the empty string -- an invalid/blank value must never be
    silently treated as "not set".
    """
    normalized = value.strip().lower()
    if normalized not in OBSERVABILITY_CONTENT_MODES:
        allowed = ", ".join(sorted(OBSERVABILITY_CONTENT_MODES))
        raise ValueError(f"{source} must be one of: {allowed}")
    return normalized


def validate_content_mode_override(content_mode: str | None) -> str | None:
    """Validate (without resolving against any default) a per-call/decorator override.

    Returns the normalized mode, or `None` when `content_mode` is `None`
    ("no override supplied"). Raises `ValueError` for an invalid value,
    including `""`. Callers that mutate other state on entry (e.g.
    `ObserveContext` setting context variables) MUST call this first, before
    any such mutation, so an invalid override never leaves partial state
    behind (M2).
    """
    if content_mode is None:
        return None
    return _validate_content_mode_value(content_mode, source="content_mode")


def _legacy_capture_content_mode() -> str | None:
    """Legacy `TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT` boolean env var.

    `true` -> `record`, `false`/unset -> `metadata`. Returns `None` when the
    env var is not set at all (as opposed to set-but-falsy), so an unset
    legacy var never counts as an explicitly-set source.
    """
    raw = os.getenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT")
    if raw is None:
        return None
    return "record" if is_truthy(raw) else "metadata"


def resolve_client_content_mode(constructor_value: str | None) -> tuple[str, bool]:
    """Most-restrictive resolution of the CLIENT-LEVEL `content_mode` (M2).

    Sources, all optional: the constructor's explicit `content_mode` value,
    env `TRAIGENT_OBSERVABILITY_CONTENT`, and the legacy
    `TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT` boolean env var. Each
    explicitly-set source is validated independently -- an invalid/empty
    value raises immediately, before any other source is even consulted --
    and the MOST RESTRICTIVE of all explicitly-set values wins (`metadata` >
    `redacted` > `record`), so an explicit privacy-on setting anywhere never
    loses to a `record` setting from another source. `metadata` is the
    result when nothing is explicitly set.

    Returns `(resolved_mode, explicitly_set)`; `explicitly_set` is `False`
    only when every source was absent, letting the caller distinguish "the
    client has no configured privacy policy" (a per-call override may pick
    any mode -- the normal opt-in path) from "the client explicitly resolved
    to metadata" (a per-call override may only tighten it further, never
    loosen it -- see `ObservabilityClient._resolve_content_mode`).
    """
    candidates: list[str] = []
    if constructor_value is not None:
        candidates.append(
            _validate_content_mode_value(constructor_value, source="content_mode")
        )
    raw_env = os.getenv("TRAIGENT_OBSERVABILITY_CONTENT")
    if raw_env is not None:
        candidates.append(
            _validate_content_mode_value(
                raw_env, source="TRAIGENT_OBSERVABILITY_CONTENT"
            )
        )
    legacy = _legacy_capture_content_mode()
    if legacy is not None:
        candidates.append(legacy)

    if not candidates:
        return "metadata", False
    return most_restrictive_content_mode(*candidates), True


def redacted_content_marker() -> dict[str, bool]:
    """Placeholder emitted in place of withheld content."""
    return {"redacted": True}


def apply_content_mode(payload: Any, mode: str, *, force_redact: bool = False) -> Any:
    """Apply the observability `content_mode` policy to a SUPPLIED content payload.

    This is the single point every entry point funnels through: direct calls
    to `ObservabilityClient.start_trace` / `record_observation` / `end_trace`,
    the `observe` decorator/context manager, buffered flush, and retry. Buffered
    flush and retry re-send the payload already gated at intake (the DTO
    snapshot stored on the trace/observation), so they never see raw content
    and cannot bypass this gate. Direct callers of the client methods are
    gated here exactly like the decorator path, closing the "decorator-only
    enforcement" gap.

    `force_redact` lets a caller (e.g. `redact_input=True` on `observe`)
    always emit the redaction placeholder regardless of the active mode.

    Callers must only invoke this when a field was actually SUPPLIED (see the
    `NOT_SUPPLIED` sentinel below) -- unlike an earlier version of this
    function, `payload is None` is no longer special-cased here. A genuinely
    supplied `None` (e.g. a decorated function that legitimately returns
    `None`) is content like any other value: `metadata` omits it, `record`
    sends it (as `null`), and `redacted` still emits the placeholder so the
    field's presence stays structurally meaningful. Treating "not supplied"
    and "supplied `None`" as identical here (as the previous implementation
    did) is exactly the bug this fixes: `record_observation`'s merge could
    not distinguish "this call didn't touch the field" from "this call
    supplied content the active mode forbids", so a later metadata-mode
    update silently kept content an earlier record-mode call had recorded.

    `record` mode returns a DEEP COPY of `payload`, never the caller's own
    object by reference. Gating runs synchronously inside
    `start_trace`/`record_observation`/`end_trace`, but the gated value is
    then stored on a long-lived DTO (`ObservabilityClient._trace_states`)
    that a LATER `flush()`/`close()`/update re-serializes from scratch --
    returning the same reference would let the caller mutate their own dict
    after the call returns and have that mutation silently reach a snapshot
    built afterward (M4's "mutation after enqueue must not leak", which
    covers not just the transport's retry buffer but any later
    re-derivation of the stored DTO). M8 (fail closed): if the deep copy
    itself raises (a caller-supplied payload can hold an uncopyable object,
    e.g. a lock or open file handle), withhold rather than raise or fall
    back to the live reference.
    """
    if force_redact or mode == "redacted":
        return redacted_content_marker()
    if mode == "record":
        try:
            return copy.deepcopy(payload)
        except Exception:
            return None
    return None


def apply_content_mode_to_text(
    text: str | None, mode: str, *, force_redact: bool = False
) -> str | None:
    """Apply the same policy to a free-form text field (e.g. an exception message).

    Returns `None` in metadata mode so the caller omits the field entirely
    rather than shipping a null placeholder. Unlike `apply_content_mode`,
    `text is None` IS still special-cased here: every call site passes a
    real, already-supplied string or `None` meaning "there is no such text
    at all" (e.g. no exception was raised, no comment was given) in a single
    one-shot call with no update/merge semantics, so there is no
    NOT_SUPPLIED-vs-supplied distinction to preserve.
    """
    if text is None:
        return None
    if force_redact or mode == "redacted":
        return "[REDACTED]"
    if mode == "record":
        return text
    return None


def wire_value_for_tightened_update(
    gated_value: Any, *, effective_content_mode: str, existing_value: Any
) -> Any:
    """M4 ingest-contract fix (parity note from the TS worker, traigent-js
    commit cf2b700da): the backend's `apply_updates` SKIPS an omitted/`None`
    field on an update and KEEPS whatever content it already has stored --
    so a `metadata`-mode-withheld field that merely OMITS (`None`) does not
    clear content a PRIOR, less restrictive call already shipped to the
    backend for this same field; the backend just keeps its stored value.

    On an UPDATE where gating produced `None` (metadata mode withheld it)
    AND there IS a prior value (`existing_value is not None` -- either real
    content from an earlier `record`/`redacted` call, or an earlier
    placeholder), send the explicit `{"redacted": true}` placeholder instead
    of `None`, so the backend actually overwrites its stored value. A
    genuine FIRST WRITE (no prior value at all) still omits cleanly -- `M1`
    says metadata omits, and there is nothing stored server-side yet to
    leak.

    `redacted` mode never reaches the `None` branch (it always produces the
    placeholder already); `record` mode never reaches it either (it returns
    the payload, deep-copied, never `None` unless the caller's own value was
    `None`). So this only ever changes behavior for `metadata`.
    """
    if gated_value is not None:
        return gated_value
    if effective_content_mode == "metadata" and existing_value is not None:
        return redacted_content_marker()
    return gated_value


class _NotSupplied:
    """Sentinel type for `NOT_SUPPLIED` (see below)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "NOT_SUPPLIED"


NOT_SUPPLIED: Any = _NotSupplied()
"""Default value for content-bearing keyword arguments that also support partial
updates (`ObservabilityClient.record_observation` / `end_trace`).

Distinguishes three states (M4): the argument was not passed at all (this
sentinel; the field is left exactly as previously recorded), the argument was
passed as an explicit `None` (real content that happens to be `None`, or an
explicit "clear this field" -- both are content states subject to
`content_mode` gating, not "leave unchanged"), and, after gating, the
`content_mode`-forbidden ("withheld") result. Using plain `None` as the
default (as this SDK originally did) conflated the first two states: a
caller that never touched a field and a caller supplying content the active
mode forbids both produced `None`, so an update could not tell "leave
existing content alone" apart from "the current mode just withheld the
content I actually sent" -- silently letting an earlier, less restrictive
call's content survive a later, more restrictive one.
"""


def _read_default_execution_context() -> dict[str, str | None]:
    """Read only explicit Traigent lineage identifiers from the environment."""
    context: dict[str, str | None] = {}
    for field_name, environment_name in _EXECUTION_CONTEXT_ENV_VARS.items():
        raw_value = os.getenv(environment_name)
        if raw_value is not None and raw_value.strip():
            context[field_name] = raw_value.strip()
    if "release_id" not in context:
        release = os.getenv("TRAIGENT_RELEASE")
        if release is not None and release.strip():
            context["release_id"] = release.strip()
    return context


@dataclass
class ObservabilityConfig:
    """Configuration for SDK-side observability delivery.

    ``health_callback`` receives batch-level local-drop and warning snapshots
    after internal transport locks are released. It may run on background
    threads, and exceptions raised by the callback are swallowed.
    """

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    jwt_token: str | None = field(
        default_factory=lambda: os.getenv("TRAIGENT_JWT_TOKEN")
    )
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    api_path: str = "/api/v1beta/observability"
    batch_size: int = 100
    max_buffer_age: float = 5.0
    max_queue_size: int = 10_000
    max_batch_bytes: int = 4 * 1024 * 1024
    flush_timeout: float = 30.0
    request_timeout: float = 10.0
    enable_atexit_flush: bool = True
    offline_mode: bool = field(default_factory=_read_observability_offline_mode)
    content_mode: str = _CONTENT_MODE_UNSET
    # Set in `__post_init__`; not a constructor argument. True iff the
    # resolved `content_mode` came from an explicitly-set source (constructor
    # value, env, or legacy env) rather than the bare "nothing set" default --
    # see `resolve_client_content_mode` and
    # `ObservabilityClient._resolve_content_mode`.
    content_mode_explicit: bool = field(default=False, init=False, repr=False)
    health_callback: Callable[[str, dict[str, Any]], None] | None = None
    default_environment: str | None = field(
        default_factory=lambda: resolve_environment_label(default=None)
    )
    default_release: str | None = field(
        default_factory=lambda: os.getenv("TRAIGENT_RELEASE")
    )
    default_execution_context: dict[str, str | None] = field(
        default_factory=_read_default_execution_context
    )
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_origin = validate_cloud_base_url(
            self.backend_origin,
            purpose="observability backend",
        )
        self.tenant_id = (
            self.tenant_id.strip() or None if self.tenant_id is not None else None
        )
        self.project_id = (
            self.project_id.strip() or None if self.project_id is not None else None
        )
        self.api_path = scope_api_path(self.api_path, self.project_id)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be less than or equal to {MAX_BATCH_SIZE}"
            )
        if self.max_buffer_age <= 0:
            raise ValueError("max_buffer_age must be greater than 0")
        if self.max_buffer_age > MAX_BUFFER_AGE_SECONDS:
            raise ValueError(
                f"max_buffer_age must be less than or equal to {MAX_BUFFER_AGE_SECONDS}"
            )
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0")
        if self.max_queue_size > MAX_QUEUE_SIZE:
            raise ValueError(
                f"max_queue_size must be less than or equal to {MAX_QUEUE_SIZE}"
            )
        if self.max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be greater than 0")
        if self.max_batch_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"max_batch_bytes must be less than or equal to {MAX_BATCH_BYTES}"
            )
        if self.flush_timeout <= 0:
            raise ValueError("flush_timeout must be greater than 0")
        if self.flush_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"flush_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")
        if self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )
        constructor_content_mode = (
            None if self.content_mode == _CONTENT_MODE_UNSET else self.content_mode
        )
        self.content_mode, self.content_mode_explicit = resolve_client_content_mode(
            constructor_content_mode
        )

    @property
    def ingest_url(self) -> str:
        return f"{self.backend_origin}{self.api_path}/ingest"

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-observability/0.1",
            **self.extra_headers,
        }
        # Blank explicit credentials are treated as absent so they can never
        # overwrite working auth supplied through extra_headers.
        if self.api_key is not None and nonblank_credential(self.api_key):
            headers["X-API-Key"] = self.api_key
        if nonblank_credential(self.jwt_token):
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers
