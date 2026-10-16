"""Content-less session create (phase 3, group 1: identity).

Opt-in (``optimize(content_less=True)`` or ``TRAIGENT_CONTENT_LESS=1``; default off). When
active, ``POST /api/v1/sessions`` carries opaque references and closed enums instead of
customer-chosen identity: it declares the ``content_less_v1`` capability, a closed
``environment`` and ``agent_ref`` / ``dataset_ref`` / ``evaluation_set_ref`` /
``evaluator_ref`` where the SDK holds them. The legacy identity fields (agent key, dataset
id, dataset / evaluation-set names, display names, evaluator ids) are not sent, and human
labels never travel here (they go only through the label service writer, ``traigent.labels``).
A Backend that does not advertise ``content_less_v1`` is refused locally with
``backend_upgrade_required``: there is no plaintext fallback.

Contract: TraigentSchema ``optimization/session_capabilities_schema.json``,
``optimization/session_create_response_schema.json``, ``labels/label_common_schema.json``.
Function-derived names are code identifiers (GNFR-PRIV-003) and stay as ``function_name``.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traigent.cloud.client import SessionContractError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

CAPABILITY = "content_less_v1"
ENVIRONMENTS = ("production", "staging", "development", "test", "default")
#: Common spellings accepted for the closed enum (case-insensitive). Anything else is refused.
ENVIRONMENT_ALIASES = {
    "prod": "production",
    "dev": "development",
    "stage": "staging",
    "stg": "staging",
    "testing": "test",
}

#: Closed local refusal codes (a subset of OptimizationRefusalCode plus two local ones).
BACKEND_UPGRADE_REQUIRED = "backend_upgrade_required"
INVALID_AGENT_REF = "invalid_agent_ref"
INVALID_ENVIRONMENT = "invalid_environment"

_REFUSAL_TEXT = {
    BACKEND_UPGRADE_REQUIRED: "Backend upgrade required.",
    INVALID_AGENT_REF: "agent_ref is not a valid agent reference.",
    INVALID_ENVIRONMENT: "environment is not one of the closed environment values.",
}

#: Schema label_common_schema.json#/definitions/AgentRef etc.
_REF_PATTERNS = {
    "agent": re.compile(r"^agentref_[0-9a-f]{32}$"),
    "dataset": re.compile(r"^datasetref_[0-9a-f]{32}$"),
    "evaluation_set": re.compile(r"^evaluationsetref_[0-9a-f]{32}$"),
    "evaluator": re.compile(r"^evaluatorref_[0-9a-f]{32}$"),
}
#: request / response field per reference kind
REF_FIELDS = {
    "agent": "agent_ref",
    "dataset": "dataset_ref",
    "evaluation_set": "evaluation_set_ref",
    "evaluator": "evaluator_ref",
}

#: Legacy identity fields a content-less create must not carry (Backend
#: ``_FORBIDDEN_*`` in services/optimization_identity.py; CHECKLIST group 1, route S1).
LEGACY_TOP_LEVEL = (
    "agent_id",
    "agent_key",
    "agent_id_source",
    "dataset_id",
    "dataset_id_source",
    "hosted_dataset_ref",
    "evaluator_id",
    "evaluator_id_source",
    "evaluator_definition_id",
    "user_id",
    "billing_tier",
    "identity_version",
    "artifact_versions",
)
LEGACY_DATASET_METADATA = ("name",)
LEGACY_METADATA = ("evaluation_set", "function_display_name", "function_slug")

#: Offline / local session-record metadata key persisting that the run opted in.
RECORD_KEY = "content_less_mode"
REPLAY_UNSUPPORTED = "content_less_replay_unsupported"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ContentLessRefusal(SessionContractError):
    """A content-less session create was refused locally. Carries a closed code only.

    A :class:`SessionContractError`, so the existing contract-failure paths re-raise it
    as itself and never absorb it into a local fallback or the legacy-shape retry.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_REFUSAL_TEXT.get(code, "Content-less session refused."))


@dataclass(frozen=True)
class ContentLessOptions:
    """Per-run content-less settings from ``optimize()``. ``None`` fields defer to env."""

    enabled: bool | None = None
    agent_ref: str | None = None
    environment: str | None = None


def is_enabled(
    options: ContentLessOptions | None, env: Mapping[str, str] | None = None
) -> bool:
    """Opt-in: the explicit ``content_less`` argument wins, else ``TRAIGENT_CONTENT_LESS``."""
    if isinstance(options, ContentLessOptions) and options.enabled is not None:
        return bool(options.enabled)
    source = os.environ if env is None else env
    return str(source.get("TRAIGENT_CONTENT_LESS", "")).strip().lower() in _TRUTHY


def is_well_formed_ref(kind: str, value: Any) -> bool:
    return isinstance(value, str) and bool(_REF_PATTERNS[kind].fullmatch(value))


def options_of(request: Any) -> ContentLessOptions | None:
    """The request's ``ContentLessOptions``, or None (ignores anything else, e.g. test doubles)."""
    value = getattr(request, "content_less", None)
    return value if isinstance(value, ContentLessOptions) else None


def resolve_agent_ref(
    options: ContentLessOptions | None, env: Mapping[str, str] | None = None
) -> str | None:
    """Explicit ``agent_ref`` > ``TRAIGENT_AGENT_REF`` > None (function-derived).

    A supplied value that does not match the Schema pattern is refused locally; the
    value is never echoed.
    """
    source = os.environ if env is None else env
    value = options.agent_ref if isinstance(options, ContentLessOptions) else None
    if value is None:
        value = (source.get("TRAIGENT_AGENT_REF") or "").strip() or None
    if value is None:
        return None
    if not is_well_formed_ref("agent", value):
        raise ContentLessRefusal(INVALID_AGENT_REF)
    return value


def _otel_environment(env: Mapping[str, str]) -> str | None:
    attrs = env.get("OTEL_RESOURCE_ATTRIBUTES") or ""
    found: dict[str, str] = {}
    for part in attrs.split(","):
        key, sep, val = part.partition("=")
        if sep:
            found[key.strip()] = val.strip()
    return found.get("deployment.environment.name") or found.get(
        "deployment.environment"
    )


_warned_default_environment = False


def resolve_environment(
    explicit: str | None = None, env: Mapping[str, str] | None = None
) -> str:
    """Closed run environment.

    Precedence: explicit > TRAIGENT_ENVIRONMENT > OTel ``deployment.environment.name`` >
    ENVIRONMENT > legacy TRAIGENT_ENV > ``default`` (with a one-time warning). The first
    non-blank source decides; its value (stripped, lower-cased, alias-mapped) must be in the closed enum,
    otherwise the create is refused locally. ``production`` is only ever sent when one of
    these sources says so; it is never guessed.
    """
    global _warned_default_environment
    source = os.environ if env is None else env
    candidates = (
        explicit,
        source.get("TRAIGENT_ENVIRONMENT"),
        _otel_environment(source),
        source.get("ENVIRONMENT"),
        source.get("TRAIGENT_ENV"),
    )
    for candidate in candidates:
        if candidate is None or not str(candidate).strip():
            continue
        value = str(candidate).strip().lower()
        value = ENVIRONMENT_ALIASES.get(value, value)
        if value not in ENVIRONMENTS:
            raise ContentLessRefusal(INVALID_ENVIRONMENT)
        return value
    if not _warned_default_environment:
        _warned_default_environment = True
        logger.warning(
            "No run environment configured; content-less sessions use 'default'. Set "
            "TRAIGENT_ENVIRONMENT to one of production|staging|development|test."
        )
    return "default"


# ------------------------------------------------------------------------------ ref cache


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path() -> Path:
    base = os.getenv("TRAIGENT_RESULTS_FOLDER")
    root = Path(base).expanduser() if base else Path.home() / ".traigent"
    return root / "content_less_refs.json"


class RefCache:
    """Local cache of Backend-issued opaque references, per (tenant, project).

    Entries are keyed by a SHA-256 of a local-only key (the function-derived code
    identifier, the dataset / evaluator label), so no name is written to the cache and no
    name or digest is sent. ``scope_index`` maps a (backend, credential preview) digest to
    the last-seen tenant/project so a fresh process finds the right scope before its first
    response. Best-effort: any I/O problem degrades to "no cached refs".
    """

    _lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _cache_path()

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "scopes": {}, "scope_index": {}}
        if not isinstance(data, dict) or data.get("version") != 1:
            return {"version": 1, "scopes": {}, "scope_index": {}}
        data.setdefault("scopes", {})
        data.setdefault("scope_index", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            logger.debug("content-less ref cache could not be written")

    @staticmethod
    def principal(backend_url: str | None, credential_preview: str | None) -> str:
        return _digest(f"{backend_url or ''}\x00{credential_preview or ''}")

    def lookup(
        self, principal: str, local_keys: Mapping[str, str | None]
    ) -> dict[str, str]:
        """Cached refs for ``local_keys`` (kind -> local key) in the principal's last scope."""
        with self._lock:
            data = self._load()
        scope = data["scope_index"].get(principal)
        entries = data["scopes"].get(scope) if isinstance(scope, str) else None
        if not isinstance(entries, dict):
            return {}
        out: dict[str, str] = {}
        for kind, key in local_keys.items():
            if not key:
                continue
            ref = (entries.get(kind) or {}).get(_digest(key))
            if is_well_formed_ref(kind, ref):
                out[kind] = ref
        return out

    def store(
        self,
        principal: str,
        tenant_id: str,
        project_id: str,
        local_keys: Mapping[str, str | None],
        refs: Mapping[str, str],
    ) -> None:
        scope = f"{tenant_id}/{project_id}"
        with self._lock:
            data = self._load()
            entries = data["scopes"].setdefault(scope, {})
            for kind, key in local_keys.items():
                ref = refs.get(kind)
                if key and is_well_formed_ref(kind, ref):
                    entries.setdefault(kind, {})[_digest(key)] = ref
            data["scope_index"][principal] = scope
            self._save(data)

    def forget(self, principal: str) -> None:
        """Drop the principal's scope entries (a cached ref was refused as foreign/unknown)."""
        with self._lock:
            data = self._load()
            scope = data["scope_index"].pop(principal, None)
            if isinstance(scope, str):
                data["scopes"].pop(scope, None)
            self._save(data)


# ----------------------------------------------------------------- payload and response


def local_keys_for(
    session_request: Any, payload_metadata: Mapping[str, Any]
) -> dict[str, str | None]:
    """Local-only cache keys per reference kind (never sent).

    agent: the function-derived code identifier. dataset / evaluation_set: the dataset's
    local label or explicit id. evaluator: the explicit evaluator id.
    """
    from traigent.cloud.models import _derive_default_dataset_id

    function_name = payload_metadata.get("function_name") or getattr(
        session_request, "function_name", None
    )
    explicit = getattr(session_request, "dataset_id", None)
    dataset_key = (
        explicit.strip()
        if isinstance(explicit, str) and explicit.strip()
        else _derive_default_dataset_id(session_request)
    )
    evaluator = getattr(session_request, "evaluator_id", None) or getattr(
        session_request, "evaluator_definition_id", None
    )
    label = (getattr(session_request, "metadata", None) or {}).get("evaluation_set")
    return {
        "agent": str(function_name) if function_name else None,
        "dataset": dataset_key,
        "evaluation_set": label if isinstance(label, str) and label.strip() else None,
        "evaluator": evaluator.strip()
        if isinstance(evaluator, str) and evaluator.strip()
        else None,
    }


def strip_legacy_identity(payload: dict[str, Any]) -> None:
    """Remove every G1 legacy identity field from a session-create payload, in place."""
    for name in LEGACY_TOP_LEVEL:
        payload.pop(name, None)
    dataset_metadata = payload.get("dataset_metadata")
    if isinstance(dataset_metadata, dict):
        for name in LEGACY_DATASET_METADATA:
            dataset_metadata.pop(name, None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for name in LEGACY_METADATA:
            metadata.pop(name, None)


def apply_to_payload(
    payload: dict[str, Any],
    *,
    options: ContentLessOptions | None,
    cached_refs: Mapping[str, str],
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Rewrite a typed session-create payload to the content-less shape, in place.

    Returns the refs that came from the local cache (so a refusal can invalidate them).
    Raises :class:`ContentLessRefusal` for an invalid ``agent_ref`` / environment before
    anything is sent.
    """
    agent_ref = resolve_agent_ref(options, env)
    environment = resolve_environment(
        options.environment if isinstance(options, ContentLessOptions) else None, env
    )
    strip_legacy_identity(payload)
    payload["capabilities"] = [CAPABILITY]
    payload["environment"] = environment
    used_cache: dict[str, str] = {}
    refs = dict(cached_refs)
    if agent_ref is not None:
        refs["agent"] = agent_ref
    for kind, ref in refs.items():
        if kind in REF_FIELDS and is_well_formed_ref(kind, ref):
            payload[REF_FIELDS[kind]] = ref
            if not (kind == "agent" and agent_ref is not None):
                used_cache[kind] = ref
    return used_cache


def refs_from_response(result: Mapping[str, Any]) -> dict[str, str]:
    """Well-formed refs on the 201 (top level, or the metadata copy legacy clients get)."""
    metadata = (
        result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    )
    out: dict[str, str] = {}
    for kind, field in REF_FIELDS.items():
        value = result.get(field)
        if value is None:
            value = metadata.get(field)
        if is_well_formed_ref(kind, value):
            out[kind] = value
    return out


def require_capability(result: Mapping[str, Any]) -> None:
    """Refuse a Backend that does not advertise ``content_less_v1`` (no plaintext fallback)."""
    caps = result.get("capabilities")
    if not isinstance(caps, list) or CAPABILITY not in caps:
        raise ContentLessRefusal(BACKEND_UPGRADE_REQUIRED)


# ------------------------------------------- shared by every session-create serializer


def prepare_payload(
    payload: dict[str, Any],
    request: Any,
    principal: str,
    env: Mapping[str, str] | None = None,
) -> tuple[Any, ...]:
    """Rewrite ``payload`` to the content-less shape; returns the response context.

    The context is ``(cache, principal, local_keys, used_cache)`` for
    :func:`accept_response` / :func:`forget_refused`.
    """
    local_keys = local_keys_for(request, payload.get("metadata") or {})
    cache = RefCache()
    cached = cache.lookup(principal, local_keys)
    used_cache = apply_to_payload(
        payload, options=options_of(request), cached_refs=cached, env=env
    )
    return (cache, principal, local_keys, used_cache)


def accept_response(result: Any, ctx: tuple[Any, ...]) -> dict[str, str]:
    """Refuse a Backend without ``content_less_v1``; cache and return the refs it issued."""
    if not isinstance(result, dict):
        raise ContentLessRefusal(BACKEND_UPGRADE_REQUIRED)
    require_capability(result)
    cache, principal, local_keys, _used = ctx
    metadata = (
        result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    )
    tenant_id = result.get("tenant_id") or metadata.get("tenant_id")
    project_id = result.get("project_id") or metadata.get("project_id")
    refs = refs_from_response(result)
    if refs and tenant_id and project_id:
        cache.store(principal, str(tenant_id), str(project_id), local_keys, refs)
    return refs


def label_texts(request: Any) -> dict[str, str | None]:
    """The human names behind a session's refs, for the label service writer only.

    These strings are customer-chosen names: they are never put on a session request.
    """
    from traigent.cloud.models import _derive_default_dataset_id

    explicit = getattr(request, "dataset_id", None)
    dataset = (
        explicit.strip()
        if isinstance(explicit, str) and explicit.strip()
        else _derive_default_dataset_id(request)
    )
    evaluation_set = (getattr(request, "metadata", None) or {}).get("evaluation_set")
    evaluator = getattr(request, "evaluator_id", None) or getattr(
        request, "evaluator_definition_id", None
    )
    agent = getattr(request, "agent_key", None)
    return {
        "agent": agent if isinstance(agent, str) else None,
        "dataset": dataset,
        "evaluation_set": evaluation_set if isinstance(evaluation_set, str) else None,
        "evaluator": evaluator if isinstance(evaluator, str) else None,
    }


def forget_refused(ctx: tuple[Any, ...]) -> None:
    """A cached reference was refused (404): drop the scope's cached refs."""
    cache, principal, _keys, used_cache = ctx
    if used_cache:
        cache.forget(principal)


def replay_refused(session_metadata: Mapping[str, Any] | None) -> bool:
    """True when an offline session must not be replayed (`traigent sync`).

    Replay builds a legacy identity body and cannot mint or resolve refs, nor verify the
    capability, so a record made in content-less mode (or any replay while
    ``TRAIGENT_CONTENT_LESS`` is set) is refused with the closed code
    ``content_less_replay_unsupported``.
    """
    if is_enabled(None):
        return True
    return bool(
        isinstance(session_metadata, Mapping) and session_metadata.get(RECORD_KEY)
    )
