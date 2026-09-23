"""Provider/model versions OBSERVED from provider responses (``ObservedProviderVersionV1``).

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` section 8 and
``schemas/agents/agent_version_manifest_v1_schema.json``. Build is what was
shipped; observation is what ran. Each observation records:

* ``provider`` -- which provider SDK the intercepted call went through;
* ``requested_model`` -- what the caller asked for, read from the calling
  client/arguments at call time. When the client does not expose it the call
  is NOT recorded (the schema requires a string, and a placeholder would be a
  guess) -- the same rule as the JS SDK;
* ``response_model`` -- what the RESPONSE says served it, or ``None``. It is
  never copied from ``requested_model``: an alias like ``gpt-4o`` is not a
  version;
* ``system_fingerprint`` -- the provider's backend fingerprint when returned;
* ``call_count``.

Observations are recorded only from REAL provider calls (the interceptors'
non-mock branches), into the per-trial capture scope
(:class:`traigent.utils.langchain_interceptor.capture_scope`). Coverage limit:
only calls that pass through the SDK's interceptors are observed, and that
includes LLM-judge calls made by metric functions inside the trial's scope.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "observation_key",
    "observations_payload",
]

_PROVIDER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_MAX_TEXT = 256
_MAX_OBSERVATIONS = 256


def _text(value: Any) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= _MAX_TEXT:
        return value
    return None


def _from_response(response: Any, names: tuple[str, ...]) -> str | None:
    """First string field named in ``names`` on the response or its metadata."""
    sources: list[Any] = [response]
    for attr in ("response_metadata", "llm_output"):
        extra = (
            response.get(attr)
            if isinstance(response, Mapping)
            else getattr(response, attr, None)
        )
        if isinstance(extra, Mapping):
            sources.append(extra)
    for source in sources:
        for name in names:
            try:
                value = (
                    source.get(name)
                    if isinstance(source, Mapping)
                    else getattr(source, name, None)
                )
            except Exception:  # noqa: BLE001 - a raising property is "unknown"
                value = None
            text = _text(value)
            if text is not None:
                return text
    return None


def observation_key(
    response: Any, *, provider: str, requested_model: Any
) -> tuple[str, str, str | None, str | None] | None:
    """``(provider, requested_model, response_model, system_fingerprint)`` of one call.

    ``None`` -- record nothing -- when the provider name is not representable
    or the requested model is unknown.
    """
    normalized_provider = provider.lower() if isinstance(provider, str) else ""
    requested = _text(requested_model)
    if not _PROVIDER_RE.fullmatch(normalized_provider) or requested is None:
        return None
    response_model = _from_response(response, ("model", "model_name", "model_id"))
    fingerprint = _from_response(response, ("system_fingerprint",))
    return normalized_provider, requested, response_model, fingerprint


def observations_payload(
    counts: Mapping[tuple[str, str, str | None, str | None], int],
) -> list[dict[str, Any]]:
    """Sorted ``ObservedProviderVersionV1`` list (at most 256 entries)."""
    payload: list[dict[str, Any]] = []
    for key in sorted(counts, key=lambda k: tuple("" if v is None else v for v in k)):
        provider, requested, response_model, fingerprint = key
        # Every optional field is present, null when unknown, in the JS SDK's
        # key order, so both SDKs serialize one observation identically.
        payload.append(
            {
                "provider": provider,
                "requested_model": requested,
                "response_model": response_model,
                "system_fingerprint": fingerprint,
                "call_count": int(counts[key]),
            }
        )
        if len(payload) >= _MAX_OBSERVATIONS:
            break
    return payload
