"""Attribute contract for the Traigent OpenTelemetry observability layer.

Independently authored from the public OpenTelemetry semantic-convention
attribute *names* (interoperability facts) and Traigent's own design.  The
allowlist below is TYPED and BOUNDED: every key is listed explicitly with a
value type and limit.  There are no wildcard or prefix entries.

Content classification is separate from the allowlist: keys that carry user
content are listed in :data:`CONTENT_ATTRIBUTE_KEYS` (plus the single
documented prefix :data:`CONTENT_METADATA_PREFIX` for ``attributes(metadata=)``
values).  In ``metadata`` mode everything that is not allowlisted is dropped,
so content classification only matters for ``redacted`` (placeholder) and
``record`` (scrubbed pass-through).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

CONTRACT_VERSION: Final = "1"

# Resource attribute by which the SDK declares its content mode to the
# receiver.  Versioned: a receiver that does not know ``.v1`` must treat the
# batch as ``metadata``.  The declaration never grants permission.
CONTENT_MODE_ATTRIBUTE: Final = "traigent.content_mode.v1"
CONTENT_MODES: Final = ("metadata", "redacted", "record")
REDACTED_PLACEHOLDER: Final = "[REDACTED]"

# Scope name used by spans created through Traigent's own ``observe``.
TRAIGENT_SCOPE_NAME: Final = "traigent.observability"

# Attributes stamped by the SDK.
ATTR_OBSERVATION_TYPE: Final = "traigent.observation_type"
ATTR_TRIAL_ID: Final = "traigent.trial_id"
ATTR_OPTIMIZATION_SESSION_ID: Final = "traigent.optimization_session_id"
ATTR_EXPERIMENT_RUN_ID: Final = "traigent.experiment_run_id"
ATTR_CONFIG_HASH: Final = "traigent.config_hash"
ATTR_SESSION_ID: Final = "session.id"
ATTR_USER_ID: Final = "user.id"
# Tags are user labels: content class (the receiver contract lists tag.tags
# among its content keys), never allowlisted.
ATTR_TAGS: Final = "tag.tags"
# Prompt reference uses the GenAI names the receiver contract reads.
ATTR_PROMPT_NAME: Final = "gen_ai.prompt.name"
ATTR_PROMPT_VERSION: Final = "gen_ai.prompt.version"
ATTR_INPUT: Final = "traigent.input"
ATTR_OUTPUT: Final = "traigent.output"
ATTR_DROPPED_ATTRS: Final = "traigent.content.dropped_attrs"
CONTENT_METADATA_PREFIX: Final = "traigent.metadata."

OBSERVATION_TYPES: Final = frozenset(
    {
        "span",
        "generation",
        "event",
        "tool_call",
        "agent",
        "chain",
        "tool",
        "retriever",
        "evaluator",
        "embedding",
        "guardrail",
    }
)


@dataclass(frozen=True)
class AttrSpec:
    """Type and bound for one allowlisted attribute."""

    kind: str  # "str" | "int" | "float" | "bool" | "str_seq" | "enum"
    max_len: int = 128
    max_items: int = 8
    lo: float | None = None
    hi: float | None = None
    choices: frozenset[str] | None = None


def _s(n: int = 128) -> AttrSpec:
    return AttrSpec("str", max_len=n)


def _i(lo: int = 0, hi: int = 2**53) -> AttrSpec:
    return AttrSpec("int", lo=lo, hi=hi)


def _e(choices) -> AttrSpec:
    values = frozenset(choices)
    return AttrSpec("enum", max_len=max(len(v) for v in values), choices=values)


def _f(lo: float = -1e12, hi: float = 1e12) -> AttrSpec:
    return AttrSpec("float", lo=lo, hi=hi)


_OI_KINDS = frozenset(
    {
        "LLM",
        "CHAIN",
        "TOOL",
        "RETRIEVER",
        "EMBEDDING",
        "AGENT",
        "GUARDRAIL",
        "EVALUATOR",
        "RERANKER",
        "PROMPT",
        "UNKNOWN",
    }
)

# Typed, bounded, exact-key allowlist for span attributes in metadata mode.
ATTRIBUTE_ALLOWLIST: Final[dict[str, AttrSpec]] = {
    # GenAI semantic-convention names (public spec)
    "gen_ai.operation.name": _s(64),
    "gen_ai.provider.name": _s(64),
    "gen_ai.system": _s(64),
    "gen_ai.request.model": _s(128),
    "gen_ai.request.temperature": _f(0, 100),
    "gen_ai.request.top_p": _f(0, 1),
    "gen_ai.request.max_tokens": _i(),
    "gen_ai.response.model": _s(128),
    "gen_ai.response.id": _s(128),
    "gen_ai.response.finish_reasons": AttrSpec("str_seq", max_len=32, max_items=8),
    "gen_ai.usage.input_tokens": _i(),
    "gen_ai.usage.output_tokens": _i(),
    "gen_ai.usage.cache_read.input_tokens": _i(),
    "gen_ai.usage.cache_creation.input_tokens": _i(),
    "gen_ai.usage.reasoning.output_tokens": _i(),
    "gen_ai.agent.name": _s(128),
    "gen_ai.agent.id": _s(128),
    "gen_ai.tool.name": _s(128),
    "gen_ai.tool.call.id": _s(128),
    "gen_ai.tool.type": _s(32),
    "gen_ai.conversation.id": _s(128),
    "error.type": _s(128),
    # OpenInference-style names (public spec)
    "openinference.span.kind": _e(_OI_KINDS),
    "llm.model_name": _s(128),
    "embedding.model_name": _s(128),
    "tool.name": _s(128),
    "enduser.id": _s(128),
    "llm.provider": _s(64),
    "llm.system": _s(64),
    "llm.token_count.prompt": _i(),
    "llm.token_count.completion": _i(),
    "llm.token_count.total": _i(),
    # Correlation
    ATTR_SESSION_ID: _s(128),
    ATTR_USER_ID: _s(128),
    # Traigent
    ATTR_OBSERVATION_TYPE: _e(OBSERVATION_TYPES),
    ATTR_TRIAL_ID: _s(128),
    ATTR_OPTIMIZATION_SESSION_ID: _s(128),
    ATTR_EXPERIMENT_RUN_ID: _s(128),
    ATTR_CONFIG_HASH: _s(128),
    ATTR_PROMPT_NAME: _s(128),
    ATTR_PROMPT_VERSION: _i(0, 1_000_000),
    ATTR_DROPPED_ATTRS: _i(0, 1_000_000),
    CONTENT_MODE_ATTRIBUTE: _e(CONTENT_MODES),
}

# Keys known to carry user content: dropped in metadata mode (they are not
# allowlisted), replaced by a placeholder in redacted mode, scrubbed and capped
# in record mode.
CONTENT_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.tool.definitions",
        "gen_ai.prompt",
        "gen_ai.completion",
        "llm.input_messages",
        "llm.output_messages",
        "llm.prompts",
        "llm.prompt_template.template",
        "llm.prompt_template.variables",
        "llm.invocation_parameters",
        "input.value",
        "output.value",
        "tool.parameters",
        "tool.description",
        "retrieval.documents",
        "embedding.text",
        "embedding.embeddings",
        ATTR_INPUT,
        ATTR_OUTPUT,
        ATTR_TAGS,
    }
)

# Resource attribute allowlist (exact keys).
RESOURCE_ALLOWLIST: Final[dict[str, AttrSpec]] = {
    "service.name": _s(128),
    "service.version": _s(128),
    "service.namespace": _s(128),
    "deployment.environment": _s(64),
    "deployment.environment.name": _s(64),
    "telemetry.sdk.name": _s(64),
    "telemetry.sdk.language": _s(32),
    "telemetry.sdk.version": _s(32),
    CONTENT_MODE_ATTRIBUTE: _e(CONTENT_MODES),
}

# Event names that survive metadata/redacted mode and the only attribute on them.
ALLOWED_EVENTS: Final[dict[str, frozenset[str]]] = {
    "exception": frozenset({"exception.type"}),
}

MAX_RECORD_ATTR_BYTES: Final = 64 * 1024
MAX_SPAN_NAME_LEN: Final = 80

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-/:]{1,80}$")
# Identifier-shaped data hides in "static-looking" names (``order_12345_john``,
# ``user:42``, ``/users/42/orders``).  A name that carries any of these shapes
# is treated as data, not a static operation name.
_DIGIT_RUN = re.compile(r"[0-9]{3,}")
_HEX_RUN = re.compile(r"[0-9A-Fa-f]{8,}")
_UUID_SHAPE = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_SEGMENT_SPLIT = re.compile(r"[:/.]")
# Scope names are module paths by OTel convention.  Free text (hyphens,
# spaces, slashes) is rejected; a dotted identifier that merely *looks* like a
# module path cannot be told apart from one (documented residual).
_SAFE_SCOPE_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,63}(\.[A-Za-z_][A-Za-z0-9_]{0,63}){0,7}$"
)
_SAFE_SCOPE_VERSION = re.compile(r"^[A-Za-z0-9_.\-+]{1,32}$")
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def is_token_charset(value: str) -> bool:
    """Charset + length only (no identifier-shape heuristics)."""
    return bool(_SAFE_TOKEN.match(value))


def is_safe_token(value: str) -> bool:
    """A static-looking name: safe charset and no identifier-shaped data.

    Rejected: a run of >=3 digits, a run of >=8 hex characters, a UUID shape,
    or a purely numeric ``:``/``/``/``.`` separated segment.
    """
    if not _SAFE_TOKEN.match(value):
        return False
    if _DIGIT_RUN.search(value) or _HEX_RUN.search(value) or _UUID_SHAPE.search(value):
        return False
    return not any(seg.isdigit() for seg in _SEGMENT_SPLIT.split(value))


def is_safe_scope_name(value: str) -> bool:
    return bool(_SAFE_SCOPE_NAME.match(value))


def is_safe_scope_version(value: str) -> bool:
    return bool(_SAFE_SCOPE_VERSION.match(value))


def is_safe_key(value: str) -> bool:
    return bool(_SAFE_KEY.match(value))
