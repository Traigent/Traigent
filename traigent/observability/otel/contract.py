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

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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


_CONTRACT_PATH = Path(__file__).with_name("otel_attribute_contract_v1.json")
#: The shared OTel attribute contract (hash-locked copy of the Schema repo's
#: file).  The SDK CONSUMES it: the egress set, the content-mode resolution
#: vectors and the usage rules below all come from here, not from private lists.
CONTRACT: Final[dict[str, Any]] = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))

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

#: Keys in the contract's egress set that belong to another channel: resource
#: attributes and the exception event.  They are never span attributes.
RESOURCE_KEYS: Final = frozenset(
    {
        "service.name",
        "service.version",
        "deployment.environment",
        "deployment.environment.name",
    }
)
EVENT_KEYS: Final = frozenset({"exception.type"})

# SDK-side refinements that only NARROW a contract type (never widen): the
# contract types these as bounded strings, the SDK accepts only the known set.
_ENUM_NARROWING: Final[dict[str, frozenset[str]]] = {
    "openinference.span.kind": _OI_KINDS,
    ATTR_OBSERVATION_TYPE: OBSERVATION_TYPES,
    CONTENT_MODE_ATTRIBUTE: frozenset(CONTENT_MODES),
}


def _spec_from_contract(key: str, rule: Mapping[str, Any]) -> AttrSpec:
    kind = rule["type"]
    if key in _ENUM_NARROWING:
        values = _ENUM_NARROWING[key]
        return AttrSpec(
            "enum",
            max_len=min(rule.get("max_length", 128), max(len(v) for v in values)),
            choices=values,
        )
    if "enum" in rule:
        values = frozenset(rule["enum"])
        return AttrSpec("enum", max_len=rule.get("max_length", 128), choices=values)
    if kind == "string":
        return AttrSpec("str", max_len=rule["max_length"])
    if kind == "string_array":
        return AttrSpec(
            "str_seq", max_len=rule["max_length"], max_items=rule["max_items"]
        )
    if kind == "non_negative_integer":
        return AttrSpec("int", lo=rule["minimum"], hi=rule["maximum"])
    if kind == "number":
        return AttrSpec("float")
    raise ValueError(f"unsupported contract attribute type {kind!r} for {key}")


_EGRESS: Final[Mapping[str, Mapping[str, Any]]] = CONTRACT["metadata_allowlist"][
    "attributes"
]

# Typed, bounded, exact-key allowlist for span attributes in metadata mode:
# the contract's single egress set minus the resource/event-only keys.
ATTRIBUTE_ALLOWLIST: Final[dict[str, AttrSpec]] = {
    key: _spec_from_contract(key, rule)
    for key, rule in _EGRESS.items()
    if key not in RESOURCE_KEYS and key not in EVENT_KEYS
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
    **{
        key: _spec_from_contract(key, rule)
        for key, rule in _EGRESS.items()
        if key in RESOURCE_KEYS
    },
    CONTENT_MODE_ATTRIBUTE: _spec_from_contract(
        CONTENT_MODE_ATTRIBUTE, _EGRESS[CONTENT_MODE_ATTRIBUTE]
    ),
}

# Event names that survive metadata/redacted mode and the only attribute on them.
ALLOWED_EVENTS: Final[dict[str, frozenset[str]]] = {
    "exception": frozenset({"exception.type"}),
}

MAX_RECORD_ATTR_BYTES: Final = 64 * 1024
MAX_SPAN_NAME_LEN: Final = 80
MAX_RECORD_NAME_LEN: Final = 256
MAX_RECORD_STATUS_LEN: Final = 1024

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
