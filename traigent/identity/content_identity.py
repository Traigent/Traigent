"""Traigent content identity v1 (``traigent.content_identity.v1``) -- SDK port.

Ported, not imported: ``pyproject.toml`` forbids a runtime dependency on
TraigentSchema (PyPI rejects direct git references, see
``scripts/ci/schema-pin.txt``), so this module is a line-for-line port of
TraigentSchema ``traigent_schema/example_identity.py`` at commit
``024d55b0514c8c961d9496c34e8c5fc3548e1eb1`` (TraigentSchema ``develop``, the merge of PR #515;
identical to the PR's final head ``86a3bac8``). The only substantive
change is the import: canonicalization uses the SDK's own fp2 port
(:mod:`traigent.utils.fp2`, itself a line-for-line port of
``traigent_schema/fp2.py``); the rest is the SDK's ruff formatting and one
local type annotation for mypy. Conformance is enforced by
``tests/unit/identity/test_content_identity_vectors.py``, which runs every
section of the vendored vector file
(``tests/unit/identity/fixtures/content_identity_v1_vectors.json``). If the
Schema reference or its vectors change, re-port and re-vendor both.

Key custody (spec section 12 / ruling D1): the SDK NEVER holds a tenant master.
The reference's master-key derivation (``hkdf_sha256`` / ``hkdf_info`` /
``derive_tenant_keys``) is therefore NOT shipped: it lives in the test support
module ``tests/unit/identity/support/key_derivation.py``, where the
conformance vectors exercise it. Production code receives the two derived
purpose keys and ``kid`` from the Backend (:mod:`traigent.identity.keys`).

----

Reference-implementation docstring follows (unchanged apart from module paths).


Normative document: ``docs/identity/content-identity-v1.md``.
Conformance vectors: ``traigent_schema/data/content_identity_v1_vectors.json``.
Wire shapes: ``schemas/datasets/content_identity_v1_schema.json``.

What this module computes, all pure and offline (no network, no filesystem,
no randomness):

* (``derive_tenant_keys`` -- not shipped in the SDK; see above.)
* ``compute_example_id`` -- HMAC-SHA-256 over the canonical INPUT (and context,
  when present). Stable while the input is unchanged.
* ``compute_example_version`` -- HMAC-SHA-256 over the example id plus the
  expected output and version-relevant metadata. Changes when the label does.
* ``compute_multiset_root`` -- an RFC 9162 Merkle Tree Hash over sorted
  ``[example_id, example_version, count]`` leaves: order-free, duplicates
  counted. Used for both a dataset's root and a trial's evaluated root.
* ``build_inclusion_proof`` / ``verify_inclusion_proof`` -- RFC 9162 audit
  paths. Verification needs NO key: leaves are already keyed tokens.
* ``compute_public_input_digest`` -- the opt-in UNKEYED input digest, for
  public-benchmark contamination checks only.
* ``compute_agent_build_digest`` -- the unkeyed agent build version over an
  ``AgentBuildManifestV1`` (digests and version strings only, no content);
  ``certifiable=True`` rejects partial coverage.
* ``compute_evaluator_version_digest`` -- the unkeyed evaluator version over an
  ``EvaluatorVersionManifestV1`` (code, judge, objectives, dependencies).
* ``key_id_of`` -- the kid of any ex1/exv1/msr1 identifier (its second
  colon-separated field).

Canonicalization is ``jcs_v1`` via :mod:`traigent.utils.fp2` (RFC 8785) plus
the strict content-identity profile: non-finite numbers, lone surrogates (in
values AND keys), non-plain types, nesting deeper than 100 containers, and ANY
number -- integer or float -- whose magnitude exceeds 2**53-1 are rejected; no
Unicode normalization. The number rule is stricter than fp2 (which accepts a
large float) on purpose: JavaScript cannot tell 2**53+1 from 2**53 once parsed,
so accepting any number beyond the safe range lets two runtimes hash different
values for the same text. Duplicate object keys can only exist in JSON *text*,
so they are rejected by :func:`parse_strict_json`.

Error messages never echo caller values: inputs are user content and exception
messages are among the most reliably logged strings in a system.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from traigent.utils import fp2

__all__ = [
    "SCHEME",
    "HKDF_SALT",
    "DOMAIN_EXAMPLE_ID",
    "DOMAIN_EXAMPLE_VERSION",
    "DOMAIN_KEY_ID",
    "DOMAIN_MULTISET_LEAF",
    "DOMAIN_PUBLIC_INPUT",
    "DOMAIN_AGENT_BUILD",
    "DOMAIN_EVALUATOR_VERSION",
    "AGENT_ASSET_CATEGORIES",
    "AGENT_COVERAGE_VALUES",
    "EVALUATOR_ORIENTATIONS",
    "RESERVED_METADATA_KEYS",
    "MAX_COUNT",
    "ContentIdentityError",
    "ConflictingExampleVersionsWarning",
    "TenantIdentityKeys",
    "ExampleProjection",
    "MultisetMember",
    "MultisetRoot",
    "InclusionProof",
    "parse_strict_json",
    "canonical_bytes",
    "example_id_payload",
    "example_version_payload",
    "compute_example_id",
    "compute_example_version",
    "compute_public_input_digest",
    "compute_agent_build_digest",
    "compute_evaluator_version_digest",
    "key_id_of",
    "project_sdk_example",
    "identify_example",
    "multiset_leaf_data",
    "compute_multiset_root",
    "build_inclusion_proof",
    "verify_inclusion_proof",
    "is_sub_multiset",
    "example_id_overlap",
]

SCHEME = "traigent.content_identity.v1"
HKDF_SALT = b"traigent.content_identity.v1"
DOMAIN_EXAMPLE_ID = "traigent.content_identity.example_id.v1"
DOMAIN_EXAMPLE_VERSION = "traigent.content_identity.example_version.v1"
DOMAIN_KEY_ID = "traigent.content_identity.key_id.v1"
DOMAIN_MULTISET_LEAF = "traigent.content_identity.multiset_leaf.v1"
DOMAIN_PUBLIC_INPUT = "traigent.content_identity.public_input.v1"
DOMAIN_AGENT_BUILD = "traigent.content_identity.agent_build.v1"
DOMAIN_EVALUATOR_VERSION = "traigent.content_identity.evaluator_version.v1"

#: Annotation keys that never participate in example_version. The SDK
#: projection (:func:`project_sdk_example`) strips them; the primitive
#: (:func:`compute_example_version`) REJECTS them so a caller that skipped the
#: projection fails loudly instead of minting a version that changes whenever
#: someone re-tags a row. ``context`` is here because it belongs to the
#: example_id payload, not to metadata.
RESERVED_METADATA_KEYS = frozenset(
    {
        "context",
        "example_id",
        "external_id",
        "source_ref",
        "tags",
        "split",
        "splits",
        "difficulty",
        "confidence",
        "status",
        "score",
        "explanation",
        "created_at",
        "updated_at",
        "supersedes",
    }
)

#: Largest multiplicity a leaf may carry, and the largest total_count of a
#: multiset: the jcs_v1 safe-integer ceiling.
MAX_COUNT = 2**53 - 1
_MAX_SAFE_DECIMAL = Decimal(2**53 - 1)

#: Backend tenant ids (``enterprise_tenants.id``, e.g. ``tenant_<hex>``) as the
#: exact stored string; the same charset as common_types ForeignKeyId. ASCII
#: only, so UTF-8 encoding is unambiguous across runtimes.
_HEX64 = "[0-9a-f]{64}"
_KID = "k[0-9a-f]{16}"
# Every pattern is applied with re.fullmatch: ``re.match`` with ``$`` would
# accept a trailing newline ("ex1:...\n"), minting a second spelling of one id.
_KEY_ID_RE = re.compile(_KID)
_EXAMPLE_ID_RE = re.compile(rf"ex1:({_KID}):{_HEX64}")
_EXAMPLE_VERSION_RE = re.compile(rf"exv1:({_KID}):{_HEX64}")
_ROOT_RE = re.compile(rf"msr1:({_KID}):({_HEX64})")
_HASH_HEX_RE = re.compile(_HEX64)
_SHA256_DIGEST_RE = re.compile(rf"sha256:{_HEX64}")
# The repo's canonical METRIC_NAME (tests/test_metric_name_pattern_consistency.py),
# capped at 128 characters.
_OBJECTIVE_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]{0,127}")


class ContentIdentityError(ValueError):
    """A value cannot be given a content identity. Messages are content-free."""


class ConflictingExampleVersionsWarning(UserWarning):
    """One example_id appears with more than one example_version in a multiset."""


class _Absent:
    """Sentinel: the field is absent (distinct from any JSON value)."""

    _instance: _Absent | None = None

    def __new__(cls) -> _Absent:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT: Any = _Absent()


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ContentIdentityError(f"non-finite number literal is not supported ({name})")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContentIdentityError("duplicate object key in JSON text")
        result[key] = value
    return result


def _parse_int(text: str) -> int:
    value = int(text)
    if abs(value) > fp2._MAX_SAFE_INTEGER:
        raise ContentIdentityError("integer literal outside +/-(2**53-1)")
    return value


def _parse_float(text: str) -> float:
    # Judge the literal's EXACT decimal value, before rounding to a double:
    # 9007199254740991.1 rounds to 9007199254740991.0 (in range) but is not in
    # range as written, and a JS parser that looks at the rounded Number would
    # disagree with one that looks at the text. Decimal(text) and copy_abs()
    # are exact and Decimal comparison is exact; abs() is NOT -- it rounds to
    # the context precision (28 digits), which accepted
    # 9007199254740991.0000000000001. 1e400 and 9007199254740993.0 are
    # rejected by the same comparison.
    if Decimal(text).copy_abs() > _MAX_SAFE_DECIMAL:
        raise ContentIdentityError("number literal outside +/-(2**53-1)")
    return float(text)


def parse_strict_json(text: str) -> Any:
    """Parse JSON text under the strict profile.

    Rejects duplicate object keys, ``NaN``/``Infinity`` literals and number
    literals (integer or fractional/exponent form) whose value is outside
    +/-(2**53-1). Lone-surrogate escapes parse here but are rejected by
    :func:`canonical_bytes`, which every identity function calls. A JavaScript
    implementation MUST enforce the same rules on the source text, because
    ``JSON.parse`` silently keeps the last duplicate key and rounds large
    numbers.
    """
    if not isinstance(text, str):
        raise ContentIdentityError("JSON text must be a str")
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except ContentIdentityError:
        raise
    except ValueError as error:
        raise ContentIdentityError(
            f"invalid JSON text ({type(error).__name__})"
        ) from None


def _reject_unsafe_numbers(root: Any) -> None:
    """Reject any float whose magnitude exceeds 2**53-1.

    Runs only AFTER fp2 accepted ``root``, so the value is acyclic, at most
    100 containers deep and made of exact plain types -- an explicit stack is
    therefore bounded. Integers were already range-checked by fp2.
    """
    stack = [root]
    while stack:
        value = stack.pop()
        if type(value) is float and abs(value) > fp2._MAX_SAFE_INTEGER:
            raise ContentIdentityError("number outside +/-(2**53-1)")
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)


def canonical_bytes(value: Any) -> bytes:
    """Return the jcs_v1 (RFC 8785, strict content-identity profile) UTF-8 bytes of ``value``."""
    try:
        text: str = fp2.canonicalize(value)
    except fp2.Fp2UnsupportedValue as error:
        raise ContentIdentityError(f"value cannot be canonicalized: {error}") from None
    _reject_unsafe_numbers(value)
    return text.encode("utf-8")


def _framed(domain: str, payload: Any) -> bytes:
    return domain.encode("utf-8") + b"\x00" + canonical_bytes(payload)


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantIdentityKeys:
    """Per-purpose keys derived from one tenant master secret.

    ``key_id`` is public and appears inside every identifier; the two keys are
    secret. ``repr`` never shows key material.
    """

    key_id: str
    example_id_key: bytes = field(repr=False)
    example_version_key: bytes = field(repr=False)


def key_id_of(identifier: str) -> str:
    """Return the kid of an ``ex1:``/``exv1:``/``msr1:`` identifier.

    The kid is, normatively, the second colon-separated field. Consumers MUST
    check that it equals the record's ``key_id`` for the record, its root and
    every member.
    """
    if isinstance(identifier, str):
        for pattern in (_EXAMPLE_ID_RE, _EXAMPLE_VERSION_RE, _ROOT_RE):
            match = pattern.fullmatch(identifier)
            if match is not None:
                return match.group(1)
    raise ContentIdentityError("not a v1 ex1/exv1/msr1 identifier")


def _check_keys(keys: TenantIdentityKeys) -> None:
    if not isinstance(keys, TenantIdentityKeys) or not _KEY_ID_RE.fullmatch(
        keys.key_id
    ):
        raise ContentIdentityError(
            "keys must be TenantIdentityKeys (the Backend-granted purpose keys)"
        )


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------


def _is_absent(value: Any) -> bool:
    return value is ABSENT or value is None


def example_id_payload(input: Any, *, context: Any = ABSENT) -> dict[str, Any]:  # noqa: A002
    """The object whose jcs_v1 bytes example_id commits to.

    ``context`` absent and ``context=None`` are the same (omitted); any other
    value, including ``{}`` or ``""``, is hashed.
    """
    if input is ABSENT:
        raise ContentIdentityError("input is required")
    payload: dict[str, Any] = {"input": input}
    if not _is_absent(context):
        payload["context"] = context
    return payload


def _check_metadata(metadata: Any) -> dict[str, Any] | None:
    if _is_absent(metadata):
        return None
    if type(metadata) is not dict:
        raise ContentIdentityError("metadata must be a plain dict or None")
    reserved = [key for key in metadata if key in RESERVED_METADATA_KEYS]
    if reserved:
        # Reserved key NAMES are schema vocabulary, not user content.
        raise ContentIdentityError(
            "metadata contains reserved annotation keys that never participate in "
            f"example_version: {sorted(reserved)}; project the example first "
            "(project_sdk_example)"
        )
    return metadata or None


def example_version_payload(
    example_id: str, *, expected: Any = ABSENT, metadata: Any = None
) -> dict[str, Any]:
    """The object whose jcs_v1 bytes example_version commits to.

    ``expected`` absent and ``None`` are the same (omitted). ``metadata``
    absent, ``None`` and ``{}`` are the same (omitted).
    """
    if not isinstance(example_id, str) or not _EXAMPLE_ID_RE.fullmatch(example_id):
        raise ContentIdentityError("example_id is not a v1 example id")
    payload: dict[str, Any] = {"example_id": example_id}
    if not _is_absent(expected):
        payload["expected"] = expected
    checked = _check_metadata(metadata)
    if checked is not None:
        payload["metadata"] = checked
    return payload


def _hmac_hex(key: bytes, domain: str, payload: Any) -> str:
    return hmac.new(key, _framed(domain, payload), hashlib.sha256).hexdigest()


def compute_example_id(
    keys: TenantIdentityKeys,
    input: Any,  # noqa: A002
    *,
    context: Any = ABSENT,
) -> str:
    """``ex1:<kid>:<hex>`` = HMAC(K_example_id, domain || 0x00 || jcs_v1({input, context?}))."""
    _check_keys(keys)
    payload = example_id_payload(input, context=context)
    return f"ex1:{keys.key_id}:{_hmac_hex(keys.example_id_key, DOMAIN_EXAMPLE_ID, payload)}"


def compute_example_version(
    keys: TenantIdentityKeys,
    example_id: str,
    *,
    expected: Any = ABSENT,
    metadata: Any = None,
) -> str:
    """``exv1:<kid>:<hex>`` = HMAC(K_example_version, domain || 0x00 ||
    jcs_v1({example_id, expected?, metadata?}))."""
    _check_keys(keys)
    payload = example_version_payload(example_id, expected=expected, metadata=metadata)
    match = _EXAMPLE_ID_RE.fullmatch(example_id)
    if match is None:  # defensive: example_version_payload already validated it
        raise ContentIdentityError("example_id is not a valid ex1 identifier")
    if match.group(1) != keys.key_id:
        raise ContentIdentityError("example_id was minted under a different key id")
    digest = _hmac_hex(keys.example_version_key, DOMAIN_EXAMPLE_VERSION, payload)
    return f"exv1:{keys.key_id}:{digest}"


def compute_public_input_digest(input: Any, *, context: Any = ABSENT) -> str:  # noqa: A002
    """``exu1:<hex>`` = SHA-256(domain || 0x00 || jcs_v1({input, context?})).

    UNKEYED: anyone who can guess the input can confirm it. Compute it only
    when the tenant has opted in, and only for examples that come from a
    public benchmark, where the input is already public.
    """
    payload = example_id_payload(input, context=context)
    return "exu1:" + hashlib.sha256(_framed(DOMAIN_PUBLIC_INPUT, payload)).hexdigest()


AGENT_ASSET_CATEGORIES = ("helper_modules", "prompts", "tool_definitions")
AGENT_COVERAGE_VALUES = ("complete", "partial")


def _require_digest(value: Any, what: str) -> None:
    if not isinstance(value, str) or not _SHA256_DIGEST_RE.fullmatch(value):
        raise ContentIdentityError(f"{what} must be a sha256:<64 hex> digest")


def compute_agent_build_digest(
    manifest: Mapping[str, Any], *, certifiable: bool = False
) -> str:
    """``sha256:<hex>`` = SHA-256(domain || 0x00 || jcs_v1(AgentBuildManifestV1)).

    Full shape validation is the schema's job
    (``schemas/agents/agent_version_manifest_v1_schema.json``). This function
    enforces the rules without which the digest would not identify a build:

    * at least one of ``code_revision`` / ``source_digest``, and
      ``source_digest`` whenever ``code_revision.dirty`` is true;
    * ``asset_digests`` with all three categories (``helper_modules``,
      ``prompts``, ``tool_definitions``; an empty map asserts "none");
    * ``applied_config_digest`` (the candidate configuration);
    * ``runtime`` with ``language`` and ``sdk_version``;
    * ``coverage`` in {"complete", "partial"}.

    With ``certifiable=True`` a manifest whose coverage is not ``complete`` is
    rejected: a partial manifest's digest cannot stand for the agent's
    behaviour in a certificate.
    """
    if type(manifest) is not dict:
        raise ContentIdentityError("agent build manifest must be a plain dict")
    revision = manifest.get("code_revision")
    if revision is None and "source_digest" not in manifest:
        raise ContentIdentityError(
            "agent build manifest needs code_revision or source_digest"
        )
    if isinstance(revision, dict) and revision.get("dirty") is True:
        if "source_digest" not in manifest:
            raise ContentIdentityError(
                "agent build manifest with a dirty code_revision needs source_digest"
            )
    assets = manifest.get("asset_digests")
    if type(assets) is not dict or sorted(assets) != sorted(AGENT_ASSET_CATEGORIES):
        raise ContentIdentityError(
            "asset_digests must have exactly the categories "
            f"{list(AGENT_ASSET_CATEGORIES)}"
        )
    for category in AGENT_ASSET_CATEGORIES:
        if type(assets[category]) is not dict:
            raise ContentIdentityError(f"asset_digests.{category} must be a map")
        for digest in assets[category].values():
            _require_digest(digest, f"asset_digests.{category} value")
    _require_digest(manifest.get("applied_config_digest"), "applied_config_digest")
    runtime = manifest.get("runtime")
    if (
        type(runtime) is not dict
        or not runtime.get("language")
        or not runtime.get("sdk_version")
    ):
        raise ContentIdentityError(
            "runtime (language, sdk_version) is required: it affects behaviour"
        )
    coverage = manifest.get("coverage")
    if coverage not in AGENT_COVERAGE_VALUES:
        raise ContentIdentityError("coverage must be 'complete' or 'partial'")
    if certifiable and coverage != "complete":
        raise ContentIdentityError(
            "a partial-coverage agent manifest is not certifiable"
        )
    return "sha256:" + hashlib.sha256(_framed(DOMAIN_AGENT_BUILD, manifest)).hexdigest()


EVALUATOR_ORIENTATIONS = ("maximize", "minimize")


def compute_evaluator_version_digest(manifest: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` = SHA-256(domain || 0x00 || jcs_v1(EvaluatorVersionManifestV1)).

    The evaluator version commits to everything that changes a score: the
    evaluator code (its fp2 efp2 digest), the judge model and its
    configuration (``judge``: null for a model-free evaluator -- explicit, never
    omitted), the objective set, and the behaviour-affecting dependency
    versions, plus -- because efp2 covers the evaluator's source only -- the
    evaluator's bound configuration (``config_digest``: thresholds, captured
    parameters) and the contents of its local helper files
    (``helper_digests``; ``{}`` asserts none). ``objectives`` MUST be sorted
    ascending by ``name`` (ASCII) with unique names, so the same set has one
    spelling; it is rejected otherwise rather than silently re-sorted. Schema:
    ``schemas/evaluation/evaluator_version_manifest_v1_schema.json``.
    """
    if type(manifest) is not dict:
        raise ContentIdentityError("evaluator version manifest must be a plain dict")
    _require_digest(manifest.get("code_digest"), "code_digest")
    _require_digest(manifest.get("config_digest"), "config_digest")
    helpers = manifest.get("helper_digests")
    if type(helpers) is not dict:
        raise ContentIdentityError("helper_digests must be a map (empty means none)")
    for digest in helpers.values():
        _require_digest(digest, "helper_digests value")
    if "judge" not in manifest:
        raise ContentIdentityError(
            "judge must be present (null for a model-free evaluator)"
        )
    judge = manifest["judge"]
    if judge is not None:
        if type(judge) is not dict or not isinstance(judge.get("model"), str):
            raise ContentIdentityError(
                "judge must be null or an object with a model id"
            )
        _require_digest(judge.get("config_digest"), "judge.config_digest")
    objectives = manifest.get("objectives")
    if type(objectives) is not list or not objectives:
        raise ContentIdentityError("objectives must be a non-empty list")
    names: list[str] = []
    for objective in objectives:
        if type(objective) is not dict:
            raise ContentIdentityError("each objective must be an object")
        name = objective.get("name")
        if not isinstance(name, str) or not _OBJECTIVE_NAME_RE.fullmatch(name):
            raise ContentIdentityError(
                "objective name must fully match [a-zA-Z_][a-zA-Z0-9_.-]{0,127}"
            )
        if objective.get("orientation") not in EVALUATOR_ORIENTATIONS:
            raise ContentIdentityError(
                "objective orientation must be maximize or minimize"
            )
        weight = objective.get("weight")
        if type(weight) not in (int, float):
            raise ContentIdentityError("objective weight must be a number")
        names.append(name)
    if names != sorted(set(names)):
        raise ContentIdentityError(
            "objectives must be sorted by name with unique names"
        )
    if type(manifest.get("dependency_versions")) is not dict:
        raise ContentIdentityError(
            "dependency_versions must be a map (empty means none)"
        )
    return (
        "sha256:"
        + hashlib.sha256(_framed(DOMAIN_EVALUATOR_VERSION, manifest)).hexdigest()
    )


@dataclass(frozen=True)
class ExampleProjection:
    """An example reduced to the four fields content identity reads."""

    input: Any
    context: Any = ABSENT
    expected: Any = ABSENT
    metadata: dict[str, Any] | None = None


def project_sdk_example(
    input_data: Any,
    expected_output: Any = ABSENT,
    metadata: Mapping[str, Any] | None = None,
) -> ExampleProjection:
    """Project an SDK ``EvaluationExample`` (input_data / expected_output / metadata).

    * input    := input_data
    * context  := metadata["context"] when present and not None
    * expected := expected_output (None == absent)
    * metadata := metadata minus RESERVED_METADATA_KEYS; empty == absent

    Both SDKs MUST implement exactly this projection; the ``projection`` vectors
    pin it.
    """
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ContentIdentityError("metadata must be a mapping or None")
    context = metadata.get("context", ABSENT)
    kept = {
        key: value
        for key, value in metadata.items()
        if key not in RESERVED_METADATA_KEYS
    }
    for key in kept:
        if type(key) is not str:
            raise ContentIdentityError("metadata keys must be plain strings")
    return ExampleProjection(
        input=input_data,
        context=ABSENT if context is None else context,
        expected=ABSENT if expected_output is None else expected_output,
        metadata=kept or None,
    )


def identify_example(
    keys: TenantIdentityKeys, projection: ExampleProjection
) -> tuple[str, str]:
    """Return ``(example_id, example_version)`` for a projected example."""
    example_id = compute_example_id(keys, projection.input, context=projection.context)
    example_version = compute_example_version(
        keys, example_id, expected=projection.expected, metadata=projection.metadata
    )
    return example_id, example_version


# --------------------------------------------------------------------------
# Multisets (dataset_root, evaluated_root)
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class MultisetMember:
    """One distinct (example_id, example_version) pair and its multiplicity."""

    example_id: str
    example_version: str
    count: int = 1

    def __post_init__(self) -> None:
        id_match = (
            _EXAMPLE_ID_RE.fullmatch(self.example_id)
            if isinstance(self.example_id, str)
            else None
        )
        version_match = (
            _EXAMPLE_VERSION_RE.fullmatch(self.example_version)
            if isinstance(self.example_version, str)
            else None
        )
        if id_match is None or version_match is None:
            raise ContentIdentityError(
                "member is not a v1 (example_id, example_version) pair"
            )
        if id_match.group(1) != version_match.group(1):
            raise ContentIdentityError("member id and version carry different key ids")
        if type(self.count) is not int or not 1 <= self.count <= MAX_COUNT:
            raise ContentIdentityError("member count must be an int in [1, 2**53-1]")

    @property
    def key_id(self) -> str:
        return self.example_id.split(":", 2)[1]


@dataclass(frozen=True)
class MultisetRoot:
    """The order-free root of a multiset plus the facts a caller must report."""

    root: str
    key_id: str
    members: tuple[MultisetMember, ...]
    distinct_count: int
    total_count: int
    conflicting_example_ids: tuple[str, ...]


def multiset_leaf_data(member: MultisetMember) -> bytes:
    """RFC 9162 leaf *data* (before the 0x00 leaf prefix) for one member."""
    return _framed(
        DOMAIN_MULTISET_LEAF, [member.example_id, member.example_version, member.count]
    )


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly less than ``n`` (n >= 2)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _mth(leaf_hashes: Sequence[bytes]) -> bytes:
    n = len(leaf_hashes)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return leaf_hashes[0]
    k = _split(n)
    return _node_hash(_mth(leaf_hashes[:k]), _mth(leaf_hashes[k:]))


def _path(m: int, leaf_hashes: Sequence[bytes]) -> list[bytes]:
    n = len(leaf_hashes)
    if n == 1:
        return []
    k = _split(n)
    if m < k:
        return _path(m, leaf_hashes[:k]) + [_mth(leaf_hashes[k:])]
    return _path(m - k, leaf_hashes[k:]) + [_mth(leaf_hashes[:k])]


def _coerce_members(items: Iterable[Any]) -> list[MultisetMember]:
    members: list[MultisetMember] = []
    for item in items:
        if isinstance(item, MultisetMember):
            members.append(item)
        elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
            members.append(MultisetMember(*item))
        else:
            raise ContentIdentityError(
                "multiset items must be MultisetMember or (example_id, example_version[, count])"
            )
    return members


def _normalize(items: Iterable[Any]) -> tuple[MultisetMember, ...]:
    """Merge equal pairs by summing counts and sort by (example_id, example_version).

    Ids are ASCII by construction, so Python string order equals byte order
    equals UTF-16 code-unit order: every implementation sorts identically.
    """
    totals: dict[tuple[str, str], int] = {}
    for member in _coerce_members(items):
        pair = (member.example_id, member.example_version)
        totals[pair] = totals.get(pair, 0) + member.count
        if totals[pair] > MAX_COUNT:
            raise ContentIdentityError("member count overflows 2**53-1")
    return tuple(
        MultisetMember(example_id, example_version, count)
        for (example_id, example_version), count in sorted(totals.items())
    )


def compute_multiset_root(
    items: Iterable[Any], *, key_id: str | None = None
) -> MultisetRoot:
    """Return the order-free multiset root ``msr1:<kid>:<hex>``.

    ``items`` may repeat a pair (each occurrence counts) or carry explicit
    counts; both are summed. All members must share one key id; an empty
    multiset needs ``key_id``. When one example_id carries several versions,
    a :class:`ConflictingExampleVersionsWarning` is emitted and the ids are
    listed in ``conflicting_example_ids`` -- the root is still defined.
    """
    members = _normalize(items)
    key_ids = {member.key_id for member in members}
    if key_id is not None:
        if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id):
            raise ContentIdentityError("key_id is not a v1 key id")
        key_ids.add(key_id)
    if not key_ids:
        raise ContentIdentityError("an empty multiset needs an explicit key_id")
    if len(key_ids) != 1:
        raise ContentIdentityError("a multiset cannot mix key ids")
    (resolved_key_id,) = key_ids

    versions_per_id: dict[str, int] = {}
    for member in members:
        versions_per_id[member.example_id] = (
            versions_per_id.get(member.example_id, 0) + 1
        )
    conflicts = tuple(sorted(eid for eid, n in versions_per_id.items() if n > 1))
    if conflicts:
        warnings.warn(
            f"{len(conflicts)} example_id(s) carry more than one example_version "
            "(conflicting labels) in one multiset",
            ConflictingExampleVersionsWarning,
            stacklevel=2,
        )

    total_count = sum(member.count for member in members)
    if total_count > MAX_COUNT:
        raise ContentIdentityError("multiset total_count overflows 2**53-1")

    root_hash = _mth([_leaf_hash(multiset_leaf_data(member)) for member in members])
    return MultisetRoot(
        root=f"msr1:{resolved_key_id}:{root_hash.hex()}",
        key_id=resolved_key_id,
        members=members,
        distinct_count=len(members),
        total_count=total_count,
        conflicting_example_ids=conflicts,
    )


@dataclass(frozen=True)
class InclusionProof:
    """RFC 9162 inclusion proof that ``member`` is a leaf of ``root``."""

    member: MultisetMember
    leaf_index: int
    tree_size: int
    audit_path: tuple[str, ...]
    root: str


def build_inclusion_proof(
    multiset: MultisetRoot, member: MultisetMember
) -> InclusionProof:
    """Prove that ``member`` (exact pair AND count) is a leaf of ``multiset``."""
    try:
        index = multiset.members.index(member)
    except ValueError:
        raise ContentIdentityError("member is not a leaf of this multiset") from None
    leaf_hashes = [_leaf_hash(multiset_leaf_data(m)) for m in multiset.members]
    return InclusionProof(
        member=member,
        leaf_index=index,
        tree_size=len(leaf_hashes),
        audit_path=tuple(node.hex() for node in _path(index, leaf_hashes)),
        root=multiset.root,
    )


def verify_inclusion_proof(proof: InclusionProof) -> bool:
    """RFC 9162 section 2.1.3.2 verification. Needs no key. Never raises on a bad proof."""
    root_match = _ROOT_RE.fullmatch(proof.root) if isinstance(proof.root, str) else None
    if root_match is None or root_match.group(1) != proof.member.key_id:
        return False
    if type(proof.leaf_index) is not int or type(proof.tree_size) is not int:
        return False
    if not 0 <= proof.leaf_index < proof.tree_size:
        return False
    path: list[bytes] = []
    for node in proof.audit_path:
        if not isinstance(node, str) or not _HASH_HEX_RE.fullmatch(node):
            return False
        path.append(bytes.fromhex(node))

    fn = proof.leaf_index
    sn = proof.tree_size - 1
    r = _leaf_hash(multiset_leaf_data(proof.member))
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = _node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = _node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(r, bytes.fromhex(root_match.group(2)))


# --------------------------------------------------------------------------
# Set relations used by the statistics (see design doc, "Statistical use")
# --------------------------------------------------------------------------


def is_sub_multiset(inner: MultisetRoot, outer: MultisetRoot) -> bool:
    """True when every (id, version) of ``inner`` occurs in ``outer`` at least as often."""
    if inner.key_id != outer.key_id:
        return False
    available = {(m.example_id, m.example_version): m.count for m in outer.members}
    return all(
        available.get((m.example_id, m.example_version), 0) >= m.count
        for m in inner.members
    )


def example_id_overlap(first: MultisetRoot, second: MultisetRoot) -> tuple[str, ...]:
    """Sorted example_ids present in both multisets, regardless of version.

    Overlap is judged on the INPUT (example_id), not on (id, version): a
    relabelled example still leaks its input from search into evaluation.
    """
    if first.key_id != second.key_id:
        raise ContentIdentityError(
            "cannot compare multisets minted under different key ids"
        )
    return tuple(
        sorted(
            {m.example_id for m in first.members}
            & {m.example_id for m in second.members}
        )
    )
