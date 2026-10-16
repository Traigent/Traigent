"""Contract constants for the Pillar Legend (v0).

Every value here mirrors TraigentSchema ``docs/identity/pillar-legend-v0.md`` and
``traigent_schema/schemas/legend/*.json``. The contract is still under review:
keep all of it in this one module so a rename or enum change is a one-file edit.
"""

from __future__ import annotations

import re
from typing import Final

SCHEMA_VERSION: Final = "0"
MANIFEST_SCHEME: Final = "traigent.recovery_manifest.v0"
MANIFEST_DOMAIN: Final = "traigent.recovery_manifest.v0"
MANIFEST_FINGERPRINT_PREFIX: Final = "rmf0:sha256:"

# EntityKind
KIND_AGENT: Final = "agent"
KIND_CONFIGURATION: Final = "configuration"
KIND_DATASET: Final = "evaluation_dataset"
KIND_EVALUATOR: Final = "evaluator"
KIND_SEARCH_SPACE: Final = "search_space"
ENTITY_KINDS: Final = frozenset(
    {KIND_AGENT, KIND_CONFIGURATION, KIND_DATASET, KIND_EVALUATOR, KIND_SEARCH_SPACE}
)

# FingerprintScheme chosen per kind by the SDK (section 2 of the design note).
SCHEME_BY_KIND: Final = {
    KIND_AGENT: "agent_build_digest_v1",
    KIND_CONFIGURATION: "applied_config_digest_v1",
    KIND_DATASET: "dataset_multiset_root_v1",
    KIND_EVALUATOR: "evaluator_version_digest_v1",
    KIND_SEARCH_SPACE: "search_space_ssp1",
}
KEYED_SCHEMES: Final = frozenset({"dataset_multiset_root_v1", "search_space_ssp1"})

IDENTITY_SDK_FINGERPRINTED: Final = "sdk_fingerprinted"
IDENTITY_CLIENT_DECLARED: Final = "client_declared"
PROVENANCE_EXPLICIT: Final = "explicit"
PROVENANCE_IMPLICIT_RUN: Final = "implicit_run"
# Client-sendable provenance values; ``implicit_legacy`` is server-only (backfill).
CLIENT_PROVENANCES: Final = frozenset({PROVENANCE_EXPLICIT, PROVENANCE_IMPLICIT_RUN})

CHANGE_REASONS: Final = frozenset(
    {
        "code",
        "prompt",
        "configuration",
        "dataset_content",
        "dependency",
        "rollback",
        "unspecified",
    }
)

# Recovery manifest
ORDERING_PRESERVED: Final = "preserved"
ORDERING_NOT_APPLICABLE: Final = "not_applicable"
REPR_SOURCE_BUNDLE: Final = "source_bundle"
REPR_PROMPT_FILES: Final = "prompt_files"
REPR_HELPER_MODULES: Final = "helper_modules"
REPR_TOOL_DEFINITIONS: Final = "tool_definitions"
REPR_CONFIG_DOCUMENT: Final = "config_document"
REPR_DEPENDENCY_LOCK: Final = "dependency_lock"
REPR_DATASET_ROWS: Final = "dataset_rows"
REPR_EVALUATOR_BUNDLE: Final = "evaluator_bundle"
REPR_JUDGE_CONFIG: Final = "judge_config_document"
REPR_SEARCH_SPACE: Final = "search_space_document"
COVERAGE_COMPONENTS: Final = (
    "code",
    "prompts",
    "helper_modules",
    "tool_definitions",
    "applied_config",
    "dependency_lock",
    "dataset_members",
    "dataset_order",
    "evaluator_code",
    "judge_config",
    "objectives",
    "search_space_definition",
    "runtime",
)
# RequiredComponents(kind), fixed by the schema. ``covered`` and ``missing`` partition
# exactly this set (disjoint, union = required); ``dependency_lock`` is optional extra
# coverage and appears in neither list (it stays a representation).
REQUIRED_BY_KIND: Final = {
    KIND_AGENT: (
        "code",
        "prompts",
        "helper_modules",
        "tool_definitions",
        "applied_config",
        "runtime",
    ),
    KIND_CONFIGURATION: ("applied_config",),
    KIND_DATASET: ("dataset_members", "dataset_order"),
    KIND_EVALUATOR: ("evaluator_code", "judge_config", "objectives", "helper_modules"),
    KIND_SEARCH_SPACE: ("search_space_definition",),
}
# The component whose truth the identity-digest reproduction vouches for; it is demoted
# to ``missing`` when an agent/evaluator capture did not reproduce the identity digests.
IDENTITY_COMPONENT_BY_KIND: Final = {
    KIND_AGENT: "code",
    KIND_EVALUATOR: "evaluator_code",
    KIND_SEARCH_SPACE: "search_space_definition",
}

# RepresentationCommitment: sha256: + hex SHA-256(domain || 0x00 || jcs_v1({"representations": [d0, ...]}))
REPRESENTATION_COMMITMENT_DOMAIN: Final = "traigent.legend.representation_commitment.v0"
EVENT_REF_RE: Final = re.compile(r"eventref_[0-9a-f]{32}")
SEMVER_RE: Final = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
SHA256_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
AVAILABILITY_STATUSES: Final = frozenset(
    {"available", "missing", "corrupted", "retired"}
)

EXPORTER_ID: Final = "traigent-sdk-legend"
EXPORTER_VERSION: Final = "0.1.0"

# Wire patterns (must match the schema; every pattern there ends in (?![\s\S])).
IDEMPOTENCY_KEY_RE: Final = re.compile(r"idk_[0-9a-f]{32}")
CONFORMANCE_KEY_PREFIX: Final = "rmfc0:sha256:"
REQUEST_DOMAIN: Final = "traigent.legend.request.v0"
VERSION_REF_RE: Final = re.compile(r"versionref_[0-9a-f]{32}")
ENTITY_REF_RE: Final = re.compile(r"entityref_[0-9a-f]{32}")
CURSOR_RE: Final = re.compile(r"cursor_[0-9a-f]{32}")
MANIFEST_FINGERPRINT_RE: Final = re.compile(r"rmf0:sha256:[0-9a-f]{64}")
KEY_ID_RE: Final = re.compile(r"k[0-9a-f]{16}")
SEARCH_SPACE_ID_RE: Final = re.compile(r"ssp1:(k[0-9a-f]{16}):[0-9a-f]{64}")
MULTISET_ROOT_RE: Final = re.compile(r"msr1:(k[0-9a-f]{16}):[0-9a-f]{64}")

QUERY_MAX_PAGE_SIZE: Final = 100

# Live Backend routes (under the API base URL; TraigentBackend pillar_legend_routes.py).
ROUTE_DECLARE: Final = "/legend/declare"
ROUTE_QUERY: Final = "/legend/query"
ROUTE_RECORD_CAPTURE: Final = "/legend/record-capture"
ROUTE_RECORD_RESTORE: Final = "/legend/record-restore"
ROUTE_RECORD_AVAILABILITY: Final = "/legend/record-availability"
ROUTE_ADMISSION_FMT: Final = "/sessions/{session_id}/legend/admission"

# Recovery state / restore grades / store statuses (pillar_legend_v0.json)
RECOVERY_STORED_UNVERIFIED: Final = "stored_unverified"
RECOVERY_ROUND_TRIP_VERIFIED: Final = "round_trip_verified"
GRADE_EXACT: Final = "exact"
GRADE_PARTIAL: Final = "partial"
GRADE_SUBSTITUTED: Final = "substituted"
GRADE_FAILED: Final = "failed"
# BackendKeyed(version) (design note section 4): ``ssp1`` search spaces and evaluators
# (keyed ``evd1`` leaves under Backend-only ``K_eval``; an unknown authoritative format
# also counts as backend-keyed). The Backend refuses a claim that differs from its
# derived value, so the SDK states it from the declared authoritative format (see
# ``backend_keyed``): unknown => keyed, plain => not.
BACKEND_KEYED_KINDS: Final = frozenset({KIND_SEARCH_SPACE})
# Evaluator authoritative manifest format as declared by the caller (reference model:
# ``plain | keyed | unknown``). An evaluator is backend-keyed iff keyed OR unknown.
AUTH_FORMAT_PLAIN: Final = "plain"
AUTH_FORMAT_KEYED: Final = "keyed"
AUTH_FORMAT_UNKNOWN: Final = "unknown"
AUTH_FORMATS: Final = frozenset(
    {AUTH_FORMAT_PLAIN, AUTH_FORMAT_KEYED, AUTH_FORMAT_UNKNOWN}
)


def backend_keyed(kind: str, authoritative_format: str = AUTH_FORMAT_UNKNOWN) -> bool:
    """``BackendKeyed(version)``: search space always; evaluator unless its format is ``plain``."""
    if kind in BACKEND_KEYED_KINDS:
        return True
    return kind == KIND_EVALUATOR and authoritative_format != AUTH_FORMAT_PLAIN


EVIDENCE_CLIENT_SDK: Final = "client_sdk_reported"
