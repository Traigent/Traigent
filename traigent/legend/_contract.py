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
REPR_CONFIG_DOCUMENT: Final = "config_document"
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
)
COVERED_BY_KIND: Final = {
    KIND_AGENT: (
        "code",
        "prompts",
        "helper_modules",
        "tool_definitions",
        "applied_config",
        "dependency_lock",
    ),
    KIND_CONFIGURATION: ("applied_config",),
    KIND_DATASET: ("dataset_members", "dataset_order"),
    KIND_EVALUATOR: ("evaluator_code", "judge_config", "objectives"),
    KIND_SEARCH_SPACE: ("search_space_definition",),
}

EXPORTER_ID: Final = "traigent-sdk-legend"
EXPORTER_VERSION: Final = "0.1.0"

# Wire patterns (must match the schema; every pattern there ends in (?![\s\S])).
VERSION_REF_RE: Final = re.compile(r"versionref_[0-9a-f]{32}")
ENTITY_REF_RE: Final = re.compile(r"entityref_[0-9a-f]{32}")
CURSOR_RE: Final = re.compile(r"cursor_[0-9a-f]{32}")
KEY_ID_RE: Final = re.compile(r"k[0-9a-f]{16}")
SEARCH_SPACE_ID_RE: Final = re.compile(r"ssp1:(k[0-9a-f]{16}):[0-9a-f]{64}")
MULTISET_ROOT_RE: Final = re.compile(r"msr1:(k[0-9a-f]{16}):[0-9a-f]{64}")

QUERY_MAX_PAGE_SIZE: Final = 100

# Future Backend routes (item 3); not final.
ROUTE_DECLARE: Final = "/api/v1/legend/versions:declare"
ROUTE_QUERY: Final = "/api/v1/legend/versions:query"
LEGEND_ENABLED_ENV: Final = "TRAIGENT_LEGEND_HTTP_ENABLED"
