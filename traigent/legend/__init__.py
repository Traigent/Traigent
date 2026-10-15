"""Pillar Legend client (SDK part a): declare, query, recovery manifests, local aliases.

Contract: TraigentSchema ``docs/identity/pillar-legend-v0.md``. Fingerprints come from
:mod:`traigent.identity`; this package only declares them and builds manifests.
"""

from traigent.legend.aliases import AliasMap
from traigent.legend.bundle import Bundle
from traigent.legend.client import declare, query, set_default_transport
from traigent.legend.errors import LegendError
from traigent.legend.exporters import (
    AgentStateExport,
    AgentStateSource,
    EvaluatorFilesSource,
    export,
    export_agent_state,
    export_pillar,
    recompute_fingerprint,
)
from traigent.legend.manifest import (
    RecoveryManifest,
    build_recovery_manifest,
    manifest_fingerprint,
    pillar_representation_commitment,
    representation_commitment,
    validate_coverage,
)
from traigent.legend.payloads import (
    build_availability_payload,
    build_capture_payload,
    build_restore_payload,
    new_idempotency_key,
    request_digest,
)
from traigent.legend.pillars import (
    AgentManifestSource,
    AgentSource,
    ConfigurationSource,
    DatasetSource,
    EvaluatorManifestSource,
    EvaluatorSource,
    Pillar,
    SearchSpaceSource,
    compute_pillar,
)
from traigent.legend.recovery import (
    ActivationResult,
    CaptureIndex,
    CaptureRecord,
    ConformanceOutcome,
    RestoreResult,
    activate,
    capture,
    conformance_check,
    record_availability,
    restore,
)
from traigent.legend.store import ClientStore, LocalDirectoryStore
from traigent.legend.transport import (
    CaptureTransport,
    DeclareResult,
    HttpLegendTransport,
    InMemoryTransport,
    LegendTransport,
    QueryResult,
)

__all__ = [
    "ActivationResult",
    "AgentStateExport",
    "AgentStateSource",
    "Bundle",
    "CaptureIndex",
    "CaptureRecord",
    "CaptureTransport",
    "ClientStore",
    "ConformanceOutcome",
    "EvaluatorFilesSource",
    "LocalDirectoryStore",
    "RestoreResult",
    "activate",
    "capture",
    "conformance_check",
    "export",
    "export_agent_state",
    "export_pillar",
    "recompute_fingerprint",
    "restore",
    "AgentManifestSource",
    "AgentSource",
    "AliasMap",
    "ConfigurationSource",
    "DatasetSource",
    "DeclareResult",
    "EvaluatorManifestSource",
    "EvaluatorSource",
    "HttpLegendTransport",
    "InMemoryTransport",
    "LegendError",
    "LegendTransport",
    "Pillar",
    "QueryResult",
    "RecoveryManifest",
    "SearchSpaceSource",
    "build_availability_payload",
    "build_capture_payload",
    "build_recovery_manifest",
    "build_restore_payload",
    "pillar_representation_commitment",
    "record_availability",
    "representation_commitment",
    "validate_coverage",
    "compute_pillar",
    "declare",
    "manifest_fingerprint",
    "new_idempotency_key",
    "query",
    "request_digest",
    "set_default_transport",
]
