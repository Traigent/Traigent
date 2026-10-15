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
)
from traigent.legend.payloads import new_idempotency_key, request_digest
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
    "build_recovery_manifest",
    "compute_pillar",
    "declare",
    "manifest_fingerprint",
    "new_idempotency_key",
    "query",
    "request_digest",
    "set_default_transport",
]
