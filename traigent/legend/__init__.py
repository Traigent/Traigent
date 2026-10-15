"""Pillar Legend client (SDK part a): declare, query, recovery manifests, local aliases.

Contract: TraigentSchema ``docs/identity/pillar-legend-v0.md``. Fingerprints come from
:mod:`traigent.identity`; this package only declares them and builds manifests.
"""

from traigent.legend.aliases import AliasMap
from traigent.legend.client import declare, query, set_default_transport
from traigent.legend.errors import LegendError
from traigent.legend.manifest import (
    RecoveryManifest,
    build_recovery_manifest,
    manifest_fingerprint,
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
from traigent.legend.transport import (
    DeclareResult,
    HttpLegendTransport,
    InMemoryTransport,
    LegendTransport,
    QueryResult,
)

__all__ = [
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
    "query",
    "set_default_transport",
]
