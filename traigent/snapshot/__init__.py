"""Customer-side execution snapshots.

Capture the dataset, configuration, agent and evaluator inputs of an
optimization into one verifiable bundle, and store it in storage you own.
Nothing here talks to Traigent: a snapshot is an unbound client declaration
("these bytes were captured"), not proof of what ran.
"""

from traigent.snapshot.capture import (
    EvaluatorInputs,
    build_execution_snapshot,
    serialize_example,
)
from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.manifest import (
    ExecutionSnapshot,
    LoadedSnapshot,
    UnavailableReason,
)
from traigent.snapshot.store import (
    CreateResult,
    SnapshotArtifactStore,
    SnapshotRef,
    SnapshotStore,
    SnapshotVerification,
    load_snapshot,
    publish_snapshot,
    verify_snapshot,
)

__all__ = [
    "CreateResult",
    "EvaluatorInputs",
    "ExecutionSnapshot",
    "LoadedSnapshot",
    "SnapshotArtifactStore",
    "SnapshotErrorCode",
    "SnapshotRef",
    "SnapshotStore",
    "SnapshotStoreError",
    "SnapshotVerification",
    "UnavailableReason",
    "build_execution_snapshot",
    "load_snapshot",
    "publish_snapshot",
    "serialize_example",
    "verify_snapshot",
]
