"""S0 / S1 capture for ``ExecutionOptions.snapshot_store`` (opt-in G1 client evidence).

S0 runs before any trial: it captures the dataset, evaluator and configuration
space, applies the presence rules, publishes the snapshot to the customer's
store and (for online G1) requires the Backend readiness probe to say
``available``. S1 runs on the terminal ``result.best_config`` of every
completion path, refuses on drift against S0, publishes the snapshot, writes the
local client-evidence file and attaches local-only handles to the result.

Everything here is local except the readiness probe, which sends no body.
Failures raise :class:`~traigent.utils.exceptions.SnapshotBindingError`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.certification.bind import BindError, check_g1_readiness
from traigent.certification.client_evidence import (
    ClientEvidenceError,
    LocalFileCustody,
    build_client_evidence,
)
from traigent.certification.transport import G1Transport
from traigent.snapshot import (
    EvaluatorInputs,
    ExecutionSnapshot,
    build_execution_snapshot,
    publish_snapshot,
)
from traigent.utils.exceptions import SnapshotBindingError

logger = logging.getLogger(__name__)

__all__ = ["BIND_GUIDANCE", "SnapshotRun", "default_transport"]

#: Shown after a run that declared G1 intent. The deferral deadline is the
#: Backend's (default 7 days from run registration).
BIND_GUIDANCE = (
    "G1 evidence captured locally. Run `certify bind` before the "
    "deferral deadline (default 7 days) or the run's certificate stays plain."
)

_DATASET = "dataset/examples.jsonl"
_CONFIG = "config/config.json"
_CONFIG_SPACE = "config/config_space.json"
_EVALUATOR_PREFIX = "evaluator/"


def default_transport(
    api_key: str | None = None, *, no_egress: bool = False
) -> G1Transport:
    """The authenticated, egress-gated Backend transport (replaced in tests).

    ``api_key`` is the caller's explicit credential (e.g. ``traigent sync
    --api-key``); without it the client resolves the ambient one.
    """
    from traigent.certification.transport import BackendG1Transport
    from traigent.cloud.backend_client import BackendIntegratedClient

    return BackendG1Transport(
        BackendIntegratedClient(api_key=api_key, no_egress=no_egress)
    )


def _part_state(snapshot: ExecutionSnapshot, path: str) -> tuple[str, str | None]:
    entry = snapshot.manifest["parts"].get(path)
    if not isinstance(entry, dict):
        return ("absent", None)
    if entry.get("status") == "unavailable":
        return ("unavailable", str(entry.get("reason")))
    return ("present", str(entry.get("sha256")))


def _evaluator_states(snapshot: ExecutionSnapshot) -> dict[str, tuple[str, str | None]]:
    return {
        path: _part_state(snapshot, path)
        for path in sorted(snapshot.manifest["parts"])
        if path.startswith(_EVALUATOR_PREFIX)
    }


def _present(state: tuple[str, str | None]) -> bool:
    return state[0] == "present"


@dataclass
class SnapshotRun:
    """Per-run S0/S1 state. Holds local references only; nothing here is serialized."""

    store: Any
    _s0_dataset: tuple[str, str | None] | None = field(default=None, repr=False)
    _s0_config_space: tuple[str, str | None] | None = field(default=None, repr=False)
    _s0_evaluators: dict[str, tuple[str, str | None]] | None = field(
        default=None, repr=False
    )
    _inputs: dict[str, Any] = field(default_factory=dict, repr=False)
    #: True once S0 passed for an online run: run registration must declare G1.
    g1_required: bool = field(default=False, repr=False)
    #: True once S0 passed for an offline run: the local session is marked so a
    #: later sync registers with G1 intent after the readiness check.
    g1_pending: bool = field(default=False, repr=False)

    def capture_s0(
        self,
        *,
        dataset: Any,
        configuration_space: Any,
        agent: Callable[..., Any],
        evaluator: EvaluatorInputs,
        offline: bool,
        transport_factory: Callable[[], G1Transport] | None = None,
    ) -> None:
        """Capture before dispatch; raise ``SnapshotBindingError`` before any spend."""
        self._inputs = {
            "dataset": dataset,
            "config_space": configuration_space,
            "agent": agent,
            "evaluator": evaluator,
        }
        try:
            snapshot = build_execution_snapshot(config=None, **self._inputs)
        except Exception:
            raise SnapshotBindingError("SNAPSHOT_CAPTURE_FAILED") from None
        dataset_state = _part_state(snapshot, _DATASET)
        space_state = _part_state(snapshot, _CONFIG_SPACE)
        evaluators = _evaluator_states(snapshot)
        if (
            not _present(dataset_state)
            or not _present(space_state)
            or not evaluators
            or not all(_present(state) for state in evaluators.values())
        ):
            raise SnapshotBindingError("SNAPSHOT_INCOMPLETE")
        try:
            publish_snapshot(self.store, snapshot)
        except Exception:
            raise SnapshotBindingError("SNAPSHOT_CAPTURE_FAILED") from None
        if not offline:
            try:
                check_g1_readiness((transport_factory or default_transport)())
            except BindError:
                raise SnapshotBindingError("READINESS_UNAVAILABLE") from None
            except Exception:
                raise SnapshotBindingError("READINESS_UNAVAILABLE") from None
        self._s0_dataset = dataset_state
        self._s0_config_space = space_state
        self._s0_evaluators = evaluators
        # Online and ready: declare G1 intent at run registration.
        self.g1_required = not offline
        # Offline capture cannot declare at registration: keep the intent locally.
        self.g1_pending = offline

    def capture_s1(self, result: Any, *, custody: Any = None) -> None:
        """Capture from the terminal ``result.best_config``; attach local-only handles."""
        if self._s0_dataset is None:
            raise SnapshotBindingError("SNAPSHOT_CAPTURE_FAILED", result=result)
        best_config = getattr(result, "best_config", None)
        if not isinstance(best_config, dict) or not best_config:
            raise SnapshotBindingError("NO_ELIGIBLE_WINNER", result=result)
        try:
            snapshot = build_execution_snapshot(
                config=dict(best_config), **self._inputs
            )
        except Exception:
            raise SnapshotBindingError(
                "SNAPSHOT_CAPTURE_FAILED", result=result
            ) from None
        if (
            _part_state(snapshot, _DATASET) != self._s0_dataset
            or _part_state(snapshot, _CONFIG_SPACE) != self._s0_config_space
            or _evaluator_states(snapshot) != self._s0_evaluators
        ):
            raise SnapshotBindingError("SNAPSHOT_INPUT_DRIFT", result=result)
        if not _present(_part_state(snapshot, _CONFIG)):
            raise SnapshotBindingError("SNAPSHOT_INCOMPLETE", result=result)
        try:
            ref = publish_snapshot(self.store, snapshot)
        except Exception:
            raise SnapshotBindingError(
                "SNAPSHOT_CAPTURE_FAILED", result=result
            ) from None
        custody = custody if custody is not None else LocalFileCustody()
        try:
            evidence = build_client_evidence(snapshot, custody=custody)
        except ClientEvidenceError:
            raise SnapshotBindingError(
                "EVIDENCE_PERSIST_FAILED", result=result
            ) from None
        except Exception:
            raise SnapshotBindingError(
                "SNAPSHOT_CAPTURE_FAILED", result=result
            ) from None
        path = (
            custody.path_for(evidence.root_hex)
            if hasattr(custody, "path_for")
            else None
        )
        result.snapshot_ref = ref
        result.client_evidence_path = path
        if self.g1_required:
            logger.warning(BIND_GUIDANCE)
