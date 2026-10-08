"""Harness contract a connector author implements to run the conformance suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..privacy import CustomerSideMinter

# Synthetic content canary; deliberately not token-shaped and not real data.
CANARY = "CANARY-synthetic-content-0xC0FFEE"


class ConformanceFailure(AssertionError):
    """A conformance case failed.

    ``reason`` is a closed code identifying why; messages never carry row or
    summary content.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class FaultKind(StrEnum):
    FAIL_BEFORE_APPLY = "fail_before_apply"  # nothing committed, error raised
    LOST_ACK = "lost_ack"  # whole batch committed, acknowledgement lost
    PARTIAL = "partial"  # first ``applied`` rows committed, then error


@dataclass(frozen=True, slots=True)
class Fault:
    kind: FaultKind
    applied: int = 0


@dataclass(frozen=True, slots=True)
class Connection:
    """One live connection: its minter and an opaque credential identity."""

    label: str
    minter: CustomerSideMinter
    credential: str


class ConnectorHarness(ABC):
    """Adapter between the suites and one connector over a fake backend.

    The fake backend must (a) really inject the requested faults, (b) really
    commit state that ``committed`` reports, and (c) record the credential of
    every request the connector makes (``request_credentials``). Rows are
    mappings with an ``"id"`` key unless ``row_key`` is overridden; the suites
    compare full row content, not just ids. ``validate_summary`` and
    ``serialize_summary`` MUST be the connector's own privacy gate and
    serializer; there is deliberately no default.
    """

    kind: str
    # Closed connector-kind vocabulary used inside summaries (schema enum).
    summary_kind: str
    transient_errors: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    expired_cursor_errors: tuple[type[BaseException], ...] = (LookupError,)

    @abstractmethod
    def open_connection(self, label: str) -> Connection:
        """Create an isolated connection with its own credentials and minter."""

    @abstractmethod
    def write(
        self, connection: Connection, rows: Sequence[Mapping[str, Any]], idem_key: str
    ) -> None:
        """Write a batch idempotently under ``idem_key``."""

    @abstractmethod
    def committed(self, connection: Connection) -> list[Mapping[str, Any]]:
        """Rows the backend durably holds for this connection."""

    @abstractmethod
    def seed(
        self,
        connection: Connection,
        rows: Sequence[Mapping[str, Any]],
        page_size: int,
        overlap: int = 0,
    ) -> None:
        """Load rows for paged reads; consecutive pages share ``overlap`` rows."""

    @abstractmethod
    def read_page(
        self, connection: Connection, cursor: str | None
    ) -> Mapping[str, Any]:
        """Return ``{"items": [...], "next_cursor": str | None}``."""

    @abstractmethod
    def arm_fault(self, fault: Fault) -> None:
        """Inject ``fault`` into the next write."""

    @abstractmethod
    def expire_cursors(self) -> None:
        """Invalidate every cursor issued so far."""

    @abstractmethod
    def crash(self) -> bool:
        """Simulate a process crash: destroy every in-memory connector object
        (sessions, cursor tables, caches) and rebuild from persisted state only.
        Return True only if in-memory state really was discarded.
        """

    @abstractmethod
    def request_credentials(self) -> list[str]:
        """Credential identity of every backend request made so far, in order."""

    @abstractmethod
    def validate_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The connector's own privacy gate over wire-form summaries."""

    @abstractmethod
    def serialize_summary(
        self, connection: Connection, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The connector's own serializer; token fields hold minted token objects."""

    def row_key(self, row: Mapping[str, Any]) -> str:
        return str(row["id"])
