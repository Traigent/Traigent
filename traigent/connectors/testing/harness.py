"""Harness contract a connector author implements to run the conformance suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import ConnectionRef
from ..privacy import CustomerSideMinter, validate_summary

# Synthetic content canary; deliberately not token-shaped and not real data.
CANARY = "CANARY-synthetic-content-0xC0FFEE"


class ConformanceFailure(AssertionError):
    """A conformance case failed. Messages are closed text, never row content."""


@dataclass(frozen=True, slots=True)
class Connection:
    """One live connection: its minter and an opaque credential identity."""

    label: str
    minter: CustomerSideMinter
    credential: str


class ConnectorHarness(ABC):
    """Adapter between the suites and one connector over a fake backend.

    The harness owns a fake (in-memory or recorded) backend that can inject
    faults. Rows are mappings with an ``"id"`` key unless ``row_key`` is
    overridden. Methods raise the error types declared below for injected
    faults; anything else is treated as a real defect.
    """

    kind: str = "dummy"
    # Summaries use the schema's closed connector-kind vocabulary.
    summary_kind: str = "git_file"
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
    def stored(self, connection: Connection) -> Sequence[Mapping[str, Any]]:
        """Rows the backend currently holds for this connection."""

    @abstractmethod
    def seed(
        self,
        connection: Connection,
        rows: Sequence[Mapping[str, Any]],
        page_size: int,
    ) -> None:
        """Load rows into the backend for paged reads."""

    @abstractmethod
    def read_page(
        self, connection: Connection, cursor: str | None
    ) -> Mapping[str, Any]:
        """Return ``{"items": [...], "next_cursor": str | None}``."""

    @abstractmethod
    def arm_lost_response(self) -> None:
        """Next write is applied by the backend but its response is lost."""

    @abstractmethod
    def arm_partial_failure(self, applied: int) -> None:
        """Next write applies only ``applied`` rows, then fails."""

    @abstractmethod
    def expire_cursors(self) -> None:
        """Invalidate every cursor issued so far."""

    def restart(self) -> None:
        """Drop in-process state (simulated crash); durable backend state stays."""
        return None

    def row_key(self, row: Mapping[str, Any]) -> str:
        return str(row["id"])

    def make_minter(self) -> CustomerSideMinter:
        return CustomerSideMinter(ConnectionRef(self.kind))

    def summary_for(
        self, connection: Connection, overrides: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """A valid run summary as the connector would emit; ``overrides`` merge in."""
        minter = connection.minter
        payload: dict[str, Any] = {
            "schema_version": "1",
            "run_token": minter.mint("run", "synthetic").value,
            "connector_kind": self.summary_kind,
            "connection_token": minter.connection_token.value,
            "command": "check",
            "status": "queued",
            "counts": {
                "observations_read": 0,
                "scores_read": 0,
                "rows_dropped_invalid": 0,
                "pages": 0,
                "items_written": 0,
            },
            "guarantees": [],
        }
        payload.update(overrides or {})
        return payload

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The privacy gate under test; route to the connector's own path if any."""
        return dict(validate_summary(payload))
