"""Opaque response contracts issued by the cold-start backend."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DiscoveryGap:
    """An opaque backend-reported gap."""

    handle: str
    status: str


@dataclass(frozen=True, slots=True)
class Receipt:
    """An opaque backend-issued receipt."""

    handle: str
    status: str


@dataclass(frozen=True, slots=True)
class ColdStartResult:
    """An opaque backend-issued result."""

    handle: str
    status: str
    gaps: tuple[DiscoveryGap, ...] = field(default_factory=tuple)
    receipts: tuple[Receipt, ...] = field(default_factory=tuple)
