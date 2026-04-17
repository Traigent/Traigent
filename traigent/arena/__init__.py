"""Public Agent Arena SDK exports."""

from __future__ import annotations

from typing import Any

from traigent.arena.client import ArenaClient
from traigent.arena.config import ArenaConfig
from traigent.arena.dtos import (
    ArenaExecutionRef,
    ArenaInvokeResult,
    ArenaLeaderboard,
    ArenaLeaderboardRow,
    ArenaProviderSource,
    ArenaRun,
    ArenaRunProviderSession,
)

_default_client: ArenaClient | None = None


def get_arena_client(config: ArenaConfig | None = None) -> ArenaClient:
    """Return a shared default Arena client or a config-specific client."""
    global _default_client
    if config is not None:
        return ArenaClient(config)
    if _default_client is None:
        _default_client = ArenaClient()
    return _default_client


def list_provider_sources(
    *, client: ArenaClient | None = None
) -> list[ArenaProviderSource]:
    """List configured Agent Arena provider sources."""
    return (client or get_arena_client()).list_provider_sources()


def create_provider_source(
    *, client: ArenaClient | None = None, **kwargs: Any
) -> ArenaProviderSource:
    """Create a new Agent Arena provider source."""
    return (client or get_arena_client()).create_provider_source(**kwargs)


def update_provider_source(
    provider_source_id: str,
    *,
    client: ArenaClient | None = None,
    **kwargs: Any,
) -> ArenaProviderSource:
    """Update an existing Agent Arena provider source."""
    return (client or get_arena_client()).update_provider_source(
        provider_source_id, **kwargs
    )


def delete_provider_source(
    provider_source_id: str, *, client: ArenaClient | None = None
) -> ArenaProviderSource:
    """Delete an Agent Arena provider source."""
    return (client or get_arena_client()).delete_provider_source(provider_source_id)


def accept_provider_consent(
    provider_source_id: str, *, client: ArenaClient | None = None
) -> dict[str, Any]:
    """Accept the current sponsored-provider disclosure for a provider source."""
    return (client or get_arena_client()).accept_provider_consent(provider_source_id)


def list_runs(*, client: ArenaClient | None = None) -> list[ArenaRun]:
    """List Agent Arena runs."""
    return (client or get_arena_client()).list_runs()


def create_run(*, client: ArenaClient | None = None, **kwargs: Any) -> ArenaRun:
    """Create an Agent Arena run."""
    return (client or get_arena_client()).create_run(**kwargs)


def get_run(run_id: str, *, client: ArenaClient | None = None) -> ArenaRun:
    """Fetch one Agent Arena run."""
    return (client or get_arena_client()).get_run(run_id)


def get_leaderboard(
    run_id: str, *, client: ArenaClient | None = None
) -> ArenaLeaderboard:
    """Fetch the leaderboard for one Agent Arena run."""
    return (client or get_arena_client()).get_leaderboard(run_id)


def invoke(*, client: ArenaClient | None = None, **kwargs: Any) -> ArenaInvokeResult:
    """Run explicit brokered Agent Arena inference."""
    return (client or get_arena_client()).invoke(**kwargs)


__all__ = [
    "ArenaClient",
    "ArenaConfig",
    "ArenaExecutionRef",
    "ArenaInvokeResult",
    "ArenaLeaderboard",
    "ArenaLeaderboardRow",
    "ArenaProviderSource",
    "ArenaRun",
    "ArenaRunProviderSession",
    "accept_provider_consent",
    "create_provider_source",
    "create_run",
    "delete_provider_source",
    "get_arena_client",
    "get_leaderboard",
    "get_run",
    "invoke",
    "list_provider_sources",
    "list_runs",
    "update_provider_source",
]
