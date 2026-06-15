"""Remote-guided optimizer (hybrid mode scaffold).

This optimizer integrates with a remote suggestion service when available.
It is designed to be privacy-aware: only metadata (e.g., indices, metrics)
should be sent upstream when `privacy_enabled` is True.

Note: Remote suggestion services are reserved for future enterprise features.
Use `execution_mode="hybrid"` for the supported portal-tracked path where
trials still execute locally.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from typing import Any, cast

from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import register_optimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


_NO_REMOTE_CLIENT_MSG = (
    "RemoteOptimizer requires both a remote_client and remote_enabled=True. "
    "Remote optimization is not yet implemented as a first-party service, so "
    "constructing this optimizer without both arguments used to silently fall "
    "back to local random search — making the result mislabeled as 'remote'. "
    "If you want random search, pass algorithm='random'. "
    "If you want portal-tracked local execution, pass execution_mode='hybrid' "
    "with one of the locally supported algorithms (grid, random). "
    "If you are wiring a remote suggestion backend, pass it explicitly via "
    "remote_client=... and remote_enabled=True."
)


class RemoteOptimizer(BaseOptimizer):
    """Optimizer that requests suggestions from an injected remote service.

    Note: Remote optimization is not implemented as a first-party service yet.
    To use this optimizer you MUST inject a ``remote_client`` and set
    ``remote_enabled=True``; otherwise construction raises ``OptimizationError``.
    There is no longer a silent local-random-search fallback at construction
    time — that path masqueraded a local run as a remote one and was the
    surface flagged by the tracked fix.

    - Async suggestion APIs are provided to align with remote access patterns.
    - Privacy: call sites can pass ``remote_context={"privacy_enabled": True}``
      to indicate indices-only behavior for any remote integration.
    - Runtime fallback: when the injected client raises at suggest-time, the
      optimizer emits a WARNING and falls back to local random search for
      that single suggestion. This is logged, not silent.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        context: Any | None = None,
        objective_weights: dict[str, float] | None = None,
        remote_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        # Fail closed: without a real remote_client, this optimizer cannot do
        # what its name advertises. See module docstring and the tracked fix.
        # Keep .pop (not .get): the client is stored on self by BaseOptimizer
        # via algorithm_config; leaving it in kwargs propagates the remote
        # client into the local fallback path, which is not what we want when
        # this construction-time guard raises. Develop's behavior is the
        # intended one; main's .get during #872 work was a regression we are
        # not propagating forward.
        remote_client = kwargs.pop("remote_client", None)
        if remote_client is None or not remote_enabled:
            raise OptimizationError(_NO_REMOTE_CLIENT_MSG)

        super().__init__(
            config_space,
            objectives,
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )
        self.remote_enabled = remote_enabled

        # Runtime fallback (only used when the remote client raises mid-call;
        # never as a substitute for a missing client — see __init__ guard).
        self._fallback = RandomSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )

        self._remote_client = remote_client

    def suggest_next_trial(self, history: list[Any]) -> dict[str, Any]:
        """Remote suggestions require the async API.

        The old sync implementation delegated to local random search, which
        recreated the fake "remote" completion bug for sync callers.
        """
        raise OptimizationError(
            "RemoteOptimizer does not support synchronous suggestions. "
            "Use suggest_next_trial_async(), generate_candidates_async(), or "
            "choose algorithm='random' for local random search."
        )

    async def suggest_next_trial_async(
        self, history: list[Any], remote_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Async suggestion, enabling remote use when implemented.

        Currently falls back to local suggestions. The `remote_context` can
        include `privacy_enabled` and other metadata; future integration will
        use these in remote calls to ensure privacy-safe behavior.
        """
        if self.remote_enabled and self._remote_client is not None:
            try:
                # Use single-suggestion path
                if hasattr(self._remote_client, "suggest_next"):
                    return cast(
                        dict[str, Any],
                        await self._remote_client.suggest_next(
                            self.config_space, history, remote_context
                        ),
                    )
                logger.debug("Remote client missing suggest methods; using fallback")
            except Exception as e:
                logger.warning(f"Remote suggestion failed, using fallback: {e}")
        return self._fallback.suggest_next_trial(history)

    def should_stop(self, history: list[Any]) -> bool:
        return self._fallback.should_stop(history)

    async def should_stop_async(
        self, history: list[Any], remote_context: dict[str, Any] | None = None
    ) -> bool:
        return await super().should_stop_async(history, remote_context)

    async def generate_candidates_async(
        self, max_candidates: int, remote_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Prefer remote batch suggestions when available."""
        if self.remote_enabled and self._remote_client is not None:
            try:
                if hasattr(self._remote_client, "suggest_batch"):
                    batch = await self._remote_client.suggest_batch(
                        self.config_space, max_candidates, [], remote_context
                    )
                    if isinstance(batch, list) and batch:
                        return batch
            except Exception as e:
                logger.warning(f"Remote batch suggestion failed, falling back: {e}")

        # Runtime fallback remains explicit: use the local optimizer object,
        # not RemoteOptimizer.suggest_next_trial(), which intentionally fails
        # loud for sync callers.
        return await self._fallback.generate_candidates_async(
            max_candidates, remote_context=remote_context
        )


class MockRemoteSuggestionClient:
    """Simple mock remote client for tests/examples.

    - Respects `privacy_enabled` by recording the flag; does not accept content.
    - Generates deterministic suggestions by walking config_space lists.
    """

    def __init__(self) -> None:
        self.calls: dict[str, Any] = {"batch": 0, "single": 0}
        self.last_context: dict[str, Any] | None = None

    async def suggest_batch(
        self,
        config_space: dict[str, Any],
        n: int,
        history: list[Any] | None = None,
        ctx: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls["batch"] += 1
        self.last_context = ctx
        # Generate n configs by cycling through available discrete lists
        keys = list(config_space.keys())
        values = [
            (v if isinstance(v, list) and v else [v])
            for v in (config_space[k] for k in keys)
        ]
        configs: list[dict[str, Any]] = []
        dataset_size = int(ctx.get("dataset_size", 0)) if isinstance(ctx, dict) else 0
        for i in range(n):
            cfg = {
                k: vals[min(i, len(vals) - 1)]
                for k, vals in zip(keys, values, strict=False)
            }
            # Provide indices-only subset split if dataset_size known
            if dataset_size > 0:
                idxs = list(range(i, dataset_size, max(1, n)))
                cfg["__subset_indices__"] = idxs
            configs.append(cfg)
        return configs

    async def suggest_next(
        self,
        config_space: dict[str, Any],
        history: list[Any] | None = None,
        ctx: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls["single"] += 1
        self.last_context = ctx
        return {
            k: (v[0] if isinstance(v, list) and v else v)
            for k, v in config_space.items()
        }


# Register under the name "remote" for easy selection
register_optimizer("remote", RemoteOptimizer)
