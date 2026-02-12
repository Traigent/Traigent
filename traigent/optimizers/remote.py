"""Remote-guided optimizer (hybrid mode scaffold).

This optimizer integrates with a remote suggestion service when available.
It is designed to be privacy-aware: only metadata (e.g., indices, metrics)
should be sent upstream when `privacy_enabled` is True.

Note: Remote optimization requires cloud or hybrid mode, which are not yet
supported in the open-source SDK. This optimizer is reserved for future
enterprise features.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from typing import Any, cast

from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import register_optimizer
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class RemoteOptimizer(BaseOptimizer):
    """Optimizer that can request suggestions from a remote service.

    Note: Remote optimization requires cloud or hybrid execution mode, which
    are not yet supported in the open-source SDK. This optimizer currently
    uses a local RandomSearchOptimizer as a placeholder.

    - Async suggestion APIs are provided to align with remote access patterns.
    - Privacy: call sites can pass `remote_context={"privacy_enabled": True}` to
      indicate indices-only behavior for any remote integration.
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
        super().__init__(
            config_space,
            objectives,
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )
        self.remote_enabled = remote_enabled

        # Fallback local optimizer
        self._fallback = RandomSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )

        # Lightweight mockable remote client; real client to be injected later
        self._remote_client = kwargs.get("remote_client")

    def suggest_next_trial(self, history: list[Any]) -> dict[str, Any]:
        """Suggest the next configuration (fallback path)."""
        # For now, just use the local fallback. Remote path will be added later.
        return self._fallback.suggest_next_trial(history)

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

        # Fallback to base implementation
        return await super().generate_candidates_async(
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
