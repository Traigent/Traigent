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

import math
from typing import Any

from traigent.api.parameter_ranges import ParameterRange
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import register_optimizer
from traigent.utils.discrete_domains import (
    float_step_divides_span,
    stepped_int_contains,
)
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


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_HISTORY_STATUSES = frozenset(
    {
        "not_started",
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "pruned",
    }
)


def _history_for_remote(history: list[Any] | None) -> list[dict[str, Any]]:
    """Project trial history to a closed set of fields before it leaves.

    Default-deny: only the trial index and a status from a fixed enum are
    forwarded, whatever the privacy setting. Trial ids, configuration values,
    metric names and values, and any free-form text never leave; an
    unrecognised status becomes ``"unknown"``.
    """
    entries: list[dict[str, Any]] = []
    for index, trial in enumerate(history or []):
        status = trial.get("status") if isinstance(trial, dict) else None
        if not isinstance(trial, dict):
            status = getattr(trial, "status", None)
        status = getattr(status, "value", status)
        if not isinstance(status, str) or status not in _HISTORY_STATUSES:
            status = "unknown"
        entries.append({"index": index, "status": status})
    return entries


def _value_problem(key: str, spec: Any, value: Any) -> str | None:
    """Return why ``value`` is outside ``spec`` for ``key``, or None when valid.

    Unknown specification shapes are rejected (fail closed).
    """
    if isinstance(spec, ParameterRange):
        spec = spec.to_config_value()
    if isinstance(spec, list):
        if not any(
            isinstance(value, bool) == isinstance(o, bool) and value == o for o in spec
        ):
            return f"value for {key!r} is outside the configured options"
        return None
    low: Any = None
    high: Any = None
    step: Any = None
    is_int = False
    if isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        is_int = isinstance(low, int) and isinstance(high, int)
    elif isinstance(spec, dict) and spec.get("type") in ("int", "float"):
        low, high, step = spec.get("low"), spec.get("high"), spec.get("step")
        is_int = spec["type"] == "int"
    if low is None:
        return f"configuration specification for {key!r} is not supported"
    if not (_is_number(low) and _is_number(high)):
        return f"configuration specification for {key!r} is not supported"
    if not _is_number(value) or not math.isfinite(value) or not low <= value <= high:
        return f"value for {key!r} is outside the configured range"
    if is_int:
        if value != int(value):
            return f"value for {key!r} is not an integer"
        if step is not None:
            if not (isinstance(step, int) and not isinstance(step, bool)) or step <= 0:
                return f"configuration specification for {key!r} is not supported"
            if not stepped_int_contains(int(low), int(high), step, int(value)):
                return f"value for {key!r} is not on the configured step"
    elif step is not None:
        if not _is_number(step) or not math.isfinite(step) or step <= 0:
            return f"configuration specification for {key!r} is not supported"
        on_grid = value == high or float_step_divides_span(low, value, step)
        if not on_grid:
            return f"value for {key!r} is not on the configured step"
    return None


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
    - Malformed responses fail closed: a suggestion that is not a dict inside
      the configuration space (or a batch that is not a non-empty list of
      such dicts) raises ``OptimizationError``; it is never replaced by local
      random search.
    """

    # The seed reaches only the local fallback; handled explicitly in __init__.
    _honours_seed = True

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
        if kwargs.get("seed") is not None:
            self._warn_seed_ignored(
                "does not control RemoteOptimizer suggestions; it seeds only "
                "the local random fallback."
            )

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
            suggestion: Any = None
            has_method = hasattr(self._remote_client, "suggest_next")
            if has_method:
                try:
                    suggestion = await self._remote_client.suggest_next(
                        self.config_space,
                        _history_for_remote(history),
                        remote_context,
                    )
                except Exception as e:
                    logger.warning(f"Remote suggestion failed, using fallback: {e}")
                    has_method = False
            else:
                logger.debug("Remote client missing suggest methods; using fallback")
            if has_method:
                problem = self._suggestion_problem(suggestion)
                if problem is not None:
                    raise OptimizationError(f"Remote suggestion rejected: {problem}")
                return dict(suggestion)
        return self._fallback.suggest_next_trial(history)

    def should_stop(self, history: list[Any]) -> bool:
        return self._fallback.should_stop(history)

    async def should_stop_async(
        self, history: list[Any], remote_context: dict[str, Any] | None = None
    ) -> bool:
        return await super().should_stop_async(history, remote_context)

    def _suggestion_problem(self, suggestion: Any) -> str | None:
        """Return why a remote suggestion is unusable, or None when valid.

        A valid suggestion is a dict that covers every config-space key, keeps
        each value inside its declared specification, and carries no unknown
        keys (dunder keys such as ``__subset_indices__`` are service metadata).
        """
        if not isinstance(suggestion, dict):
            return f"expected a dict, got {type(suggestion).__name__}"
        for key, spec in self.config_space.items():
            if key not in suggestion:
                return f"missing config key {key!r}"
            problem = _value_problem(key, spec, suggestion[key])
            if problem is not None:
                return problem
        for key in suggestion:
            if key not in self.config_space and not str(key).startswith("__"):
                return f"unknown config key {key!r}"
        return None

    async def generate_candidates_async(
        self,
        max_candidates: int,
        remote_context: dict[str, Any] | None = None,
        history: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Prefer remote batch suggestions when available.

        ``history`` (prior trials) is forwarded to the remote service as a
        closed projection of index and status only (see ``_history_for_remote``); omitted means an
        empty list. A client that raises is logged at WARNING and replaced by
        the local fallback. A response that is not a non-empty list of valid
        suggestions raises ``OptimizationError`` (fail closed).
        """
        if self.remote_enabled and self._remote_client is not None:
            batch: Any = None
            called = False
            if hasattr(self._remote_client, "suggest_batch"):
                try:
                    batch = await self._remote_client.suggest_batch(
                        self.config_space,
                        max_candidates,
                        _history_for_remote(history),
                        remote_context,
                    )
                    called = True
                except Exception as e:
                    logger.warning(f"Remote batch suggestion failed, falling back: {e}")
            if called:
                if not isinstance(batch, list) or not batch:
                    raise OptimizationError(
                        "Remote batch rejected: expected a non-empty list, "
                        f"got {type(batch).__name__}"
                    )
                if len(batch) > max_candidates:
                    raise OptimizationError(
                        "Remote batch rejected: returned more candidates than requested"
                    )
                for candidate in batch:
                    problem = self._suggestion_problem(candidate)
                    if problem is not None:
                        raise OptimizationError(f"Remote batch rejected: {problem}")
                return [dict(c) for c in batch]

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
