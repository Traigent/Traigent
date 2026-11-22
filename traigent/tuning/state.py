"""The compiled policy and its validation state, attached as ``fn.tuning``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from traigent.tuning.context import (
    DEFAULT_SNAPSHOT_ID,
    derive_module_context,
    merge_context,
)
from traigent.tuning.errors import (
    E_TVL_VALIDATION_FAILED,
    E_TVL_VALIDATOR_UNAVAILABLE,
    TuningError,
    TvlValidationFailed,
    ValidatorUnavailable,
)
from traigent.tuning.policy import Every, Tuning
from traigent.tuning.validator import Diagnostic, lint_document, validator_available


@dataclass
class TuningState:
    """Result of compiling and validating a policy at decoration time.

    ``validated`` is True only when the TVL linter ran and reported no
    error-severity diagnostic.  When the toolchain is unavailable every executing
    path refuses with ``E_TVL_VALIDATOR_UNAVAILABLE``; when the linter reported
    errors it refuses with ``E_TVL_VALIDATION_FAILED``, carrying the TVL codes
    unchanged (:meth:`require_validated`).
    """

    agent: str | None
    policy: Tuning
    tvl: dict[str, Any]
    validated: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # The inner ``@traigent.optimize`` function (``fn.tuning.optimized``), when the
    # decorated callable wraps one; observe() returns a plain function, so this is
    # how the CLI and callers reach ``.optimize()`` through the documented order.
    optimized: Any = None
    # The function under decoration; its module source gives the default code version.
    source: Any = None
    # The deployment environment the decorated function runs in (``observe(environment=)``);
    # the store and the CLI follow it, never the production default.
    environment: str | None = None

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def require_validated(self) -> None:
        """Refuse unless TVL validated this policy with no error diagnostic."""
        if self.errors:
            raise TvlValidationFailed(self.errors)
        if not self.validated:
            raise ValidatorUnavailable("policy was not validated by the TVL toolchain")


_MAX_UNWRAP_DEPTH = 16


def find_optimized(fn: Any) -> Any:
    """The object with a callable ``.optimize`` at or under ``fn`` (walks
    ``__wrapped__``, bounded depth), or None."""
    seen: set[int] = set()
    current = fn
    for _ in range(_MAX_UNWRAP_DEPTH):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if callable(getattr(current, "optimize", None)):
            return current
        current = getattr(current, "__wrapped__", None)
    return None


def initial_config_of(optimized: Any) -> dict[str, Any]:
    """The optimized function's default configuration: its ``default_config``,
    else the first value of each configuration-space dimension; {} if unknown."""
    if optimized is None:
        return {}
    default = getattr(optimized, "default_config", None)
    if isinstance(default, dict) and default:
        return dict(default)
    space = getattr(optimized, "configuration_space", None)
    if not isinstance(space, dict):
        return {}
    config: dict[str, Any] = {}
    for key, values in space.items():
        if isinstance(values, (list, tuple)):
            if values:
                config[key] = values[0]
        elif not isinstance(values, (set, dict)):
            config[key] = values
    return config


def _snapshot_id_for(tune: Tuning) -> str:
    """The environment snapshot a derived context names.

    TVL's schedule lint (``schedule_anchor_out_of_range``) needs every ``Every``
    anchor to be at most one year after ``environment.snapshot_id``, so the 1970
    placeholder makes any modern schedule fail.  With no explicit
    ``module_context`` snapshot, the snapshot is the EARLIEST schedule anchor: the
    moment the policy starts (a legal value: the anchor is then zero days after it).
    Without a schedule the deterministic placeholder is kept.  An explicit
    ``module_context["environment"]["snapshot_id"]`` always wins (the merge).
    """
    anchors = []
    for trigger in getattr(tune, "on", ()):
        if isinstance(trigger, Every):
            try:
                parsed = datetime.fromisoformat(trigger.anchor.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                anchors.append((parsed, trigger.anchor))
            except ValueError:
                continue  # an unparseable anchor is the linter's finding
    return min(anchors)[1] if anchors else DEFAULT_SNAPSHOT_ID


def attach_tuning(
    target: Any,
    agent: str | None,
    tune: Tuning,
    *,
    source: Any = None,
    environment: str | None = None,
) -> TuningState:
    """Compile and validate ``tune`` (local, no network); return the state.

    The module context comes from ``source`` (the function under decoration),
    overridden by any ``module_context`` given to ``Tuning``.
    """
    derived = derive_module_context(
        source if source is not None else target,
        snapshot_id=_snapshot_id_for(tune),
    )
    context = merge_context(derived, tune.module_context)
    doc = tune.to_tvl(context)
    if validator_available():
        diagnostics = lint_document(doc)
        validated = not any(d.is_error for d in diagnostics)
    else:
        diagnostics, validated = [], False
    state = TuningState(
        agent=agent,
        policy=tune,
        tvl=doc,
        validated=validated,
        diagnostics=diagnostics,
        optimized=find_optimized(source if source is not None else target),
        source=source if source is not None else target,
        environment=environment,
    )
    return state


__all__ = [
    "TuningState",
    "attach_tuning",
    "find_optimized",
    "initial_config_of",
    "E_TVL_VALIDATION_FAILED",
    "E_TVL_VALIDATOR_UNAVAILABLE",
    "TuningError",
]
