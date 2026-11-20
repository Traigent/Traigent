"""The compiled policy and its validation state, attached as ``fn.tuning``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traigent.tuning.context import derive_module_context, merge_context
from traigent.tuning.errors import E_TVL_VALIDATOR_UNAVAILABLE, TuningError
from traigent.tuning.policy import Tuning
from traigent.tuning.validator import Diagnostic, lint_document, validator_available


@dataclass
class TuningState:
    """Result of compiling and validating a policy at decoration time.

    ``validated`` is False when the TVL toolchain is unavailable; every executing
    path then refuses via :meth:`require_validated`.
    """

    agent: str | None
    policy: Tuning
    tvl: dict[str, Any]
    validated: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def require_validated(self) -> None:
        """Raise ``E_TVL_VALIDATOR_UNAVAILABLE`` unless TVL validated this policy."""
        if not self.validated:
            from traigent.tuning.errors import ValidatorUnavailable

            raise ValidatorUnavailable("policy was not validated by the TVL toolchain")


def attach_tuning(
    target: Any, agent: str | None, tune: Tuning, *, source: Any = None
) -> TuningState:
    """Compile and validate ``tune`` (local, no network); return the state.

    The module context comes from ``source`` (the function under decoration),
    overridden by any ``module_context`` given to ``Tuning``.
    """
    derived = derive_module_context(source if source is not None else target)
    context = merge_context(derived, tune.module_context)
    doc = tune.to_tvl(context)
    if validator_available():
        diagnostics = lint_document(doc)
        validated = True
    else:
        diagnostics, validated = [], False
    state = TuningState(
        agent=agent, policy=tune, tvl=doc, validated=validated, diagnostics=diagnostics
    )
    return state


__all__ = ["TuningState", "attach_tuning", "E_TVL_VALIDATOR_UNAVAILABLE", "TuningError"]
