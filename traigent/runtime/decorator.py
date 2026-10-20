"""``@traigent.requirement(...)``: declare a latency requirement and its policy.

Network-free: nothing is contacted at import or decoration time.  It works
above or below ``@traigent.optimize`` and stores a one-element
:class:`RequirementList` on the function as ``fn.requirements``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from traigent.runtime import apply as apply_module
from traigent.runtime.model import (
    DEFAULT_ENVIRONMENT,
    Issue,
    Receipt,
    RuntimeEvent,
    RuntimeRequirement,
    RuntimeRequirementError,
)
from traigent.runtime.sugar import from_kwargs
from traigent.runtime.tvl_fragment import (
    E_COMPILER,
    compile_full_policy,
    import_compiler,
    load_fragment,
    module_name_for,
    token_entries,
)

F = TypeVar("F")
E_DUPLICATE = "duplicate_requirement"


class RequirementList(list[RuntimeRequirement]):
    """The requirements declared on one function (exactly one in PY-1)."""

    owner: str = "agent"

    def to_tvl(self) -> str:
        return "\n---\n".join(req.to_tvl() for req in self)

    def validate(self) -> list[Issue]:
        """Offline validation; adds compiler lints when ``tvl-spec`` is installed."""
        issues: list[Issue] = []
        for req in self:
            issues.extend(req.validate())
            if any(i.severity == "error" for i in issues):
                continue
            try:
                import_compiler()
            except RuntimeRequirementError:
                if req.source == "tvl":
                    issues.append(Issue(E_COMPILER, "tvl", "warning"))
                continue
            from traigent.runtime import manifest

            module = module_name_for(self.owner)
            token_map = manifest.ensure_tokens(
                manifest.default_state_dir(), token_entries(req, module), persist=False
            )
            try:
                lints = compile_full_policy(req, token_map, module)
            except RuntimeRequirementError as exc:
                issues.append(Issue(exc.code, exc.path or "policy", "error"))
                continue
            for lint in lints:
                code = lint.get("code")
                if isinstance(code, str):
                    issues.append(
                        Issue(
                            code,
                            "policy",
                            "warning" if lint.get("severity") != "error" else "error",
                        )
                    )
        return issues

    def apply(
        self, project_id: str, agent_id: str | None = None, dry_run: bool = False
    ) -> Receipt | None:
        """Register the requirement (idempotent); ``None`` for a dry run."""
        if len(self) != 1:
            raise RuntimeRequirementError(E_DUPLICATE)
        result = apply_module.apply_requirement(
            self[0],
            project_id=project_id,
            agent_id=agent_id,
            dry_run=dry_run,
            module=module_name_for(self.owner),
        )
        return result.receipt


def _resolve_fragment(path: str | Path, caller_file: str | None) -> Path:
    target = Path(path)
    if not target.is_absolute() and not target.exists() and caller_file:
        candidate = Path(caller_file).resolve().parent / target
        if candidate.exists():
            return candidate
    return target


def requirement(
    *,
    latency_ms: int | None = None,
    max_violation_rate: float | None = None,
    window: str = "6h",
    environment: str = DEFAULT_ENVIRONMENT,
    min_samples: int | None = None,
    confidence: float | None = None,
    on_breach: str = "alert",
    approval: str = "human",
    max_fix_cost_usd: float | None = None,
    name: str = "latency",
    agent_id: str | None = None,
    on_event: Callable[[RuntimeEvent], None] | None = None,
    tvl: str | Path | None = None,
) -> Callable[[F], F]:
    """Declare a runtime latency requirement and its log / alert / fix policy.

    Either the simplified kwargs or ``tvl=`` (a fragment of TVL runtime blocks)
    may be given, not both.  ``max_fix_cost_usd``, ``agent_id`` and
    ``on_event`` are execution metadata and may accompany ``tvl=``.
    """
    caller_file: str | None = None
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        caller_file = frame.f_back.f_code.co_filename
    del frame

    if tvl is not None:
        sugar_given = (
            latency_ms is not None
            or max_violation_rate is not None
            or window != "6h"
            or environment != DEFAULT_ENVIRONMENT
            or min_samples is not None
            or confidence is not None
            or on_breach != "alert"
            or approval != "human"
            or name != "latency"
        )
        if sugar_given:
            raise TypeError(
                "requirement() takes either simplified kwargs or tvl=, not both"
            )
        declared = load_fragment(
            _resolve_fragment(tvl, caller_file),
            max_fix_cost_usd=max_fix_cost_usd,
            agent_id=agent_id,
            on_event=on_event,
        )
    else:
        declared = from_kwargs(
            latency_ms=latency_ms,
            max_violation_rate=max_violation_rate,
            window=window,
            environment=environment,
            min_samples=min_samples,
            confidence=confidence,
            on_breach=on_breach,
            approval=approval,
            max_fix_cost_usd=max_fix_cost_usd,
            name=name,
            agent_id=agent_id,
            on_event=on_event,
        )

    def decorate(target: Any) -> Any:
        if getattr(target, "requirements", None):
            raise RuntimeRequirementError(E_DUPLICATE)
        collection = RequirementList([declared])
        inner = getattr(target, "func", target)
        owner = getattr(inner, "__qualname__", None) or getattr(inner, "__name__", None)
        collection.owner = owner if isinstance(owner, str) else "agent"
        target.requirements = collection
        return target

    return decorate  # type: ignore[return-value]
