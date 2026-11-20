"""The TVL reference toolchain bridge.

TVL is the source of truth: validation is ``tvl.lints.lint_module`` and nothing
else.  There is no schema-only fallback; when the toolchain is absent or too
old for TVL 1.7 every executing path refuses with ``E_TVL_VALIDATOR_UNAVAILABLE``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from traigent.tuning.errors import ValidatorUnavailable


@dataclass(frozen=True)
class Diagnostic:
    """A TVL diagnostic, carried verbatim from ``lint_module``."""

    code: str
    severity: str
    path: tuple[Any, ...] = field(default_factory=tuple)
    message: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": list(self.path),
            "message": self.message,
        }


# A 1.7 module whose only fault is an unresolved ``protects`` name.  A toolchain
# that understands TVL 1.7 answers ``protects_unknown_requirement``.
_PROBE_MODULE: dict[str, Any] = {
    "tvl_version": "1.7",
    "tvl": {"module": "probe.tuning"},
    "environment": {"snapshot_id": "2026-11-20T00:00:00Z"},
    "evaluation_set": {"dataset": "test://probe"},
    "tvars": [{"name": "model", "type": "enum[str]", "domain": ["probe/a", "probe/b"]}],
    "objectives": [{"name": "quality", "direction": "maximize"}],
    "requirements": [],
    "monitors": [],
    "responses": [
        {
            "name": "tuning",
            "rules": [
                {
                    "id": "probe",
                    "trigger": {
                        "schedule": {"every_days": 7, "anchor": "2026-12-01T00:00:00Z"}
                    },
                    "do": "optimize",
                    "approval": "human",
                    "deploy": "if_promoted",
                    "protects": ["nothing"],
                }
            ],
        }
    ],
}
_PROBE_CODE = "protects_unknown_requirement"

_support_cache: dict[str, str | None] = {}


def _unavailable_reason() -> str | None:
    """None when the toolchain supports TVL 1.7, else a short reason."""
    if "reason" in _support_cache:
        return _support_cache["reason"]
    reason: str | None
    try:
        lints = importlib.import_module("tvl.lints")
        importlib.import_module("tvl.runtime_conditions")
        issues = lints.lint_module(_PROBE_MODULE)
        codes = {i.get("code") for i in issues if isinstance(i, dict)}
        reason = (
            None if _PROBE_CODE in codes else "tvl package does not support TVL 1.7"
        )
    except ImportError:
        reason = "tvl package is not importable"
    except Exception:  # noqa: BLE001 - any failure means "cannot validate"
        reason = "tvl package failed its TVL 1.7 probe"
    _support_cache["reason"] = reason
    return reason


def reset_validator_cache() -> None:
    """Forget the cached toolchain probe (tests)."""
    _support_cache.clear()


def validator_available() -> bool:
    return _unavailable_reason() is None


def require_validator() -> None:
    """Raise ``ValidatorUnavailable`` unless the toolchain supports TVL 1.7."""
    reason = _unavailable_reason()
    if reason is not None:
        raise ValidatorUnavailable(reason)


def lint_document(doc: dict[str, Any]) -> list[Diagnostic]:
    """Run ``tvl.lints.lint_module``; return its diagnostics verbatim."""
    require_validator()
    from tvl.lints import lint_module

    out: list[Diagnostic] = []
    for issue in lint_module(doc):
        out.append(
            Diagnostic(
                code=str(issue.get("code", "")),
                severity=str(issue.get("severity", "error")),
                path=tuple(issue.get("path", ()) or ()),
                message=str(issue.get("message", "")),
            )
        )
    return out


def conditions_trace(module: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    """Run the TVL reference relation over a recorded history."""
    require_validator()
    from tvl.runtime_conditions import conditions_trace as _trace

    return dict(_trace(module, history))
