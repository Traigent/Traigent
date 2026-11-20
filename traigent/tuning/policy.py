"""Continuous Tuning sugar: builders that emit a TVL 1.7 module.

The builders only *emit* TVL (requirements, monitors, responses plus
``tvl_version``) merged with the module context.  They validate nothing
themselves; ``Tuning.validate`` runs the TVL reference linter and returns its
diagnostics unchanged.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from traigent.tuning.errors import (
    E_FLOAT_RATE,
    E_INVALID_TUNING,
    E_NOT_ROUNDTRIPPABLE,
    E_RATE_PRECISION,
    TuningError,
)
from traigent.tuning.validator import Diagnostic, lint_document

TVL_VERSION = "1.7"
RESPONSE_NAME = "tuning"
WINDOW_HOURS = {"6h": 6, "12h": 12, "24h": 24}
CLOSURE_SCOPES = ("any_generation", "since_head_move")
DEPLOY_MODES = ("if_promoted", "never")
SAMPLING_MODE = "hash_bernoulli_committed_v1"
_BODY_KEYS = ("requirements", "monitors", "responses")

# Profile defaults, identical to the runtime monitors of the slice case.
_RUNTIME_DECISION = {
    "method": "clopper_pearson_rational_v2",
    "confidence": 0.95,
    "false_alarm_budget": 0.05,
    "min_samples": 200,
    "max_unresolved_fraction": 0.05,
    "horizon_days": 30,
    "model": "iid_bernoulli_within_window_v1",
}
_GATE_DECISION = {
    "method": "clopper_pearson_rational_v2",
    "confidence": 0.95,
    "false_alarm_budget": 0.05,
}
_EPISODE_DEFAULTS = {"closure_windows": 2, "closure_max_age_hours": 168}


# -- decimals -----------------------------------------------------------------
def to_decimal(value: Any, field: str) -> Decimal:
    """A rate input: ``str`` or ``Decimal``; floats and bools are rejected."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TuningError(
            E_FLOAT_RATE,
            f"{field}: pass a decimal string or decimal.Decimal, not {type(value).__name__}",
        )
    if isinstance(value, (str, int, Decimal)):
        try:
            dec = Decimal(value) if not isinstance(value, Decimal) else value
        except InvalidOperation:
            raise TuningError(E_INVALID_TUNING, f"{field}: not a decimal") from None
        if not dec.is_finite():
            raise TuningError(E_INVALID_TUNING, f"{field}: not a finite decimal")
        return dec
    raise TuningError(E_INVALID_TUNING, f"{field}: not a decimal")


def decimal_to_json(dec: Decimal) -> int | float:
    """The JSON number TVL carries for an exact decimal (int when integral)."""
    if dec == dec.to_integral_value():
        return int(dec)
    as_float = float(dec)
    if Decimal(repr(as_float)) != dec:
        raise TuningError(E_RATE_PRECISION, "decimal is not exactly representable")
    return as_float


def decimal_to_str(dec: Decimal) -> str:
    """Minimal decimal string (no exponent, no trailing zeros)."""
    text = format(dec.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _rate(value: Any, field: str) -> Decimal:
    dec = to_decimal(value, field)
    if not Decimal(0) <= dec <= Decimal(1):
        raise TuningError(E_INVALID_TUNING, f"{field}: must be within [0, 1]")
    return dec


def _str_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or not all(isinstance(v, str) for v in value):
        raise TuningError(E_INVALID_TUNING, f"{field}: expected a list of strings")
    return tuple(value)


def _opt_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TuningError(E_INVALID_TUNING, f"{field}: expected a non-empty string")
    return value


# -- builders -----------------------------------------------------------------
@dataclass(frozen=True)
class FailureRate:
    """Keep the share of failing invocations (per ``checker``) at most ``at_most``."""

    checker: str
    at_most: Any
    window: str = "24h"
    name: str | None = None
    monitor: str | None = None
    gate: str | None = None

    def __post_init__(self) -> None:
        _opt_str(self.checker, "FailureRate.checker")
        if self.checker is None:
            raise TuningError(E_INVALID_TUNING, "FailureRate.checker is required")
        object.__setattr__(self, "at_most", _rate(self.at_most, "FailureRate.at_most"))
        if self.window not in WINDOW_HOURS:
            raise TuningError(
                E_INVALID_TUNING,
                f"FailureRate.window must be one of {sorted(WINDOW_HOURS)}",
            )
        for attr in ("name", "monitor", "gate"):
            _opt_str(getattr(self, attr), f"FailureRate.{attr}")


@dataclass(frozen=True)
class RandomAudit:
    """Sample production calls for the customer-side check (committed hash sampling)."""

    rate: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _rate(self.rate, "RandomAudit.rate"))


@dataclass(frozen=True)
class ModelReleased:
    """React to an authenticated model release (external trigger) with an optimization."""

    family: Any = ()
    tier: Any = ()
    admit: str | None = None
    protects: Any = None
    id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", _str_list(self.family, "ModelReleased.family")
        )
        object.__setattr__(self, "tier", _str_list(self.tier, "ModelReleased.tier"))
        _opt_str(self.admit, "ModelReleased.admit")
        _opt_str(self.id, "ModelReleased.id")
        if self.protects is not None:
            object.__setattr__(
                self, "protects", _str_list(self.protects, "ModelReleased.protects")
            )


@dataclass(frozen=True)
class OnBreach:
    """Optimize when a runtime monitor opens a breach episode (one rule per kept rate)."""

    id: str | None = None

    def __post_init__(self) -> None:
        _opt_str(self.id, "OnBreach.id")


@dataclass(frozen=True)
class Every:
    """Optimize on a schedule: every ``days`` days from ``anchor`` (RFC 3339 UTC)."""

    days: int
    anchor: str
    protects: Any = None
    id: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.days, bool)
            or not isinstance(self.days, int)
            or self.days < 1
        ):
            raise TuningError(E_INVALID_TUNING, "Every.days must be a positive integer")
        if not isinstance(self.anchor, str) or not self.anchor:
            raise TuningError(
                E_INVALID_TUNING, "Every.anchor must be an RFC 3339 string"
            )
        if self.protects is not None:
            object.__setattr__(
                self, "protects", _str_list(self.protects, "Every.protects")
            )
        _opt_str(self.id, "Every.id")


@dataclass(frozen=True)
class ProposeFix:
    """How an optimization outcome reaches production."""

    approval: str = "human"
    deploy: str = "if_promoted"

    def __post_init__(self) -> None:
        if self.approval == "none":
            raise TuningError(
                E_INVALID_TUNING,
                "ProposeFix.approval='none' is not allowed; TVL rejects it",
            )
        if not isinstance(self.approval, str) or not self.approval:
            raise TuningError(E_INVALID_TUNING, "ProposeFix.approval must be a string")
        if self.deploy not in DEPLOY_MODES:
            raise TuningError(
                E_INVALID_TUNING,
                f"ProposeFix.deploy must be one of {list(DEPLOY_MODES)}",
            )


_Trigger = ModelReleased | OnBreach | Every


# -- the policy ---------------------------------------------------------------
class Tuning:
    """A continuous-tuning policy: builder over TVL 1.7 runtime blocks.

    ``Tuning(keep=[...], audit=..., on=[...], fix=..., recover=...)`` or the
    alternative ``Tuning(tvl="policy.tvl.yml")`` (a TVL fragment; exclusive).

    ``checkers`` maps a checker name (``"customer:ticket_route_check"``) to a local
    ``(example, output) -> label`` function; ``checker_labels`` maps the same name to
    the full set of labels it may answer with (an answer outside it is unresolved);
    ``holdout`` is a JSONL file of ``{"input": {...}, "output": expected}`` examples,
    every row with its expected output, kept apart from the optimization's own
    evaluation set.  Together they give the promotion monitors real evidence; all stay
    in the process (only counts, verdicts and a digest of the holdout file are
    recorded).  They are not part of the TVL.
    """

    def __init__(
        self,
        keep: Sequence[FailureRate] = (),
        audit: RandomAudit | None = None,
        on: Sequence[_Trigger] = (),
        fix: ProposeFix | None = None,
        recover: str = "since_head_move",
        *,
        tvl: str | Path | None = None,
        module_context: Mapping[str, Any] | None = None,
        checkers: Mapping[str, Callable[[Any, Any], str]] | None = None,
        holdout: str | Path | None = None,
        checker_labels: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        caller_file = None
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_file = frame.f_back.f_code.co_filename
        del frame
        self.keep = tuple(keep)
        self.audit = audit
        self.on = tuple(on)
        self.fix = fix
        self.recover = recover
        # Customer-side promotion evidence: never compiled into TVL, never sent anywhere.
        self.checkers: dict[str, Callable[[Any, Any], str]] = dict(checkers or {})
        self.checker_labels: dict[str, tuple[str, ...]] = {
            str(name): tuple(str(x) for x in labels)
            for name, labels in (checker_labels or {}).items()
        }
        self.holdout: Path | None = (
            _resolve_fragment(holdout, caller_file) if holdout is not None else None
        )
        self.tvl_path: Path | None = None
        self._fragment: dict[str, Any] | None = None
        self.module_context: dict[str, Any] | None = (
            copy.deepcopy(dict(module_context)) if module_context is not None else None
        )
        if tvl is not None:
            if (
                self.keep
                or audit is not None
                or self.on
                or fix is not None
                or recover != "since_head_move"
            ):
                raise TypeError(
                    "Tuning() takes either the builder kwargs or tvl=, not both"
                )
            self._fragment = _load_fragment(_resolve_fragment(tvl, caller_file))
            self.tvl_path = Path(tvl)
            return
        for item in self.keep:
            if not isinstance(item, FailureRate):
                raise TuningError(E_INVALID_TUNING, "keep entries must be FailureRate")
        if audit is not None and not isinstance(audit, RandomAudit):
            raise TuningError(E_INVALID_TUNING, "audit must be RandomAudit")
        for trigger in self.on:
            if not isinstance(trigger, (ModelReleased, OnBreach, Every)):
                raise TuningError(
                    E_INVALID_TUNING,
                    "on entries must be ModelReleased, OnBreach or Every",
                )
        if fix is not None and not isinstance(fix, ProposeFix):
            raise TuningError(E_INVALID_TUNING, "fix must be ProposeFix")
        if recover not in CLOSURE_SCOPES:
            raise TuningError(
                E_INVALID_TUNING, f"recover must be one of {list(CLOSURE_SCOPES)}"
            )

    # -- compile ----------------------------------------------------------------
    def _keep_names(self) -> list[str]:
        out: list[str] = []
        for index, item in enumerate(self.keep):
            out.append(
                item.name
                or (
                    "failure_rate"
                    if len(self.keep) == 1
                    else f"failure_rate_{index + 1}"
                )
            )
        return out

    def _runtime_blocks(self) -> dict[str, list[dict[str, Any]]]:
        if self._fragment is not None:
            return copy.deepcopy({k: self._fragment.get(k, []) for k in _BODY_KEYS})
        names = self._keep_names()
        fix = self.fix or ProposeFix()
        requirements: list[dict[str, Any]] = []
        monitors: list[dict[str, Any]] = []
        runtime_monitor_of: dict[str, str] = {}
        for item, name in zip(self.keep, names, strict=True):
            prod = item.monitor or f"{name}_prod"
            gate = item.gate or f"{name}_gate"
            runtime_monitor_of[name] = prod
            requirements.append(
                {
                    "name": name,
                    "kind": "rate_bound",
                    "unit": "invocation",
                    "violation": {"checker": item.checker, "label_in": ["fail"]},
                    "max_rate": decimal_to_json(item.at_most),
                }
            )
            evidence: dict[str, Any] = {
                "source": "customer_side",
                "deployment_environment": "production",
                "attribution": "agent_key_exact_v1",
                "resolution": "terminal_status_v1",
            }
            if self.audit is not None:
                evidence["sampling"] = {
                    "mode": SAMPLING_MODE,
                    "rate": decimal_to_json(self.audit.rate),
                }
            monitors.append(
                {
                    "name": prod,
                    "requirement": name,
                    "phase": "runtime",
                    "evidence": evidence,
                    "decision": dict(_RUNTIME_DECISION),
                    "window": {
                        "tumbling_hours": WINDOW_HOURS[item.window],
                        "grace_minutes": 0,
                    },
                    "episode": {**_EPISODE_DEFAULTS, "closure_scope": self.recover},
                }
            )
            monitors.append(
                {
                    "name": gate,
                    "requirement": name,
                    "phase": "promotion",
                    "evidence": {"source": "evaluation_set"},
                    "decision": dict(_GATE_DECISION),
                }
            )
        rules: list[dict[str, Any]] = []
        used: set[str] = set()

        def rule_id(preferred: str | None, base: str) -> str:
            candidate = preferred or base
            n = 2
            while candidate in used:
                candidate = f"{preferred or base}_{n}"
                n += 1
            used.add(candidate)
            return candidate

        def common() -> dict[str, Any]:
            return {"do": "optimize", "approval": fix.approval, "deploy": fix.deploy}

        for trig in self.on:
            if isinstance(trig, OnBreach):
                for name in names:
                    base = "on_breach" if len(names) == 1 else f"on_breach_{name}"
                    rules.append(
                        {
                            "id": rule_id(trig.id if len(names) == 1 else None, base),
                            "trigger": {
                                "monitor": runtime_monitor_of[name],
                                "event": "episode_opened",
                            },
                            **common(),
                        }
                    )
            elif isinstance(trig, ModelReleased):
                match: dict[str, Any] = {}
                if trig.family:
                    match["provider_family"] = list(trig.family)
                if trig.tier:
                    match["tier"] = list(trig.tier)
                trigger: dict[str, Any] = {"external": "model_released"}
                if match:
                    trigger["match"] = match
                rule: dict[str, Any] = {
                    "id": rule_id(trig.id, "on_new_model"),
                    "trigger": trigger,
                    **common(),
                }
                if trig.admit is not None:
                    rule["admit"] = {"tvar": trig.admit}
                rule["protects"] = list(
                    names if trig.protects is None else trig.protects
                )
                rules.append(rule)
            else:
                rule = {
                    "id": rule_id(trig.id, "on_schedule"),
                    "trigger": {
                        "schedule": {"every_days": trig.days, "anchor": trig.anchor}
                    },
                    **common(),
                    "protects": list(names if trig.protects is None else trig.protects),
                }
                rules.append(rule)
        responses = [{"name": RESPONSE_NAME, "rules": rules}] if rules else []
        return {
            "requirements": requirements,
            "monitors": monitors,
            "responses": responses,
        }

    def to_tvl(self, module_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The full TVL 1.7 module: ``module_context`` plus this policy's blocks."""
        context = module_context if module_context is not None else self.module_context
        if context is None:
            context = {}
        doc: dict[str, Any] = {"tvl_version": TVL_VERSION}
        for key, value in context.items():
            if key not in _BODY_KEYS and key != "tvl_version":
                doc[key] = copy.deepcopy(value)
        for key, value in self._runtime_blocks().items():
            if value:
                doc[key] = value
        return doc

    def validate(
        self, module_context: Mapping[str, Any] | None = None
    ) -> list[Diagnostic]:
        """TVL diagnostics for the compiled module, verbatim from ``lint_module``.

        Raises ``ValidatorUnavailable`` (``E_TVL_VALIDATOR_UNAVAILABLE``) when the
        TVL toolchain is missing or does not support TVL 1.7.
        """
        return lint_document(self.to_tvl(module_context))

    # -- language-neutral sugar -------------------------------------------------
    def to_sugar(self) -> dict[str, Any]:
        """The language-neutral JSON description (shared with the JS SDK goldens)."""
        if self._fragment is not None:
            raise TuningError(E_NOT_ROUNDTRIPPABLE, "a tvl= policy has no sugar form")
        sugar: dict[str, Any] = {
            "keep": [
                _drop_none(
                    {
                        "checker": k.checker,
                        "at_most": decimal_to_str(k.at_most),
                        "window": k.window,
                        "name": k.name,
                        "monitor": k.monitor,
                        "gate": k.gate,
                    }
                )
                for k in self.keep
            ],
            "recover": self.recover,
        }
        if self.audit is not None:
            sugar["audit"] = {"rate": decimal_to_str(self.audit.rate)}
        if self.fix is not None:
            sugar["fix"] = {"approval": self.fix.approval, "deploy": self.fix.deploy}
        on: list[dict[str, Any]] = []
        for t in self.on:
            if isinstance(t, ModelReleased):
                on.append(
                    _drop_none(
                        {
                            "kind": "model_released",
                            "family": list(t.family),
                            "tier": list(t.tier),
                            "admit": t.admit,
                            "protects": list(t.protects)
                            if t.protects is not None
                            else None,
                            "id": t.id,
                        }
                    )
                )
            elif isinstance(t, OnBreach):
                on.append(_drop_none({"kind": "breach", "id": t.id}))
            else:
                on.append(
                    _drop_none(
                        {
                            "kind": "every",
                            "days": t.days,
                            "anchor": t.anchor,
                            "protects": list(t.protects)
                            if t.protects is not None
                            else None,
                            "id": t.id,
                        }
                    )
                )
        sugar["on"] = on
        return sugar

    @classmethod
    def from_sugar(cls, sugar: Mapping[str, Any]) -> Tuning:
        """Build from the language-neutral JSON description."""
        keep = [
            FailureRate(
                k["checker"],
                at_most=k["at_most"],
                window=k.get("window", "24h"),
                name=k.get("name"),
                monitor=k.get("monitor"),
                gate=k.get("gate"),
            )
            for k in sugar.get("keep", [])
        ]
        on: list[_Trigger] = []
        for t in sugar.get("on", []):
            kind = t.get("kind")
            if kind == "model_released":
                on.append(
                    ModelReleased(
                        family=t.get("family", ()),
                        tier=t.get("tier", ()),
                        admit=t.get("admit"),
                        protects=t.get("protects"),
                        id=t.get("id"),
                    )
                )
            elif kind == "breach":
                on.append(OnBreach(id=t.get("id")))
            elif kind == "every":
                on.append(
                    Every(
                        t["days"],
                        t["anchor"],
                        protects=t.get("protects"),
                        id=t.get("id"),
                    )
                )
            else:
                raise TuningError(E_INVALID_TUNING, "unknown trigger kind")
        audit = RandomAudit(sugar["audit"]["rate"]) if sugar.get("audit") else None
        fix = (
            ProposeFix(
                sugar["fix"].get("approval", "human"),
                sugar["fix"].get("deploy", "if_promoted"),
            )
            if sugar.get("fix")
            else None
        )
        return cls(
            keep=keep,
            audit=audit,
            on=on,
            fix=fix,
            recover=sugar.get("recover", "since_head_move"),
        )

    # -- from TVL ----------------------------------------------------------------
    @classmethod
    def from_tvl(cls, doc: Mapping[str, Any]) -> Tuning:
        """Rebuild the builder for a document produced by ``to_tvl``.

        Documents outside the builder's shape raise ``E_NOT_ROUNDTRIPPABLE``; use
        ``Tuning(tvl=...)`` for arbitrary TVL fragments.
        """
        try:
            tuning = _parse_doc(doc)
        except (KeyError, TypeError, IndexError, AttributeError):
            raise TuningError(
                E_NOT_ROUNDTRIPPABLE, "document is not in builder shape"
            ) from None
        context = {
            k: copy.deepcopy(v)
            for k, v in doc.items()
            if k not in _BODY_KEYS and k != "tvl_version"
        }
        tuning.module_context = context
        if _canonical(tuning.to_tvl()) != _canonical(dict(doc)):
            raise TuningError(E_NOT_ROUNDTRIPPABLE, "document is not in builder shape")
        return tuning


# -- helpers --------------------------------------------------------------------
def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _canonical(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, sort_keys=True))


def _parse_doc(doc: Mapping[str, Any]) -> Tuning:
    reqs = {r["name"]: r for r in doc.get("requirements", [])}
    runtime = [m for m in doc.get("monitors", []) if m["phase"] == "runtime"]
    gates = [m for m in doc.get("monitors", []) if m["phase"] == "promotion"]
    inv_hours = {v: k for k, v in WINDOW_HOURS.items()}
    keep: list[FailureRate] = []
    runtime_of: dict[str, str] = {}
    audit: RandomAudit | None = None
    recover = "since_head_move"
    for name, req in reqs.items():
        prod = next(m for m in runtime if m["requirement"] == name)
        gate = next(m for m in gates if m["requirement"] == name)
        runtime_of[prod["name"]] = name
        keep.append(
            FailureRate(
                req["violation"]["checker"],
                at_most=_num_str(req["max_rate"]),
                window=inv_hours[prod["window"]["tumbling_hours"]],
                name=name,
                monitor=prod["name"],
                gate=gate["name"],
            )
        )
        sampling = prod["evidence"].get("sampling")
        if sampling is not None:
            audit = RandomAudit(_num_str(sampling["rate"]))
        recover = prod["episode"]["closure_scope"]
    fix: ProposeFix | None = None
    on: list[_Trigger] = []
    for response in doc.get("responses", []):
        for rule in response["rules"]:
            fix = ProposeFix(rule["approval"], rule["deploy"])
            trig = rule["trigger"]
            if "monitor" in trig:
                if not any(isinstance(t, OnBreach) for t in on):
                    on.append(OnBreach(id=rule["id"]) if len(keep) == 1 else OnBreach())
            elif "external" in trig:
                match = trig.get("match", {})
                on.append(
                    ModelReleased(
                        family=match.get("provider_family", ()),
                        tier=match.get("tier", ()),
                        admit=rule.get("admit", {}).get("tvar"),
                        protects=rule.get("protects"),
                        id=rule["id"],
                    )
                )
            else:
                sched = trig["schedule"]
                on.append(
                    Every(
                        sched["every_days"],
                        sched["anchor"],
                        protects=rule.get("protects"),
                        id=rule["id"],
                    )
                )
    return Tuning(keep=keep, audit=audit, on=on, fix=fix, recover=recover)


def _num_str(value: Any) -> str:
    return decimal_to_str(
        Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
    )


_FRAGMENT_KEYS = {"tvl_version", *_BODY_KEYS}


def _resolve_fragment(path: str | Path, caller_file: str | None) -> Path:
    target = Path(path)
    if not target.is_absolute() and not target.exists() and caller_file:
        candidate = Path(caller_file).resolve().parent / target
        if candidate.exists():
            return candidate
    return target


def _load_fragment(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise TuningError(E_INVALID_TUNING, "tvl fragment could not be read") from None
    if (
        not isinstance(data, dict)
        or not set(data) <= _FRAGMENT_KEYS
        or not (set(data) & set(_BODY_KEYS))
    ):
        raise TuningError(
            E_INVALID_TUNING,
            "tvl fragment must hold only tvl_version, requirements, monitors, responses",
        )
    return data


__all__ = [
    "Every",
    "FailureRate",
    "ModelReleased",
    "OnBreach",
    "ProposeFix",
    "RandomAudit",
    "Tuning",
    "RESPONSE_NAME",
    "TVL_VERSION",
    "decimal_to_json",
    "decimal_to_str",
    "to_decimal",
]
