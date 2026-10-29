"""TVL fragment loading and (lazy) compilation of the hosted subset.

A fragment holds only runtime blocks (``requirements``, ``monitors``,
``responses``).  The loader accepts exactly the shape the backend can check and
refuses everything else with a closed code and a structural path (never a
value).  The TVL compiler (``tvl-spec``, Python >= 3.12) is imported lazily and
only by :func:`compile_hosted_subset` / :func:`compile_full_policy`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from traigent.runtime.model import (
    ACTIONS,
    DEFAULT_ENVIRONMENT,
    DEPLOY_MODES,
    E_ACTION_UNSUPPORTED,
    E_CAPABILITY,
    E_COMPILER,
    E_FRAGMENT,
    E_NOTIFY_ONLY,
    E_VERSION_BELOW,
    E_VERSION_REQUIRED,
    EVENTS,
    GUARD_STATES,
    OUTCOME_EVENTS,
    PROFILE_DEFAULTS,
    SEVERITIES,
    UNKNOWN_FOR,
    NotifyGuard,
    PromotionMonitor,
    ResponseRule,
    RuntimeEvent,
    RuntimeRequirement,
    RuntimeRequirementError,
    normalise_environment,
    positive_finite,
)

MONITORS_1 = "monitors[1]"

_MAX_FRAGMENT_BYTES = 64 * 1024
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_PLACEHOLDER = "<key>"


def _unsupported(path: str) -> RuntimeRequirementError:
    return RuntimeRequirementError(E_CAPABILITY, path)


def _invalid(path: str) -> RuntimeRequirementError:
    return RuntimeRequirementError(E_FRAGMENT, path)


def _closed_keys(
    node: Any, allowed: set[str], required: set[str], path: str
) -> Mapping[str, Any]:
    """A mapping whose keys are inside ``allowed``; unknown keys are never echoed."""
    if not isinstance(node, Mapping):
        raise _invalid(path)
    for key in node:
        if key not in allowed:
            raise _unsupported(f"{path}.{_PLACEHOLDER}")
    for key in required:
        if key not in node:
            raise _invalid(f"{path}.{key}")
    return node


def _name(node: Any, path: str) -> str:
    if not isinstance(node, str) or not _NAME_RE.fullmatch(node):
        raise _invalid(path)
    return node


def _int(node: Any, path: str) -> int:
    if type(node) is not int:
        raise _invalid(path)
    return node


def _num(node: Any, path: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise _invalid(path)
    return float(node)


def _parse_requirement(node: Any) -> tuple[str, int, float]:
    path = "requirements[0]"
    req = _closed_keys(
        node,
        {"name", "kind", "unit", "violation", "max_rate"},
        {"name", "kind", "unit", "violation", "max_rate"},
        path,
    )
    name = _name(req["name"], f"{path}.name")
    if req["kind"] != "rate_bound":
        raise _unsupported(f"{path}.kind")
    if req["unit"] != "trace":
        raise _unsupported(f"{path}.unit")
    violation = _closed_keys(
        req["violation"],
        {"metric", "exceeds"},
        {"metric", "exceeds"},
        f"{path}.violation",
    )
    if violation["metric"] != "wall_clock_latency_ms":
        raise _unsupported(f"{path}.violation.metric")
    threshold = _int(violation["exceeds"], f"{path}.violation.exceeds")
    return name, threshold, _num(req["max_rate"], f"{path}.max_rate")


def _parse_promotion(node: Any, requirement_name: str) -> PromotionMonitor:
    path = MONITORS_1
    mon = _closed_keys(
        node,
        {"name", "requirement", "phase", "evidence", "decision"},
        {"name", "requirement", "phase", "evidence", "decision"},
        path,
    )
    name = _name(mon["name"], f"{path}.name")
    if mon["requirement"] != requirement_name:
        raise _invalid(f"{path}.requirement")
    evidence = _closed_keys(mon["evidence"], {"source"}, {"source"}, f"{path}.evidence")
    if evidence["source"] != "evaluation_set":
        raise _unsupported(f"{path}.evidence.source")
    decision = _closed_keys(
        mon["decision"],
        {"method", "confidence", "false_alarm_budget"},
        {"method", "confidence", "false_alarm_budget"},
        f"{path}.decision",
    )
    if decision["method"] != "clopper_pearson_rational_v2":
        raise _unsupported(f"{path}.decision.method")
    return PromotionMonitor(
        name,
        _num(decision["confidence"], f"{path}.decision.confidence"),
        _num(decision["false_alarm_budget"], f"{path}.decision.false_alarm_budget"),
    )


def _split_monitors(mons: list[Any]) -> tuple[Any, Any | None]:
    """The one runtime monitor and the optional promotion monitor, in either order."""
    if len(mons) > 2:
        raise _unsupported("monitors[2]")
    promotion = [
        m for m in mons if isinstance(m, Mapping) and m.get("phase") == "promotion"
    ]
    runtime = [m for m in mons if m not in promotion]
    if len(promotion) > 1:
        raise _unsupported(MONITORS_1)
    if len(runtime) != 1:
        raise _unsupported(MONITORS_1) if len(runtime) > 1 else _invalid("monitors")
    return runtime[0], (promotion[0] if promotion else None)


def _parse_monitor(node: Any, requirement_name: str) -> tuple[str, dict[str, Any]]:
    path = "monitors[0]"
    mon = _closed_keys(
        node,
        {
            "name",
            "requirement",
            "phase",
            "profile",
            "applies_to",
            "evidence",
            "window",
            "decision",
            "episode",
        },
        {"name", "requirement", "phase", "profile", "evidence", "decision"},
        path,
    )
    name = _name(mon["name"], f"{path}.name")
    if mon["requirement"] != requirement_name:
        raise _invalid(f"{path}.requirement")
    if mon["phase"] != "runtime":
        raise _unsupported(f"{path}.phase")
    if mon["profile"] != "backend_v1":
        raise _unsupported(f"{path}.profile")
    if mon.get("applies_to", "agent") != "agent":
        raise _unsupported(f"{path}.applies_to")
    evidence = _closed_keys(
        mon["evidence"],
        {"source", "deployment_environment", "attribution", "resolution", "sampling"},
        {"source"},
        f"{path}.evidence",
    )
    if evidence["source"] != "backend_trace_metadata":
        raise _unsupported(f"{path}.evidence.source")
    for key, fixed in (
        ("attribution", "agent_key_exact_v1"),
        ("resolution", "terminal_status_v1"),
        ("sampling", "unverified"),
    ):
        if key in evidence and evidence[key] != fixed:
            raise _unsupported(f"{path}.evidence.{key}")
    environment = evidence.get("deployment_environment", DEFAULT_ENVIRONMENT)
    if not isinstance(environment, str):
        raise _invalid(f"{path}.evidence.deployment_environment")
    decision = _closed_keys(
        mon["decision"],
        {
            "method",
            "model",
            "confidence",
            "false_alarm_budget",
            "horizon_days",
            "min_samples",
            "max_unresolved_fraction",
        },
        {"method"},
        f"{path}.decision",
    )
    if decision["method"] != "clopper_pearson_exact_v1":
        raise _unsupported(f"{path}.decision.method")
    if "model" in decision and decision["model"] != "iid_bernoulli_within_window_v1":
        raise _unsupported(f"{path}.decision.model")
    window = _closed_keys(
        mon.get("window", {}),
        {"tumbling_hours", "grace_minutes"},
        set(),
        f"{path}.window",
    )
    episode = _closed_keys(
        mon.get("episode", {}),
        {"closure_windows", "closure_max_age_hours"},
        set(),
        f"{path}.episode",
    )
    fields: dict[str, Any] = {"environment": environment}
    for key, section, wire in (
        ("window_hours", window, "tumbling_hours"),
        ("grace_minutes", window, "grace_minutes"),
        ("closure_windows", episode, "closure_windows"),
        ("closure_max_age_hours", episode, "closure_max_age_hours"),
        ("min_samples", decision, "min_samples"),
        ("horizon_days", decision, "horizon_days"),
    ):
        if wire in section:
            fields[key] = _int(section[wire], f"{path}.{wire}")
    for key in ("confidence", "false_alarm_budget", "max_unresolved_fraction"):
        if key in decision:
            fields[key] = _num(decision[key], f"{path}.decision.{key}")
    return name, fields


def _parse_trigger(
    node: Any, monitor_name: str, path: str
) -> tuple[str | None, str, int | None]:
    """``(trigger_rule, event, hours)``: a monitor trigger or an outcome trigger."""
    trigger = _closed_keys(node, {"monitor", "rule", "event", "hours"}, {"event"}, path)
    event = trigger["event"]
    if "rule" in trigger:
        # Outcome trigger (RFC 0004 section 2.2): {rule, event} and nothing else.
        if "monitor" in trigger or "hours" in trigger:
            raise _invalid(path)
        if event not in OUTCOME_EVENTS:
            raise _unsupported(f"{path}.event")
        return _name(trigger["rule"], f"{path}.rule"), event, None
    if "monitor" not in trigger:
        raise _invalid(f"{path}.monitor")
    if trigger["monitor"] != monitor_name:
        raise _invalid(f"{path}.monitor")
    if event not in EVENTS and event != UNKNOWN_FOR:
        raise _unsupported(f"{path}.event")
    hours: int | None = None
    if event == UNKNOWN_FOR:
        if "hours" not in trigger:
            raise _invalid(f"{path}.hours")
        hours = _int(trigger["hours"], f"{path}.hours")
    elif "hours" in trigger:
        raise _invalid(f"{path}.hours")
    return None, event, hours


def _parse_guards(node: Any, monitor_name: str, path: str) -> tuple[NotifyGuard, ...]:
    if node is None:
        return ()
    if not isinstance(node, list):
        raise _invalid(path)  # an empty list is valid and projection-neutral
    guards: list[NotifyGuard] = []
    for index, raw in enumerate(node):
        where = f"{path}[{index}]"
        if not isinstance(raw, Mapping):
            raise _invalid(where)
        kind = "monitor" if "monitor" in raw else "rule" if "rule" in raw else None
        if kind is None:
            raise _invalid(where)
        guard = _closed_keys(raw, {kind, "state"}, {kind, "state"}, where)
        if guard["state"] != GUARD_STATES[kind]:
            raise _unsupported(f"{where}.state")
        ref = _name(guard[kind], f"{where}.{kind}")
        if kind == "monitor" and ref != monitor_name:
            # One monitor per fragment: a guard on any other monitor cannot resolve.
            raise _unsupported(f"{where}.monitor")
        guards.append(NotifyGuard(kind, ref))
    return tuple(guards)


def _parse_responses(
    node: Any, monitor_name: str
) -> tuple[str | None, tuple[ResponseRule, ...]]:
    if node is None:
        return None, ()
    if not isinstance(node, list):
        raise _invalid("responses")
    if len(node) > 1:
        raise _unsupported("responses[1]")
    if not node:
        return None, ()
    response = _closed_keys(
        node[0], {"name", "rules"}, {"name", "rules"}, "responses[0]"
    )
    ladder = _name(response["name"], "responses[0].name")
    raw_rules = response["rules"]
    if not isinstance(raw_rules, list):
        raise _invalid("responses[0].rules")
    rules: list[ResponseRule] = []
    for index, raw in enumerate(raw_rules):
        path = f"responses[0].rules[{index}]"
        rule = _closed_keys(
            raw,
            {
                "id",
                "trigger",
                "do",
                "severity",
                "approval",
                "deploy",
                "only_while",
                "unless",
            },
            {"id", "trigger", "do"},
            path,
        )
        rule_id = _name(rule["id"], f"{path}.id")
        trigger_rule, event, hours = _parse_trigger(
            rule["trigger"], monitor_name, f"{path}.trigger"
        )
        only_while = _parse_guards(
            rule.get("only_while"), monitor_name, f"{path}.only_while"
        )
        unless = _parse_guards(rule.get("unless"), monitor_name, f"{path}.unless")
        action = rule["do"]
        if action not in ACTIONS:
            raise RuntimeRequirementError(E_ACTION_UNSUPPORTED, f"{path}.do")
        severity = rule.get("severity")
        approval = rule.get("approval")
        deploy = rule.get("deploy")
        reactive = bool(trigger_rule or event == UNKNOWN_FOR or only_while or unless)
        if reactive and action != "notify":
            raise RuntimeRequirementError(E_NOTIFY_ONLY, f"{path}.do")
        if deploy is not None and (action != "optimize" or deploy not in DEPLOY_MODES):
            raise _invalid(f"{path}.deploy")
        if action == "notify":
            if severity is None:
                severity = "critical"  # RFC 0003 section 7.2: the default severity
            if severity not in SEVERITIES:
                raise _invalid(f"{path}.severity")
            if approval is not None:
                raise _invalid(f"{path}.approval")
        elif action == "optimize":
            if severity is not None:
                raise _invalid(f"{path}.severity")
            # Owner ruling: ``approval: standing`` is refused locally with a closed code.
            if approval not in (None, "human"):
                raise RuntimeRequirementError(E_ACTION_UNSUPPORTED, f"{path}.approval")
            approval = "human"
        elif severity is not None or approval is not None:
            raise _invalid(f"{path}.do")
        rules.append(
            ResponseRule(
                rule_id,
                event,
                action,
                severity,
                approval,
                trigger_rule=trigger_rule,
                hours=hours,
                only_while=only_while,
                unless=unless,
                deploy=deploy,
            )
        )
    return ladder, tuple(rules)


def parse_fragment(
    doc: Any,
    *,
    max_fix_cost_usd: float | None = None,
    agent_id: str | None = None,
    on_event: Callable[[RuntimeEvent], None] | None = None,
    fragment_path: str | None = None,
) -> RuntimeRequirement:
    """Validate a fragment mapping and build the requirement it declares."""
    top = _closed_keys(
        doc,
        {"tvl_version", "requirements", "monitors", "responses"},
        {"requirements", "monitors"},
        "<root>",
    )
    declared = top.get("tvl_version", "1.3")
    if declared not in ("1.3", "1.4"):
        raise _unsupported("tvl_version")
    reqs, mons = top["requirements"], top["monitors"]
    if not isinstance(reqs, list) or not reqs:
        raise _invalid("requirements")
    if len(reqs) > 1:
        raise _unsupported("requirements[1]")
    if not isinstance(mons, list) or not mons:
        raise _invalid("monitors")
    runtime_node, promotion_node = _split_monitors(mons)
    name, threshold, rate = _parse_requirement(reqs[0])
    monitor_name, fields = _parse_monitor(runtime_node, name)
    promotion = (
        _parse_promotion(promotion_node, name) if promotion_node is not None else None
    )
    ladder, rules = _parse_responses(top.get("responses"), monitor_name)
    if any(r.is_reactive() for r in rules):
        if "tvl_version" not in top:
            raise RuntimeRequirementError(E_VERSION_REQUIRED, "tvl_version")
        if declared == "1.3":
            raise RuntimeRequirementError(E_VERSION_BELOW, "tvl_version")
    if max_fix_cost_usd is not None and not positive_finite(max_fix_cost_usd):
        raise RuntimeRequirementError("fix_ceiling_required", "max_fix_cost_usd")
    values: dict[str, Any] = {key: PROFILE_DEFAULTS[key] for key in PROFILE_DEFAULTS}
    values.update(fields)
    on_breach = (
        "fix"
        if any(r.action == "optimize" for r in rules)
        else (
            "alert"
            if any(
                r.action == "notify" and r.severity != "info" and not r.is_reactive()
                for r in rules
            )
            else "log"
        )
    )
    requirement = RuntimeRequirement(
        name=name,
        latency_threshold_ms=threshold,
        max_violation_rate=rate,
        rules=rules,
        response_name=ladder,
        monitor_label=monitor_name,
        on_breach=on_breach,
        max_fix_cost_usd=max_fix_cost_usd,
        agent_id=agent_id,
        source="tvl",
        fragment_path=fragment_path,
        promotion=promotion,
        on_event=on_event,
        **values,
    )
    errors = [i for i in requirement.validate() if i.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    return requirement


def load_fragment(
    path: str | Path,
    *,
    max_fix_cost_usd: float | None = None,
    agent_id: str | None = None,
    on_event: Callable[[RuntimeEvent], None] | None = None,
) -> RuntimeRequirement:
    """Read a fragment file with the safe YAML loader and parse it."""
    target = Path(path)
    doc: Any = None
    failed = False
    try:
        if target.stat().st_size > _MAX_FRAGMENT_BYTES:
            raise RuntimeRequirementError(E_FRAGMENT, "<file>")
        doc = yaml.safe_load(target.read_text(encoding="utf-8"))
    except RuntimeRequirementError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        failed = True
    # Raised outside the except block: a YAML error message quotes source text, and
    # leaving it as __context__ would carry that text into any traceback or log.
    if failed:
        raise RuntimeRequirementError(E_FRAGMENT, "<file>")
    return parse_fragment(
        doc,
        max_fix_cost_usd=max_fix_cost_usd,
        agent_id=agent_id,
        on_event=on_event,
        fragment_path=str(target),
    )


# --- compilation (lazy) -----------------------------------------------------------


def import_compiler() -> Any:
    """Import the TVL compiler lazily; closed error when it is not installed."""
    try:
        import tvl.runtime_compiler as compiler  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeRequirementError(E_COMPILER) from None
    return compiler


def module_name_for(qualname: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.]", "_", qualname).strip(".") or "agent"
    return f"traigent.runtime.{cleaned}"[:120]


def module_document(
    requirement: RuntimeRequirement, module: str, *, with_responses: bool
) -> dict[str, Any]:
    """A complete TVL module: a minimal synthesized header plus the runtime blocks."""
    header: dict[str, Any] = {
        "tvl": {"module": module},
        "tvl_version": "1.4" if requirement.is_reactive() else "1.3",
        "environment": {"snapshot_id": "2026-10-16T00:00:00Z"},
        "evaluation_set": {"dataset": "traigent://runtime"},
        "tvars": [
            {"name": "runtime_placeholder", "type": "enum[str]", "domain": ["a", "b"]}
        ],
        "objectives": [{"name": "quality", "direction": "maximize"}],
        "promotion_policy": {
            "dominance": "epsilon_pareto",
            "alpha": 0.05,
            "min_effect": {"quality": 0.02},
        },
    }
    blocks = requirement.to_doc()
    header.update({k: v for k, v in blocks.items() if k != "tvl_version"})
    if not with_responses:
        header["responses"] = []
    return header


def token_entries(
    requirement: RuntimeRequirement, module: str
) -> list[tuple[str, str]]:
    """The (namespace, local name) pairs the compiler needs tokens for."""
    entries: list[tuple[str, str]] = [
        ("module", module),
        ("requirement", requirement.name),
        ("monitor", requirement.monitor_name),
        ("environment", normalise_environment(requirement.environment)),
    ]
    if requirement.promotion is not None:
        entries.append(("monitor", requirement.promotion.name))
    if requirement.rules:
        entries.append(("response", requirement.ladder_name))
        entries.extend(("rule", rule.id) for rule in requirement.rules)
    return entries


def _toolchain(requirement: RuntimeRequirement) -> dict[str, str]:
    return {} if requirement.is_reactive() else {"toolchain_version": "1.3.0"}


def compile_hosted_subset(
    requirement: RuntimeRequirement, token_map: Mapping[str, Any], module: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Compile the one-requirement, one-monitor hosted contract, responses included.

    A contract with a TVL 1.4 reactive construct compiles to
    ``tvl-runtime-contract/2``; any other contract stays ``/1`` and keeps the
    1.3.0 toolchain stamp, so a Backend that has not yet accepted ``1.4.0``
    still reads it (RFC 0004 section 3).
    """
    compiler = import_compiler()
    try:
        contract, local, issues = compiler.compile_runtime_document(
            module_document(requirement, module, with_responses=True),
            token_map_doc=token_map,
            **_toolchain(requirement),
        )
        return dict(contract), dict(local), list(issues)
    except Exception as exc:  # noqa: BLE001 - closed re-raise, no echo
        code = getattr(exc, "code", None)
        raise RuntimeRequirementError(
            (
                code
                if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,64}", code)
                else E_FRAGMENT
            ),
            _safe_path(getattr(exc, "path", None)),
        ) from None


def compile_full_policy(
    requirement: RuntimeRequirement, token_map: Mapping[str, Any], module: str
) -> list[dict[str, Any]]:
    """Compile the full module (responses included) to lint it; returns the lint issues."""
    compiler = import_compiler()
    try:
        _contract, _local, issues = compiler.compile_runtime_document(
            module_document(requirement, module, with_responses=True),
            token_map_doc=token_map,
            **_toolchain(requirement),
        )
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        raise RuntimeRequirementError(
            (
                code
                if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,64}", code)
                else E_FRAGMENT
            ),
            _safe_path(getattr(exc, "path", None)),
        ) from None
    return list(issues)


def _safe_path(path: Any) -> str | None:
    if not isinstance(path, (list, tuple)):
        return None
    parts = [str(p) for p in path if isinstance(p, (int, str))]
    text = ".".join(parts)
    return text if re.fullmatch(r"[A-Za-z0-9_.\[\]]{1,120}", text) else None
