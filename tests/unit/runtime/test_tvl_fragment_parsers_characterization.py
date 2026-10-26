"""Golden characterization for tvl_fragment._parse_monitor and _parse_responses.

A mutation matrix over a fully-populated monitor and a three-rule response
ladder; the expected table was recorded from the pre-refactor implementation.
"""

from __future__ import annotations

import copy

import pytest

from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.tvl_fragment import _parse_monitor, _parse_responses

MON = {
    "name": "mon",
    "requirement": "req",
    "phase": "runtime",
    "profile": "backend_v1",
    "applies_to": "agent",
    "evidence": {
        "source": "backend_trace_metadata",
        "deployment_environment": "staging",
        "attribution": "agent_key_exact_v1",
        "resolution": "terminal_status_v1",
        "sampling": "unverified",
    },
    "window": {"tumbling_hours": 2, "grace_minutes": 3},
    "decision": {
        "method": "clopper_pearson_exact_v1",
        "model": "iid_bernoulli_within_window_v1",
        "confidence": 0.9,
        "false_alarm_budget": 0.01,
        "horizon_days": 5,
        "min_samples": 7,
        "max_unresolved_fraction": 0.2,
    },
    "episode": {"closure_windows": 2, "closure_max_age_hours": 9},
}
RESP = [
    {
        "name": "lad",
        "rules": [
            {
                "id": "a",
                "trigger": {"monitor": "mon", "event": "episode_opened"},
                "do": "notify",
                "severity": "warning",
            },
            {
                "id": "b",
                "trigger": {"monitor": "mon", "event": "episode_resolved"},
                "do": "optimize",
                "approval": "human",
            },
            {
                "id": "c",
                "trigger": {"monitor": "mon", "event": "episode_opened"},
                "do": "open_next_step",
            },
        ],
    }
]
DEL = object()


def apply(base, path, val):
    d = copy.deepcopy(base)
    if not path:
        return val
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    if val is DEL:
        cur.pop(path[-1], None)
    else:
        cur[path[-1]] = val
    return d


def run(fn, node, *a):
    try:
        r = fn(node, *a)
    except RuntimeRequirementError as e:
        return ("err", e.code, e.path)
    return ("ok", repr(r))


mons = [("base", (), None)]
for k in MON:
    mons.append((f"del {k}", (k,), DEL))
    mons.append((f"bad {k}", (k,), "zzz"))
    mons.append((f"none {k}", (k,), None))
for sec in ("evidence", "window", "decision", "episode"):
    for k in MON[sec]:
        mons.append((f"del {sec}.{k}", (sec, k), DEL))
        mons.append((f"bad {sec}.{k}", (sec, k), "zzz"))
        mons.append((f"float {sec}.{k}", (sec, k), 1.5))
        mons.append((f"bool {sec}.{k}", (sec, k), True))
    mons.append((f"extra {sec}", (sec, "extra"), 1))
mons.append(("extra top", ("extra",), 1))
mons.append(("node list", (), [1]))
mons.append(("name bad re", ("name",), "1abc"))
mons.append(("no applies", ("applies_to",), DEL))
out = {}
for label, path, val in mons:
    node = MON if label == "base" else (val if path == () else apply(MON, path, val))
    out[label] = run(_parse_monitor, node, "req")
rsp = [("base", (), None), ("none", None, None)]
res = {}
res["none"] = run(_parse_responses, None, "mon")
for label, node in [
    ("str", "x"),
    ("dict", {}),
    ("empty", []),
    ("two", [RESP[0], RESP[0]]),
    ("elem list", [[1]]),
]:
    res[label] = run(_parse_responses, node, "mon")


def R(path, val):
    return copy.deepcopy(RESP) if path == () else apply(RESP, path, val)


rm = [("base", (), None)]
for k in ("name", "rules", "extra"):
    rm.append((f"del {k}", (0, k), DEL))
    rm.append((f"bad {k}", (0, k), "zzz")) if k != "extra" else rm.append(
        ("extra", (0, "extra"), 1)
    )
rm.append(("rules dict", (0, "rules"), {}))
rm.append(("rules empty", (0, "rules"), []))
for ri in range(3):
    for k in ("id", "trigger", "do", "severity", "approval"):
        rm.append((f"r{ri} del {k}", (0, "rules", ri, k), DEL))
        rm.append((f"r{ri} bad {k}", (0, "rules", ri, k), "zzz"))
        rm.append((f"r{ri} none {k}", (0, "rules", ri, k), None))
        rm.append((f"r{ri} set {k}=human", (0, "rules", ri, k), "human"))
        rm.append((f"r{ri} set {k}=critical", (0, "rules", ri, k), "critical"))
    for k in ("monitor", "event"):
        rm.append((f"r{ri} trig del {k}", (0, "rules", ri, "trigger", k), DEL))
        rm.append((f"r{ri} trig bad {k}", (0, "rules", ri, "trigger", k), "zzz"))
    rm.append((f"r{ri} extra", (0, "rules", ri, "extra"), 1))
    rm.append((f"r{ri} raw int", (0, "rules", ri), 5))
    rm.append((f"r{ri} do notify", (0, "rules", ri, "do"), "notify"))
    rm.append((f"r{ri} do optimize", (0, "rules", ri, "do"), "optimize"))
    rm.append((f"r{ri} do open_next_step", (0, "rules", ri, "do"), "open_next_step"))
for label, path, val in rm:
    res[label] = run(_parse_responses, R(path, val), "mon")
# combos: do x severity x approval
for do in ("notify", "optimize", "open_next_step"):
    for sev in (DEL, None, "info", "warning", "critical", "zzz"):
        for ap in (DEL, None, "human", "zzz"):
            rule = {
                "id": "a",
                "trigger": {"monitor": "mon", "event": "episode_opened"},
                "do": do,
            }
            if sev is not DEL:
                rule["severity"] = sev
            if ap is not DEL:
                rule["approval"] = ap
            lab = f"combo {do} {'-' if sev is DEL else sev} {'-' if ap is DEL else ap}"
            res[lab] = run(_parse_responses, [{"name": "l", "rules": [rule]}], "mon")
res["order two bad"] = run(
    _parse_responses,
    [
        {
            "name": "l",
            "rules": [
                {
                    "id": "a",
                    "trigger": {"monitor": "mon", "event": "episode_opened"},
                    "do": "zzz",
                },
                {
                    "id": "b2",
                    "trigger": {"monitor": "zzz", "event": "episode_opened"},
                    "do": "notify",
                },
            ],
        }
    ],
    "mon",
)


MONITOR_EXPECTED = {
    "base": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, 'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "del name": ("err", "tvl_fragment_invalid", "monitors[0].name"),
    "bad name": (
        "ok",
        "('zzz', {'environment': 'staging', 'window_hours': 2, 'grace_minutes': "
        "3, 'closure_windows': 2, 'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "none name": ("err", "tvl_fragment_invalid", "monitors[0].name"),
    "del requirement": ("err", "tvl_fragment_invalid", "monitors[0].requirement"),
    "bad requirement": ("err", "tvl_fragment_invalid", "monitors[0].requirement"),
    "none requirement": ("err", "tvl_fragment_invalid", "monitors[0].requirement"),
    "del phase": ("err", "tvl_fragment_invalid", "monitors[0].phase"),
    "bad phase": ("err", "runtime_unsupported_capability", "monitors[0].phase"),
    "none phase": ("err", "runtime_unsupported_capability", "monitors[0].phase"),
    "del profile": ("err", "tvl_fragment_invalid", "monitors[0].profile"),
    "bad profile": ("err", "runtime_unsupported_capability", "monitors[0].profile"),
    "none profile": ("err", "runtime_unsupported_capability", "monitors[0].profile"),
    "del applies_to": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad applies_to": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].applies_to",
    ),
    "none applies_to": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].applies_to",
    ),
    "del evidence": ("err", "tvl_fragment_invalid", "monitors[0].evidence"),
    "bad evidence": ("err", "tvl_fragment_invalid", "monitors[0].evidence"),
    "none evidence": ("err", "tvl_fragment_invalid", "monitors[0].evidence"),
    "del window": (
        "ok",
        "('mon', {'environment': 'staging', 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad window": ("err", "tvl_fragment_invalid", "monitors[0].window"),
    "none window": ("err", "tvl_fragment_invalid", "monitors[0].window"),
    "del decision": ("err", "tvl_fragment_invalid", "monitors[0].decision"),
    "bad decision": ("err", "tvl_fragment_invalid", "monitors[0].decision"),
    "none decision": ("err", "tvl_fragment_invalid", "monitors[0].decision"),
    "del episode": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad episode": ("err", "tvl_fragment_invalid", "monitors[0].episode"),
    "none episode": ("err", "tvl_fragment_invalid", "monitors[0].episode"),
    "del evidence.source": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].evidence.source",
    ),
    "bad evidence.source": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.source",
    ),
    "float evidence.source": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.source",
    ),
    "bool evidence.source": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.source",
    ),
    "del evidence.deployment_environment": (
        "ok",
        "('mon', {'environment': 'production', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': "
        "7, 'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad evidence.deployment_environment": (
        "ok",
        "('mon', {'environment': 'zzz', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': "
        "7, 'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "float evidence.deployment_environment": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].evidence.deployment_environment",
    ),
    "bool evidence.deployment_environment": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].evidence.deployment_environment",
    ),
    "del evidence.attribution": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bad evidence.attribution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.attribution",
    ),
    "float evidence.attribution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.attribution",
    ),
    "bool evidence.attribution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.attribution",
    ),
    "del evidence.resolution": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bad evidence.resolution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.resolution",
    ),
    "float evidence.resolution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.resolution",
    ),
    "bool evidence.resolution": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.resolution",
    ),
    "del evidence.sampling": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bad evidence.sampling": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.sampling",
    ),
    "float evidence.sampling": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.sampling",
    ),
    "bool evidence.sampling": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.sampling",
    ),
    "extra evidence": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].evidence.<key>",
    ),
    "del window.tumbling_hours": (
        "ok",
        "('mon', {'environment': 'staging', 'grace_minutes': 3, "
        "'closure_windows': 2, 'closure_max_age_hours': 9, "
        "'min_samples': 7, 'horizon_days': 5, 'confidence': "
        "0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad window.tumbling_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].tumbling_hours",
    ),
    "float window.tumbling_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].tumbling_hours",
    ),
    "bool window.tumbling_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].tumbling_hours",
    ),
    "del window.grace_minutes": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'closure_windows': 2, 'closure_max_age_hours': 9, "
        "'min_samples': 7, 'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bad window.grace_minutes": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].grace_minutes",
    ),
    "float window.grace_minutes": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].grace_minutes",
    ),
    "bool window.grace_minutes": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].grace_minutes",
    ),
    "extra window": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].window.<key>",
    ),
    "del decision.method": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.method",
    ),
    "bad decision.method": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.method",
    ),
    "float decision.method": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.method",
    ),
    "bool decision.method": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.method",
    ),
    "del decision.model": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, 'horizon_days': "
        "5, 'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad decision.model": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.model",
    ),
    "float decision.model": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.model",
    ),
    "bool decision.model": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.model",
    ),
    "del decision.confidence": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad decision.confidence": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.confidence",
    ),
    "float decision.confidence": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 1.5, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bool decision.confidence": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.confidence",
    ),
    "del decision.false_alarm_budget": (
        "ok",
        "('mon', {'environment': 'staging', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, 'closure_max_age_hours': "
        "9, 'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'max_unresolved_fraction': "
        "0.2})",
    ),
    "bad decision.false_alarm_budget": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.false_alarm_budget",
    ),
    "float decision.false_alarm_budget": (
        "ok",
        "('mon', {'environment': 'staging', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, 'closure_max_age_hours': "
        "9, 'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': 1.5, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bool decision.false_alarm_budget": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.false_alarm_budget",
    ),
    "del decision.horizon_days": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': 7, "
        "'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad decision.horizon_days": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].horizon_days",
    ),
    "float decision.horizon_days": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].horizon_days",
    ),
    "bool decision.horizon_days": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].horizon_days",
    ),
    "del decision.min_samples": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad decision.min_samples": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].min_samples",
    ),
    "float decision.min_samples": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].min_samples",
    ),
    "bool decision.min_samples": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].min_samples",
    ),
    "del decision.max_unresolved_fraction": (
        "ok",
        "('mon', {'environment': 'staging', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, "
        "'closure_max_age_hours': 9, 'min_samples': "
        "7, 'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01})",
    ),
    "bad decision.max_unresolved_fraction": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.max_unresolved_fraction",
    ),
    "float decision.max_unresolved_fraction": (
        "ok",
        "('mon', {'environment': 'staging', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, "
        "'closure_max_age_hours': 9, "
        "'min_samples': 7, 'horizon_days': 5, "
        "'confidence': 0.9, 'false_alarm_budget': "
        "0.01, 'max_unresolved_fraction': 1.5})",
    ),
    "bool decision.max_unresolved_fraction": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].decision.max_unresolved_fraction",
    ),
    "extra decision": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].decision.<key>",
    ),
    "del episode.closure_windows": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': "
        "2, 'grace_minutes': 3, 'closure_max_age_hours': 9, "
        "'min_samples': 7, 'horizon_days': 5, 'confidence': "
        "0.9, 'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad episode.closure_windows": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_windows",
    ),
    "float episode.closure_windows": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_windows",
    ),
    "bool episode.closure_windows": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_windows",
    ),
    "del episode.closure_max_age_hours": (
        "ok",
        "('mon', {'environment': 'staging', "
        "'window_hours': 2, 'grace_minutes': 3, "
        "'closure_windows': 2, 'min_samples': 7, "
        "'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, "
        "'max_unresolved_fraction': 0.2})",
    ),
    "bad episode.closure_max_age_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_max_age_hours",
    ),
    "float episode.closure_max_age_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_max_age_hours",
    ),
    "bool episode.closure_max_age_hours": (
        "err",
        "tvl_fragment_invalid",
        "monitors[0].closure_max_age_hours",
    ),
    "extra episode": (
        "err",
        "runtime_unsupported_capability",
        "monitors[0].episode.<key>",
    ),
    "extra top": ("err", "runtime_unsupported_capability", "monitors[0].<key>"),
    "node list": ("err", "tvl_fragment_invalid", "monitors[0]"),
    "name bad re": ("err", "tvl_fragment_invalid", "monitors[0].name"),
    "no applies": (
        "ok",
        "('mon', {'environment': 'staging', 'window_hours': 2, "
        "'grace_minutes': 3, 'closure_windows': 2, 'closure_max_age_hours': 9, "
        "'min_samples': 7, 'horizon_days': 5, 'confidence': 0.9, "
        "'false_alarm_budget': 0.01, 'max_unresolved_fraction': 0.2})",
    ),
}
RESPONSES_EXPECTED = {
    "none": ("ok", "(None, ())"),
    "str": ("err", "tvl_fragment_invalid", "responses"),
    "dict": ("err", "tvl_fragment_invalid", "responses"),
    "empty": ("ok", "(None, ())"),
    "two": ("err", "runtime_unsupported_capability", "responses[1]"),
    "elem list": ("err", "tvl_fragment_invalid", "responses[0]"),
    "base": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', action='notify', "
        "severity='warning', approval=None), ResponseRule(id='b', "
        "event='episode_resolved', action='optimize', severity=None, "
        "approval='human'), ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "del name": ("err", "tvl_fragment_invalid", "responses[0].name"),
    "bad name": (
        "ok",
        "('zzz', (ResponseRule(id='a', event='episode_opened', action='notify', "
        "severity='warning', approval=None), ResponseRule(id='b', "
        "event='episode_resolved', action='optimize', severity=None, "
        "approval='human'), ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "del rules": ("err", "tvl_fragment_invalid", "responses[0].rules"),
    "bad rules": ("err", "tvl_fragment_invalid", "responses[0].rules"),
    "del extra": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', action='notify', "
        "severity='warning', approval=None), ResponseRule(id='b', "
        "event='episode_resolved', action='optimize', severity=None, "
        "approval='human'), ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "extra": ("err", "runtime_unsupported_capability", "responses[0].<key>"),
    "rules dict": ("err", "tvl_fragment_invalid", "responses[0].rules"),
    "rules empty": ("ok", "('lad', ())"),
    "r0 del id": ("err", "tvl_fragment_invalid", "responses[0].rules[0].id"),
    "r0 bad id": (
        "ok",
        "('lad', (ResponseRule(id='zzz', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', action='optimize', "
        "severity=None, approval='human'), ResponseRule(id='c', "
        "event='episode_opened', action='open_next_step', severity=None, "
        "approval=None)))",
    ),
    "r0 none id": ("err", "tvl_fragment_invalid", "responses[0].rules[0].id"),
    "r0 set id=human": (
        "ok",
        "('lad', (ResponseRule(id='human', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 set id=critical": (
        "ok",
        "('lad', (ResponseRule(id='critical', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 del trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[0].trigger"),
    "r0 bad trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[0].trigger"),
    "r0 none trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[0].trigger"),
    "r0 set trigger=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].trigger",
    ),
    "r0 set trigger=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].trigger",
    ),
    "r0 del do": ("err", "tvl_fragment_invalid", "responses[0].rules[0].do"),
    "r0 bad do": ("err", "response_action_unsupported", "responses[0].rules[0].do"),
    "r0 none do": ("err", "response_action_unsupported", "responses[0].rules[0].do"),
    "r0 set do=human": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[0].do",
    ),
    "r0 set do=critical": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[0].do",
    ),
    "r0 del severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 bad severity": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "r0 none severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 set severity=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "r0 set severity=critical": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, "
        "approval=None)))",
    ),
    "r0 del approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 bad approval": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "r0 none approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r0 set approval=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "r0 set approval=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "r0 trig del monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].trigger.monitor",
    ),
    "r0 trig bad monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].trigger.monitor",
    ),
    "r0 trig del event": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].trigger.event",
    ),
    "r0 trig bad event": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[0].trigger.event",
    ),
    "r0 extra": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[0].<key>",
    ),
    "r0 raw int": ("err", "tvl_fragment_invalid", "responses[0].rules[0]"),
    "r0 do notify": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', action='optimize', "
        "severity=None, approval='human'), ResponseRule(id='c', "
        "event='episode_opened', action='open_next_step', severity=None, "
        "approval=None)))",
    ),
    "r0 do optimize": ("err", "tvl_fragment_invalid", "responses[0].rules[0].severity"),
    "r0 do open_next_step": ("err", "tvl_fragment_invalid", "responses[0].rules[0].do"),
    "r1 del id": ("err", "tvl_fragment_invalid", "responses[0].rules[1].id"),
    "r1 bad id": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', action='notify', "
        "severity='warning', approval=None), ResponseRule(id='zzz', "
        "event='episode_resolved', action='optimize', severity=None, "
        "approval='human'), ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 none id": ("err", "tvl_fragment_invalid", "responses[0].rules[1].id"),
    "r1 set id=human": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='human', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 set id=critical": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='critical', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 del trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[1].trigger"),
    "r1 bad trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[1].trigger"),
    "r1 none trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[1].trigger"),
    "r1 set trigger=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].trigger",
    ),
    "r1 set trigger=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].trigger",
    ),
    "r1 del do": ("err", "tvl_fragment_invalid", "responses[0].rules[1].do"),
    "r1 bad do": ("err", "response_action_unsupported", "responses[0].rules[1].do"),
    "r1 none do": ("err", "response_action_unsupported", "responses[0].rules[1].do"),
    "r1 set do=human": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[1].do",
    ),
    "r1 set do=critical": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[1].do",
    ),
    "r1 del severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 bad severity": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].severity",
    ),
    "r1 none severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 set severity=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].severity",
    ),
    "r1 set severity=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].severity",
    ),
    "r1 del approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 bad approval": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[1].approval",
    ),
    "r1 none approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 set approval=human": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r1 set approval=critical": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[1].approval",
    ),
    "r1 trig del monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].trigger.monitor",
    ),
    "r1 trig bad monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].trigger.monitor",
    ),
    "r1 trig del event": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[1].trigger.event",
    ),
    "r1 trig bad event": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[1].trigger.event",
    ),
    "r1 extra": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[1].<key>",
    ),
    "r1 raw int": ("err", "tvl_fragment_invalid", "responses[0].rules[1]"),
    "r1 do notify": ("err", "tvl_fragment_invalid", "responses[0].rules[1].approval"),
    "r1 do optimize": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', action='optimize', "
        "severity=None, approval='human'), ResponseRule(id='c', "
        "event='episode_opened', action='open_next_step', severity=None, "
        "approval=None)))",
    ),
    "r1 do open_next_step": ("err", "tvl_fragment_invalid", "responses[0].rules[1].do"),
    "r2 del id": ("err", "tvl_fragment_invalid", "responses[0].rules[2].id"),
    "r2 bad id": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', action='notify', "
        "severity='warning', approval=None), ResponseRule(id='b', "
        "event='episode_resolved', action='optimize', severity=None, "
        "approval='human'), ResponseRule(id='zzz', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 none id": ("err", "tvl_fragment_invalid", "responses[0].rules[2].id"),
    "r2 set id=human": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='human', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 set id=critical": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='critical', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 del trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[2].trigger"),
    "r2 bad trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[2].trigger"),
    "r2 none trigger": ("err", "tvl_fragment_invalid", "responses[0].rules[2].trigger"),
    "r2 set trigger=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].trigger",
    ),
    "r2 set trigger=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].trigger",
    ),
    "r2 del do": ("err", "tvl_fragment_invalid", "responses[0].rules[2].do"),
    "r2 bad do": ("err", "response_action_unsupported", "responses[0].rules[2].do"),
    "r2 none do": ("err", "response_action_unsupported", "responses[0].rules[2].do"),
    "r2 set do=human": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[2].do",
    ),
    "r2 set do=critical": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[2].do",
    ),
    "r2 del severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 bad severity": ("err", "tvl_fragment_invalid", "responses[0].rules[2].do"),
    "r2 none severity": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 set severity=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].do",
    ),
    "r2 set severity=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].do",
    ),
    "r2 del approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 bad approval": ("err", "tvl_fragment_invalid", "responses[0].rules[2].do"),
    "r2 none approval": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "r2 set approval=human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].do",
    ),
    "r2 set approval=critical": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].do",
    ),
    "r2 trig del monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].trigger.monitor",
    ),
    "r2 trig bad monitor": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].trigger.monitor",
    ),
    "r2 trig del event": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[2].trigger.event",
    ),
    "r2 trig bad event": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[2].trigger.event",
    ),
    "r2 extra": (
        "err",
        "runtime_unsupported_capability",
        "responses[0].rules[2].<key>",
    ),
    "r2 raw int": ("err", "tvl_fragment_invalid", "responses[0].rules[2]"),
    "r2 do notify": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', action='optimize', "
        "severity=None, approval='human'), ResponseRule(id='c', "
        "event='episode_opened', action='notify', severity='critical', "
        "approval=None)))",
    ),
    "r2 do optimize": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', action='optimize', "
        "severity=None, approval='human'), ResponseRule(id='c', "
        "event='episode_opened', action='optimize', severity=None, "
        "approval='human')))",
    ),
    "r2 do open_next_step": (
        "ok",
        "('lad', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None), "
        "ResponseRule(id='b', event='episode_resolved', "
        "action='optimize', severity=None, approval='human'), "
        "ResponseRule(id='c', event='episode_opened', "
        "action='open_next_step', severity=None, approval=None)))",
    ),
    "combo notify - -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None),))",
    ),
    "combo notify - None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None),))",
    ),
    "combo notify - human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify - zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify None -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None),))",
    ),
    "combo notify None None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None),))",
    ),
    "combo notify None human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify None zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify info -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='info', approval=None),))",
    ),
    "combo notify info None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='info', approval=None),))",
    ),
    "combo notify info human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify info zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify warning -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', approval=None),))",
    ),
    "combo notify warning None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='warning', "
        "approval=None),))",
    ),
    "combo notify warning human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify warning zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify critical -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', approval=None),))",
    ),
    "combo notify critical None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='notify', severity='critical', "
        "approval=None),))",
    ),
    "combo notify critical human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify critical zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].approval",
    ),
    "combo notify zzz -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo notify zzz None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo notify zzz human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo notify zzz zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize - -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, approval='human'),))",
    ),
    "combo optimize - None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, approval='human'),))",
    ),
    "combo optimize - human": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, approval='human'),))",
    ),
    "combo optimize - zzz": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[0].approval",
    ),
    "combo optimize None -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, approval='human'),))",
    ),
    "combo optimize None None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, approval='human'),))",
    ),
    "combo optimize None human": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='optimize', severity=None, "
        "approval='human'),))",
    ),
    "combo optimize None zzz": (
        "err",
        "response_action_unsupported",
        "responses[0].rules[0].approval",
    ),
    "combo optimize info -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize info None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize info human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize info zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize warning -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize warning None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize warning human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize warning zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize critical -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize critical None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize critical human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize critical zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize zzz -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize zzz None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize zzz human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo optimize zzz zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].severity",
    ),
    "combo open_next_step - -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='open_next_step', severity=None, "
        "approval=None),))",
    ),
    "combo open_next_step - None": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='open_next_step', severity=None, "
        "approval=None),))",
    ),
    "combo open_next_step - human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step - zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step None -": (
        "ok",
        "('l', (ResponseRule(id='a', event='episode_opened', "
        "action='open_next_step', severity=None, "
        "approval=None),))",
    ),
    "combo open_next_step None None": (
        "ok",
        "('l', (ResponseRule(id='a', "
        "event='episode_opened', action='open_next_step', "
        "severity=None, approval=None),))",
    ),
    "combo open_next_step None human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step None zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step info -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step info None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step info human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step info zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step warning -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step warning None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step warning human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step warning zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step critical -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step critical None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step critical human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step critical zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step zzz -": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step zzz None": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step zzz human": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "combo open_next_step zzz zzz": (
        "err",
        "tvl_fragment_invalid",
        "responses[0].rules[0].do",
    ),
    "order two bad": ("err", "response_action_unsupported", "responses[0].rules[0].do"),
}


@pytest.mark.parametrize("label", list(MONITOR_EXPECTED))
def test_parse_monitor_matrix(label):
    assert out[label] == MONITOR_EXPECTED[label]


@pytest.mark.parametrize("label", list(RESPONSES_EXPECTED))
def test_parse_responses_matrix(label):
    assert res[label] == RESPONSES_EXPECTED[label]


def test_matrix_covers_same_labels():
    assert set(out) == set(MONITOR_EXPECTED)
    assert set(res) == set(RESPONSES_EXPECTED)
