"""Golden characterization for tvl_fragment._parse_monitor.

A mutation matrix over a fully-populated monitor; the expected table was recorded from the pre-refactor implementation.
"""

from __future__ import annotations

import copy

import pytest

from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.tvl_fragment import _parse_monitor

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


@pytest.mark.parametrize("label", list(MONITOR_EXPECTED))
def test_parse_monitor_matrix(label):
    assert out[label] == MONITOR_EXPECTED[label]


def test_matrix_covers_same_labels():
    assert set(out) == set(MONITOR_EXPECTED)
