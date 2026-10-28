"""Local guards read the artifact the REAL TVL compiler produces (RFC 0003 section 15.3)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from traigent.runtime.guards import LocalGuardSet
from traigent.runtime.model import RuntimeRequirementError

compiler = pytest.importorskip(
    "tvl.runtime_compiler", reason="needs the runtime-tvl extra (Python >= 3.12)"
)

MODULE = """
tvl: {module: guardmod}
tvl_version: "%s"
environment: {snapshot_id: "2026-10-16T00:00:00Z"}
evaluation_set: {dataset: "traigent://runtime"}
tvars: [{name: runtime_placeholder, type: "enum[str]", domain: [a, b]}]
objectives: [{name: quality, direction: maximize}]
promotion_policy: {dominance: epsilon_pareto, alpha: 0.05, min_effect: {quality: 0.02}}
requirements:
  - name: payment_needs_authorization
    kind: zero_tolerance
    unit: invocation
    violation: { pattern: precedence, a: { event: authorization_granted, ref: payment_auth },
                 b: { event: tool_call, tool: payments.charge }, mode: single_use, correlate_by: correlation_id }
monitors:
  - name: payment_needs_authorization_prod
    requirement: payment_needs_authorization
    phase: runtime
    evidence: { source: customer_side, deployment_environment: production }
    window: { tumbling_hours: 24, grace_minutes: 60 }
    decision: { method: exact_count_v1 }
    episode: { closure_windows: 1, closure_max_age_hours: 48 }
    limits: { max_active_obligations: 64, max_tracked_correlations: 1024 }
responses:
  - name: payment_guard
    rules:
      - { id: block_charge, trigger: { monitor: payment_needs_authorization_prod, event: unit_violation },
          do: block_action, on_check_error: block }
      - { id: page, trigger: { monitor: payment_needs_authorization_prod, event: episode_opened },
          do: notify, severity: critical }
%s"""

REACTIVE_RULE = """      - { id: undecided, trigger: { monitor: payment_needs_authorization_prod, event: unknown_for, hours: 24 },
          do: notify, severity: warning }
"""


def _compile(version: str, extra: str = "") -> dict[str, Any]:
    doc = yaml.safe_load(MODULE % (version, extra))
    names = [
        ("module", "guardmod"),
        ("requirement", "payment_needs_authorization"),
        ("monitor", "payment_needs_authorization_prod"),
        ("environment", "production"),
        ("response", "payment_guard"),
        ("rule", "block_charge"),
        ("rule", "page"),
        ("selector_ref", "payment_auth"),
    ]
    if extra:
        names.append(("rule", "undecided"))
    token_map = {
        "format": "tvl-token-map/1",
        "tokens": [
            {"namespace": ns, "local": local, "hosted": f"t{ns}{i:04d}"}
            for i, (ns, local) in enumerate(names)
        ],
    }
    _contract, local, _issues = compiler.compile_runtime_document(
        doc, token_map_doc=token_map
    )
    return dict(local)


@pytest.mark.parametrize(
    ("version", "extra", "fmt"),
    [
        ("1.3", "", "tvl-runtime-local/1"),
        ("1.4", REACTIVE_RULE, "tvl-runtime-local/2"),
    ],
    ids=["local-v1", "local-v2"],
)
def test_guard_reads_the_compiled_local_artifact(
    version: str, extra: str, fmt: str
) -> None:
    local = _compile(version, extra)
    assert local["format"] == fmt
    assert [g["action"] for g in local["guards"]] == ["block_action"]
    unit = LocalGuardSet.from_local(local).new_unit()
    charge = {
        "type": "tool_call",
        "tool": "payments.charge",
        "producer": "sdk",
        "correlation_id": "c1",
    }
    decision = unit.check({**charge, "sequence": 0})
    assert (decision.verdict, decision.reason) == ("block", "violation_established")
    unit.observe(
        {
            "type": "authorization_granted",
            "ref": "payment_auth",
            "producer": "sdk",
            "sequence": 1,
            "correlation_id": "c1",
        }
    )
    assert unit.check({**charge, "sequence": 2}).verdict == "allow"
    assert unit.check({**charge, "sequence": 3}).verdict == "block"  # single_use


def test_the_local_artifact_declares_the_guard_and_its_monitor() -> None:
    """The local artifact names the guard, its on_check_error and the monitor it guards."""
    local = _compile("1.3")
    assert local["guards"][0]["on_check_error"] == "block"
    assert "guards" in local and local["customer_monitors"] == [
        "payment_needs_authorization_prod"
    ]


def test_an_artifact_without_guards_yields_an_empty_set() -> None:
    local = _compile("1.3")
    local["guards"] = []
    unit = LocalGuardSet.from_local(local).new_unit()
    assert (
        unit.check(
            {"type": "tool_call", "tool": "x.y", "producer": "p", "sequence": 0}
        ).verdict
        == "allow"
    )


def test_a_non_enforceable_pattern_is_refused_at_load() -> None:
    local = _compile("1.3")
    local["declarations"]["requirements"][0]["violation"] = {
        "pattern": "bounded_response",
        "a": {"event": "tool_call", "tool": "x.y"},
        "b": {"event": "tool_result", "tool": "x.y"},
        "within_ms": 10,
    }
    with pytest.raises(RuntimeRequirementError) as err:
        LocalGuardSet.from_local(local)
    assert err.value.code == "guard_not_enforceable"
