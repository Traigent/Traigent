"""Hosted-subset compile against the pinned TVL compiler (needs Python >= 3.12 + the extra)."""

from __future__ import annotations

import json

import pytest
import yaml

from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.tvl_fragment import (
    compile_full_policy,
    compile_hosted_subset,
    load_fragment,
    parse_fragment,
    token_entries,
)

from .conftest import FIXTURES

pytest.importorskip(
    "tvl.runtime_compiler", reason="needs the runtime-tvl extra (Python >= 3.12)"
)
pytest.importorskip("scipy")


def test_compiled_contract_matches_checked_in_fixture() -> None:
    fixture = json.loads((FIXTURES / "hosted_contract.json").read_text())
    req = load_fragment(FIXTURES / "latency.tvl.yml", max_fix_cost_usd=5.0)
    contract, _local, issues = compile_hosted_subset(
        req, fixture["token_map"], fixture["module"]
    )
    assert contract == fixture["contract"]
    assert [i["code"] for i in issues] == [
        "runtime_liveness_missing"
    ]  # expected warning
    # Responses are uploaded with the contract (they used to be stripped to []).
    rules = contract["declarations"]["responses"][0]["rules"]
    assert [r["do"] for r in rules] == [
        "notify",
        "open_next_step",
        "optimize",
        "notify",
    ]
    # No 1.4 construct: still /1, 1.3, and the 1.3.0 toolchain stamp.
    assert (contract["format"], contract["tvl_version"]) == (
        "tvl-runtime-contract/1",
        "1.3",
    )
    assert contract["toolchain"]["version"] == "1.3.0"


def test_full_policy_lints_with_only_the_liveness_warning() -> None:
    fixture = json.loads((FIXTURES / "hosted_contract.json").read_text())
    req = load_fragment(FIXTURES / "latency.tvl.yml", max_fix_cost_usd=5.0)
    codes = [
        i["code"]
        for i in compile_full_policy(req, fixture["token_map"], fixture["module"])
    ]
    assert codes == ["runtime_liveness_missing"]


def _tokens(req, module: str) -> dict:
    return {
        "format": "tvl-token-map/1",
        "tokens": [
            {"namespace": ns, "local": local, "hosted": f"t{ns}{i:04d}"}
            for i, (ns, local) in enumerate(token_entries(req, module))
        ],
    }


def test_reactive_fragment_compiles_to_the_v2_contract() -> None:
    req = load_fragment(FIXTURES / "reactive_fragment.yml", max_fix_cost_usd=5.0)
    contract, local, issues = compile_hosted_subset(req, _tokens(req, "m"), "m")
    assert (contract["format"], contract["tvl_version"]) == (
        "tvl-runtime-contract/2",
        "1.4",
    )
    assert local["format"] == "tvl-runtime-local/2"
    assert contract["toolchain"]["version"] == "1.4.0"
    assert [i["code"] for i in issues] == ["runtime_liveness_missing"]
    rules = contract["declarations"]["responses"][0]["rules"]
    triggers = [r["trigger"] for r in rules]
    # Outcome triggers name the hosted ref of the optimize rule, never its local id.
    assert {t["event"] for t in triggers if "rule" in t} == {
        "optimize_no_candidate",
        "optimize_failed",
        "deploy_refused",
        "deploy_failed",
    }
    assert any(t.get("event") == "unknown_for" and t["hours"] == 12 for t in triggers)
    wire = json.dumps(contract)
    for local_name in ("fix", "undecided", "latency_2s_prod", "latency_repair"):
        assert f'"{local_name}"' not in wire
    assert [m["phase"] for m in contract["declarations"]["monitors"]] == [
        "runtime",
        "promotion",
    ]
    assert any(
        r.get("deploy") == "if_promoted" and r["do"] == "optimize" for r in rules
    )


def test_the_compiler_refuses_an_unless_guard_on_its_own_monitor() -> None:
    doc = yaml.safe_load((FIXTURES / "reactive_fragment.yml").read_text())
    rule = next(r for r in doc["responses"][0]["rules"] if r["id"] == "undecided")
    rule["unless"] = [{"monitor": "latency_2s_prod", "state": "episode_open"}]
    req = parse_fragment(doc, max_fix_cost_usd=5.0)
    with pytest.raises(RuntimeRequirementError) as err:
        compile_hosted_subset(req, _tokens(req, "m"), "m")
    assert err.value.code == "runtime_guard_cycle"


def test_the_compiler_warns_about_a_redundant_only_while_guard() -> None:
    doc = yaml.safe_load((FIXTURES / "reactive_fragment.yml").read_text())
    rule = next(r for r in doc["responses"][0]["rules"] if r["id"] == "undecided")
    rule["only_while"] = [{"monitor": "latency_2s_prod", "state": "episode_open"}]
    req = parse_fragment(doc, max_fix_cost_usd=5.0)
    codes = [i["code"] for i in compile_full_policy(req, _tokens(req, "m"), "m")]
    assert "runtime_guard_redundant" in codes


def test_unknown_for_hours_equal_to_the_window_is_accepted() -> None:
    """The boundary of RFC 0004 section 2.3: hours == tumbling_hours is in range."""
    doc = yaml.safe_load((FIXTURES / "reactive_fragment.yml").read_text())
    rule = next(r for r in doc["responses"][0]["rules"] if r["id"] == "undecided")
    rule["trigger"]["hours"] = 6  # == tumbling_hours: allowed
    req = parse_fragment(doc, max_fix_cost_usd=5.0)
    contract, _local, _issues = compile_hosted_subset(req, _tokens(req, "m"), "m")
    assert contract["format"] == "tvl-runtime-contract/2"
