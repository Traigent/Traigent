"""Hosted-subset compile against the pinned TVL compiler (needs Python >= 3.12 + the extra)."""

from __future__ import annotations

import json

import pytest

from traigent.runtime.tvl_fragment import (
    compile_full_policy,
    compile_hosted_subset,
    load_fragment,
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
    assert contract["declarations"]["responses"] == []


def test_full_policy_lints_with_only_the_liveness_warning() -> None:
    fixture = json.loads((FIXTURES / "hosted_contract.json").read_text())
    req = load_fragment(FIXTURES / "latency.tvl.yml", max_fix_cost_usd=5.0)
    codes = [
        i["code"]
        for i in compile_full_policy(req, fixture["token_map"], fixture["module"])
    ]
    assert codes == ["runtime_liveness_missing"]
