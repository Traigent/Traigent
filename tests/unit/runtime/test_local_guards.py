"""Local guards (``block_action`` / ``halt_unit``) against the SDK-owned parity vectors.

The vectors in ``tests/fixtures/runtime/local_guard_vectors.json`` are derived from
RFC 0003 section 7.5 and formal semantics Definition 12.6/12.8 (each case cites its
sections).  The JS SDK vendors the same file byte-exact; ``SOURCE.md`` pins its sha256.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from traigent.runtime.guards import (
    LocalGuardError,
    LocalGuardSet,
    current_unit,
    guarded,
    rules_from_local,
)
from traigent.runtime.model import RuntimeRequirementError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "runtime"
VECTORS = FIXTURES / "local_guard_vectors.json"
DOC = json.loads(VECTORS.read_text())
PATTERNS = ("absence", "bounded_existence", "precedence", "protocol")


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    return [c["id"] for c in cases]


@pytest.mark.parametrize("case", DOC["cases"], ids=_ids(DOC["cases"]))
def test_parity_vector(case: dict[str, Any]) -> None:
    unit = LocalGuardSet.from_local(case["local"]).new_unit(**case["unit"])
    for index, step in enumerate(case["steps"]):
        got = unit.check(step["event"]).as_dict()
        want = step["expect"]
        assert {k: got[k] for k in want} == want, f"step {index}"


@pytest.mark.parametrize("case", DOC["load_cases"], ids=_ids(DOC["load_cases"]))
def test_load_vector(case: dict[str, Any]) -> None:
    if case["expect_error"] is None:
        assert LocalGuardSet.from_local(case["local"]).rules
        return
    with pytest.raises(RuntimeRequirementError) as err:
        rules_from_local(case["local"])
    assert err.value.code == case["expect_error"]


def test_vectors_cover_the_plan_matrix() -> None:
    """Per pattern: a violating prefix, a clean prefix, undecidable under both
    ``on_check_error`` values, and an out-of-population unit never acted on."""
    for pattern in PATTERNS:
        mine = [
            c
            for c in DOC["cases"]
            if c["local"]["declarations"]["requirements"][0]["violation"]["pattern"]
            == pattern
        ]
        reasons = {
            (s["expect"]["verdict"], s["expect"]["reason"])
            for c in mine
            for s in c["steps"]
        }
        assert ("block", "violation_established") in reasons or (
            "halt",
            "violation_established",
        ) in reasons, pattern
        assert any(
            r in reasons for r in (("allow", "no_violation"), ("allow", "not_relevant"))
        )
        assert ("block", "check_error_block") in reasons, pattern
        assert ("allow", "check_error_allow") in reasons, pattern
        assert ("allow", "not_in_population") in reasons, pattern


def test_every_case_cites_the_rfc() -> None:
    for case in (*DOC["cases"], *DOC["load_cases"]):
        assert case["cites"] and all(
            c.startswith(("RFC0003 §", "FS Def ", "FS §")) for c in case["cites"]
        ), case["id"]


def test_vector_file_matches_its_source_note() -> None:
    note = (FIXTURES / "SOURCE.md").read_text()
    sha = re.search(r"local_guard_vectors\.json sha256: ([0-9a-f]{64})", note)
    assert sha, "SOURCE.md must pin the vectors' sha256"
    assert sha.group(1) == hashlib.sha256(VECTORS.read_bytes()).hexdigest()


# -- behaviour that is not a parity vector ---------------------------------------------------


def _charge_set() -> LocalGuardSet:
    case = next(
        c for c in DOC["cases"] if c["id"] == "precedence/authorized-charge-is-allowed"
    )
    return LocalGuardSet.from_local(case["local"])


def test_decision_carries_no_event_values() -> None:
    unit = LocalGuardSet.from_local(DOC["cases"][0]["local"]).new_unit()
    decision = unit.check(
        {
            "type": "tool_call",
            "tool": "shell.exec",
            "producer": "sdk",
            "sequence": 0,
            "correlation_id": "secret-customer-id",
        }
    )
    assert "secret-customer-id" not in json.dumps(decision.as_dict())


def test_malformed_event_is_a_closed_error_without_echo() -> None:
    unit = LocalGuardSet.from_local(DOC["cases"][0]["local"]).new_unit()
    with pytest.raises(RuntimeRequirementError) as err:
        unit.check({"type": "SECRET-TYPE", "producer": "p", "sequence": 0})
    assert err.value.code == "guard_event_invalid" and "SECRET" not in str(err.value)


def test_guards_need_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local guards keep working when remote authority is unreachable (section 7.5)."""
    import socket

    def refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a local guard opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    unit = _charge_set().new_unit()
    assert (
        unit.check(
            {
                "type": "tool_call",
                "tool": "payments.charge",
                "producer": "sdk",
                "sequence": 0,
                "correlation_id": "c",
            }
        ).verdict
        == "block"
    )


def _absence_set(action: str = "block_action") -> LocalGuardSet:
    wanted = (
        "absence/violating-event-is-blocked"
        if action == "block_action"
        else "absence/halt-unit-stops-the-unit"
    )
    return LocalGuardSet.from_local(
        next(c for c in DOC["cases"] if c["id"] == wanted)["local"]
    )


def test_decorator_blocks_before_the_effect() -> None:
    ran: list[str] = []

    @guarded(tool="shell.exec")
    def run_shell() -> None:
        ran.append("shell")

    @guarded(tool="fs.read")
    def read_file() -> str:
        ran.append("read")
        return "ok"

    guards = _absence_set()
    with guards.unit() as unit:
        assert read_file() == "ok"
        with pytest.raises(LocalGuardError) as err:
            run_shell()
        assert err.value.code == "local_guard_blocked"
        assert err.value.decision.reason == "violation_established"
        assert read_file() == "ok"  # block_action leaves the unit running
        assert not unit.halted
    assert ran == ["read", "read"]  # the blocked body never ran
    assert current_unit() is None


def test_decorator_halt_stops_every_later_call() -> None:
    @guarded(tool="shell.exec")
    def run_shell() -> None: ...

    @guarded(tool="fs.read")
    def read_file() -> None: ...

    with _absence_set("halt_unit").unit() as unit:
        with pytest.raises(LocalGuardError) as first:
            run_shell()
        assert first.value.code == "local_guard_halted"
        with pytest.raises(LocalGuardError) as later:
            read_file()
        assert later.value.decision.reason == "unit_halted"
        assert unit.halted


def test_decorator_without_a_unit_fails_closed() -> None:
    @guarded(tool="fs.read")
    def read_file() -> None: ...

    with pytest.raises(RuntimeRequirementError) as err:
        read_file()
    assert err.value.code == "guard_unit_missing"


def test_decorator_records_results_and_correlation_for_precedence() -> None:
    @guarded(tool="payments.charge")
    def charge() -> str:
        return "charged"

    guards = _charge_set()
    with guards.unit() as unit:
        with unit.correlated("c1"):
            unit.observe(
                {
                    "type": "authorization_granted",
                    "ref": "payment_auth",
                    "producer": "sdk",
                    "sequence": unit.next_sequence(),
                    "correlation_id": "c1",
                }
            )
            assert charge() == "charged"
            with pytest.raises(LocalGuardError):  # single_use: the grant is spent
                charge()


def test_decorator_handles_async_and_failures() -> None:
    @guarded(tool="fs.read")
    async def read_async() -> str:
        return "async-ok"

    @guarded(tool="fs.read")
    def boom() -> None:
        raise ValueError("body failed")

    with _absence_set().unit() as unit:
        assert asyncio.run(read_async()) == "async-ok"
        with pytest.raises(ValueError, match="body failed"):
            boom()
        assert not unit.halted


def test_decorator_refuses_a_name_that_is_not_a_code_identifier() -> None:
    with pytest.raises(RuntimeRequirementError) as err:

        @guarded()
        def local_fn() -> None: ...  # qualname contains "<locals>"

    assert err.value.code == "guard_tool_not_code_identifier"
