"""Regression tests for fix round 4 (fail-closed lock, exact numbers, token-map lock, guard order)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import traigent
from traigent.runtime import manifest
from traigent.runtime.apply import _equal, profile_diff
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.respond import respond

from .conftest import AGENT
from .test_reactive_invariants import _kw


def test_missing_flock_fails_closed_before_the_protected_section(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    monkeypatch.setattr(manifest, "fcntl", None)
    entered: list[bool] = []
    with pytest.raises(RuntimeRequirementError) as err:
        with manifest._ledger_lock(state_dir, state_dir / "x.json"):
            entered.append(True)
    assert err.value.code == "state_lock_unavailable"
    assert entered == []
    with pytest.raises(RuntimeRequirementError) as err2:
        manifest.ensure_tokens(state_dir, [("requirement", "a")])
    assert err2.value.code == "state_lock_unavailable"
    assert not (state_dir / "tvl-token-map.json").exists()
    with pytest.raises(RuntimeRequirementError) as err3:
        manifest.reserve_occurrence(state_dir, AGENT, "b|fix|0", {"run_id": None})
    assert err3.value.code == "state_lock_unavailable"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (1e-12, 1e-14, False),
        (1e-10, 1e-12, False),
        (1e-12, 1e-12, True),
        (1, 1.0, True),
        (0.05, 0.05000000001, False),
    ],
)
def test_number_equality_is_exact_after_canonical_normalisation(
    a: float, b: float, same: bool
) -> None:
    assert _equal(a, b) is same
    req = SimpleNamespace(profile=lambda: {"false_alarm_budget": a})
    assert (profile_diff(req, {"false_alarm_budget": b}) == []) is same  # type: ignore[arg-type]


def test_concurrent_token_minting_loses_no_mapping(state_dir: Path) -> None:
    manifest.check_state_dir(
        state_dir, create=True
    )  # marker race is separate from this test
    workers = 12
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def mint(i: int) -> None:
        try:
            barrier.wait(10)
            manifest.ensure_tokens(state_dir, [("requirement", f"name{i}")])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=mint, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    doc = manifest.ensure_tokens(state_dir, [], persist=False)
    locals_ = {t["local"] for t in doc["tokens"]}
    assert locals_ == {f"name{i}" for i in range(workers)}


async def test_non_optimizable_function_consumes_nothing(state_dir: Path) -> None:
    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="fix", max_fix_cost_usd=2.0
    )
    def plain() -> None: ...

    assert not hasattr(plain, "optimize")
    asked: list[object] = []
    kwargs = _kw(state_dir, confirm=lambda f: asked.append(f) or True)
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(plain, plain.requirements[0], **kwargs)
    assert err.value.code == "function_not_optimizable"
    assert asked == []  # no approval requested
    assert not list(state_dir.rglob("*occurr*")) if state_dir.exists() else True
