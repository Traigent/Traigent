"""Sugar kwargs -> TVL (golden), refusals, and decorator placement."""

from __future__ import annotations

import pytest

import traigent
from traigent.runtime.model import RuntimeRequirementError

from .conftest import GOLDEN


def _declare(**kw):
    @traigent.requirement(**kw)
    def fn() -> None: ...

    return fn.requirements[0]


@pytest.mark.parametrize("mode", ["log", "alert", "fix"])
def test_sugar_compiles_to_golden_tvl(mode: str) -> None:
    kw = {"latency_ms": 2000, "max_violation_rate": 0.05, "on_breach": mode}
    if mode == "fix":
        kw["max_fix_cost_usd"] = 5.0
    assert _declare(**kw).to_tvl() == (GOLDEN / f"{mode}.tvl.yml").read_text()


def test_optimize_alias_is_fix() -> None:
    req = _declare(
        latency_ms=2000,
        max_violation_rate=0.05,
        on_breach="optimize",
        max_fix_cost_usd=1,
    )
    assert req.on_breach == "fix"


@pytest.mark.parametrize("mode", ["fix_and_deploy", "optimize_and_deploy"])
def test_deploy_is_refused(mode: str) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        _declare(
            latency_ms=2000, max_violation_rate=0.05, on_breach=mode, max_fix_cost_usd=1
        )
    assert err.value.code == "response_action_unsupported"


@pytest.mark.parametrize("window", ["2h", "6", "", "48h"])
def test_bad_window_refused(window: str) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        _declare(latency_ms=2000, max_violation_rate=0.05, window=window)
    assert err.value.code == "invalid_window"


def test_window_values_map_to_hours() -> None:
    for text, hours in {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}.items():
        assert (
            _declare(latency_ms=2000, max_violation_rate=0.05, window=text).window_hours
            == hours
        )


def test_fix_requires_a_ceiling() -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        _declare(latency_ms=2000, max_violation_rate=0.05, on_breach="fix")
    assert err.value.code == "fix_ceiling_required"


@pytest.mark.parametrize("ceiling", [0, -1, float("inf"), float("nan")])
def test_ceiling_must_be_finite_positive(ceiling: float) -> None:
    with pytest.raises(RuntimeRequirementError):
        _declare(
            latency_ms=2000,
            max_violation_rate=0.05,
            on_breach="fix",
            max_fix_cost_usd=ceiling,
        )


def test_standing_approval_is_refused_not_downgraded() -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        _declare(latency_ms=2000, max_violation_rate=0.05, approval="standing")
    assert err.value.code == "response_action_unsupported"


def test_kwargs_and_tvl_together_is_type_error() -> None:
    with pytest.raises(TypeError):
        traigent.requirement(latency_ms=2000, tvl="x.yml")
    with pytest.raises(TypeError):
        traigent.requirement(on_breach="log", tvl="x.yml")


def test_incomplete_and_out_of_domain_values_refused() -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        traigent.requirement(latency_ms=2000)
    assert err.value.code == "requirement_incomplete"
    for kw in ({"max_violation_rate": 1.5}, {"latency_ms": 0}, {"confidence": 0.5}):
        base = {"latency_ms": 2000, "max_violation_rate": 0.05, **kw}
        with pytest.raises(RuntimeRequirementError):
            traigent.requirement(**base)


def test_works_above_and_below_optimize() -> None:
    cfg = {
        "configuration_space": {"t": [0.1, 0.2]},
        "objectives": ["accuracy"],
        "injection_mode": "seamless",
    }

    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05)
    @traigent.optimize(eval_dataset=None, **cfg)
    def above(q: str, t: float = 0.1) -> str:
        return q

    @traigent.optimize(eval_dataset=None, **cfg)
    @traigent.requirement(latency_ms=1000, max_violation_rate=0.05)
    def below(q: str, t: float = 0.1) -> str:
        return q

    assert above.requirements[0].latency_threshold_ms == 2000
    assert below.requirements[0].latency_threshold_ms == 1000
    assert below.requirements.owner.endswith("below")


def test_declaration_is_network_free(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def boom(*_a, **_k):
        raise AssertionError("network used at decoration time")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    _declare(latency_ms=2000, max_violation_rate=0.05, on_breach="alert")


def test_second_declaration_on_one_function_refused() -> None:
    with pytest.raises(RuntimeRequirementError) as err:

        @traigent.requirement(latency_ms=1000, max_violation_rate=0.05)
        @traigent.requirement(latency_ms=2000, max_violation_rate=0.05)
        def fn() -> None: ...

    assert err.value.code == "duplicate_requirement"
