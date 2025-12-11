"""SEAMLESS injection mode coverage for CostEnforcer binding."""

from __future__ import annotations

import pytest

from traigent.api.decorators import optimize
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig


def _assert_invariants_hold() -> None:
    """Sanity-check CostEnforcer invariants."""
    enforcer = CostEnforcer(CostEnforcerConfig(limit=1.0))
    permit = enforcer.acquire_permit()
    if permit.is_granted:
        enforcer.track_cost(0.05, permit=permit)
    assert not enforcer._verify_invariants(), "CostEnforcer invariants violated"


@optimize(
    configuration_space={"model": ["gpt-4o-mini"]},
    objectives=["accuracy"],
    injection={"injection_mode": "seamless"},
    mock={"enabled": True},
)
def _seamless_function(prompt: str = "test") -> str:
    """Function that should have its local model variable rewritten."""
    model = "default-model"  # noqa: F841 - will be rewritten by seamless injection
    return model  # type: ignore[return-value]


def test_seamless_injection_rewrites_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEAMLESS mode rewrites local assignments using injected config."""
    desired = "seamless-model"
    # Force safety check to allow AST transformation in tests
    from traigent.config.providers import SeamlessParameterProvider

    monkeypatch.setattr(
        SeamlessParameterProvider, "_is_safe_function", lambda *args, **kwargs: True
    )

    _seamless_function.set_config({"model": desired})

    result = _seamless_function("hello")

    assert result == desired
    _assert_invariants_hold()
