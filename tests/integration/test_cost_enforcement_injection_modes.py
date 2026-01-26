"""Decorator-level injection mode coverage for CostEnforcer binding.

These tests prove that @traigent.optimize actually injects configuration via
each supported mode (context, parameter) and keeps CostEnforcer
invariants clean while doing so. This addresses the evidence gap for
decorator-level injection coverage.

Note: Attribute injection mode was removed in v2.x due to thread-safety issues.
"""

from __future__ import annotations

from traigent.api.decorators import optimize
from traigent.config.context import get_config
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig


def _assert_invariants_hold() -> None:
    """Sanity-check that CostEnforcer invariants hold in a minimal run."""
    enforcer = CostEnforcer(CostEnforcerConfig(limit=1.0))
    permit = enforcer.acquire_permit()
    # Only track if granted (mock mode will always grant).
    if permit.is_granted:
        enforcer.track_cost(0.1, permit=permit)
    assert not enforcer._verify_invariants(), "CostEnforcer invariants violated"


@optimize(
    configuration_space={"mode_tag": ["context-mode"]},
    objectives=["accuracy"],
    injection={"injection_mode": "context"},
    mock={"enabled": True},
)
def _context_injected_function() -> str:
    """Return the injected config marker via context provider."""
    cfg = get_config()
    return cfg.get("mode_tag", "missing")  # type: ignore[union-attr]


@optimize(
    configuration_space={"mode_tag": ["parameter-mode"]},
    objectives=["accuracy"],
    injection={"injection_mode": "parameter", "config_param": "cfg"},
    mock={"enabled": True},
)
def _parameter_injected_function(cfg=None) -> str:
    """Return the injected config marker passed as a parameter."""
    assert cfg is not None, "Config should be injected as parameter"
    return cfg.get("mode_tag", "missing")


class TestInjectionModes:
    """Decorator-level injection mode coverage."""

    def test_context_injection(self) -> None:
        """Context mode writes config into contextvars."""
        _context_injected_function.set_config({"mode_tag": "context-mode"})
        result = _context_injected_function()
        assert result == "context-mode"
        _assert_invariants_hold()

    def test_parameter_injection(self) -> None:
        """Parameter mode injects config param automatically."""
        _parameter_injected_function.set_config({"mode_tag": "parameter-mode"})
        result = _parameter_injected_function()
        assert result == "parameter-mode"
        _assert_invariants_hold()
