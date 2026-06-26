from __future__ import annotations

import pytest

from traigent.api.decorators import optimize


def test_decorator_privacy_alias_maps_to_cloud_brain_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

    with pytest.warns(DeprecationWarning):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="privacy",
        )
        def f(v: int) -> int:
            return v

    assert f.execution_policy.intent.value == "cloud_brain"
    assert f.execution_policy.offline is False
    assert getattr(f, "execution_mode", None) == "hybrid"

    # The optimized function exposes traigent_config on optimize() path; just ensure callable and configured
    async def run():
        return 42

    # Basic property checks
    assert hasattr(f, "optimize")
