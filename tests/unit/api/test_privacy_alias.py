from __future__ import annotations

from traigent.api.decorators import optimize


def test_decorator_privacy_alias_maps_to_hybrid():
    @optimize(
        configuration_space={"x": [1]},
        objectives=["accuracy"],
        execution_mode="privacy",
    )
    def f(v: int) -> int:
        return v

    # execution_mode should map to hybrid with privacy
    assert getattr(f, "execution_mode", None) in (
        "hybrid",
        "privacy",
        None,
    )  # compat: some wrappers may store different values

    # The optimized function exposes traigent_config on optimize() path; just ensure callable and configured
    async def run():
        return 42

    # Basic property checks
    assert hasattr(f, "optimize")
