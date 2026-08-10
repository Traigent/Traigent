"""The cold-start public surface stays exactly six names, never re-exported."""

from __future__ import annotations

import traigent
from traigent import generation
from traigent.generation import coldstart


def test_exports_are_closed() -> None:
    assert coldstart.__all__ == [
        "ColdStartOutcome",
        "ColdStartResult",
        "DiscoveryGap",
        "LocalVerifier",
        "ScoreReceipt",
        "build_cold_start_eval_set",
    ]
    for name in coldstart.__all__:
        assert hasattr(coldstart, name)


def test_build_cold_start_eval_set_not_promoted_to_top_level() -> None:
    assert not hasattr(traigent, "build_cold_start_eval_set")


def test_coldstart_survives_generation_package_edits() -> None:
    """coldstart must stay exported from traigent.generation.

    Regression guard: restoring guided generation (revert of 114d9386) rewrote
    traigent/generation/__init__.py wholesale and dropped coldstart from both
    the imports and __all__. The subpackage stayed importable by path, so
    nothing failed loudly -- it simply vanished from the package's public
    surface.
    """
    assert "coldstart" in generation.__all__
    assert generation.coldstart is coldstart
