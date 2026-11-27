"""The egress-canary detectors do not watch litellm's native price fetch.

litellm 1.111 fetches its price table at import through a native client, which
the socket, aiohttp, requests, httpx and urllib patches in ``harness.py`` and
in ``test_may_send_at_send_points.py`` cannot see. That blind spot is only
acceptable while litellm can never fetch under these tests, so this file pins
that precondition: the local-cost-map pin is set before litellm can import.
Both detectors enforce it themselves at start (``require_litellm_local_cost_map``).
The native fetch is watched by ``scripts/security/behaviour_audit.py``.
"""

from __future__ import annotations

import os

import pytest

from .harness import capture_egress, require_litellm_local_cost_map


def test_litellm_cost_map_pin_is_set_before_litellm_can_import() -> None:
    # Checked before any litellm import, so a missing pin goes red without a fetch.
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
    assert os.environ.get("TRAIGENT_LITELLM_LIVE_PRICES") is None
    import litellm  # noqa: F401 - the import is what would fetch
    from litellm.litellm_core_utils.get_model_cost_map import (
        get_model_cost_map_source_info,
    )

    info = get_model_cost_map_source_info()
    assert info["is_env_forced"] is True, info
    assert info["source"] == "local", info
    assert info["url"] is None, info


@pytest.mark.parametrize("value", ["False", "", None])
def test_capture_refuses_to_start_without_the_pin(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    else:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", value)
    with pytest.raises(AssertionError, match="LITELLM_LOCAL_MODEL_COST_MAP"):
        require_litellm_local_cost_map()
    with pytest.raises(AssertionError, match="LITELLM_LOCAL_MODEL_COST_MAP"):
        with capture_egress():
            pass
