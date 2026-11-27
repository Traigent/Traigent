"""The egress-canary detectors do not watch litellm's native price fetch.

litellm 1.111 fetches its price table at import through a native client, which
the socket, aiohttp, requests, httpx and urllib patches in ``harness.py`` and
in ``test_may_send_at_send_points.py`` cannot see. That blind spot is only
acceptable while litellm can never fetch under these tests, so this file pins
that precondition: the local-cost-map pin is set before litellm can import.
The native fetch is watched by ``scripts/security/behaviour_audit.py``.
"""

from __future__ import annotations

import os


def test_litellm_cost_map_pin_is_set_before_litellm_can_import() -> None:
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
    assert os.environ.get("TRAIGENT_LITELLM_LIVE_PRICES") is None


def test_litellm_does_not_attempt_a_remote_fetch_under_the_harness() -> None:
    import litellm  # noqa: F401 - the import is what would fetch
    from litellm.litellm_core_utils.get_model_cost_map import (
        get_model_cost_map_source_info,
    )

    info = get_model_cost_map_source_info()
    assert info["is_env_forced"] is True, info
    assert info["source"] == "local", info
    assert info["url"] is None, info
