"""Head-resolving config loader: the EXACT version the agent head pins.

Two other reads exist and are wrong for this purpose: the config GET returns the
*current* version of a best config, and the listing omits specs.  The head can pin an
older version, so a process that wants "what is deployed" reads the head's pinned
version and its generation in ONE request (``GET .../agent-heads/{agent}/config``;
ETag = generation).  The result is customer content and stays on the customer side.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.runtime.hosted import PinnedConfig, ReactiveClient
from traigent.runtime.model import E_BACKEND, RuntimeRequirementError

E_HEAD_MOVED = "head_generation_changed"


def load_pinned_config(
    client: ReactiveClient, *, expected_generation: int | None = None
) -> PinnedConfig:
    """Read the pinned version.  With ``expected_generation`` a moved head is refused."""
    pinned = client.get_pinned_config()
    if expected_generation is not None and pinned.generation != expected_generation:
        raise RuntimeRequirementError(E_HEAD_MOVED)
    return pinned


def config_values(pinned: PinnedConfig) -> Mapping[str, Any]:
    """The knob values of the pinned spec (``spec["config"]``), fail closed if absent."""
    values = pinned.spec.get("config")
    if not isinstance(values, dict):
        raise RuntimeRequirementError(E_BACKEND, "response")
    return values
