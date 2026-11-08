"""Validation for user-supplied seed configurations (issue #1665).

``optimize(seed_configs=[...])`` lets a caller start a search from configs they
already trust. Seeds are customer input, so every error here is value-free: it
names the seed's position and the *declared* parameter names, never a
client-supplied key or value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from traigent.core.comparison import _value_in_domain


def validate_seed_configs(
    seed_configs: Sequence[Mapping[str, Any]] | None,
    configuration_space: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return validated seed configs (``[]`` when none were given).

    Each seed must be a dict that names every declared parameter exactly and
    holds only in-space values; duplicates are rejected because they would
    spend trials re-evaluating the same point.

    Raises:
        ValueError: A seed is malformed, outside the space, or duplicated.
    """
    if seed_configs is None:
        return []
    if isinstance(seed_configs, (str, bytes)) or not isinstance(seed_configs, Sequence):
        raise ValueError("seed_configs must be a list of config dicts.")
    if not seed_configs:
        return []
    if not configuration_space:
        raise ValueError(
            "seed_configs requires a declared configuration space to validate "
            "the seeds against."
        )
    declared = sorted(configuration_space)
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, seed in enumerate(seed_configs):
        if not isinstance(seed, Mapping):
            raise ValueError(f"Seed config #{index} must be a dict.")
        extra = [key for key in seed if key not in configuration_space]
        if extra:
            raise ValueError(
                f"Seed config #{index} has {len(extra)} parameter(s) outside "
                f"the declared configuration space (declared: {declared})."
            )
        missing = [name for name in declared if name not in seed]
        if missing:
            raise ValueError(
                f"Seed config #{index} is not fully specified; missing "
                f"declared parameter(s): {missing}."
            )
        bad = [
            name
            for name in declared
            if not _value_in_domain(configuration_space[name], seed[name])
        ]
        if bad:
            raise ValueError(
                f"Seed config #{index} has value(s) outside the declared "
                f"configuration space for parameter(s): {bad}."
            )
        key = json.dumps({n: seed[n] for n in declared}, sort_keys=True, default=repr)
        if key in seen:
            raise ValueError(f"Seed config #{index} duplicates an earlier seed.")
        seen.add(key)
        cleaned.append({name: seed[name] for name in declared})
    return cleaned
