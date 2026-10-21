"""The REAL pinned TVL compiler accepts the token map this SDK writes.

The token-map format string is the compiler's contract (``tvl-token-map/1``); local
state versioning lives on ``state-owner.json`` only.  Skipped, with the reason, when
``tvl-spec`` (the ``runtime-tvl`` extra, Python >= 3.12) is not importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import traigent
from traigent.runtime import manifest
from traigent.runtime.tvl_fragment import (
    compile_full_policy,
    compile_hosted_subset,
    module_name_for,
    token_entries,
)

pytest.importorskip(
    "tvl.runtime_compiler",
    reason="tvl-spec (the runtime-tvl extra, Python >= 3.12) is not installed",
)

FIXTURES = Path(__file__).parent / "fixtures"
UNRESOLVED = "runtime_unresolved_alias"


def _decorated():
    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05)
    def fn() -> None: ...

    return fn


def _fragment():
    @traigent.requirement(tvl=str(FIXTURES / "latency.tvl.yml"), max_fix_cost_usd=5.0)
    def fn() -> None: ...

    return fn


def test_token_map_format_is_the_compilers_contract() -> None:
    assert manifest.TOKEN_MAP_FORMAT == "tvl-token-map/1"
    assert manifest.STATE_VERSION == 2  # versioning stays on the marker


@pytest.mark.parametrize("make", [_decorated, _fragment], ids=["decorator", "fragment"])
def test_real_compiler_resolves_tokens(make, tmp_path: Path) -> None:
    fn = make()
    req = fn.requirements[0]
    module = module_name_for(fn.requirements.owner)
    token_map = manifest.ensure_tokens(
        tmp_path / "state", token_entries(req, module), persist=False
    )
    contract, _local, _issues = compile_hosted_subset(req, token_map, module)
    assert contract
    compile_full_policy(req, token_map, module)
    assert not [i for i in fn.requirements.validate() if i.code == UNRESOLVED]
