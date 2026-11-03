"""Upload sends the compiled contract, responses included (real compiler, fake Backend)."""

from __future__ import annotations

import json

import pytest

import traigent
from traigent.runtime.apply import apply_requirement

from .conftest import AGENT, FIXTURES, PROJECT, FakeBackend, head

pytestmark = pytest.mark.backend_online

pytest.importorskip(
    "tvl.runtime_compiler", reason="needs the runtime-tvl extra (Python >= 3.12)"
)
pytest.importorskip("scipy")


def _apply(backend: FakeBackend, fragment: str) -> dict:
    @traigent.requirement(tvl=str(FIXTURES / fragment), max_fix_cost_usd=3.0)
    def fn() -> None: ...

    backend.on("POST", "/production-requirements/tvl-import", 201, head())
    apply_requirement(fn.requirements[0], project_id=PROJECT, agent_id=AGENT)
    return backend.requests[0]["body"]


def test_upload_carries_the_responses_of_a_13_fragment(backend: FakeBackend) -> None:
    body = _apply(backend, "latency.tvl.yml")
    contract = body["contract"]
    assert contract["format"] == "tvl-runtime-contract/1"
    assert len(contract["declarations"]["responses"][0]["rules"]) == 4


def test_backend_refusal_of_responses_propagates_closed(backend: FakeBackend) -> None:
    """Mocked fail-closed propagation: a 422 ``response_action_unsupported`` reaches the caller.

    The fake Backend refuses any non-empty ``responses`` the way TraigentBackend
    ``develop`` ``tvl_import.py:326`` does today (``_single_requirement_and_monitor``).
    This test claims only that the SDK surfaces such a refusal as that closed code and
    does not swallow it; it makes no claim about any deployed Backend.
    """
    from traigent.runtime.model import RuntimeRequirementError

    def strict(body):
        responses = body["contract"]["declarations"]["responses"]
        if responses != []:
            return 422, {"error_code": "response_action_unsupported"}
        return 201, head()

    backend.routes[("POST", "/production-requirements/tvl-import")] = strict

    @traigent.requirement(tvl=str(FIXTURES / "latency.tvl.yml"), max_fix_cost_usd=3.0)
    def fn() -> None: ...

    with pytest.raises(RuntimeRequirementError) as err:
        apply_requirement(fn.requirements[0], project_id=PROJECT, agent_id=AGENT)
    assert err.value.code == "response_action_unsupported"


def test_upload_of_a_reactive_fragment_is_the_v2_contract(backend: FakeBackend) -> None:
    body = _apply(backend, "reactive_fragment.yml")
    contract = body["contract"]
    assert (contract["format"], contract["tvl_version"]) == (
        "tvl-runtime-contract/2",
        "1.4",
    )
    wire = json.dumps(body)
    for local in ("fix_found_nothing", "latency_repair", "latency_2s_prod"):
        assert f'"{local}"' not in wire
