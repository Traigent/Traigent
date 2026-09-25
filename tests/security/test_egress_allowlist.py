"""Egress allowlist audit for the Python SDK.

Runs real SDK workloads in fresh interpreters under the PEP 578 recorder
(scripts/security/behaviour_audit.py) and asserts that every host the SDK
tries to reach is on an explicit allowlist. Nothing leaves the machine: the
backend is a loopback stub, LLM calls are mocked, and every non-loopback DNS
lookup / connect is recorded and then refused by the recorder.

Not part of the PR gate: the tests are marked ``egress_audit`` and the
default ``-m`` selection in pyproject.toml excludes that marker. Run with::

    pytest tests/security/test_egress_allowlist.py -m egress_audit -n 0

A new host showing up here is a customer-facing change (firewall rules, proxy
allowlists, EDR review), so adding one to the allowlist must be a reviewed
decision, recorded in docs/security/network-and-behaviour-manifest.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.egress_audit, pytest.mark.security]

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "security"))

import behaviour_audit  # noqa: E402

#: First-party hosts the SDK may contact with its BUILT-IN backend setting.
#: Only the ``backend="default"`` runs see these; stub runs must see none.
FIRST_PARTY_HOSTS = frozenset({"portal.traigent.ai"})

#: Third-party hosts the SDK is known to reach unless the documented pins
#: (behaviour_audit.NETWORK_PINS) are set. They are NOT allowed; they are
#: listed so the xfail cases below can name them.
PINNABLE_THIRD_PARTY_HOSTS = frozenset({"raw.githubusercontent.com", "huggingface.co"})

#: Workloads whose run should reach no host other than the loopback stub.
STUB_WORKLOADS = (
    "optimize_mock",
    "optimize_seamless",
    "import_litellm_first",
    "token_count_llama",
)

#: Owner ruling 2026-09-25: the SDK pins litellm's bundled price table and
#: Anthropic beta-header config on every ``import traigent``; opt out with
#: TRAIGENT_LITELLM_LIVE_PRICES=1. Implemented on a separate branch.
_LITELLM_PIN_FIX = (
    "fixed by fix/litellm-local-cost-map-by-default (owner ruling 2026-09-25): "
    "today a default run reaches raw.githubusercontent.com for litellm's price "
    "table"
)
_LITELLM_FIRST_RESIDUAL = (
    "documented residual risk: when the application imports litellm BEFORE "
    "traigent, litellm fetches its price table at its own import time, before "
    "the SDK can set any pin; only the customer's environment "
    "(LITELLM_LOCAL_MODEL_COST_MAP=True) prevents it"
)
_HF_PENDING = (
    "known finding, no owner ruling yet: pricing a Llama-family model through "
    "litellm.token_counter downloads a tokenizer from huggingface.co unless "
    "HF_HUB_OFFLINE=1 (8 blocked attempts, ~25 s added latency when the host "
    "is unreachable)"
)

_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


def _run(
    workload: str,
    *,
    pins: bool,
    backend: str = "stub",
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    key = (workload, pins, backend, tuple(sorted((extra_env or {}).items())))
    if key not in _cache:
        _cache[key] = behaviour_audit.run_workload(
            workload, pins=pins, backend=backend, extra_env=extra_env, timeout=600
        )
    report = _cache[key]
    assert report["exit"] == "ok", (
        f"workload {workload} did not complete: {report.get('error')}\n"
        f"{report['stderr_tail']}"
    )
    return report


def _unexpected(report: dict[str, Any], allowed: frozenset[str]) -> set[str]:
    return behaviour_audit.external_hosts(report) - allowed


@pytest.mark.parametrize("workload", STUB_WORKLOADS)
def test_pinned_run_reaches_only_the_loopback_stub(workload: str) -> None:
    report = _run(workload, pins=True)
    assert _unexpected(report, frozenset()) == set(), behaviour_audit.to_markdown(
        [report]
    )
    hosts = set(report["summary"]["hosts"])
    assert not hosts & PINNABLE_THIRD_PARTY_HOSTS


@pytest.mark.parametrize("workload", STUB_WORKLOADS)
def test_pinned_run_spawns_no_process_and_loads_no_foreign_library(
    workload: str,
) -> None:
    report = _run(workload, pins=True)
    assert report["summary"]["processes"] == []
    # ``ctypes.dlopen: None`` is ``import ctypes`` opening the running process
    # itself (ctypes/__init__.py: pythonapi = PyDLL(None)); any other library
    # load is new behaviour for the manifest.
    assert set(report["summary"]["native"]) <= {"ctypes.dlopen: None"}
    assert not any("chmod +x" in w for w in report["summary"]["file_writes"])


def test_keyless_quickstart_reaches_no_external_host() -> None:
    # The quickstart sets LITELLM_LOCAL_MODEL_COST_MAP itself
    # (traigent/__init__.py quickstart bootstrap), so no pins are passed here.
    report = _run("quickstart", pins=False)
    assert _unexpected(report, frozenset()) == set(), behaviour_audit.to_markdown(
        [report]
    )


def test_default_backend_reaches_only_first_party_hosts_when_pinned() -> None:
    report = _run("optimize_mock", pins=True, backend="default")
    assert _unexpected(report, FIRST_PARTY_HOSTS) == set(), behaviour_audit.to_markdown(
        [report]
    )


@pytest.mark.xfail(strict=True, reason=_LITELLM_PIN_FIX)
def test_default_run_reaches_no_third_party_host() -> None:
    report = _run("optimize_mock", pins=False)
    assert _unexpected(report, frozenset()) == set()


@pytest.mark.xfail(strict=True, reason=_LITELLM_FIRST_RESIDUAL)
def test_litellm_imported_first_reaches_no_third_party_host() -> None:
    report = _run("import_litellm_first", pins=False)
    assert _unexpected(report, frozenset()) == set()


def test_live_prices_opt_out_reaches_github_only() -> None:
    """TRAIGENT_LITELLM_LIVE_PRICES=1 is the documented opt-out: the ONE
    extra host it may add is litellm's price-table origin."""
    report = _run(
        "optimize_mock",
        pins=False,
        extra_env={"TRAIGENT_LITELLM_LIVE_PRICES": "1"},
    )
    assert behaviour_audit.external_hosts(report) == {"raw.githubusercontent.com"}


@pytest.mark.xfail(strict=True, reason=_HF_PENDING)
def test_default_llama_token_count_reaches_no_third_party_host() -> None:
    report = _run("token_count_llama", pins=False)
    assert _unexpected(report, frozenset()) == set()


def test_known_finding_is_exactly_the_documented_host() -> None:
    """Pin the shape of the xfails: they must fail for the documented reason
    and nothing else, or a new host would hide behind an expected failure."""
    for workload in ("optimize_mock", "import_litellm_first", "token_count_llama"):
        report = _run(workload, pins=False)
        assert behaviour_audit.external_hosts(report) <= PINNABLE_THIRD_PARTY_HOSTS


def test_recorder_sees_and_blocks_a_new_host() -> None:
    """Sensitivity control: an unlisted host must make the allowlist fail."""
    report = _run("canary_new_host", pins=True)
    assert _unexpected(report, frozenset()) == {"egress-canary.invalid"}
    assert any("egress-canary.invalid" in b for b in report["blocked"])
