"""Environment handling of the egress audit driver (scripts/security/behaviour_audit.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "security"))

import behaviour_audit  # noqa: E402


def _child_env(monkeypatch: pytest.MonkeyPatch, **proxy: str) -> dict[str, str]:
    for name in ("NO_PROXY", "no_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in proxy.items():
        monkeypatch.setenv(name, value)
    return behaviour_audit.child_env(backend_url=None, pins=False, extra=None)


def test_star_spelling_stays_star_while_other_spelling_gets_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _child_env(
        monkeypatch,
        NO_PROXY="*",
        no_proxy="internal.example",
        HTTPS_PROXY="http://proxy.example:3128",
    )
    assert env["NO_PROXY"] == "*"
    assert "127.0.0.1" in env["no_proxy"].split(",")
    assert "internal.example" in env["no_proxy"].split(",")


def test_single_spelling_is_extended_and_both_unset_gets_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _child_env(monkeypatch, no_proxy="a.example")
    assert env["no_proxy"].split(",") == ["a.example", "127.0.0.1"]
    assert "127.0.0.1" in env["NO_PROXY"].split(",")
    env = _child_env(monkeypatch)
    assert env["NO_PROXY"] == env["no_proxy"] == "127.0.0.1"


def test_probe_leaves_the_workload_bytecode_cache_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "pycache"
    workload_env = dict(
        behaviour_audit.child_env(
            backend_url=None,
            pins=False,
            extra={"PYTHONPYCACHEPREFIX": str(cache)},
        )
    )
    behaviour_audit.resolve_litellm_pricing_url(workload_env, str(tmp_path))
    assert not cache.exists() or not any(cache.rglob("*"))


def test_uppercase_only_no_proxy_is_mirrored_to_both_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _child_env(monkeypatch, NO_PROXY="corp.example")
    assert env["NO_PROXY"] == env["no_proxy"] == "corp.example,127.0.0.1"


def test_lowercase_only_no_proxy_is_mirrored_to_both_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _child_env(monkeypatch, no_proxy="corp.example")
    assert env["NO_PROXY"] == env["no_proxy"] == "corp.example,127.0.0.1"


def test_uppercase_only_star_stays_star_in_both_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _child_env(monkeypatch, NO_PROXY="*")
    assert env["NO_PROXY"] == env["no_proxy"] == "*"


_MIRROR_URL = "https://mirror.example:8443/prices.json"


def _report(**over: object) -> dict:
    rep: dict = {
        "workload": "w",
        "pins": False,
        "backend": "stub",
        "block_egress": True,
        "exit": "ok",
        "returncode": 0,
        "wall_clock_s": 0.1,
        "litellm_version": "1.111.0",
        "stub_backend_requests": [],
        "events": [],
        "pricing_listener_url": "http://127.0.0.1:1/x",
        "pricing_listener_requests": [],
        "litellm_default_pricing_url": _MIRROR_URL,
        "litellm_load_record": {
            "source": "remote",
            "url": _MIRROR_URL,
            "is_env_forced": False,
            "fallback_reason": f"failed fetching {_MIRROR_URL}: boom",
        },
    }
    rep.update(over)
    return rep


_LISTENER_URL = "http://127.0.0.1:1/x"
_HEADER_BLOCKED = "egress blocked after recording;"
_HEADER_PARTIAL = (
    "egress blocked for Python-level connections only; "
    "see the native-fetch lines below;"
)
_L_LISTENER = "- the loopback listener received a pricing request"
_L_RECORD = "- litellm reported a pricing fetch attempt"
_L_NOT_REDIRECTED = (
    "- litellm reported a pricing fetch attempt at a different URL "
    "from the configured loopback listener"
)
_NO_RECORD = {"source": "local", "url": None, "is_env_forced": True}
_REDIRECTED_RECORD = {
    "source": "remote",
    "url": _LISTENER_URL,
    "is_env_forced": False,
}
_ELSEWHERE_RECORD = {"source": "remote", "url": _MIRROR_URL, "is_env_forced": False}

_MATRIX = [
    # id, overrides, header, exact detail lines
    (
        "allowed",
        {
            "block_egress": False,
            "pricing_listener_url": None,
            "litellm_load_record": _NO_RECORD,
        },
        "egress allowed;",
        [],
    ),
    (
        "allowed-with-load-record",
        {
            "block_egress": False,
            "pricing_listener_url": None,
            "litellm_load_record": _ELSEWHERE_RECORD,
        },
        "egress allowed;",
        [_L_RECORD + " to host `mirror.example`"],
    ),
    ("blocked-no-witness", {"litellm_load_record": _NO_RECORD}, _HEADER_BLOCKED, []),
    (
        "listener-only",
        {"litellm_load_record": _NO_RECORD, "pricing_listener_requests": ["/x"]},
        _HEADER_PARTIAL,
        [_L_LISTENER],
    ),
    (
        "record-only-redirected",
        {"litellm_load_record": _REDIRECTED_RECORD},
        _HEADER_PARTIAL,
        [_L_RECORD + " to host `127.0.0.1`"],
    ),
    (
        "record-only-not-redirected",
        {"litellm_load_record": _ELSEWHERE_RECORD},
        _HEADER_PARTIAL,
        [_L_RECORD + " to host `mirror.example`", _L_NOT_REDIRECTED],
    ),
    (
        "both-witnesses",
        {
            "litellm_load_record": _REDIRECTED_RECORD,
            "pricing_listener_requests": ["/x"],
        },
        _HEADER_PARTIAL,
        [_L_LISTENER, _L_RECORD + " to host `127.0.0.1`"],
    ),
    (
        "both-and-not-redirected",
        {"litellm_load_record": _ELSEWHERE_RECORD, "pricing_listener_requests": ["/x"]},
        _HEADER_PARTIAL,
        [_L_LISTENER, _L_RECORD + " to host `mirror.example`", _L_NOT_REDIRECTED],
    ),
]


@pytest.mark.parametrize(
    ("overrides", "header", "details"),
    [pytest.param(o, h, d, id=i) for i, o, h, d in _MATRIX],
)
def test_markdown_wording_claims_only_what_each_witness_shows(
    overrides: dict, header: str, details: list[str]
) -> None:
    rep = _report(**overrides)
    rep["summary"] = behaviour_audit.summarize(rep)
    lines = behaviour_audit.to_markdown([rep]).splitlines()
    assert header in lines[2]
    end = next(i for i, ln in enumerate(lines) if ln.startswith("- stub backend"))
    assert lines[3:end] == details  # the complete detail section
    assert "observed" not in "\n".join(lines[2:end])
