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
