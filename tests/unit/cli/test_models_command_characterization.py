"""Characterization of the ``traigent models`` command output matrix."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import traigent.cli.main as main
from traigent.cli.main import cli


@pytest.fixture
def env(monkeypatch):
    state = {"ids": ["m1", "m2"], "known": True, "verdict": "live", "live": ["m1"]}
    calls = {"verdict": [], "discover": [], "printed": []}
    console = MagicMock(name="console")

    def discover(provider, refresh, region):
        calls["discover"].append((provider, refresh, region))
        return state["ids"]

    def verdict(provider, model_id, valid, live_ids):
        calls["verdict"].append((provider, model_id, valid, live_ids))
        return state["verdict"]

    monkeypatch.setattr(main, "_discover_model_ids", discover)
    monkeypatch.setattr(main, "_is_known_model_id", lambda p, m, ids: state["known"])
    monkeypatch.setattr(main, "_model_verdict", verdict)
    monkeypatch.setattr(main, "_live_ids_for", lambda p: state["live"])
    monkeypatch.setattr(
        main, "_print_model_ids", lambda p, r, ids: calls["printed"].append((p, r, ids))
    )
    monkeypatch.setattr(main, "console", console)
    return state, calls, console


def _invoke(*args):
    return CliRunner().invoke(cli, ["models", *args])


def _text(console):
    return "\n".join(str(c.args[0]) for c in console.print.call_args_list)


def test_list_only_no_check_prints_table_and_exits_zero(env):
    _state, calls, console = env
    r = _invoke("--provider", "OpenAI")
    assert r.exit_code == 0
    assert calls["discover"] == [("openai", False, None)]
    assert calls["printed"] == [("openai", None, ["m1", "m2"])]
    assert calls["verdict"] == []
    console.print.assert_not_called()


def test_json_without_check(env):
    r = _invoke("--provider", "openai", "--json", "--refresh")
    assert r.exit_code == 0
    assert json.loads(r.output) == {
        "provider": "openai",
        "region": None,
        "model": None,
        "valid": None,
        "verdict": None,
        "models": ["m1", "m2"],
        "note": None,
    }


@pytest.mark.parametrize(
    "verdict,valid,note_is_live,code",
    [
        ("live", True, False, 0),
        ("snapshot-only", True, True, 0),
        ("not-live", False, False, 1),
        ("unknown-shape", False, False, 1),
    ],
)
def test_json_with_check(env, verdict, valid, note_is_live, code):
    state, calls, _ = env
    state["verdict"] = verdict
    r = _invoke("--provider", "openai", "--check", "m1", "--json")
    assert r.exit_code == code
    payload = json.loads(r.output)
    assert payload["valid"] is valid and payload["verdict"] == verdict
    assert (payload["note"] == main._LIVE_NOTE) is note_is_live
    assert calls["verdict"] == [("openai", "m1", True, ["m1"])]


def test_json_bedrock_region_and_note(env):
    state, calls, _ = env
    state["verdict"] = "live"
    r = _invoke(
        "--provider", "bedrock", "--region", "us-east-1", "--check", "m1", "--json"
    )
    payload = json.loads(r.output)
    assert r.exit_code == 0
    assert payload["region"] == "us-east-1"
    assert payload["note"].startswith("Bedrock validation lists regional")
    assert calls["discover"] == [("bedrock", False, "us-east-1")]


def test_json_region_dropped_for_non_bedrock(env):
    r = _invoke("--provider", "openai", "--region", "eu", "--json")
    assert json.loads(r.output)["region"] is None


def test_text_live_pass(env):
    state, _, console = env
    r = _invoke("--provider", "openai", "--check", "m1")
    assert r.exit_code == 0
    out = _text(console)
    assert "Model preflight passed (live):[/green] m1" in out
    assert "Snapshot/shape" not in out


def test_text_snapshot_only_pass_has_yellow_note(env):
    state, _, console = env
    state["verdict"] = "snapshot-only"
    r = _invoke("--provider", "openai", "--check", "m1")
    assert r.exit_code == 0
    out = _text(console)
    assert "passed (snapshot-only)" in out and main._LIVE_NOTE in out


def test_text_not_live_fails_with_retired_message(env):
    state, _, console = env
    state["verdict"] = "not-live"
    r = _invoke("--provider", "openai", "--check", "gone")
    assert r.exit_code == 1
    out = _text(console)
    assert "is not in the live openai catalog (retired or unavailable" in out
    assert "known openai model surface" not in out


def test_text_unknown_shape_fails_with_pick_hint(env):
    state, _, console = env
    state["verdict"] = "unknown-shape"
    r = _invoke("--provider", "openai", "--check", "banana")
    assert r.exit_code == 1
    out = _text(console)
    assert "is not in the known openai model surface" in out
    assert "Pick a model from the table above" in out


def test_text_bedrock_adds_dim_note_even_without_check(env):
    _, _, console = env
    r = _invoke("--provider", "bedrock")
    assert r.exit_code == 0
    assert "Bedrock preflight lists regional foundation models only" in _text(console)


def test_bedrock_pass_message_then_note_order(env):
    _, _, console = env
    r = _invoke("--provider", "bedrock", "--check", "m1")
    assert r.exit_code == 0
    out = _text(console)
    assert out.index("preflight passed") < out.index("Bedrock preflight lists")
