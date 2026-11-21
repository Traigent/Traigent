"""route_shape: the one function that turns a path into a loggable shape."""

from __future__ import annotations

import pytest

from traigent.egress.log import route_shape


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        ("/api/v1/projects", "/api/v1/projects"),
        ("/api/v1beta/projects/", "/api/v1beta/projects/"),
        (
            "/api/v2/sessions/3f2b8c1e-aaaa-bbbb-cccc-123456789abc/attempts",
            "/api/v2/sessions/{}/attempts",
        ),
        ("/api/v1/projects/12345", "/api/v1/projects/{}"),
        ("/api/v1/projects/Alice_has_diabetes", "/api/v1/projects/{}"),
        ("/api/v1/projects/MixedCase", "/api/v1/projects/{}"),
        ("/api/v1/projects/abc123", "/api/v1/projects/{}"),
        ("/api/v1/projects/a" + "b" * 80, "/api/v1/projects/{}"),
        ("/api/v1/projects/%61lice", "/api/v1/projects/{}"),
        ("/api/v1/optimization/plan", "/api/v1/optimization/plan"),
        ("/api/v1/projects?token=abc#frag", "/api/v1/projects"),
        ("/", "/"),
        ("", "/"),
        ("/v1/projects/v10alpha", "/v1/projects/v10alpha"),
        ("/vX/1v", "/{}/{}"),
    ],
)
def test_route_shape(path, shape):
    assert route_shape(path) == shape


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        ("/api/v1/projects/alice_has_diabetes", "/api/v1/projects/{}"),
        ("/api/v1/projects/alice", "/api/v1/projects/{}"),
        (
            "/api/v1/sessions/my-secret-experiment/attempts",
            "/api/v1/sessions/{}/attempts",
        ),
        ("/api/v1/projects/" + "x" * 40, "/api/v1/projects/{}"),
        ("/api/v1/example-sets/alice/examples", "/api/v1/example-sets/{}/examples"),
        ("/api/v1beta/projects", "/api/v1beta/projects"),
        ("/api/v1/optimization/plan", "/api/v1/optimization/plan"),
        ("/api/v1/best-config-head", "/api/v1/best-config-head"),
    ],
)
def test_route_shape_keeps_only_allowlisted_words(path, shape):
    assert route_shape(path) == shape


def test_a_lowercase_canary_in_a_path_is_never_logged(tmp_path):
    from traigent.egress.log import RelayLog

    canary = "canary_alice_has_diabetes"
    log = RelayLog(tmp_path / "relay.jsonl")
    try:
        seq = log.log_request("GET", f"/api/v1/projects/{canary}/sessions", 0, "0" * 64)
        log.log_response(seq, 200, 0)
    finally:
        log.close()
    text = (tmp_path / "relay.jsonl").read_text()
    assert canary not in text
    assert "/api/v1/projects/{}/sessions" in text
    assert '"route_source":"allowlist"' in text


def test_logs_written_with_the_old_route_source_are_still_readable(tmp_path):
    import json

    from traigent.egress.log import iter_lines

    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seq": 1,
                "ts": "2026-01-01T00:00:00Z",
                "kind": "request",
                "method": "GET",
                "route": "/api/v1/projects",
                "route_source": "heuristic",
                "bytes": 0,
                "sha256": "0" * 64,
            }
        )
        + "\n"
    )
    assert [e["route"] for e in iter_lines(path)] == ["/api/v1/projects"]


def test_every_static_sdk_route_word_is_in_the_allowlist():
    from pathlib import Path

    import traigent
    from tests.unit.egress.route_words_scan import scan_route_words
    from traigent.egress.route_words import ROUTE_WORDS

    used = scan_route_words(Path(traigent.__file__).parent)
    assert used, "the scan found no route words: the scanner is broken"
    missing = {w: where for w, where in used.items() if w not in ROUTE_WORDS}
    assert not missing, (
        "static route words used by the SDK but missing from "
        f"traigent/egress/route_words.py: {sorted(missing)} (first use: {missing})"
    )


def test_allowlist_is_a_frozenset_of_plain_words():
    import re

    from traigent.egress.route_words import ROUTE_WORDS

    assert isinstance(ROUTE_WORDS, frozenset)
    assert all(re.fullmatch(r"[a-z_-]+", w) for w in ROUTE_WORDS)
