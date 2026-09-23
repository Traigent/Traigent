"""No purpose-key grant => the SDK sends exactly what develop sent (owner-delegated
M2 review decision: "no grant, emit nothing new").

``fixtures/no_grant_payloads_develop.json`` is :func:`build_snapshot` run on the
base develop commit ``bad5953b`` (the branch point of ``feat/content-identity-v1``)
with this same ``no_grant_snapshot.py`` copied into a detached worktree. It
captures, for one deterministic offline optimize() run with no grant:

* the orchestrator -> BackendSessionManager.create_session call (every kwarg
  name, every JSON-able value);
* the full body of BOTH typed session-create serializers;
* every trial's Backend metadata payload, its example results and its
  metadata keys.

Timing values, generated ids and the SDK version string are normalized; every
other key and value must be identical. On the reviewed head ``cc20b4aa`` this
comparison failed (trials carried ``content_identity`` and results carried
``external_id``). Regenerate only from a develop commit, never from this branch: in that
worktree, write ``json.dumps(build_snapshot(), indent=1, sort_keys=True)`` to
the fixture path.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.identity.no_grant_snapshot import build_snapshot
from traigent.identity.keys import (
    clear_content_identity_keys,
    get_content_identity_keys,
)

GOLDEN = Path(__file__).parent / "fixtures" / "no_grant_payloads_develop.json"


def test_no_grant_payloads_are_identical_to_develop() -> None:
    clear_content_identity_keys()
    assert get_content_identity_keys() is None
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = build_snapshot()
    assert actual == expected
    text = json.dumps(actual)
    for marker in ("content_identity", "external_id", "example_version", "ex1:"):
        assert marker not in text, marker
