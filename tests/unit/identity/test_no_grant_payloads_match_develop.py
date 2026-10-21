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
``external_id``). Regenerate only from a develop commit, never from a feature branch: in that
worktree, write ``json.dumps(build_snapshot(), indent=1, sort_keys=True)`` to
the fixture path.

Exception, 2026-10-17: the fixture was regenerated from the honest-cost branch
(PR #2615), because honest cost accounting (#2517) deliberately reports an
unmeasured cost instead of $0. The delta from the ``bad5953b`` snapshot is
limited to the cost keys removed (``cost``, ``total_cost``, ``input_cost``,
``output_cost``, ``cost_per_example_mean``, the ``*_cost`` entries of
``per_metric_coverage``, and ``total_example_cost`` in ``metadata_keys``) plus
``cost_unmeasured`` added. After #2615 merges, regenerate only from a develop
commit again.

Exception, 2026-10-20: SDK #1906 — ``backend_metadata`` input/output/total_tokens
are now per-trial totals (sum over examples), not per-example means, matching the
per-trial cost total the backend reads. Owner-approved via decision card
product-2026-10-20-sdk-1906-token-totals-golden-v2.
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


def test_no_grant_payloads_are_identical_to_develop(monkeypatch) -> None:
    # The keyed per-example outcome signals (example_digest / output_digest /
    # signal_key_id) are emitted only when a project API key is configured, so
    # their PRESENCE would otherwise depend on the machine. The fixture was
    # captured with them present; pin a fixed dummy key so every environment
    # matches (their values are normalized to "<hex>" by the snapshot).
    from traigent.utils import outcome_signals

    monkeypatch.setattr(
        outcome_signals,
        "_resolve_signal_key",
        lambda: outcome_signals._cached_key_pair("no-grant-snapshot-fixed-key"),
    )
    clear_content_identity_keys()
    assert get_content_identity_keys() is None
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = build_snapshot()
    assert actual == expected
    text = json.dumps(actual)
    for marker in ("content_identity", "external_id", "example_version", "ex1:"):
        assert marker not in text, marker
