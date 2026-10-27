"""The GitHub store gates its own requests session; an injected one is left as-is."""

from __future__ import annotations

import requests

from traigent.snapshot.backends import GitHubStore


def test_default_session_is_gated_and_injected_session_untouched() -> None:
    default = GitHubStore("owner/repo", "main")
    assert type(default._session)._traigent_gated is True
    assert default._session._traigent_oo_external is True

    injected = requests.Session()
    store = GitHubStore("owner/repo", "main", session=injected)
    assert store._session is injected
    assert not getattr(type(injected), "_traigent_gated", False)
