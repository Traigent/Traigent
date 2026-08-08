"""An unset ``max_trials`` must not be persisted as an invented number.

Traigent#2049.

`backend_session_manager` substituted a literal ``10`` when the caller supplied
no ``max_trials``, and stamped it into ``session_metadata`` -- where it is
indistinguishable on disk from a user who genuinely asked for 10.

For an exhaustive grid the trial count is a property of the configuration space,
not a budget, so a 24-configuration run recorded ``max_trials: 10``. A completion
ratio then renders ``24/10`` and a progress bar computed from it exceeds 100%.
The honest representation of "no budget was set" is no value, which is what the
sibling ``max_total_examples`` in the same dict already does.

`traigent/cli/local_commands.py` documented this defect as the reason it could
not show a "completed/total" ratio at all: *"There is no honest denominator to
put back: metadata['max_trials'] is fabricated upstream"*. That justification is
what this fix removes.
"""

from __future__ import annotations

from traigent.core import backend_session_manager as bsm


# --------------------------------------------------------------------------
# Behavioural: capture the metadata actually handed to the backend client.
# --------------------------------------------------------------------------


class _CapturingClient:
    def __init__(self) -> None:
        self.metadata: dict | None = None

    def create_session(self, **kwargs):
        self.metadata = kwargs.get("metadata")
        raise _StopHere()


class _StopHere(Exception):
    """Ends create_session once the metadata has been captured."""


class _Descriptor:
    identifier = "mod.fn"
    display_name = "fn"
    slug = "fn_abc123"
    module = "mod"
    relative_path = "mod.py"


class _Dataset:
    name = "ds"

    def __len__(self) -> int:
        return 3


def _capture_metadata(monkeypatch, max_trials):
    """Drive create_session far enough to capture the metadata dict."""
    manager = bsm.BackendSessionManager.__new__(bsm.BackendSessionManager)
    client = _CapturingClient()
    manager._backend_client = client
    manager._optimization_id = "opt-1"
    manager._smart_pruning = None
    manager._objectives = ["accuracy"]
    manager._traigent_config = None
    manager._optimizer = type("O", (), {"config_space": {"model": ["a"]}})()
    monkeypatch.setattr(manager, "_egress_disabled", lambda: False, raising=False)
    monkeypatch.setattr(
        manager, "_local_sequencing_active", lambda: False, raising=False
    )
    monkeypatch.setattr(bsm, "policy_from_config", lambda _cfg: None, raising=False)

    try:
        manager.create_session(
            func=lambda: None,
            dataset=_Dataset(),
            function_descriptor=_Descriptor(),
            max_trials=max_trials,
            start_time=0.0,
        )
    except _StopHere:
        pass
    return client.metadata


def test_an_unset_max_trials_is_not_persisted_as_a_number(monkeypatch):
    """THE defect: with no budget the metadata carried a fabricated 10."""
    metadata = _capture_metadata(monkeypatch, None)

    assert metadata is not None, "metadata was never built"
    assert metadata.get("max_trials") is None, (
        f"an unset max_trials was persisted as {metadata.get('max_trials')!r}; "
        f"a consumer cannot tell that from a real budget"
    )


def test_a_real_budget_is_persisted_unchanged(monkeypatch):
    """Non-vacuity: the fix must not drop a budget the caller DID set."""
    metadata = _capture_metadata(monkeypatch, 10)

    assert metadata is not None
    assert metadata.get("max_trials") == 10


def test_the_cli_records_that_the_denominator_is_no_longer_fabricated():
    """The stale justification must not outlive the defect.

    `local_commands` dropped its "completed/total" ratio partly because the
    denominator could not be trusted -- *"There is no honest denominator to put
    back: metadata['max_trials'] is fabricated upstream"*. That reason is gone
    now, and leaving it stated as current would mislead the next person who
    tries to restore the ratio.
    """
    import inspect

    from traigent.cli import local_commands

    assert "#2049" in inspect.getsource(local_commands)
