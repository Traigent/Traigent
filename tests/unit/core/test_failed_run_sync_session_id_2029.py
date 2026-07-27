"""Regression tests for #2029: a run that RAISES must still name its sync id.

#2020 put ``sync_session_id`` on ``OptimizationResult`` — but a run that fails
mid-flight never returns a result. The trials that DID complete are already
durably on disk (trials persist per-trial via ``add_trial_result``, not at
finalize), and ``traigent sync --all`` skips failed sessions, so those trials
were strandable with no supported way to name them.

The contract pinned here: the syncable local session id rides along on the
raised exception as ``sync_session_id`` — on ``OptimizationError`` and on the
typed ``ResolutionError`` alike — obtained via the SAME locality + durability
predicate the success path uses (``syncable_local_session_id``). It is ``None``
when there is no readable local record.

Why this file exists instead of new cases in ``tests/unit/core/test_orchestrator.py``:
``pyproject.toml``'s ``addopts`` carries ``--ignore=tests/unit/core/test_orchestrator.py``
(it hangs locally), so anything added there does not run by default. These tests
must run by default — they are the acceptance evidence for #2029.
"""

from __future__ import annotations

from typing import Any

import pytest

import traigent
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core import orchestrator as orchestrator_module
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.execution_policy_runtime import CloudBrainUnavailableError
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.knobs import ResolutionError
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.error_handler import APIKeyError
from traigent.utils.exceptions import CostLimitExceeded, OptimizationError

_SPACE = {"x": ["a", "b"]}


def _dataset() -> Dataset:
    return Dataset(
        [EvaluationExample({"text": "case-0"}, "ok")],
        name="failed_run_sync_session_id_2029",
    )


def _isolated_env(monkeypatch, tmp_path) -> None:
    """No key, no network, no LLM spend, and a private results folder.

    Copied deliberately from ``test_result_sync_session_id_2020.py`` rather than
    imported: this harness is as much the thing under test as the assertions
    are, and a cross-module import would let a change over there silently
    re-point these tests at a different environment.

    Offline mode is explicitly turned OFF. ``tests/conftest.py``'s autouse
    ``jwt_development_mode`` fixture forces ``TRAIGENT_OFFLINE_MODE=true`` for
    everything outside ``tests/unit/cloud/``, which would reroute these no-key
    runs through the offline branch instead of the local-fallback branch they
    exercise.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


def _capture_predicate_input(monkeypatch) -> list[str | None]:
    """Record every ``session_id`` the #2020 predicate is asked about.

    Gives the tests the run's real session id without reaching into
    orchestrator internals, so "the id on the exception is THIS run's session
    id" is assertable rather than assumed.
    """
    seen: list[str | None] = []
    original = BackendSessionManager.syncable_local_session_id

    def spy(self, session_id: str | None) -> str | None:
        seen.append(session_id)
        return original(self, session_id)

    monkeypatch.setattr(BackendSessionManager, "syncable_local_session_id", spy)
    return seen


def _optimized(func):
    """Wrap ``func`` as a real ``@traigent.optimize`` grid run.

    The evaluated function must be ``async``: a sync function is executed in a
    thread pool (``_execute_sync_in_thread``), which captures its exception and
    hands back an error string, so nothing would ever propagate out of the run
    and the test would silently prove nothing. Custom knobs arrive on
    ``config.custom_params`` — ``x`` is not a ``TraigentConfig`` field.
    """
    return traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
    )(func)


# ---------------------------------------------------------------------------
# T1 — headline regression: a REAL failing run whose id `SyncManager` accepts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_run_exposes_a_sync_id_the_sync_manager_accepts(
    monkeypatch, tmp_path
) -> None:
    """The issue repro, end to end. Nothing mocked: not the optimization loop,
    not the finalizer, not the storage.

    ``APIKeyError`` is load-bearing, not an arbitrary "some exception":

    * It is a ``TraigentError`` (``utils/error_handler.py``), which
      ``_execute_function`` re-raises explicitly instead of folding into a
      failed ``ExampleResult``. A plain ``RuntimeError`` there becomes a merely
      failed example and the run COMPLETES normally — proving nothing.
    * ``TrialLifecycleManager`` re-raises it specifically to stop the whole run.
    * It is NOT an ``OptimizationError``, so it exercises the *wrap* branch —
      the shape where the object the caller catches is one the orchestrator
      constructed, and therefore the one an attach can most easily miss.

    Trial 1 ('a') is durably on disk before trial 2 ('b') raises, because trials
    persist per-trial rather than at finalize. Those stranded-but-complete
    trials are exactly what the id exists to let the caller upload.
    """
    _isolated_env(monkeypatch, tmp_path)
    seen_sessions = _capture_predicate_input(monkeypatch)

    @_optimized
    async def answer(text: str, config=None) -> str:
        if config.custom_params.get("x") == "b":
            raise APIKeyError("OPENAI_API_KEY")
        return "ok"

    with pytest.raises(OptimizationError, match="Optimization failed") as excinfo:
        await answer.optimize(algorithm="grid")

    # The wrap contract is intact: a new OptimizationError chained to the cause.
    assert isinstance(excinfo.value.__cause__, APIKeyError)

    # Pre-fix this attribute did not exist (AttributeError) — the whole issue.
    sync_id = excinfo.value.sync_session_id
    assert isinstance(sync_id, str) and sync_id

    # It is THIS run's session, not some other record in the store.
    assert sync_id in seen_sessions

    # The id names the single record in the isolated store, and that record
    # went terminal as "failed" — the state `traigent sync --all` skips.
    store = tmp_path / "results"
    session_files = list((store / "sessions").glob("*.json"))
    assert len(session_files) == 1
    assert session_files[0].stem == sync_id

    storage = LocalStorageManager(str(store))
    session = storage.load_session(sync_id)
    assert session is not None
    assert session.status == "failed"
    # The stranded completed trial really is on disk, mid-flight raise and all.
    assert len(session.trials) == 1

    # … and `traigent sync <id>` accepts it, carrying that trial with it.
    sync = SyncManager(TraigentConfig.from_environment())
    outcome = sync.sync_session_to_cloud(sync_id, dry_run=True)
    assert outcome["status"] == "success"
    assert outcome["trials_converted"] >= 1


# ---------------------------------------------------------------------------
# T3-T8 — contract tests. The optimization loop is mocked to raise a chosen
# exception type; the real orchestrator, session creation, local store and
# finalizers still run around it. The real-run criterion is already covered by
# T1 here and by the strict-e2e ResolutionError case (T2, in
# tests/integration/test_knobs_strict_e2e.py).
# ---------------------------------------------------------------------------


def _raise_from_loop(monkeypatch, exc: BaseException) -> None:
    """Make the optimization loop fail with exactly ``exc``."""

    async def boom(self, *args: Any, **kwargs: Any) -> int:
        raise exc

    monkeypatch.setattr(OptimizationOrchestrator, "_run_optimization_loop", boom)


async def _run_failing(monkeypatch, exc: BaseException) -> BaseException:
    """Drive a real run whose loop raises ``exc``; return the escaped exception."""
    _raise_from_loop(monkeypatch, exc)

    @_optimized
    async def answer(text: str, config=None) -> str:
        return "ok"

    with pytest.raises(BaseException) as excinfo:
        await answer.optimize(algorithm="grid")
    return excinfo.value


@pytest.mark.asyncio
async def test_original_optimization_error_is_reraised_and_carries_the_id(
    monkeypatch, tmp_path
) -> None:
    """T3: an ``OptimizationError`` from the loop is re-raised as-is.

    Identity matters: the object (and its traceback) is the caller's, so the id
    must be attached to it rather than to a replacement.
    """
    _isolated_env(monkeypatch, tmp_path)
    seen_sessions = _capture_predicate_input(monkeypatch)

    original = OptimizationError("optimizer boom")
    raised = await _run_failing(monkeypatch, original)

    assert raised is original
    assert isinstance(raised.sync_session_id, str) and raised.sync_session_id
    assert raised.sync_session_id in seen_sessions


@pytest.mark.asyncio
async def test_generic_exception_is_wrapped_and_the_wrapper_carries_the_id(
    monkeypatch, tmp_path
) -> None:
    """T4: a non-Traigent exception is wrapped — the id goes on the WRAPPER.

    The caller only ever sees the wrapper, so an id left on the original (still
    reachable via ``__cause__``) would be invisible in practice.
    """
    _isolated_env(monkeypatch, tmp_path)

    original = RuntimeError("infra boom")
    raised = await _run_failing(monkeypatch, original)

    assert type(raised) is OptimizationError
    assert raised.__cause__ is original
    assert isinstance(raised.sync_session_id, str) and raised.sync_session_id


@pytest.mark.asyncio
async def test_resolution_error_escapes_unwrapped_and_carries_the_id(
    monkeypatch, tmp_path
) -> None:
    """T5: the RFC 0001 §3.4 typed rejection keeps its exact type.

    Callers match on ``ResolutionError`` to tell a governance rejection from a
    generic failure, so the id must be attached WITHOUT wrapping. Asserting the
    exact type (not ``isinstance``) is the point: an ``OptimizationError``
    wrapper would satisfy a looser check while breaking every caller.
    """
    _isolated_env(monkeypatch, tmp_path)

    original = ResolutionError((), "mocked rejection")
    raised = await _run_failing(monkeypatch, original)

    assert type(raised) is ResolutionError
    assert raised is original
    assert isinstance(raised.sync_session_id, str) and raised.sync_session_id


@pytest.mark.asyncio
async def test_no_local_record_leaves_the_id_none_and_never_leaks_the_raw_id(
    monkeypatch, tmp_path
) -> None:
    """T6: when the predicate declines, the attribute stays ``None``.

    The tempting wrong implementation is to attach the orchestrator's raw
    ``session_id`` directly. That id is not necessarily one ``traigent sync``
    accepts — an ephemeral ``local_session_<uuid>`` with no record on disk is
    the #2020 case — so a caller handed it would get a rejection. The predicate
    is the only source, and its ``None`` must survive.
    """
    _isolated_env(monkeypatch, tmp_path)

    raw_session_ids: list[str | None] = []

    def declining_predicate(self, session_id: str | None) -> str | None:
        raw_session_ids.append(session_id)
        return None

    monkeypatch.setattr(
        BackendSessionManager, "syncable_local_session_id", declining_predicate
    )

    raised = await _run_failing(monkeypatch, RuntimeError("infra boom"))

    real_ids = [sid for sid in raw_session_ids if sid]
    # Non-vacuity: the predicate really was consulted with a real session id.
    assert real_ids
    assert raised.sync_session_id is None
    assert raised.sync_session_id not in real_ids


_DISTINCT_SYNCABLE_ID = "syncable-id-distinct-from-the-raw-session-id"


def _predicate_returning_a_distinct_id(monkeypatch) -> list[str | None]:
    """Force the predicate to return an id that is NOT the raw ``session_id``.

    In the natural harness the predicate returns its own argument, so the raw
    orchestrator ``session_id`` and the syncable id are the SAME string and
    "the code attached the predicate's answer" is indistinguishable from "the
    code attached the raw id it already had". Splitting them makes the
    distinction observable.
    """
    raw_session_ids: list[str | None] = []

    def distinct_predicate(self, session_id: str | None) -> str | None:
        raw_session_ids.append(session_id)
        return _DISTINCT_SYNCABLE_ID if session_id else None

    monkeypatch.setattr(
        BackendSessionManager, "syncable_local_session_id", distinct_predicate
    )
    return raw_session_ids


@pytest.mark.parametrize(
    ("loop_failure", "expected_type"),
    [
        pytest.param(
            OptimizationError("optimizer boom"), OptimizationError, id="reraise-branch"
        ),
        pytest.param(
            ResolutionError((), "mocked rejection"),
            ResolutionError,
            id="resolution-branch",
        ),
        pytest.param(RuntimeError("infra boom"), OptimizationError, id="wrap-branch"),
    ],
)
@pytest.mark.asyncio
async def test_no_branch_leaks_the_raw_session_id_instead_of_the_predicates(
    monkeypatch, tmp_path, loop_failure: BaseException, expected_type: type
) -> None:
    """T10: the raw-id-never-leaked guarantee holds on ALL THREE branches.

    T6 covers only the generic-wrapper branch, and T3/T5 run in a setup where
    the predicate returns its own argument — so the raw session id and the
    syncable id coincide there and a refactor that surfaced ``session_id``
    directly on the re-raise or ``ResolutionError`` branch would keep every
    other test green. Here the two differ by construction, so each branch has
    to prove it forwarded the predicate's answer rather than the id it already
    had in hand.
    """
    _isolated_env(monkeypatch, tmp_path)
    raw_session_ids = _predicate_returning_a_distinct_id(monkeypatch)

    raised = await _run_failing(monkeypatch, loop_failure)

    assert type(raised) is expected_type
    real_ids = [sid for sid in raw_session_ids if sid]
    # Non-vacuity: the predicate really was consulted with a real session id,
    # so "the raw id was not attached" is a claim about a real alternative.
    assert real_ids
    assert raised.sync_session_id == _DISTINCT_SYNCABLE_ID
    assert raised.sync_session_id not in real_ids


@pytest.mark.asyncio
async def test_attach_is_unconditional_and_never_leaves_a_previous_runs_id(
    monkeypatch, tmp_path
) -> None:
    """T11: assignment is unconditional, and lands on the INSTANCE.

    Assigning only when the predicate returns a string is the seductive
    version — "don't clobber a good id with None" — and it is wrong. Exception
    instances are reusable objects: the same instance can be raised by a second
    run (as here), or shared by two concurrent runs. If run B's declining
    predicate leaves run A's id in place, the caller reads a stale id and
    uploads the WRONG session. The attribute must always describe THIS raise.

    Also pins the target of the write. Writing to the class instead of the
    instance would satisfy a naive ``exc.sync_session_id == ...`` check while
    poisoning every other exception object in the process, including
    already-raised ones a caller is still holding.
    """
    _isolated_env(monkeypatch, tmp_path)

    staged: list[str | None] = ["run-a-syncable-id", None]
    calls: list[str | None] = []

    def staged_predicate(self, session_id: str | None) -> str | None:
        calls.append(session_id)
        return staged[min(len(calls) - 1, len(staged) - 1)]

    monkeypatch.setattr(
        BackendSessionManager, "syncable_local_session_id", staged_predicate
    )

    # One exception object, raised by two consecutive runs.
    shared = OptimizationError("reused instance")

    first = await _run_failing(monkeypatch, shared)
    assert first is shared
    assert shared.sync_session_id == "run-a-syncable-id"
    # The write went to the instance, not the class.
    assert "sync_session_id" in vars(shared)
    assert OptimizationError.sync_session_id is None
    assert OptimizationError("bystander").sync_session_id is None

    second = await _run_failing(monkeypatch, shared)
    assert second is shared
    # Non-vacuity: run B really did consult the predicate (and got a decline).
    assert len(calls) == 2
    # The whole point: run A's id must NOT survive into run B's raise.
    assert shared.sync_session_id is None


@pytest.mark.asyncio
async def test_a_broken_logging_handler_cannot_replace_the_real_failure(
    monkeypatch, tmp_path
) -> None:
    """T12: the attach helper's own recovery logging must not escape.

    The helper absorbs a failing predicate and reports it at WARNING — but that
    warning is itself a call into application code. A logging handler whose
    ``emit()`` raises (a misconfigured Sentry/Datadog transport is the usual
    culprit) would otherwise propagate out of a best-effort diagnostic and
    REPLACE the caller's optimization failure with a logging error; for a
    ``ResolutionError`` that also destroys the RFC 0001 §3.4 typed identity.
    Both the probe and the report of the probe's failure are guarded.
    """
    _isolated_env(monkeypatch, tmp_path)

    def exploding_predicate(self, session_id: str | None) -> str | None:
        raise RuntimeError("local store probe exploded")

    monkeypatch.setattr(
        BackendSessionManager, "syncable_local_session_id", exploding_predicate
    )

    exploded_warnings: list[str] = []

    class _ExplodingWarningLogger:
        """Delegates everything, but blows up on the attach recovery warning.

        Targeted rather than a real ``logging.Handler``: ``logging`` routes
        handler errors through ``handleError`` and prints them, so a handler
        that raises would never reach the code under test.
        """

        def __init__(self, real: Any) -> None:
            self._real = real

        def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            if "traigent sync` id for the failed" in str(msg):
                exploded_warnings.append(str(msg))
                raise RuntimeError("logging handler emit() exploded")
            self._real.warning(msg, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    monkeypatch.setattr(
        orchestrator_module,
        "logger",
        _ExplodingWarningLogger(orchestrator_module.logger),
    )

    original = RuntimeError("infra boom")
    raised = await _run_failing(monkeypatch, original)

    # Non-vacuity: the guarded diagnostic really did raise.
    assert exploded_warnings, "the recovery warning never fired — test is vacuous"
    # The caller still gets the REAL failure, not the logging error.
    assert type(raised) is OptimizationError
    assert raised.__cause__ is original
    # And the attribute still describes this raise: no id could be determined.
    assert raised.sync_session_id is None


@pytest.mark.asyncio
async def test_a_broken_error_log_cannot_replace_the_typed_failure(
    monkeypatch, tmp_path
) -> None:
    """T15: the handler's OWN ``logger.error`` must not escape either.

    T12 covers the recovery ``logger.warning`` inside ``_attach_sync_session_id``,
    but the handler's main "Optimization %s failed" diagnostic sits on the
    normal path — it runs on EVERY generic failure, not just the ones where the
    storage probe declined — so it is the far likelier place for a broken
    application logging handler to strike. Unguarded it propagates out of the
    ``except`` block and REPLACES the caller's exception: for a
    ``ResolutionError`` that destroys the RFC 0001 §3.4 typed identity callers
    match on, and the caller receives an ``OSError`` with no
    ``sync_session_id`` — the exact loss #2029 exists to prevent, arriving via
    a best-effort log line.

    ``ResolutionError`` is the load-bearing choice: it takes the branch that
    re-raises the caller's own object, so a replacement here is maximally
    destructive and maximally invisible to a type-agnostic assertion.
    ``OSError`` because that is what a dead log socket / full disk actually
    raises.
    """
    _isolated_env(monkeypatch, tmp_path)
    seen_sessions = _capture_predicate_input(monkeypatch)

    exploded_errors: list[str] = []

    class _ExplodingErrorLogger:
        """Delegates everything, but blows up on the handler's failure log.

        Targeted rather than a real ``logging.Handler`` for the same reason as
        T12's stub: ``logging`` routes handler errors through ``handleError``
        and prints them, so a raising handler would never reach the code under
        test.
        """

        def __init__(self, real: Any) -> None:
            self._real = real

        def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
            if "failed:" in str(msg):
                exploded_errors.append(str(msg))
                raise OSError("logging handler emit() exploded")
            self._real.error(msg, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    monkeypatch.setattr(
        orchestrator_module, "logger", _ExplodingErrorLogger(orchestrator_module.logger)
    )

    original = ResolutionError((), "mocked rejection")
    raised = await _run_failing(monkeypatch, original)

    # Non-vacuity: the guarded diagnostic really did raise.
    assert exploded_errors, "the failure log never fired — test is vacuous"
    # The caller still gets the REAL failure with its exact type, not an OSError.
    assert type(raised) is ResolutionError
    assert raised is original
    # … and the id survived the broken handler.
    sync_id = raised.sync_session_id
    assert isinstance(sync_id, str) and sync_id
    assert sync_id in seen_sessions


@pytest.mark.parametrize(
    "second_probe",
    ["declines", "raises"],
    ids=["second-probe-declines", "second-probe-raises"],
)
@pytest.mark.asyncio
async def test_the_wrapper_branch_probes_storage_exactly_once(
    monkeypatch, tmp_path, second_probe: str
) -> None:
    """T16: one probe, one answer, shared by the original and the wrapper.

    The wrap branch labels two objects: the caught exception and the
    ``OptimizationError`` the caller actually receives. Resolving the id
    separately for each means a SECOND trip to local storage, and storage is
    exactly the thing that is unreliable when a run has just failed. If that
    second probe declines or throws, the wrapper — the only object the caller
    ever sees — carries ``None`` while the valid id sits on ``__cause__``,
    reachable only by a caller who already knows to look there.

    Both halves are pinned. The call count is the direct statement of the
    invariant; the hostile-second-probe cases are what makes it matter, and
    they fail loudly on any re-probing implementation even if a future refactor
    made the count assertion look incidental.
    """
    _isolated_env(monkeypatch, tmp_path)

    probe_calls: list[str | None] = []

    def once_then_hostile(self, session_id: str | None) -> str | None:
        probe_calls.append(session_id)
        if len(probe_calls) == 1:
            return _DISTINCT_SYNCABLE_ID
        if second_probe == "raises":
            raise RuntimeError("local store probe exploded on the second call")
        return None

    monkeypatch.setattr(
        BackendSessionManager, "syncable_local_session_id", once_then_hostile
    )

    original = RuntimeError("infra boom")
    raised = await _run_failing(monkeypatch, original)

    # The invariant: the failing run consulted storage exactly once.
    assert len(probe_calls) == 1, (
        f"expected a single storage probe, got {len(probe_calls)}: {probe_calls}"
    )
    # And the object the caller actually holds is the correctly labelled one.
    assert type(raised) is OptimizationError
    assert raised.__cause__ is original
    assert raised.sync_session_id == _DISTINCT_SYNCABLE_ID
    # The cause agrees rather than being the only place the id survived.
    assert original.sync_session_id == _DISTINCT_SYNCABLE_ID


class _UnprintableError(Exception):
    """An exception whose ``__str__`` raises — the #2029 finding-C repro.

    Not exotic: any exception that renders itself from attributes set after
    ``__init__`` (or from a lazily-loaded field) behaves this way when it is
    raised from a half-built state.
    """

    def __str__(self) -> str:
        raise RuntimeError("__str__ exploded")


@pytest.mark.asyncio
async def test_unprintable_exception_still_finalizes_and_carries_the_id(
    monkeypatch, tmp_path
) -> None:
    """T13: a failure whose ``__str__`` raises must not strand the run.

    The handler rendered the caught exception into ``logger.error`` BEFORE
    finalizing the session and before attaching the id, and rendered it again
    into the ``OptimizationError`` wrapper message. Both are f-string calls
    into user code. With a raising ``__str__`` that meant: no FAILED finalize,
    no id, and a ``RuntimeError('__str__ exploded')`` reaching the caller in
    place of the real failure. All three consequences are asserted against.
    """
    _isolated_env(monkeypatch, tmp_path)
    seen_sessions = _capture_predicate_input(monkeypatch)

    original = _UnprintableError()
    raised = await _run_failing(monkeypatch, original)

    # 1. The caller gets the real failure, wrapped — not the formatting error.
    assert type(raised) is OptimizationError
    assert raised.__cause__ is original
    # The wrapper message degraded gracefully instead of exploding.
    assert "<unprintable _UnprintableError>" in str(raised)

    # 2. The id was attached.
    sync_id = raised.sync_session_id
    assert isinstance(sync_id, str) and sync_id
    assert sync_id in seen_sessions

    # 3. The local session was still finalized terminal FAILED.
    storage = LocalStorageManager(str(tmp_path / "results"))
    session = storage.load_session(sync_id)
    assert session is not None
    assert session.status == "failed"


class _HardExit(BaseException):
    """A BaseException that is neither ``Exception`` nor the interrupt pair.

    Stands in for ``SystemExit`` / ``GeneratorExit`` / a test framework's
    outcome exceptions — the class of exits that used to slip past every
    handler in ``_run_optimization_with_tracing``.
    """


@pytest.mark.asyncio
async def test_non_exception_baseexception_is_finalized_attached_and_reraised(
    monkeypatch, tmp_path
) -> None:
    """T14: the BaseException gap between the code and the docs is closed.

    ``KeyboardInterrupt``/``CancelledError`` are handled (they return a partial
    result) and ``Exception`` is handled — everything else fell through: the
    run was never marked FAILED and no id was attached, while the docs promised
    the id for "a run that raises". Now such an exit is finalized and labelled
    on its way out.

    The three properties that make this safe are all pinned here: the exit is
    NEVER swallowed, it is NEVER wrapped (its exact type is the contract), and
    the recovery work happens before it propagates.
    """
    _isolated_env(monkeypatch, tmp_path)
    seen_sessions = _capture_predicate_input(monkeypatch)

    original = _HardExit("hard exit")
    raised = await _run_failing(monkeypatch, original)

    # Never swallowed, never wrapped: the caller's own object, exact type.
    assert type(raised) is _HardExit
    assert raised is original

    # No class-level default on a foreign BaseException — hence `getattr`, and
    # hence the docs telling callers to read this family that way.
    sync_id = getattr(raised, "sync_session_id", None)
    assert isinstance(sync_id, str) and sync_id
    assert sync_id in seen_sessions

    # The run really was finalized terminal, not left pending.
    storage = LocalStorageManager(str(tmp_path / "results"))
    session = storage.load_session(sync_id)
    assert session is not None
    assert session.status == "failed"


def test_exception_types_default_the_id_to_none() -> None:
    """T7: the attribute exists and defaults to ``None`` on a bare instance.

    Callers write ``exc.sync_session_id`` unconditionally, including for
    exceptions the orchestrator never touched, so the default is part of the
    contract rather than an implementation detail. The two subclasses are here
    on purpose: putting the default on ``OptimizationError`` means every
    subclass inherits it.

    ``CloudBrainUnavailableError`` is a genuine carrier — ``next-trial`` raises
    it from inside the optimization loop, which lands in the failure handler.
    ``CostLimitExceeded`` is here only for its default: it is raised by the
    PRE-RUN ``_check_cost_approval`` gate, ahead of the orchestrator's try
    block, so it never reaches an attach site and always reads ``None``. The
    docs say so rather than listing it as a carrier.
    """
    assert OptimizationError("x").sync_session_id is None
    assert ResolutionError((), "").sync_session_id is None
    assert CostLimitExceeded(1.0, 0.5).sync_session_id is None
    assert CloudBrainUnavailableError("s", "r").sync_session_id is None


@pytest.mark.asyncio
async def test_backend_finalize_failure_does_not_starve_the_attach(
    monkeypatch, tmp_path
) -> None:
    """T8: a blowing-up backend finalize must not cost the caller the id.

    The failure path finalizes the backend session before the caller ever sees
    the exception, and that finalize is the fragile part — it talks to the
    network. If the id were attached inside that block, or after it without the
    block containing its own failure, a backend hiccup would silently strip the
    only handle to the stranded local trials, precisely when the user most
    needs it.

    ``backend_tracking_enabled`` is forced on because the no-key environment
    otherwise short-circuits before ``finalize_session`` is reached — the
    non-vacuity assertion below pins that the fragile call really did run and
    really did raise. Forcing it does NOT suppress the id: locality is still
    established by the unacknowledged-trials clause (a dead backend URL
    acknowledges nothing).
    """
    _isolated_env(monkeypatch, tmp_path)

    finalize_calls: list[str | None] = []

    def exploding_finalize(self, session_id, *args: Any, **kwargs: Any):
        finalize_calls.append(session_id)
        raise RuntimeError("HTTP 500 finalize exploded")

    monkeypatch.setattr(BackendSessionManager, "finalize_session", exploding_finalize)
    monkeypatch.setattr(
        BackendSessionManager,
        "backend_tracking_enabled",
        property(lambda self: True),
    )

    raised = await _run_failing(monkeypatch, RuntimeError("infra boom"))

    # Non-vacuity: the fragile branch was actually entered and actually failed.
    assert finalize_calls, "finalize_session was never reached — test is vacuous"
    assert isinstance(raised.sync_session_id, str) and raised.sync_session_id


@pytest.mark.asyncio
async def test_local_finalize_failure_does_not_starve_the_attach(
    monkeypatch, tmp_path
) -> None:
    """T8b: the UNGUARDED half of the finalizer must not starve the attach either.

    T8 covers the backend finalize, which ``_finalize_failed_backend_session``
    already wraps in its own try/except. Its *local* finalize is not wrapped —
    ``finalize_local_session`` resolves a storage handle and consults backend
    state outside any try — so "finalize, then attach" as two plain statements
    loses the id whenever that half throws. The handler pairs them as
    ``try: finalize … finally: attach`` for exactly this case.

    DOCUMENTED BOUNDARY, asserted rather than glossed over. When the local
    finalize throws, the optimization failure is NOT what reaches the caller.
    The storage error propagates in place of ``e`` (the finalizer is called
    from the handler body, so its exception replaces the one being handled),
    and ``OptimizedFunction._run_optimization``'s own generic handler then
    re-wraps it into a fresh ``OptimizationError`` that quotes the STORAGE
    message and — being a layer that knows nothing about #2029 — carries no
    ``sync_session_id`` at all.

    That substitution is pre-existing #1939 behaviour and deliberately OUT of
    #2029's scope: #2029 owns only that the exception the run started with is
    still labelled before anything else can go wrong. An earlier draft of this
    test asserted against ``original`` while discarding what actually escaped,
    so it read as "the caller gets a labelled failure" when the caller in fact
    gets an unlabelled storage error. Both halves are pinned below instead. If
    #1939 is ever fixed so the original failure survives, the escaped-object
    assertions are where that change announces itself.
    """
    _isolated_env(monkeypatch, tmp_path)

    local_finalize_calls: list[str | None] = []

    def exploding_local_finalize(self, session_id, *args: Any, **kwargs: Any):
        local_finalize_calls.append(session_id)
        raise RuntimeError("local session store is unwritable")

    monkeypatch.setattr(
        BackendSessionManager, "finalize_local_session", exploding_local_finalize
    )

    original = OptimizationError("optimizer boom")
    escaped = await _run_failing(monkeypatch, original)

    # Non-vacuity: the unguarded call really was reached and really did raise.
    assert local_finalize_calls, "finalize_local_session was never reached"

    # The #1939 boundary, stated as it really is: what escapes describes the
    # STORAGE failure, not "optimizer boom", and nobody labelled it.
    assert escaped is not original
    assert "local session store is unwritable" in str(escaped)
    assert "optimizer boom" not in str(escaped)
    assert getattr(escaped, "sync_session_id", None) is None
    # The finalizer's own RuntimeError is the thing that displaced `original`;
    # the outer OptimizedFunction handler is what turned it into this wrapper.
    assert type(escaped.__cause__) is RuntimeError
    assert escaped.__cause__.__context__ is original

    # What #2029 does own: the original failure was still labelled on the way
    # out — the `finally` is what makes this true. A caller that walks the
    # chain, or that already held the instance, can still name the session.
    assert isinstance(original.sync_session_id, str) and original.sync_session_id


# ---------------------------------------------------------------------------
# T9 — cancelled-path guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_cancelled_run_still_returns_a_result_carrying_the_id(
    monkeypatch, tmp_path
) -> None:
    """T9: a user interrupt still RETURNS a result — it does not raise.

    NOT evidence for the #2029 fix: this passes on the pre-fix code too, since
    it exercises the #2020 result-path assignment. It is a guard, and its green
    must not be read as proof of anything. It exists so that a future
    unification of the failure and cancellation finalizers breaks a test
    instead of quietly turning a returned partial result into a raise.
    """
    _isolated_env(monkeypatch, tmp_path)

    @_optimized
    async def answer(text: str, config=None) -> str:
        if config.custom_params.get("x") == "b":
            raise KeyboardInterrupt
        return "ok"

    result = await answer.optimize(algorithm="grid")

    # At least one trial completed before the interrupt — the partial result is
    # real, which is what makes carrying an id worthwhile.
    assert len(result.trials) >= 1
    assert isinstance(result.sync_session_id, str) and result.sync_session_id
