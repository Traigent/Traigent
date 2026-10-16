"""Pillar Legend run binding: declared refs, drift, admission, and the flag-off no-op.

No network: a fake transport records every call. Enablement is driven only by the
``legend`` block of the session-create response, so most tests build an orchestrator
whose session creation returns (or omits) that block.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.session_context import SessionContext
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.legend import _contract as c
from traigent.legend import (
    AgentManifestSource,
    ConfigurationSource,
    InMemoryTransport,
    LegendError,
    declare,
)
from traigent.legend.pillars import compute_pillar
from traigent.legend.run_binding import (
    DeclaredVersion,
    LegendOptions,
    LegendRunBinder,
    parse_block,
    start_binder,
)
from traigent.legend.transport import DeclareResult
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError

CANARY = "Alice_has_diabetes"
CANARY_PATH = "/home/alice/secret_project/agent.py"


def vref(n: int) -> str:
    return f"versionref_{n:032x}"


def block(
    run_bindings: list[tuple[str, str]] | None = None,
    trial: tuple[str, list[tuple[str, str, str]]] | None = None,
) -> dict[str, Any]:
    """The Backend's ``legend`` block: ``[(role, ref)]`` run slots, optional trial slots."""
    out: dict[str, Any] = {
        "enabled": True,
        "run_ref": f"runref_{1:032x}",
        "bindings": [
            {
                "binding_role": role,
                "slot_scope": "run",
                "binding_slot": 0,
                "version_ref": ref,
            }
            for role, ref in (run_bindings or [])
        ],
    }
    if trial is not None:
        out["trial_ref"] = trial[0]
        out["bindings"] += [
            {
                "binding_role": role,
                "slot_scope": scope,
                "binding_slot": 0,
                "version_ref": ref,
            }
            for role, scope, ref in trial[1]
        ]
    return out


class FakeServer(InMemoryTransport):
    """In-memory declare plus a recording admission route; counts every call."""

    def __init__(self, fail_declare: int = 0, fail_admit: int = 0) -> None:
        super().__init__()
        self.admissions: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.declare_attempts: list[dict[str, Any]] = []
        self._fail_declare = fail_declare
        self._fail_admit = fail_admit

    def declare(self, payload: Mapping[str, Any]) -> DeclareResult:
        self.calls.append("declare")
        self.declare_attempts.append(dict(payload))
        if self._fail_declare > 0:
            self._fail_declare -= 1
            raise LegendError("store_unavailable")
        return super().declare(payload)

    def admit(self, session_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append("admit")
        if self._fail_admit > 0:
            self._fail_admit -= 1
            raise LegendError("internal_error")
        self.admissions.append(
            {"session_id": session_id, **json.loads(json.dumps(body))}
        )
        return {"run_tuple_digest": "sha256:" + "0" * 64, "replayed": False}


class _Optimizer(BaseOptimizer):
    def __init__(self, space: dict[str, Any], objectives: list[str], **kw: Any):
        super().__init__(space, objectives, **kw)
        self.plan: list[dict[str, Any]] = []
        self.n = 0

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        cfg = dict(self.plan[self.n])
        self.n += 1
        return cfg

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self.n >= len(self.plan)

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        return None

    def is_finished(self) -> bool:
        return self.n >= len(self.plan)


class _Evaluator(BaseEvaluator):
    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.count = 0
        self.configs: list[dict[str, Any]] = []

    async def evaluate(self, func, config, dataset, **_kw):  # type: ignore[no-untyped-def]
        self.count += 1
        self.configs.append(dict(config))
        metrics = {"accuracy": 0.5, "examples_attempted": 1}
        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=1,
            successful_examples=1,
            duration=0.01,
            metrics=metrics,
            outputs=["o"],
            errors=[None],
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = 1
        return result


@pytest.fixture(autouse=True)
def _logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "logs"))


async def _run(
    legend_block: dict[str, Any] | None,
    options: Any,
    plan: list[dict[str, Any]],
    *,
    privacy: bool = False,
    default_config: dict[str, Any] | None = None,
    agent_key: str | None = None,
) -> tuple[_Evaluator, OptimizationOrchestrator, list[dict[str, Any]]]:
    optimizer = _Optimizer({"param1": [0, 1, 2]}, ["accuracy"])
    optimizer.plan = plan
    evaluator = _Evaluator()
    extra: dict[str, Any] = {"agent_key": agent_key} if agent_key else {}
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=len(plan) + 1,  # +1: the default_config baseline runs first
        config=TraigentConfig(
            no_egress=True,
            enable_usage_analytics=False,
            **({"privacy_enabled": True} if privacy else {}),
        ),
        default_config=default_config if default_config is not None else {"param1": 1},
        **extra,
    )
    orchestrator.legend_options = options
    create_calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> SessionContext:
        create_calls.append(
            {k: v for k, v in kwargs.items() if k not in ("func", "start_time")}
        )
        return SessionContext(
            session_id="sess_1",
            dataset_name="d",
            function_name="f",
            optimization_id="o",
            start_time=0.0,
            legend=legend_block,
        )

    orchestrator.backend_session_manager.create_session = fake_create  # type: ignore[method-assign]

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    ds = Dataset([EvaluationExample({"q": "a"}, "b")], name="ds")
    await orchestrator.optimize(f, ds)
    return evaluator, orchestrator, create_calls


def _config_ref(server: FakeServer, config: dict[str, Any]) -> DeclareResult:
    return declare(c.KIND_CONFIGURATION, ConfigurationSource(config), transport=server)


# ----------------------------------------------------------------------- block parsing
def test_parse_block_absent_off_and_malformed_are_none() -> None:
    assert parse_block(None) is None
    assert parse_block({}) is None
    assert parse_block({**block(), "enabled": False}) is None
    assert parse_block({**block(), "run_ref": "x"}) is None
    bad = block([("dataset", "not-a-ref")])
    assert parse_block(bad) is None
    wrong_role = block([("not_a_role", vref(1))])
    assert parse_block(wrong_role) is None
    ok = parse_block(block([("run_base_configuration", vref(2))]))
    assert ok is not None and ok.trial_ref is None and len(ok.bindings) == 1


def test_options_validate_refs_and_mode() -> None:
    assert LegendOptions.coerce(None) is None
    opts = LegendOptions.coerce(
        {"declared": {c.KIND_AGENT: vref(1)}, "mode": "permitted"}
    )
    assert opts is not None and opts.declared[c.KIND_AGENT] == DeclaredVersion(vref(1))
    for bad in (
        {"declared": {c.KIND_AGENT: "Alice"}},
        {"declared": {"not_a_kind": vref(1)}},
        {"mode": "lenient"},
        {"surprise": 1},
    ):
        with pytest.raises(LegendError):
            LegendOptions.coerce(bad)


# ----------------------------------------------------------------------- flag off
@pytest.mark.asyncio
@pytest.mark.parametrize("off", [None, {"enabled": False, "run_ref": "x"}, {}])
async def test_flag_off_is_byte_identical_with_zero_calls(off: Any) -> None:
    server = FakeServer()
    plan = [{"param1": 1}, {"param1": 2}]
    ev_off, orch, calls_off = await _run(
        off,
        LegendOptions(transport=server),
        plan,  # type: ignore[arg-type]
    )
    ev_plain, _, calls_plain = await _run(None, None, plan)
    assert server.calls == []  # not one transport call
    assert orch._legend_binder is None
    assert ev_off.count == ev_plain.count == 3
    assert json.dumps(calls_off, default=str, sort_keys=True) == json.dumps(
        calls_plain, default=str, sort_keys=True
    )  # the legacy create request is unchanged


@pytest.mark.asyncio
async def test_block_on_without_a_transport_or_options_does_nothing() -> None:
    ev, orch, _ = await _run(
        block([("run_base_configuration", vref(9))]), None, [{"param1": 1}]
    )
    assert orch._legend_binder is None and ev.count == 2


@pytest.mark.asyncio
async def test_block_on_without_a_transport_refuses_a_declaration() -> None:
    ev = _Evaluator()
    with pytest.raises(LegendError) as err:
        await _run(
            block([("run_base_configuration", vref(9))]),
            LegendOptions(declared={c.KIND_CONFIGURATION: vref(1)}),
            [{"param1": 1}],
        )
    assert err.value.code == "local_transport_unavailable" and ev.count == 0


# ----------------------------------------------------------------------- match / drift
@pytest.mark.asyncio
async def test_match_reuses_the_declared_version() -> None:
    server = FakeServer()
    declared = _config_ref(server, {"param1": 1})  # the run-base config
    server.calls.clear()
    server.declare_attempts.clear()
    opts = LegendOptions(
        declared={
            c.KIND_CONFIGURATION: DeclaredVersion(
                declared.version_ref, declared.entity_ref
            )
        },
        transport=server,
    )
    ev, orch, _ = await _run(
        block([("run_base_configuration", vref(7))]), opts, [{"param1": 1}]
    )
    assert ev.count == 2
    assert server.calls == ["declare", "admit"]  # no child: the observed one matched
    run_obs = server.admissions[0]["run_observed"]
    assert run_obs == [
        {
            "binding_role": "run_base_configuration",
            "slot_scope": "run",
            "binding_slot": 0,
            "version_ref": declared.version_ref,
        }
    ]
    assert server.admissions[0]["mode"] == "strict"


@pytest.mark.asyncio
async def test_permitted_drift_declares_a_same_entity_child() -> None:
    server = FakeServer()
    declared = _config_ref(server, {"param1": 99})  # not what the run applies
    server.calls.clear()
    server.declare_attempts.clear()
    opts = LegendOptions(
        declared={
            c.KIND_CONFIGURATION: DeclaredVersion(
                declared.version_ref, declared.entity_ref
            )
        },
        mode="permitted",
        transport=server,
    )
    ev, _, _ = await _run(
        block([("run_base_configuration", vref(7))]), opts, [{"param1": 1}]
    )
    assert ev.count == 2
    child = server.declare_attempts[-1]
    assert child["entity_ref"] == declared.entity_ref
    assert child["previous_version_ref"] == declared.version_ref
    assert child["change_reason"] == "unspecified"
    admitted = server.admissions[0]["run_observed"][0]["version_ref"]
    assert admitted not in (declared.version_ref, vref(7))
    assert server.versions[admitted]["entity_ref"] == declared.entity_ref
    assert server.admissions[0]["mode"] == "permitted"


@pytest.mark.asyncio
async def test_strict_mismatch_refuses_before_any_evaluator_call() -> None:
    server = FakeServer()
    declared = _config_ref(server, {"param1": 99})
    server.calls.clear()
    opts = LegendOptions(
        declared={
            c.KIND_CONFIGURATION: DeclaredVersion(
                declared.version_ref, declared.entity_ref
            )
        },
        transport=server,
    )
    optimizer_seen: list[Any] = []
    with pytest.raises(LegendError) as err:
        evaluator, _, _ = await _run(
            block([("run_base_configuration", vref(7))]), opts, [{"param1": 1}]
        )
        optimizer_seen.append(evaluator)
    assert err.value.code == "manifest_mismatch"
    assert "admit" not in server.calls  # refused before admission, hence before running


@pytest.mark.asyncio
async def test_strict_refusal_leaves_the_evaluator_uncalled() -> None:
    server = FakeServer()
    declared = _config_ref(server, {"param1": 99})
    opts = LegendOptions(
        declared={c.KIND_CONFIGURATION: DeclaredVersion(declared.version_ref)},
        transport=server,
    )
    evaluator = _Evaluator()
    optimizer = _Optimizer({"param1": [0, 1]}, ["accuracy"])
    optimizer.plan = [{"param1": 1}]
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        default_config={"param1": 1},
    )
    orch.legend_options = opts
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "s", "d", "f", "o", 0.0, legend=block([("run_base_configuration", vref(7))])
    )

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    with pytest.raises(LegendError):
        await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    assert evaluator.count == 0


# ----------------------------------------------------------------------- per trial
def _trial_block(
    trial_n: int,
    cfg_ref: str,
    extra_run: list[tuple[str, str]],
    entity: str | None = None,
) -> dict[str, Any]:
    """A trial legend block; ``entity`` is the ``entity_ref`` the Backend advertises."""
    blk = block(
        extra_run,
        trial=(
            f"trialref_{trial_n:032x}",
            [("trial_configuration", "trial", cfg_ref)],
        ),
    )
    if entity is not None:
        for b in blk["bindings"]:
            if b["binding_role"] == "trial_configuration":
                b["entity_ref"] = entity
    return blk


@pytest.mark.asyncio
async def test_multi_trial_observations_are_distinct_and_complete() -> None:
    server = FakeServer()
    s1 = _config_ref(server, {"param1": 1})
    s2 = _config_ref(server, {"param1": 2})
    server.calls.clear()
    run_cfg = vref(7)
    plan = [
        {
            "param1": 1,
            "_traigent_backend_trial_id": "t_1",
            "_traigent_legend_trial": _trial_block(
                1, s1.version_ref, [("run_base_configuration", run_cfg)], s1.entity_ref
            ),
        },
        {
            "param1": 2,
            "_traigent_backend_trial_id": "t_2",
            "_traigent_legend_trial": _trial_block(
                2, s2.version_ref, [("run_base_configuration", run_cfg)], s2.entity_ref
            ),
        },
    ]
    ev, _, _ = await _run(
        block([("run_base_configuration", run_cfg)]),
        LegendOptions(mode="permitted", transport=server),
        plan,
    )
    assert ev.count == 3
    trial_calls = [a for a in server.admissions if "trial_id" in a]
    assert [a["trial_id"] for a in trial_calls] == ["t_1", "t_2"]
    refs = [
        [o["version_ref"] for o in a["trial_observed"] if o["slot_scope"] == "trial"]
        for a in trial_calls
    ]
    assert refs[0] != refs[1]
    # each trial reports the identity of the config it actually ran, in the run's entity
    for (ref,), n in zip(refs, (1, 2), strict=True):
        assert (
            server.versions[ref]["fingerprint"]
            == compute_pillar(
                c.KIND_CONFIGURATION, ConfigurationSource({"param1": n})
            ).fingerprint
        )
    run_obs = [a["run_observed"] for a in trial_calls]
    assert run_obs[0] == run_obs[1] == server.admissions[0]["run_observed"]
    # complete: the inherited run slot is repeated next to the trial slot
    keys = {
        (o["binding_role"], o["slot_scope"]) for o in trial_calls[0]["trial_observed"]
    }
    assert keys == {("run_base_configuration", "run"), ("trial_configuration", "trial")}


@pytest.mark.asyncio
async def test_per_trial_recheck_catches_a_config_changed_after_suggestion() -> None:
    server = FakeServer()
    suggested = _config_ref(server, {"param1": 1})
    server.calls.clear()
    plan = [
        {
            "param1": 2,  # what actually runs differs from what was suggested (param1=1)
            "_traigent_backend_trial_id": "t_1",
            "_traigent_legend_trial": _trial_block(
                1,
                suggested.version_ref,
                [("run_base_configuration", vref(7))],
                suggested.entity_ref,
            ),
        }
    ]
    optimizer = _Optimizer({"param1": [0, 1, 2]}, ["accuracy"])
    optimizer.plan = plan
    evaluator = _Evaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        default_config={"param1": 1},
    )
    orch.legend_options = LegendOptions(transport=server)  # strict
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "s", "d", "f", "o", 0.0, legend=block([("run_base_configuration", vref(7))])
    )

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    with pytest.raises(OptimizationError) as err:
        await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    assert isinstance(err.value.__cause__, LegendError)
    assert err.value.__cause__.code == "manifest_mismatch"
    # only the legend-free baseline (param1=1) ran; the drifted config never reached it
    assert [cfg.get("param1") for cfg in evaluator.configs] == [1]
    assert not [a for a in server.admissions if "trial_id" in a]


# ----------------------------------------------------------------------- retries
def _binder(server: FakeServer, mode: str = "permitted") -> LegendRunBinder:
    b = start_binder(
        session_id="sess_1",
        raw_block=block([("run_base_configuration", vref(7))]),
        options=LegendOptions(mode=mode, transport=server),
        privacy_enabled=False,
    )
    assert b is not None
    return b


def test_transient_failures_retry_with_the_same_idempotency_key() -> None:
    from traigent.legend.run_binding import pillars_from_content_identity

    server = FakeServer(fail_declare=2)
    binder = _binder(server)
    binder.admit_run(pillars_from_content_identity(None, {"param1": 1}))
    keys = {a["idempotency_key"] for a in server.declare_attempts}
    digests = {a["request_digest"] for a in server.declare_attempts}
    assert len(server.declare_attempts) == 3 and len(keys) == 1 and len(digests) == 1


def test_a_failed_admission_can_be_retried_without_new_keys() -> None:
    from traigent.legend.run_binding import pillars_from_content_identity

    server = FakeServer(fail_admit=3)  # exhausts with_retries once
    binder = _binder(server)
    pillars = pillars_from_content_identity(None, {"param1": 1})
    with pytest.raises(LegendError):
        binder.admit_run(pillars)
    first = [a["idempotency_key"] for a in server.declare_attempts]
    binder.admit_run(pillars)
    assert [
        a["idempotency_key"] for a in server.declare_attempts
    ] == first  # no re-declare
    assert len(server.admissions) == 1


# ----------------------------------------------------------------------- privacy
@pytest.mark.asyncio
async def test_privacy_mode_withholds_identity() -> None:
    server = FakeServer()
    ev, orch, _ = await _run(
        block([("run_base_configuration", vref(7))]),
        LegendOptions(transport=server),
        [{"param1": 1}],
        privacy=True,
    )
    assert server.calls == [] and orch._legend_binder is None and ev.count == 2
    with pytest.raises(LegendError) as err:
        await _run(
            block([("run_base_configuration", vref(7))]),
            LegendOptions(declared={c.KIND_CONFIGURATION: vref(1)}, transport=server),
            [{"param1": 1}],
            privacy=True,
        )
    assert err.value.code == "local_privacy_withholds_identity"
    assert server.calls == []


@pytest.mark.asyncio
async def test_no_names_or_paths_cross_the_wire(tmp_path: Path) -> None:
    server = FakeServer()
    prop = _config_ref(server, {"param1": 1})
    plan = [
        {
            "param1": 1,
            "_traigent_backend_trial_id": "t_1",
            "_traigent_legend_trial": _trial_block(
                1,
                prop.version_ref,
                [("run_base_configuration", vref(7))],
                prop.entity_ref,
            ),
        }
    ]
    await _run(
        block([("run_base_configuration", vref(7))]),
        LegendOptions(mode="permitted", transport=server),
        plan,
        agent_key=CANARY,
        default_config={"param1": 1},
    )
    wire = json.dumps(
        {
            "declares": server.declare_attempts,
            "admits": server.admissions,
            "ev": server.events,
        }
    )
    for secret in (CANARY, CANARY_PATH, "secret_project", str(tmp_path)):
        assert secret not in wire
    # only opaque refs, closed enums and digests: every admission value is one of them
    for admission in server.admissions:
        for item in admission["run_observed"]:
            assert c.VERSION_REF_RE.fullmatch(item["version_ref"])
    # refusals carry a fixed closed message, never the offending value
    with pytest.raises(LegendError) as err:
        LegendOptions.coerce({"declared": {c.KIND_AGENT: CANARY}})
    assert CANARY not in str(err.value)


# ----------------------------------------------------------------------- entry points
@pytest.mark.filterwarnings("ignore")
def test_decorator_and_call_level_options_are_typed_and_forwarded(
    tmp_path: Path,
) -> None:
    import traigent

    ds = tmp_path / "d.jsonl"
    ds.write_text(json.dumps({"input": {"q": "a"}, "output": "b"}) + "\n")
    opts = LegendOptions(declared={c.KIND_AGENT: vref(5)})

    @traigent.optimize(
        eval_dataset=str(ds),
        objectives=["accuracy"],
        configuration_space={"x": [1, 2]},
        legend={"declared": {c.KIND_AGENT: vref(5)}, "mode": "permitted"},
    )
    def fn(q: str) -> str:
        return q

    assert fn.legend is not None and fn.legend.mode == "permitted"
    assert fn._legend_run_kwargs(None)["legend"].declared == opts.declared
    override = LegendOptions(mode="strict")
    assert fn._legend_run_kwargs(override)["legend"] is override  # call-level wins

    @traigent.optimize(
        eval_dataset=str(ds), objectives=["accuracy"], configuration_space={"x": [1]}
    )
    def plain(q: str) -> str:
        return q

    assert plain._legend_run_kwargs(None) == {}  # nothing given => nothing forwarded


def test_backend_block_is_carried_verbatim_and_absent_stays_none() -> None:
    from traigent.cloud.api_operations import TraigentSessionApiResult
    from traigent.core.session_types import SessionCreationResult

    blk = block([("run_base_configuration", vref(7))])
    assert TraigentSessionApiResult("s", "e", "r", legend=blk).legend == blk
    assert TraigentSessionApiResult("s", "e", "r").legend is None
    assert SessionCreationResult.connected("s", legend=blk).legend == blk
    assert SessionCreationResult.connected("s").legend is None


# ----------------------------------------------------------------- never fill from proposals
@pytest.mark.asyncio
async def test_unobserved_slot_is_never_copied_from_the_proposal_strict_refuses() -> (
    None
):
    server = FakeServer()
    # a dataset slot is advertised but this run has no key grant, so none is observed
    adv = block([("run_base_configuration", vref(7)), ("dataset", vref(8))])
    with pytest.raises(LegendError) as err:
        await _run(adv, LegendOptions(transport=server), [{"param1": 1}])
    assert err.value.code == "local_observation_incomplete"
    assert server.calls == []  # no declaration, no admission: nothing was written


@pytest.mark.asyncio
async def test_unobserved_slot_leaves_the_run_unadmitted_when_permitted() -> None:
    server = FakeServer()
    adv = block([("run_base_configuration", vref(7)), ("dataset", vref(8))])
    plan = [
        {
            "param1": 1,
            "_traigent_backend_trial_id": "t_1",
            "_traigent_legend_trial": _trial_block(
                1, vref(3), [("run_base_configuration", vref(7)), ("dataset", vref(8))]
            ),
        }
    ]
    ev, orch, _ = await _run(
        adv, LegendOptions(mode="permitted", transport=server), plan
    )
    assert ev.count == 2  # the run proceeds (baseline + one trial)
    assert server.calls == [] and server.admissions == []
    assert orch._legend_binder is not None
    assert orch._legend_binder.unadmitted_reason == "local_observation_incomplete"
    assert vref(8) not in json.dumps(server.admissions)


@pytest.mark.asyncio
async def test_search_space_is_backend_attested_and_everything_else_observed() -> None:
    server = FakeServer()
    adv = block([("run_base_configuration", vref(7)), ("search_space", vref(9))])
    await _run(adv, LegendOptions(mode="permitted", transport=server), [{"param1": 1}])
    obs = {
        o["binding_role"]: o["version_ref"]
        for o in server.admissions[0]["run_observed"]
    }
    assert obs["search_space"] == vref(9)
    assert obs["run_base_configuration"] != vref(7)  # observed, not the proposal


def test_trial_candidate_is_derived_from_the_executed_build_never_the_suggestion() -> (
    None
):
    from traigent.legend.run_binding import candidate_pillar

    class _Run:
        def _version_for(self, cfg):  # type: ignore[no-untyped-def]
            return {
                "agent_id": "a",
                "build_digest": "sha256:" + "a" * 64,
                "manifest": {},
            }, None

    assert candidate_pillar(None, {"x": 1}) is None  # no snapshot => missing
    pillar = candidate_pillar(_Run(), {"x": 1})
    assert pillar is not None and pillar.fingerprint == "sha256:" + "a" * 64

    server = FakeServer()
    binder = _binder_for(server, [("run_base_configuration", vref(7))])
    from traigent.legend.run_binding import pillars_from_content_identity

    binder.admit_run(pillars_from_content_identity(None, {"param1": 1}))
    tb = parse_block(
        block(
            [("run_base_configuration", vref(7))],
            trial=(
                f"trialref_{1:032x}",
                [
                    ("trial_configuration", "trial", vref(3)),
                    ("trial_candidate", "trial", vref(4)),
                ],
            ),
        )
    )
    assert tb is not None
    cfg = pillars_from_content_identity(None, {"param1": 2})
    # candidate not derivable: permitted leaves the trial unadmitted, nothing copied
    binder.admit_trial(trial_id="t_1", trial_block=tb, observed=cfg)
    assert [a for a in server.admissions if "trial_id" in a] == []
    assert binder.unadmitted_reason == "local_observation_incomplete"
    binder.unadmitted_reason = None
    # derivable now, but the Backend names no entity: still not admitted (the proposal
    # is never copied)
    binder.admit_trial(
        trial_id="t_1",
        trial_block=tb,
        observed={**cfg, c.KIND_AGENT: pillar},
    )
    assert [a for a in server.admissions if "trial_id" in a] == []
    # with the advertised entity named, the executed candidate is declared under it
    proposal = declare(
        c.KIND_AGENT,
        AgentManifestSource(_manifest()),
        transport=server,
    )
    tb2 = parse_block(
        {
            **block(
                [("run_base_configuration", vref(7))],
                trial=(
                    f"trialref_{1:032x}",
                    [
                        ("trial_configuration", "trial", vref(3)),
                        ("trial_candidate", "trial", proposal.version_ref),
                    ],
                ),
            ),
        }
    )
    assert tb2 is not None
    cfg_proposal = _config_ref(server, {"param1": 2})
    raw = block(
        [("run_base_configuration", vref(7))],
        trial=(
            f"trialref_{1:032x}",
            [
                ("trial_configuration", "trial", cfg_proposal.version_ref),
                ("trial_candidate", "trial", proposal.version_ref),
            ],
        ),
    )
    for b in raw["bindings"]:
        if b["binding_role"] == "trial_candidate":
            b["entity_ref"] = proposal.entity_ref
        if b["binding_role"] == "trial_configuration":
            b["entity_ref"] = cfg_proposal.entity_ref
    tb3 = parse_block(raw)
    assert tb3 is not None
    binder.admit_trial(
        trial_id="t_1",
        trial_block=tb3,
        observed={**cfg, c.KIND_AGENT: pillar},
    )
    sent = [a for a in server.admissions if "trial_id" in a][0]["trial_observed"]
    roles = {o["binding_role"]: o["version_ref"] for o in sent}
    cand = server.versions[roles["trial_candidate"]]
    assert (
        cand["fingerprint"] == pillar.fingerprint
    )  # what executed, not the suggestion
    assert cand["entity_ref"] == proposal.entity_ref


def _manifest() -> dict[str, Any]:
    from traigent.identity.agent_build import collect_agent_build_base

    def agent(q: str) -> str:
        return q

    return collect_agent_build_base(agent, agent_id="agent_x").manifest(
        "sha256:" + "0" * 64
    )


def _binder_for(server: FakeServer, bindings: list[tuple[str, str]]) -> LegendRunBinder:
    b = start_binder(
        session_id="sess_1",
        raw_block=block(bindings),
        options=LegendOptions(mode="permitted", transport=server),
        privacy_enabled=False,
    )
    assert b is not None
    return b


@pytest.mark.asyncio
async def test_parallel_refusal_finalizes_the_issued_trial_as_not_executed() -> None:
    from traigent.api.types import TrialStatus
    from traigent.legend.run_binding import LegendTrialRefused

    server = FakeServer()
    suggested = _config_ref(server, {"param1": 1})
    plan = [
        {
            "param1": 2,  # differs from the suggestion => strict refusal pre-evaluation
            "_traigent_backend_trial_id": "issued_1",
            "_traigent_legend_trial": _trial_block(
                1,
                suggested.version_ref,
                [("run_base_configuration", vref(7))],
                suggested.entity_ref,
            ),
        }
    ]
    optimizer = _Optimizer({"param1": [0, 1, 2]}, ["accuracy"])
    optimizer.plan = plan
    evaluator = _Evaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        parallel_trials=2,
        default_config={"param1": 1},
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
    )
    orch.legend_options = LegendOptions(transport=server)
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "s", "d", "f", "o", 0.0, legend=block([("run_base_configuration", vref(7))])
    )
    handled: list[Any] = []
    real = orch._handle_trial_result

    async def spy(**kw: Any) -> Any:
        handled.append(kw["trial_result"])
        return await real(**kw)

    orch._handle_trial_result = spy  # type: ignore[method-assign]

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    with pytest.raises((LegendTrialRefused, OptimizationError)):
        await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    refused = [t for t in handled if t.trial_id == "issued_1"]
    assert refused and refused[0].status == TrialStatus.CANCELLED  # not FAILED
    assert refused[0].metadata["backend_trial_id_acquired"] is True
    assert 2 not in [cfg.get("param1") for cfg in evaluator.configs]


# ----------------------------------------------------------------------- HTTP transport
class _FakeClient:
    """Stands in for the authenticated backend client: records route + body."""

    def __init__(self, server: FakeServer) -> None:
        self.server = server
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def legend_post_sync(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((suffix, json.loads(json.dumps(payload))))
        if suffix == "/legend/declare":
            r = self.server.declare(payload)
            return {
                "version_ref": r.version_ref,
                "entity_ref": r.entity_ref,
                "event_ref": r.event_ref,
                "created_version": r.created,
            }
        return self.server.admit("x", payload)


@pytest.mark.asyncio
async def test_backend_flag_on_binds_through_the_authenticated_client(
    schemas: Any,
) -> None:
    server = FakeServer()
    client = _FakeClient(server)
    optimizer = _Optimizer({"param1": [0, 1]}, ["accuracy"])
    optimizer.plan = []
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=_Evaluator(),
        max_trials=1,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        default_config={"param1": 1},
    )
    orch.backend_client = client  # type: ignore[assignment]
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "sess_1",
        "d",
        "f",
        "o",
        0.0,
        legend=block([("run_base_configuration", vref(7))]),
    )

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    routes = [r for r, _ in client.posts]
    assert routes == ["/legend/declare", "/sessions/sess_1/legend/admission"]
    schemas.validate("legend_requests_v0.json", "DeclareRequestV0", client.posts[0][1])
    assert set(client.posts[1][1]) == {"run_observed", "mode"}


def test_backend_client_posts_with_auth_bounded_timeout_and_closed_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import requests

    from traigent.cloud.backend_client import BackendIntegratedClient
    from traigent.legend.run_binding import BackendLegendTransport

    client = BackendIntegratedClient.__new__(BackendIntegratedClient)
    client.api_base_url = "https://api.example.test/api/v1"  # type: ignore[attr-defined]
    client.timeout = 120.0  # type: ignore[attr-defined]
    client._raise_if_backend_egress_disabled = lambda *_a, **_k: None  # type: ignore[method-assign]
    client._get_sync_auth_headers = lambda target="backend": {
        "Authorization": "Bearer t"
    }  # type: ignore[method-assign]
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: Any, headers: Any, timeout: Any) -> Any:
        seen.update(url=url, json=json, headers=headers, timeout=timeout)
        resp = MagicMock()
        resp.status_code = status["code"]
        resp.json.return_value = status["body"]
        resp.text = ""
        resp.headers = {}
        return resp

    status: dict[str, Any] = {
        "code": 200,
        "body": {"data": {"run_tuple_digest": "sha256:" + "0" * 64}},
    }
    monkeypatch.setattr(requests, "post", fake_post)
    out = BackendLegendTransport(client).admit(
        "s1", {"run_observed": [], "mode": "strict"}
    )
    assert out["run_tuple_digest"].startswith("sha256:")
    assert seen["url"] == "https://api.example.test/api/v1/sessions/s1/legend/admission"
    assert seen["headers"]["Authorization"] == "Bearer t" and seen["timeout"] == 30.0
    assert seen["json"] == {"run_observed": [], "mode": "strict"}
    status.update(code=409, body={"error_code": "manifest_mismatch", "message": CANARY})
    with pytest.raises(LegendError) as err:
        BackendLegendTransport(client).admit(
            "s1", {"run_observed": [], "mode": "strict"}
        )
    assert err.value.code == "manifest_mismatch" and CANARY not in str(err.value)


@pytest.mark.asyncio
async def test_sequential_refusal_finalizes_the_issued_trial_as_not_executed() -> None:
    from traigent.api.types import TrialStatus

    server = FakeServer()
    suggested = _config_ref(server, {"param1": 1})
    plan = [
        {
            "param1": 2,
            "_traigent_backend_trial_id": "issued_1",
            "_traigent_legend_trial": _trial_block(
                1,
                suggested.version_ref,
                [("run_base_configuration", vref(7))],
                suggested.entity_ref,
            ),
        }
    ]
    optimizer = _Optimizer({"param1": [0, 1, 2]}, ["accuracy"])
    optimizer.plan = plan
    evaluator = _Evaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        default_config={"param1": 1},
    )
    orch.legend_options = LegendOptions(transport=server)
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "s", "d", "f", "o", 0.0, legend=block([("run_base_configuration", vref(7))])
    )
    handled: list[Any] = []
    real = orch._handle_trial_result

    async def spy(**kw: Any) -> Any:
        handled.append(kw["trial_result"])
        return await real(**kw)

    orch._handle_trial_result = spy  # type: ignore[method-assign]

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    with pytest.raises(OptimizationError):
        await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    refused = [t for t in handled if t.trial_id == "issued_1"]
    assert len(refused) == 1 and refused[0].status == TrialStatus.CANCELLED
    assert refused[0].metadata["backend_trial_id_acquired"] is True
    assert [cfg.get("param1") for cfg in evaluator.configs] == [1]  # drifted never ran


# ----------------------------------------------- D1/D2: entity-scoped declare semantics
def _entity_block(
    server: FakeServer, cfg: dict[str, Any]
) -> tuple[DeclareResult, dict[str, Any]]:
    """The Backend's trial proposal: a version in its own entity, entity named in the block."""
    proposal = _config_ref(server, cfg)
    blk = _trial_block(1, proposal.version_ref, [("run_base_configuration", vref(7))])
    for b in blk["bindings"]:
        if b["binding_role"] == "trial_configuration":
            b["entity_ref"] = proposal.entity_ref
    return proposal, blk


def _admit_one(
    server: FakeServer, mode: str, blk: dict[str, Any], cfg: dict[str, Any]
) -> None:
    from traigent.legend.run_binding import (
        admit_trial_config,
        pillars_from_content_identity,
    )

    binder = start_binder(
        session_id="sess_1",
        raw_block=block([("run_base_configuration", vref(7))]),
        options=LegendOptions(mode=mode, transport=server),
        privacy_enabled=False,
    )
    assert binder is not None
    binder.admit_run(pillars_from_content_identity(None, {"param1": 1}))
    admit_trial_config(binder, "t_1", blk, cfg, None)


def test_declared_under_the_advertised_entity_matches_in_strict_mode() -> None:
    server = FakeServer()
    proposal, blk = _entity_block(server, {"param1": 3})
    entities_before = len(server.entities)
    _admit_one(server, "strict", blk, {"param1": 3})
    sent = [a for a in server.admissions if "trial_id" in a][0]["trial_observed"]
    ref = [
        o["version_ref"] for o in sent if o["binding_role"] == "trial_configuration"
    ][0]
    assert ref == proposal.version_ref  # same fingerprint, same entity => same version
    assert (
        len(server.entities) == entities_before + 1
    )  # only the run's own, no stray one


def test_genuine_drift_under_the_advertised_entity_is_refused_or_recorded_as_a_child() -> (
    None
):
    server = FakeServer()
    proposal, blk = _entity_block(server, {"param1": 3})
    with pytest.raises(LegendError) as err:
        _admit_one(server, "strict", blk, {"param1": 4})
    assert err.value.code == "manifest_mismatch"
    _admit_one(server, "permitted", blk, {"param1": 4})
    sent = [a for a in server.admissions if "trial_id" in a][0]["trial_observed"]
    ref = [
        o["version_ref"] for o in sent if o["binding_role"] == "trial_configuration"
    ][0]
    assert ref != proposal.version_ref
    assert (
        server.versions[ref]["entity_ref"] == proposal.entity_ref
    )  # same-entity child
    link = [
        e
        for e in server.events
        if e.get("previous_version_ref") == proposal.version_ref
    ]
    assert link and link[0]["version_ref"] == ref  # the drift is recorded, not silent


def test_without_an_advertised_entity_the_slot_stays_unadmitted_never_copied() -> None:
    server = FakeServer()
    proposal = _config_ref(server, {"param1": 3})
    blk = _trial_block(1, proposal.version_ref, [("run_base_configuration", vref(7))])
    _admit_one(
        server, "permitted", blk, {"param1": 3}
    )  # permitted: skipped, run goes on
    assert [a for a in server.admissions if "trial_id" in a] == []
    with pytest.raises(LegendError) as err:  # strict: closed refusal
        _admit_drift_strict(server, blk)
    assert err.value.code == "local_observation_incomplete"


def test_repeated_drifted_trials_reuse_the_advertised_entity() -> None:
    server = FakeServer()
    proposal, blk = _entity_block(server, {"param1": 3})
    for n in (4, 5, 4, 5):
        _admit_one(server, "permitted", blk, {"param1": n})
    for n in (4, 5):
        fp = compute_pillar(
            c.KIND_CONFIGURATION, ConfigurationSource({"param1": n})
        ).fingerprint
        homes = {
            v["entity_ref"]
            for v in server.versions.values()
            if v["fingerprint"] == fp and v["entity_ref"] == proposal.entity_ref
        }
        # the drifted identity lives in the advertised entity (once), not in a fresh one per trial
        assert homes == {proposal.entity_ref}
        in_advertised = [
            v
            for v in server.versions.values()
            if v["fingerprint"] == fp and v["entity_ref"] == proposal.entity_ref
        ]
        assert len(in_advertised) == 1


def _admit_drift_strict(server: FakeServer, blk: dict[str, Any]) -> None:
    from traigent.legend.run_binding import (
        admit_trial_config,
        pillars_from_content_identity,
    )

    binder = start_binder(
        session_id="sess_1",
        raw_block=block([("run_base_configuration", vref(7))]),
        options=LegendOptions(transport=server),
        privacy_enabled=False,
    )
    assert binder is not None
    binder.admit_run(pillars_from_content_identity(None, {"param1": 1}))
    admit_trial_config(binder, "t_1", blk, {"param1": 4}, None)


def test_entity_reuse_via_the_alias_map_dedupes_across_declares(tmp_path: Path) -> None:
    from traigent.legend import AliasMap

    server = FakeServer()
    first = _config_ref(server, {"param1": 5})
    aliases = AliasMap(tmp_path / "aliases.json")
    aliases.set("prod config", c.KIND_CONFIGURATION, first.entity_ref)
    entity = aliases.resolve("prod config", c.KIND_CONFIGURATION)
    again = declare(
        c.KIND_CONFIGURATION,
        ConfigurationSource({"param1": 5}),
        entity=entity,
        transport=server,
    )
    assert again.version_ref == first.version_ref and again.created is False
    assert len(server.entities) == 1  # no new entity minted
    bare = _config_ref(server, {"param1": 5})  # no entity: starts another one
    assert bare.entity_ref != first.entity_ref and len(server.entities) == 2


# ----------------------------------------------- D3: the Legend client over HTTP end to end
def test_legend_client_works_end_to_end_over_the_backend_transport(
    tmp_path: Path, schemas: Any
) -> None:
    from traigent.legend import (
        BackendLegendTransport,
        CaptureIndex,
        LocalDirectoryStore,
        capture,
        query,
        record_availability,
        restore,
    )

    server = FakeServer()
    posts: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def legend_post_sync(
            self, suffix: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            posts.append((suffix, json.loads(json.dumps(payload))))
            if suffix == "/legend/declare":
                r = server.declare(payload)
                return {
                    "version_ref": r.version_ref,
                    "entity_ref": r.entity_ref,
                    "event_ref": r.event_ref,
                    "ledger_position": r.ledger_position,
                    "created_version": r.created,
                    "replayed": r.replayed,
                }
            if suffix == "/legend/query":
                q = server.query(payload)
                return {
                    "version_refs": list(q.version_refs),
                    "next_page_cursor": q.next_page_cursor,
                }
            handler = {
                "/legend/record-capture": server.record_capture,
                "/legend/record-restore": server.record_restore,
                "/legend/record-availability": server.record_availability,
            }[suffix]
            return {"event_ref": handler(payload)}

    http = BackendLegendTransport(Client())
    store = LocalDirectoryStore(tmp_path / "store")
    idx = CaptureIndex()
    first = declare(c.KIND_CONFIGURATION, ConfigurationSource({"a": 1}), transport=http)
    assert isinstance(first.ledger_position, int) and first.replayed is False
    page = query(
        first.entity_ref, as_of_ledger_position=first.ledger_position, transport=http
    )
    assert page.version_refs == (first.version_ref,)
    rec = capture(
        first.version_ref,
        ConfigurationSource({"a": 1}),
        store=store,
        transport=http,
        index=idx,
    )
    assert rec.recovery_state == "round_trip_verified"
    res = restore(
        first.version_ref, tmp_path / "dest", store=store, transport=http, index=idx
    )
    assert res.grade == "exact"
    assert record_availability(
        first.version_ref, rec.manifest_fingerprint, "available", transport=http
    )
    routes = [r for r, _ in posts]
    assert routes[0] == "/legend/declare" and "/legend/query" in routes
    for route in (
        "/legend/record-capture",
        "/legend/record-restore",
        "/legend/record-availability",
    ):
        assert route in routes
    shapes = {
        "/legend/declare": "DeclareRequestV0",
        "/legend/query": "QueryRequestV0",
        "/legend/record-capture": "RecordCaptureRequestV0",
        "/legend/record-restore": "RecordRestoreRequestV0",
        "/legend/record-availability": "RecordAvailabilityRequestV0",
    }
    for route, body in posts:
        schemas.validate("legend_requests_v0.json", shapes[route], body)


@pytest.mark.asyncio
async def test_matching_suggestion_fingerprint_does_not_substitute_for_the_advertised_entity() -> (
    None
):
    """The live path the earlier fallback used: trial_lifecycle with a suggested-fingerprint
    key in the config and a block that names NO entity. Digest equality must not admit the
    advertised version; strict refuses with a closed code and nothing is posted."""
    server = FakeServer()
    proposal = _config_ref(
        server, {"param1": 1}
    )  # the Backend's proposal for this trial
    fingerprint = compute_pillar(
        c.KIND_CONFIGURATION, ConfigurationSource({"param1": 1})
    ).fingerprint
    plan = [
        {
            "param1": 1,  # exactly what was suggested: the fingerprints are equal
            "_traigent_backend_trial_id": "t_1",
            "_traigent_legend_suggested": fingerprint,  # what the older code keyed on
            "_traigent_legend_trial": _trial_block(
                1, proposal.version_ref, [("run_base_configuration", vref(7))]
            ),  # no entity_ref
        }
    ]
    optimizer = _Optimizer({"param1": [0, 1, 2]}, ["accuracy"])
    optimizer.plan = plan
    evaluator = _Evaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        default_config={"param1": 1},
    )
    orch.legend_options = LegendOptions(transport=server)  # strict
    orch.backend_session_manager.create_session = lambda **kw: SessionContext(  # type: ignore[method-assign]
        "s", "d", "f", "o", 0.0, legend=block([("run_base_configuration", vref(7))])
    )

    async def f(input_data: dict[str, Any], **cfg: Any) -> Any:
        return "x"

    with pytest.raises(OptimizationError) as err:
        await orch.optimize(f, Dataset([EvaluationExample({"q": "a"}, "b")], name="ds"))
    assert isinstance(err.value.__cause__, LegendError)
    assert err.value.__cause__.code == "local_observation_incomplete"
    assert not [a for a in server.admissions if "trial_id" in a]
    assert proposal.version_ref not in json.dumps(server.admissions)
