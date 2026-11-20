"""Merge-review blockers: code version, authority, clock, store environment,
holdout identity, domains and journal durability."""

from __future__ import annotations

import os
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import controller as controller_module
from traigent.tuning.authority import ApproverAuthority
from traigent.tuning.code_version import CODE_VERSION_ENV, resolve_code_version
from traigent.tuning.errors import TuningError

from .test_controller import (  # noqa: F401  (_holdout is an autouse fixture)
    APPROVERS,
    KIT,
    NEW,
    OLD,
    FakeOptimizer,
    _deploy_awaiting,
    _holdout,
    at,
    controller,
    e_ref,
    event,
    head,
    make,
    state,
)

# -- 1. the code version hashes the agent, not the SDK wrapper --------------------------
_COUNTER = [0]


def _agent_module(tmp_path: Path, body: str) -> Any:
    """Import ``body`` as a fresh real module (kept in sys.modules, so inspect finds it)."""
    import importlib.util

    _COUNTER[0] += 1
    name = f"agent_mod_under_test_{_COUNTER[0]}"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_default_code_version_hashes_the_agent_under_the_sdk_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    import traigent

    template = """
        import traigent

        @traigent.optimize(
            configuration_space={"model": ["m"]}, objectives=["accuracy"],
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}], offline=True,
            max_trials=1,
        )
        def agent(q):
            return "VERSION"
        """
    first = _agent_module(tmp_path, template.replace("VERSION", "one"))
    second = _agent_module(tmp_path, template.replace("VERSION", "two"))
    assert type(first.agent).__module__.startswith("traigent.core")
    v1 = resolve_code_version(None, first.agent)
    v2 = resolve_code_version(None, second.agent)
    assert v1 != v2  # editing the agent changes the digest; the SDK wrapper does not
    assert traigent is not None


def test_a_functools_wrapper_does_not_hide_the_agent_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    mod = _agent_module(tmp_path, "def agent(q):\n    return 1\n")

    def wrapper(*a: Any, **k: Any) -> Any:  # defined in THIS module, no functools.wraps
        return mod.agent(*a, **k)

    wrapper.func = mod.agent  # type: ignore[attr-defined]
    assert resolve_code_version(None, wrapper) == resolve_code_version(None, mod.agent)


def test_unreadable_agent_source_is_refused_even_if_a_wrapper_has_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CODE_VERSION_ENV, raising=False)
    namespace: dict[str, Any] = {"__name__": "no_such_source_module"}
    exec("def agent(q):\n    return 1\n", namespace)  # noqa: S102 - no file behind it

    def wrapper(*a: Any, **k: Any) -> Any:
        return None

    wrapper.func = namespace["agent"]  # type: ignore[attr-defined]
    with pytest.raises(TuningError) as err:
        resolve_code_version(None, wrapper)
    assert err.value.code == "E_NO_CODE_VERSION"


# -- 2. authority fails closed -----------------------------------------------------------
def _hook(authority_hook: Any) -> ApproverAuthority:
    return ApproverAuthority(
        frozenset(APPROVERS), mode="platform", authenticator=authority_hook
    )


def test_only_a_literal_true_from_a_synchronous_hook_authenticates() -> None:
    async def async_false(approver: str, context: Any) -> bool:
        return False

    assert _hook(async_false).authenticated("boss", {}) is False  # coroutine: refused
    assert _hook(lambda a, c: "yes").authenticated("boss", {}) is False
    assert _hook(lambda a, c: 1).authenticated("boss", {}) is False
    assert _hook(lambda a, c: None).authenticated("boss", {}) is False
    assert _hook(lambda a, c: True).authenticated("boss", {}) is True


def test_a_saved_approval_is_not_used_once_the_environment_is_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    monkeypatch.setenv(
        "ENVIRONMENT", "production"
    )  # the process now runs in production
    report = c.step(at(10))
    assert opt.calls == [] and report.awaiting_approval == [e_ref("E1")]


def test_a_saved_acceptance_authority_is_not_used_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(TuningError) as err:
        c.submit_acceptance(record, at(31))
    assert err.value.code == "E_LOCAL_AUTHORITY_IN_PRODUCTION"
    assert c._on_list_now("approver-1", record) is False
    assert head(tmp_path)["generation"] == 0


# -- 3. the clock is re-read before each paid run and at the release commit -------------
def _fake_clock(monkeypatch: pytest.MonkeyPatch, start: Any) -> list[Any]:
    clock = [start]
    real_utc = controller_module._utc
    monkeypatch.setattr(
        controller_module,
        "_utc",
        lambda value: real_utc(value) if value is not None else clock[0],
    )
    return clock


def test_a_second_paid_run_is_not_started_after_the_grant_expired_during_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .test_controller import _standing_state

    clock = _fake_clock(monkeypatch, at(40))
    ran: list[int] = []

    class SlowOptimizer(FakeOptimizer):
        def __call__(self, space: dict[str, Any]) -> Any:
            ran.append(len(ran))
            clock[0] = clock[0] + timedelta(hours=3)  # the first run outlasts the grant
            return super().__call__(space)

    opt = SlowOptimizer()
    c = make(_standing_state(), tmp_path / "store", opt, now=at(0))
    c.record_event(event("E1", NEW["model"], 39), at(40))
    c.record_event(event("E6", "anthropic/claude-haiku-5-6", 39), at(40))
    c.record_standing_grant(
        "G1", "tuning/on_new_model", 5, at(40) + timedelta(hours=2), "issuer-1", at(40)
    )
    c.step(ingestion_closed=True)  # no explicit time: the controller reads the clock
    assert len(ran) == 1


def test_a_release_whose_acceptance_expires_during_authentication_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _deploy_awaiting(tmp_path)
    record = c.build_acceptance(e_ref("E1"), "approver-1", 0, now=at(31))
    clock = _fake_clock(monkeypatch, at(32))

    def slow_hook(approver: str, context: Any) -> bool:
        clock[0] = clock[0] + timedelta(days=30)  # authentication is slow
        return True

    c.authority = ApproverAuthority(
        frozenset(APPROVERS), mode="platform", authenticator=slow_hook
    )
    outcome = c.submit_acceptance(record)
    assert outcome.result != "written"
    assert head(tmp_path)["generation"] == 0


# -- N1. --tvl runs any lint-clean module, even outside the builder's shape --------------
def test_cli_tvl_file_outside_the_builder_shape_runs_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import yaml
    from click.testing import CliRunner

    from traigent.cli.main import cli
    from traigent.tuning.policy import Tuning

    from .test_controller import _cli_fixture, module

    auth, common = _cli_fixture(tmp_path, monkeypatch)
    doc = module()
    for monitor in doc["monitors"]:
        if monitor.get("phase") == "promotion":
            monitor["decision"]["min_samples"] = 777  # custom, not the builder's value
    for req in doc["requirements"]:
        req["violation"]["label_in"] = ["fail", "error"]
    (tmp_path / "module.yml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(TuningError):
        Tuning.from_tvl(doc)  # the premise: outside the builder shape
    result = CliRunner().invoke(
        cli,
        ["tune", "status", *common, "--authority", str(auth)],
    )
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    assert "Traceback" not in result.output
    assert "E_NOT_ROUNDTRIPPABLE" not in result.output
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)


# -- 4. the store is per environment -----------------------------------------------------
def test_default_store_path_includes_the_environment() -> None:
    from traigent.tuning.controller import default_store_dir

    assert default_store_dir("agent-1", "staging") != default_store_dir(
        "agent-1", "production"
    )
    assert "staging" in default_store_dir("agent-1", "staging").parts


def test_reopening_a_store_for_another_environment_is_refused(tmp_path: Path) -> None:
    from traigent.tuning.controller import E_STORE_ENVIRONMENT_MISMATCH, HeadLoader

    store = tmp_path / "store"
    make(state(), store, FakeOptimizer(), environment="staging")
    with pytest.raises(TuningError) as err:
        make(state(), store, FakeOptimizer(), environment="production")
    assert err.value.code == E_STORE_ENVIRONMENT_MISMATCH
    make(state(), store, FakeOptimizer(), environment="staging")  # same env: fine
    with pytest.raises(TuningError) as err:
        HeadLoader(store, "production").head()
    assert err.value.code == E_STORE_ENVIRONMENT_MISMATCH
    assert HeadLoader(store, "staging").head() is not None


# -- 5. holdout identity for every supported eval_dataset form ---------------------------
def test_eval_set_identity_covers_every_supported_form(tmp_path: Path) -> None:
    from traigent.evaluators.base import Dataset, EvaluationExample
    from traigent.tuning.evidence import (
        EvidenceUnavailable,
        eval_set_identity,
        examples_digest,
    )

    rows = [{"input": {"x": i}, "output": i} for i in range(3)]
    path = tmp_path / "h.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    wanted = examples_digest(rows)
    for form in (
        str(path),
        [str(path)],  # a list of paths
        Dataset(examples=[EvaluationExample({"x": i}, i) for i in range(3)]),
        rows,
    ):
        digests, loaded = eval_set_identity(form)
        assert wanted in digests and loaded, form
    with pytest.raises(EvidenceUnavailable) as err:  # unreadable: refuse, never empty
        eval_set_identity([str(tmp_path / "missing.jsonl")])
    assert err.value.reason == "eval_set_identity_unknown"
    with pytest.raises(EvidenceUnavailable):
        eval_set_identity(object())


def test_a_holdout_listed_as_a_path_list_eval_set_is_refused(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())

    class Optimized:
        eval_dataset = [str(KIT["holdout"])]  # the SAME file as the holdout

        def __call__(self, x: int) -> str:
            return "billing"

        def _dataset_from_source(self, source: Any) -> Any:
            from traigent.core.optimized_function import OptimizedFunction

            return OptimizedFunction._dataset_from_source(OptimizedFunction, source)

    c.state.optimized = Optimized()
    c.record_event(event("E1", NEW["model"], 9), at(10))
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    c.step(at(10))
    assert any(
        f.get("reason") == "holdout_is_optimization_eval_set"
        for f in c.controller_facts()
    )


# -- 6. TVL domains are normalized; explicit strict pricing comes before exemptions ------
def test_a_set_domain_is_normalized_before_computing_admitted_models(
    tmp_path: Path,
) -> None:
    from .test_controller import module

    doc = module()
    declared = list(doc["tvars"][0]["domain"])
    doc["tvars"][0]["domain"] = {"set": declared}
    st = state(doc)
    c = make(st, tmp_path / "store", FakeOptimizer())
    fact = {"tvar": "model", "domain": [*declared, NEW["model"]]}
    assert c._admitted_values(fact) == [NEW["model"]]  # not every declared model
    assert c._search_space(fact)["model"] == [*declared, NEW["model"]]
    assert c._admitted_values({"tvar": "model", "domain": declared}) == []


def test_explicit_strict_cost_accounting_beats_the_unpriced_model_exemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traigent
    from traigent.core.optimized_function import UnknownModelError

    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    for name in ("TRAIGENT_COST_APPROVED", "TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT"):
        monkeypatch.delenv(name, raising=False)
    unpriced = "anthropic/claude-haiku-9-9"

    @traigent.optimize(
        configuration_space={"model": ["anthropic/claude-haiku-4-5", unpriced]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=2,
        default_config={"model": "anthropic/claude-haiku-4-5"},
    )
    def agent(q: str) -> str:
        return q

    agent._unpriced_models_allowed = frozenset({unpriced})
    space = {"model": ["anthropic/claude-haiku-4-5", unpriced]}
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    agent._preflight_model_cost_coverage(space)  # exempt: passes without strict
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    with pytest.raises(UnknownModelError):
        agent._preflight_model_cost_coverage(space)


def test_an_unreadable_price_table_means_unpriced_never_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traigent.utils.cost_calculator as calc

    def boom(_ids: Any) -> Any:
        raise RuntimeError("price table unreadable")

    monkeypatch.setattr(calc, "find_models_missing_price_coverage", boom)
    assert controller_module._unpriced_models_of({"model": "some/model-x"}) == [
        "some/model-x"
    ]


# -- 7. journal durability: the directory entry is fsynced -------------------------------
def test_creating_and_renaming_journal_files_fsyncs_the_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    synced_dirs: list[int] = []
    real = os.fsync

    def spy(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_dirs.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    controller_module._write_json(tmp_path / "head.json", {"generation": 1})
    assert len(synced_dirs) == 1  # the rename of head.json is durable
    synced_dirs.clear()
    controller_module._append_line(tmp_path / "commits.jsonl", {"fact": "x"})
    assert len(synced_dirs) == 1  # the creation of commits.jsonl is durable
    synced_dirs.clear()
    controller_module._append_line(tmp_path / "commits.jsonl", {"fact": "y"})
    assert synced_dirs == []  # an append to an existing file adds no directory entry


# -- 11 (SDK side). a whole-second instant reaches the relation as ONE step --------------
def test_events_with_the_same_instant_form_one_relation_step(tmp_path: Path) -> None:
    c = controller(tmp_path, FakeOptimizer())
    lines = [
        {"instant": 100, "category": "events", "record": {"id": "from-Z"}},
        {"instant": 100, "category": "events", "record": {"id": "from-A"}},
        {"instant": 101, "category": "events", "record": {"id": "later"}},
    ]
    steps = c._history(lines=lines)["steps"]
    assert [s["instant"] for s in steps] == [100, 101]
    assert steps[0]["events"] == [{"id": "from-Z"}, {"id": "from-A"}]
