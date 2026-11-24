"""SDK-129 group A regressions for the tuning loop (N60, N71, N122, N123, N1255, N1256, N1258)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.tuning.controller import traigent_optimize_fn
from traigent.tuning.errors import TuningError
from traigent.tuning.evidence import _accuracy, examples_digest
from traigent.tuning.state import initial_config_of

from .conftest import tvl_free
from .test_controller import (
    FakeOptimizer,
    _cli_fixture,
)
from .test_controller import _holdout as _kit_holdout  # noqa: F401 - _holdout is an autouse fixture the kit needs
from .test_controller import at, controller, e_ref


def test_n1255_examples_digest_ignores_row_order() -> None:
    a = {"input": {"q": "x"}, "output": 1}
    b = {"input": {"q": "y"}, "output": 2}
    assert examples_digest([a, b]) == examples_digest([b, a])
    assert examples_digest([a, b]) != examples_digest([a, a])


def test_n1258_accuracy_does_not_equate_bool_and_int() -> None:
    examples = [{"output": 1}, {"output": True}, {"output": [1]}, {"output": 1.0}]
    runs = [(True, True), (True, 1), (True, [True]), (True, 1.0)]
    assert _accuracy(examples, runs) == [0.0, 0.0, 0.0, 1.0]


def test_n1258_accuracy_nested_bool_in_non_dict_mapping() -> None:
    from collections import UserDict

    examples = [{"output": {"ok": 1}}, {"output": {"ok": 1}}]
    runs = [(True, UserDict({"ok": True})), (True, UserDict({"ok": 1}))]
    assert _accuracy(examples, runs) == [0.0, 1.0]  # the plain-int mapping matches


def test_n123_unresolvable_dimension_gives_empty_initial_config() -> None:
    optimized = types.SimpleNamespace(
        default_config={},
        configuration_space={
            "model": ["a", "b"],
            "temperature": {"low": 0.0, "high": 1.0},
        },
    )
    assert initial_config_of(optimized) == {}


def test_n122_spend_limit_shows_the_decorator_cost_limit() -> None:
    fn = types.SimpleNamespace(
        max_trials=10, timeout=None, kwargs={"cost_limit": 0.5, "max_trials": 10}
    )
    run = traigent_optimize_fn(fn)
    assert run.spend_limit == {"max_trials": 10, "cost_limit": 0.5}  # type: ignore[attr-defined]


def test_n60_same_generation_head_with_wrong_fingerprint_is_refused(
    tmp_path: Path,
) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    head_path = tmp_path / "store" / "head.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    head["config"] = {"tampered": True}
    head_path.write_text(json.dumps(head), encoding="utf-8")
    with pytest.raises(TuningError) as err:
        c.step(at(10))
    assert err.value.code == "E_HEAD_DIVERGED"
    assert opt.calls == []


def test_n71_tvl_and_target_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _auth, common = _cli_fixture(tmp_path, monkeypatch)
    target = tmp_path / "agent.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["tune", "status", *common, "--target", f"{target}:x"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_n1256_target_module_that_exits_is_a_load_failure(tmp_path: Path) -> None:
    target = tmp_path / "agent.py"
    target.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["tune", "compile", f"{target}:fn"])
    assert result.exit_code != 0
    assert "module_load_failed" in result.output


def test_g7_approvers_mapping_is_refused_not_reduced_to_its_names() -> None:
    from traigent.tuning.authority import authority_from_config

    with pytest.raises(TuningError):
        authority_from_config({"approvers": {"alice": {"role": "x"}}})
    authority, _ = authority_from_config({"approvers": ["alice"]})
    assert authority is not None and authority.authorized("alice")


def test_n1258_accuracy_nested_bool_across_container_types() -> None:
    from collections import UserList

    examples = [{"output": [1]}, {"output": [1]}]
    runs = [(True, UserList([True])), (True, [1])]
    assert _accuracy(examples, runs) == [0.0, 1.0]  # the plain-int mapping matches


def test_n122_real_decorated_function_declared_cost_limit(monkeypatch) -> None:
    import traigent

    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "0.73")

    @traigent.optimize(
        configuration_space={"x": [1, 2]}, objectives=["accuracy"], cost_limit=0.5
    )
    def fn(x=1):
        return x

    run = traigent_optimize_fn(fn)
    # declared settings only: the explicit decorator cap, labelled as declared
    assert run.spend_limit.get("cost_limit") == 0.5  # type: ignore[attr-defined]


def test_sol1_missing_head_is_refused_before_the_marker(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    (tmp_path / "store" / "head.json").unlink()
    with pytest.raises(TuningError) as err:
        c.step(at(10))
    assert err.value.code == "E_HEAD_DIVERGED"
    assert opt.calls == []


def test_sol2_legacy_order_sensitive_digest_still_refuses_the_holdout(
    tmp_path,
) -> None:
    import json

    from tests.unit.tuning.test_evidence import _build, _holdout
    from traigent.tuning.evidence import EvidenceUnavailable, _legacy_examples_digest

    rows = [
        {"input": {"q": "y"}, "output": 2},
        {"input": {"q": "x"}, "output": 1},
    ]
    # reverse-sorted rows: the sorted (order-independent) digest differs from legacy
    assert _legacy_examples_digest(rows) != _legacy_examples_digest(rows[::-1])
    assert _legacy_examples_digest(rows) != examples_digest(rows)
    holdout = _holdout(tmp_path)
    held = [json.loads(line) for line in holdout.read_text().splitlines()]
    with pytest.raises(EvidenceUnavailable) as err:
        _build(tmp_path, eval_set_digests={_legacy_examples_digest(held)})
    assert err.value.reason == "holdout_is_optimization_eval_set"


def test_sol3_tuple_never_equals_list() -> None:
    assert _accuracy([{"output": [1]}], [(True, (1,))]) == [0.0]
    assert _accuracy([{"output": [1]}], [(True, [1])]) == [1.0]


def test_sol4_nested_bool_in_resolved_domain_member_is_refused() -> None:
    from traigent.tvl.spec_loader import TVLValidationError
    from traigent.tvl.spec_loader import _check_default_in_resolved_domain as check

    with pytest.raises(TVLValidationError):
        check("t", (True, "x"), [(1, "x"), (2, "x")])
    check("t", (1, "x"), [(1, "x"), (2, "x")])


def test_sol5_rounding_cannot_turn_untestable_into_reject() -> None:
    from traigent.tvl.promotion_gate import (
        ObjectiveSpec,
        PromotionGate,
        PromotionPolicy,
    )

    gate = PromotionGate(
        PromotionPolicy(min_effect={"a": 0.02}), [ObjectiveSpec("a", "maximize")]
    )
    decision = gate.evaluate({"a": [0.03] * 13}, {"a": [0.01] * 13})
    assert decision.decision == "no_decision"


@tvl_free
def test_sol6_context_mode_keeps_the_declared_config() -> None:
    import traigent

    @traigent.optimize(
        configuration_space={"model": ["a", "b"], "t": (0.0, 1.0)},
        default_config={"model": "a"},
        objectives=["accuracy"],
    )
    def fn(model="a", t=0.5):
        return model

    # default (context) injection mode: no signature default is what runs
    assert initial_config_of(fn) == {"model": "a"}


@tvl_free
def test_sol6_seamless_partial_default_keeps_declared_settings() -> None:
    def agent():
        t = 0.5
        return t

    optimized = types.SimpleNamespace(
        func=agent,
        injection_mode="seamless",
        default_config={"mode": "declared"},
        configuration_space={"mode": ["declared", "other"], "t": (0.0, 1.0)},
    )
    assert initial_config_of(optimized) == {"mode": "declared"}


def test_sol7_implicit_max_trials_is_not_declared() -> None:
    import traigent

    @traigent.optimize(configuration_space={"x": [1, 2]}, objectives=["accuracy"])
    def omitted(x=1):
        return x

    @traigent.optimize(
        configuration_space={"x": [1, 2]}, objectives=["accuracy"], max_trials=10
    )
    def explicit(x=1):
        return x

    assert "max_trials" not in traigent_optimize_fn(omitted).spend_limit  # type: ignore[attr-defined]
    assert traigent_optimize_fn(explicit).spend_limit["max_trials"] == 10  # type: ignore[attr-defined]


def test_sol8_second_approver_gets_already_approved_naming_the_first(
    tmp_path: Path,
) -> None:
    c = controller(
        tmp_path,
        FakeOptimizer(),
        authorized_approvers=frozenset({"approver-1", "approver-2"}),
    )
    first = c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    second = c.approve_optimization(e_ref("E1"), "approver-2", at(11))
    assert first == {"approver": "approver-1", "already_approved": False}
    assert second == {"approver": "approver-1", "already_approved": True}


@tvl_free
def test_unserializable_signature_default_is_omitted_sdk129_bar_r2() -> None:
    import json

    fallback = object()

    def agent(format=fallback):
        return format

    optimized = types.SimpleNamespace(
        func=agent,
        injection_mode="seamless",
        default_config={"mode": "live"},
        configuration_space={"mode": ["live", "x"], "format": ["short", "long"]},
    )
    config = initial_config_of(optimized)
    assert config == {"mode": "live"}
    json.dumps(config)


def test_masked_numpy_bool_is_not_a_number_sdk129_bar_r2() -> None:
    np = pytest.importorskip("numpy")
    from traigent.tuning.evidence import _exact_match

    assert not _exact_match(np.ma.array([True]), [1])
    assert not _exact_match(np.ma.array([1]), [True])
    assert _exact_match(np.ma.array([1, 2]), [1, 2])
    assert not _exact_match(np.ma.array([1, 2], mask=[False, True]), [1, 2])
