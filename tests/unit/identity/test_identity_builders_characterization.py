"""Characterization of identity digest builders (output equality is the contract).

These pin exact digests, reason strings, ``source`` values and ``gaps`` so a
behaviour-preserving refactor of ``build_evaluator_binding``,
``collect_agent_build_base``, ``_bound_state`` and ``_declared_assets_of`` is
provable: certificates bind to these bytes.
"""

from __future__ import annotations

import functools
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.local import LocalEvaluator
from traigent.identity import agent_build
from traigent.identity.agent_build import (
    _bound_state,
    _declared_assets_of,
    afp2_source_digest,
    collect_agent_build_base,
    declare_agent_assets,
)
from traigent.identity.evaluator_version import (
    build_evaluator_binding,
    declare_evaluator,
)

H = "sha256:" + "b" * 64
C = "sha256:" + "a" * 64
ACC = [{"name": "accuracy", "orientation": "maximize", "weight": 1.0}]
UNAVAILABLE = "evaluator_manifest_unavailable"


@pytest.fixture(autouse=True)
def _pin_sdk_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_build, "sdk_version", lambda: "9.9.9")
    monkeypatch.setattr(agent_build, "_sdk_version", lambda: "9.9.9")


def _score(output: Any, expected: Any, **_: Any) -> float:
    return float(output == expected)


def _bind(evaluator: Any, objectives: Any = ACC, evaluator_id: str | None = None):
    return build_evaluator_binding(
        evaluator, objectives=objectives, evaluator_id=evaluator_id
    )


def _declared(target: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("judge", None)
    kwargs.setdefault("dependency_versions", {})
    kwargs.setdefault("helper_digests", {})
    return declare_evaluator(target, **kwargs)


def _custom(**kwargs: Any) -> Any:
    return LocalEvaluator(metric_functions={"accuracy": _score}, **kwargs)


def _summary(result: Any) -> tuple[Any, ...]:
    binding, source, reason = result
    if binding is None:
        return (None, source, reason)
    return (
        binding["version_digest"],
        source,
        reason,
        binding["manifest"],
        binding["evaluator_id"],
        binding["resolution"],
    )


def _closure_fn(threshold: float) -> Any:
    def scorer(output: Any, expected: Any, **_: Any) -> float:
        return float(len(str(output)) > threshold)

    return scorer


def _kw_scorer(output: Any, expected: Any, scale: float = 1.0, **_: Any) -> float:
    return scale * float(output == expected)


class _Holder:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    def score(self, output: Any, expected: Any, **_: Any) -> float:
        return float(len(str(output)) > self.limit)


class _ClassEvaluator:
    metrics = ["accuracy"]


def test_binding_none_and_early_reasons() -> None:
    assert _bind(None) == (None, None, UNAVAILABLE)
    assert _bind(LocalEvaluator(metrics=["accuracy"]), evaluator_id="bad id!") == (
        None,
        None,
        "evaluator_id_unavailable",
    )
    # conflicting declarations -> declaration error, source None
    one = _declared(_closure_fn(1.0))
    two = _declared(_closure_fn(2.0), judge={"provider": "openai", "model": "m"})
    conflicting = LocalEvaluator(metric_functions={"a": one, "b": two})
    assert _bind(conflicting) == (None, None, UNAVAILABLE)


def test_binding_check_order_and_sources() -> None:
    # judge undeclared on custom code
    assert _bind(_custom()) == (None, "fallback", UNAVAILABLE)
    assert _bind(_custom(), evaluator_id="ev1") == (None, "declared", UNAVAILABLE)
    # judge declared without config -> no config_digest
    nocfg = _declared(
        _custom(), judge={"provider": "openai", "model": "gpt"}, evaluator_id="ev1"
    )
    assert _bind(nocfg) == (None, "declared", UNAVAILABLE)
    # judge undeclared beats missing dependency_versions
    undecl = _custom()
    declare_evaluator(undecl)
    assert _bind(undecl) == (None, "fallback", UNAVAILABLE)
    # dependency_versions missing on custom code
    nodeps = _custom()
    declare_evaluator(nodeps, judge=None)
    assert _bind(nodeps) == (None, "fallback", UNAVAILABLE)
    # judge config_digest check precedes dependency_versions check
    both = _custom()
    declare_evaluator(both, judge={"provider": "openai", "model": "gpt"})
    assert _bind(both) == (None, "fallback", UNAVAILABLE)
    # evaluator class without config_digest
    cls_eval = _ClassEvaluator()
    _declared(cls_eval)
    assert _bind(cls_eval, evaluator_id="ev2") == (None, "declared", UNAVAILABLE)
    # builtin metric outside the deterministic set
    assert _bind(LocalEvaluator(metrics=["faithfulness"])) == (
        None,
        "fallback",
        UNAVAILABLE,
    )


def test_binding_failures_inside_the_try() -> None:
    band = [{"name": "accuracy", "orientation": "band", "weight": 1.0}]
    assert _bind(_declared(_custom()), band) == (None, "fallback", UNAVAILABLE)
    assert _bind(_declared(_custom()), []) == (None, "fallback", UNAVAILABLE)
    # unreadable source (lambda has no def)
    lam = LocalEvaluator(metric_functions={"accuracy": _declared(lambda o, e: 1.0)})
    assert _bind(lam) == (None, "fallback", UNAVAILABLE)
    # unserializable closure value
    sentinel = object()

    def leaky(output: Any, expected: Any, **_: Any) -> float:
        return float(sentinel is output)

    assert _bind(LocalEvaluator(metric_functions={"accuracy": _declared(leaky)})) == (
        None,
        "fallback",
        UNAVAILABLE,
    )
    # bad weight type
    bad = [{"name": "accuracy", "orientation": "maximize", "weight": "x"}]
    assert _bind(_declared(_custom()), bad) == (None, "fallback", UNAVAILABLE)
    # source resolved from helper_digests lookup failure: builtin scorer file
    # is excluded so helper_digests stays {}, but a declared-less project
    # function whose file cannot be read yields None helper_digests.
    nofile = _custom()
    declare_evaluator(nofile, judge=None, dependency_versions={})
    ev = LocalEvaluator(
        metric_functions={"accuracy": _declared(eval("lambda o, e: 1"))}
    )
    assert _bind(ev) == (None, "fallback", UNAVAILABLE)


def _expected_cases() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    cases["builtin"] = _summary(_bind(LocalEvaluator(metrics=["accuracy"])))
    cases["builtin_declared_id"] = _summary(
        _bind(LocalEvaluator(metrics=["accuracy"]), evaluator_id=" ev_1 ")
    )
    cases["function"] = _summary(_bind(_declared(_custom())))
    closure_ev = LocalEvaluator(
        metric_functions={"accuracy": _declared(_closure_fn(2.0))}
    )
    cases["closure"] = _summary(_bind(closure_ev))
    part = functools.partial(_kw_scorer, scale=0.5)
    cases["partial_kwargs"] = _summary(
        _bind(_declared(LocalEvaluator(metric_functions={"accuracy": part})))
    )
    bound = _Holder(3).score
    cases["bound_method"] = _summary(
        _bind(_declared(LocalEvaluator(metric_functions={"accuracy": bound})))
    )
    judged = _declared(
        _custom(),
        judge={"provider": "openai", "model": "gpt", "config": {"t": 0}},
        config_digest=C,
        helper_digests={"pkg/p.py": H},
        dependency_versions={"ragas": "0.2.1"},
        evaluator_id="ev_j",
    )
    cases["judged"] = _summary(_bind(judged))
    cls_ok = _ClassEvaluator()
    _declared(cls_ok, config_digest=C)
    cases["class_with_config"] = _summary(_bind(cls_ok))
    two = LocalEvaluator(
        metric_functions={
            "b": _declared(_closure_fn(1.0)),
            "a": _declared(_closure_fn(1.0)),
        }
    )
    cases["two_slots"] = _summary(
        _bind(two, [{"name": "z", "orientation": "minimize", "weight": 2}, *ACC])
    )
    return cases


def test_binding_exact_outputs() -> None:
    assert _expected_cases() == EXPECTED_BINDINGS


def test_bound_state_shapes() -> None:
    def f(x: Any) -> Any:
        return x

    assert _bound_state(None, f) == {}
    p = functools.partial(f, 1, 2, k=3)
    assert _bound_state(p, f) == {"partial_args": [1, 2], "partial_kwargs": {"k": 3}}
    assert _bound_state(functools.partial(f), f) == {}

    a, b = 1, "two"

    def g() -> Any:
        return a, b

    assert _bound_state(None, g) == {"closure": {"a": 1, "b": "two"}}

    def make_empty() -> Any:
        def h() -> Any:
            return late

        if False:  # pragma: no cover
            late = 1  # noqa: F841
        return h

    assert _bound_state(None, make_empty()) is None

    holder = _Holder(7)
    assert _bound_state(None, holder.score) == {"instance": {"limit": 7}}

    class Slotted:
        __slots__ = ()

        def m(self) -> None:
            return None

    assert _bound_state(None, Slotted().m) is None

    class Plain:
        def m(self) -> None:
            return None

    assert _bound_state(None, Plain().m) == {}

    import math

    assert _bound_state(None, math.sqrt) == {}
    # all three at once, in key order
    state = _bound_state(functools.partial(holder.score, 1, z=2), holder.score)
    assert state == {
        "partial_args": [1],
        "partial_kwargs": {"z": 2},
        "instance": {"limit": 7},
    }
    assert list(state) == ["partial_args", "partial_kwargs", "instance"]


class _FalsyMapping(dict):  # type: ignore[type-arg]
    def __bool__(self) -> bool:
        return False


class _FalsyState:
    """Instance whose ``__dict__`` is a non-empty mapping that is falsy.

    A property named ``__dict__`` supplies the mapping (plain classes cannot be
    assigned a dict subclass as ``__dict__``).
    """

    @property
    def __dict__(self) -> Any:  # type: ignore[override]
        return _FalsyMapping(secret=1)

    def m(self) -> None:
        return None


class _TruthyEmptyCopy(dict):  # type: ignore[type-arg]
    def __bool__(self) -> bool:
        return True

    def keys(self) -> Any:
        return []

    def __iter__(self) -> Any:
        return iter(())


class _TruthyState:
    @property
    def __dict__(self) -> Any:  # type: ignore[override]
        return _TruthyEmptyCopy(secret=1)

    def m(self) -> None:
        return None


def test_bound_state_truth_tests_the_raw_instance_dict() -> None:
    # raw __dict__ is falsy -> no "instance" key, even though a copy is non-empty
    assert _bound_state(None, _FalsyState().m) == {}
    # raw __dict__ is truthy -> "instance" holds dict(raw), here an empty copy
    assert _bound_state(None, _TruthyState().m) == {"instance": {}}


def test_afp2_digest_exact() -> None:
    assert afp2_source_digest(_kw_scorer) == EXPECTED_AFP2["plain"]
    assert (
        afp2_source_digest(functools.partial(_kw_scorer, scale=2.0))
        == EXPECTED_AFP2["partial"]
    )
    assert afp2_source_digest(_closure_fn(2.0)) == EXPECTED_AFP2["closure"]
    assert afp2_source_digest(_Holder(3).score) == EXPECTED_AFP2["method"]
    assert afp2_source_digest(lambda: 1) is None


def test_declared_assets_layers() -> None:
    def inner() -> None:
        return None

    declare_agent_assets(
        inner, prompts={"a": "x"}, coverage="partial", tool_definitions={}
    )

    @functools.wraps(inner)
    def outer() -> None:
        return None

    declare_agent_assets(
        outer, prompts={"b": "y"}, helper_modules={"m.py": b"1"}, coverage="complete"
    )
    found = _declared_assets_of(outer)
    # outer layer wins per category; inner fills categories outer lacks
    assert found == {
        "prompts": {"b": agent_build._declared_category({"b": "y"})["b"]},
        "tool_definitions": {},
        "helper_modules": agent_build._declared_category({"m.py": b"1"}),
        "coverage": "complete",
    }
    assert _declared_assets_of(inner)["coverage"] == "partial"
    assert _declared_assets_of(lambda: 1) == {}
    setattr(inner, agent_build._DECLARED_ASSETS_ATTR, {"prompts": "str", "coverage": 5})
    assert _declared_assets_of(inner) == {}
    setattr(inner, agent_build._DECLARED_ASSETS_ATTR, "nope")
    assert _declared_assets_of(inner) == {}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Any:
    import importlib

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(agent_build, "_SDK_ROOT", Path("/nonexistent-sdk-root"))
    return importlib.import_module(name).agent


def _project(tmp_path: Path, name: str, *, git: bool, dirty: bool = False) -> Path:
    (tmp_path / f"{name}_helper.py").write_text("RULE = 1\n", encoding="utf-8")
    (tmp_path / f"{name}.py").write_text(
        f"import {name}_helper\n\n\ndef agent(q):\n    return {name}_helper.RULE\n",
        encoding="utf-8",
    )
    if git:
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-q", "-m", "init")
        if dirty:
            (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    return tmp_path


def _base_view(base: Any) -> Any:
    if base is None:
        return None
    return (
        base.agent_id,
        base.coverage,
        base.gaps,
        base.source_digest,
        base.asset_digests,
        base.code_revision is not None and sorted(base.code_revision),
        base.code_revision and base.code_revision["dirty"],
        base.dependency_lock_digest,
        {
            **base.runtime,
            "language_version": base.runtime["language_version"]
            == platform.python_version()[:64],
        },
        base.code_revision
        and (
            base.code_revision["vcs"],
            bool(re.fullmatch(r"[0-9a-f]{40}", base.code_revision["commit"])),
            base.code_revision["dirty"],
        ),
    )


def test_collect_bad_agent_id_and_no_revision_no_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, "agb0", git=False)
    agent = _load(tmp_path, monkeypatch, "agb0")
    assert collect_agent_build_base(agent, agent_id=None) is None
    assert collect_agent_build_base(agent, agent_id="has spaces") is None
    monkeypatch.setattr(agent_build, "afp2_source_digest", lambda f: None)
    assert collect_agent_build_base(agent, agent_id="a1") is None


def test_collect_gaps_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "nogit"
    root.mkdir()
    _project(root, "agb1", git=False)
    agent = _load(root, monkeypatch, "agb1")
    view = _base_view(collect_agent_build_base(agent, agent_id="a1"))
    assert view == EXPECTED_COLLECT["nogit_enumerated"]

    declare_agent_assets(
        agent,
        prompts={"s": "p"},
        tool_definitions={},
        helper_modules={"agb1.py": (root / "agb1.py").read_bytes()},
        coverage="complete",
    )
    view = _base_view(collect_agent_build_base(agent, agent_id="a1"))
    assert view == EXPECTED_COLLECT["nogit_declared"]


def test_collect_git_clean_and_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    _project(clean, "agb2", git=True)
    agent = _load(clean, monkeypatch, "agb2")
    view = _base_view(collect_agent_build_base(agent, agent_id="a1"))
    assert view == EXPECTED_COLLECT["git_clean"]

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    _project(dirty, "agb3", git=True, dirty=True)
    agent = _load(dirty, monkeypatch, "agb3")
    view = _base_view(collect_agent_build_base(agent, agent_id="a1"))
    assert view == EXPECTED_COLLECT["git_dirty"]
    # dirty tree without a source digest -> withheld
    monkeypatch.setattr(agent_build, "afp2_source_digest", lambda f: None)
    assert collect_agent_build_base(agent, agent_id="a1") is None


def test_collect_no_source_file(monkeypatch: pytest.MonkeyPatch) -> None:
    base = collect_agent_build_base(len, agent_id="a1")  # builtin: no source file
    assert base is None
    monkeypatch.setattr(
        agent_build, "afp2_source_digest", lambda f: "sha256:" + "c" * 64
    )
    base = collect_agent_build_base(len, agent_id="a1")
    assert _base_view(base) == EXPECTED_COLLECT["no_source_file"]


EXPECTED_BINDINGS: dict[str, Any] = {
    "bound_method": (
        "sha256:42c1ca21c0a58b0776828f636b4d98394e032249a7863ae77ae813d995b12129",
        "fallback",
        None,
        {
            "code_digest": "sha256:48ee20e142bf626a1af37380a2d914192b9732f42da3c6627b6c1897c448c58b",
            "config_digest": "sha256:95c5df062a7151c08f44c4e12d4cf6a43150680582d5a47b0e920fa1c0ffd479",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "builtin": (
        "sha256:37af35dc2be195d158e799d2af44a473093234b497f23ba3737f528aafae1c91",
        "fallback",
        None,
        {
            "code_digest": "sha256:db5fdfc0b39c4d52e0db2018cd5b04afd185e03df9452d29d266e5bb48a6a138",
            "config_digest": "sha256:c11de4432a8b40d4533c65afc2fcd51b9a6fa51c1dbe79e158a5a73ec89cd7bb",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "builtin_declared_id": (
        "sha256:6fc4a5f3709828e6f8a64eed3f365f6994999cb1c3488f095e634c078fa2e63b",
        "declared",
        None,
        {
            "code_digest": "sha256:db5fdfc0b39c4d52e0db2018cd5b04afd185e03df9452d29d266e5bb48a6a138",
            "config_digest": "sha256:c11de4432a8b40d4533c65afc2fcd51b9a6fa51c1dbe79e158a5a73ec89cd7bb",
            "dependency_versions": {},
            "evaluator_id": "ev_1",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "ev_1",
        "declared_at_session_start",
    ),
    "class_with_config": (
        "sha256:0f1fda8d67c81414b90e7f89fd29be08795ce5bacf36e671f387103af9b3d25c",
        "fallback",
        None,
        {
            "code_digest": "sha256:4877e1eea0d092b0db998041d6cd9f414a2fda3c80df838b8a66c2c6a3b11f52",
            "config_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "closure": (
        "sha256:5b08f685029ed4041c61a0914e18c74790d25a8cac9db95248c9bc7538d23808",
        "fallback",
        None,
        {
            "code_digest": "sha256:af74a9e0b7d73de026df13fd547194f91babe49ce0b7e37e70fb3d5d59a09994",
            "config_digest": "sha256:99f44223d78c5095dd7464c779c9a952846aff574431eb586516972101218f39",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "function": (
        "sha256:16ec0b0dde182f0df2c6097f2ef309ca5f2f782a23037bdc56193084bccfa564",
        "fallback",
        None,
        {
            "code_digest": "sha256:26c3fc6e7ea4b6e0b3ea29d0f35bbe341e98f2b381b0e3551efba8c85d3763c5",
            "config_digest": "sha256:c11de4432a8b40d4533c65afc2fcd51b9a6fa51c1dbe79e158a5a73ec89cd7bb",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "judged": (
        "sha256:6a835f210a18f2f7c948afc80fd004f02e760b8d99325dc71dc3a79022767449",
        "declared",
        None,
        {
            "code_digest": "sha256:26c3fc6e7ea4b6e0b3ea29d0f35bbe341e98f2b381b0e3551efba8c85d3763c5",
            "config_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dependency_versions": {"ragas": "0.2.1"},
            "evaluator_id": "ev_j",
            "helper_digests": {
                "pkg/p.py": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "judge": {
                "config_digest": "sha256:746d69ef020c2af9b6792fe7c12f9dd9ac437bbe2a65f0b0c41919e53c5dceab",
                "model": "gpt",
                "provider": "openai",
            },
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "ev_j",
        "declared_at_session_start",
    ),
    "partial_kwargs": (
        "sha256:c11bf46a245be8dc54c1ba096ed8164665b66d184f15dabb73f45269ea2d498a",
        "fallback",
        None,
        {
            "code_digest": "sha256:d45f2c48bcb69773bddd36e9935c1db49d15a90aca6c4ad8f79b6601acfa4643",
            "config_digest": "sha256:150b80d27bad00b69847c33950ac1bc0b9f1f08a9019b9f47851e8b2e91e96e7",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
    "two_slots": (
        "sha256:905fc901b9a3767d2a08e8bfa7b2684efaf81ea0852671fb13f766fd3d3dfda1",
        "fallback",
        None,
        {
            "code_digest": "sha256:70724df3b5e6297146d9f787edf4ea8fdd394eea4fb3e14abeb9985cc7dd1ed7",
            "config_digest": "sha256:637545994cd694d9a4d9f9da6cf4e4572fd9a4c2f4dd2ed9cc69fe03a6f233c4",
            "dependency_versions": {},
            "evaluator_id": "sdk_local_evaluator",
            "helper_digests": {},
            "judge": None,
            "manifest_version": 1,
            "objectives": [
                {"name": "accuracy", "orientation": "maximize", "weight": 1.0},
                {"name": "z", "orientation": "minimize", "weight": 2},
            ],
        },
        "sdk_local_evaluator",
        "declared_at_session_start",
    ),
}

EXPECTED_AFP2: dict[str, Any] = {
    "closure": "sha256:cc4a422b19020b092aab433d88720e5d24db4454ec7778e43f4b09e4bbf440a2",
    "method": "sha256:76abb4eacceb50b331b046b63bd0909d1df70404be9c65e15f4d99b73f0f1f46",
    "partial": "sha256:33cc07fc44ff68e59288cc59049847c6022cec5d679737e4d38c9b7813759ca3",
    "plain": "sha256:ed85871a7273c8aca3a1a2097eada82dd3957be72cbee20cb0714f64d6973e11",
}
EXPECTED_COLLECT: dict[str, Any] = {
    "git_clean": (
        "a1",
        "partial",
        (
            "coverage_not_declared_complete",
            "helper_modules_not_declared",
            "prompts_not_declared",
            "tool_definitions_not_declared",
        ),
        "sha256:e240669f25c8c1127b9702f8455a7cb6707e2afbce62d10c97e00830b387389c",
        {
            "helper_modules": {
                "agb2.py": "sha256:034f192f231f9eecf55b1d6ea2675f85f53f2691fe68b346e90c278318f48ed8",
                "agb2_helper.py": "sha256:fb9f4293d1b691689d64b398ed908e450b93a3760f223ce66affd8cd51fd2261",
            },
            "prompts": {},
            "tool_definitions": {},
        },
        ["commit", "dirty", "vcs"],
        False,
        None,
        {"language": "python", "language_version": True, "sdk_version": "9.9.9"},
        ("git", True, False),
    ),
    "git_dirty": (
        "a1",
        "partial",
        (
            "coverage_not_declared_complete",
            "dirty_files_outside_manifest",
            "helper_modules_not_declared",
            "prompts_not_declared",
            "tool_definitions_not_declared",
        ),
        "sha256:26f6a6634cff460858c22957f2f6e0a474ea2dd3204a86409d7a7caa2508f777",
        {
            "helper_modules": {
                "agb3.py": "sha256:e3014d61d1d0b282aa00c30939d9ce8da52c8b58a99089532450cb16e5b97ed5",
                "agb3_helper.py": "sha256:fb9f4293d1b691689d64b398ed908e450b93a3760f223ce66affd8cd51fd2261",
            },
            "prompts": {},
            "tool_definitions": {},
        },
        ["commit", "dirty", "vcs"],
        True,
        None,
        {"language": "python", "language_version": True, "sdk_version": "9.9.9"},
        ("git", True, True),
    ),
    "no_source_file": (
        "a1",
        "partial",
        (
            "coverage_not_declared_complete",
            "entry_source_file_unknown",
            "helper_modules_not_declared",
            "prompts_not_declared",
            "tool_definitions_not_declared",
        ),
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        {"helper_modules": {}, "prompts": {}, "tool_definitions": {}},
        False,
        None,
        None,
        {"language": "python", "language_version": True, "sdk_version": "9.9.9"},
        None,
    ),
    "nogit_declared": (
        "a1",
        "partial",
        ("no_code_revision",),
        "sha256:7bc2cd408d33cc763dd5e8b2ab98dcaec028569fbd5bae9af87ea831e6e14687",
        {
            "helper_modules": {
                "agb1.py": "sha256:2044bf941fa3e3633b92575aff14df6d7469cefa6386e86e38e8b95390468b26"
            },
            "prompts": {
                "s": "sha256:148de9c5a7a44d19e56cd9ae1a554bf67847afb0c58f6e12fa29ac7ddfca9940"
            },
            "tool_definitions": {},
        },
        False,
        None,
        None,
        {"language": "python", "language_version": True, "sdk_version": "9.9.9"},
        None,
    ),
    "nogit_enumerated": (
        "a1",
        "partial",
        (
            "coverage_not_declared_complete",
            "helper_modules_not_declared",
            "no_code_revision",
            "prompts_not_declared",
            "tool_definitions_not_declared",
        ),
        "sha256:7bc2cd408d33cc763dd5e8b2ab98dcaec028569fbd5bae9af87ea831e6e14687",
        {
            "helper_modules": {
                "agb1.py": "sha256:2044bf941fa3e3633b92575aff14df6d7469cefa6386e86e38e8b95390468b26",
                "agb1_helper.py": "sha256:fb9f4293d1b691689d64b398ed908e450b93a3760f223ce66affd8cd51fd2261",
            },
            "prompts": {},
            "tool_definitions": {},
        },
        False,
        None,
        None,
        {"language": "python", "language_version": True, "sdk_version": "9.9.9"},
        None,
    ),
}
