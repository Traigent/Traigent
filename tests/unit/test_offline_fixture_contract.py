"""Contract for the autouse offline fixture's connected/offline decision (#2033).

This file lives OUTSIDE ``tests/unit/cloud/`` on purpose. Before #2033 the
``jwt_development_mode`` fixture decided offline mode from the test's file path
(``"tests/unit/cloud/" in _node_path``), so a connected test's semantics changed
when the file moved — it silently flipped offline and kept passing while no
longer exercising the path it named (PR #2026 / #2020).

The properties under test:

* **the marker decides, the directory does not.** Proven by *running* a marked
  and an unmarked probe from INSIDE ``tests/unit/cloud/`` — the exact directory
  the deleted carve-out named — and requiring OPPOSITE outcomes there. A grep
  for the old path literal would not prove this: the carve-out can be
  reinstated without ever spelling that literal (a subdirectory ``conftest.py``
  is itself a path-derived condition), so behaviour is asserted, not spelling.
* **offline-by-default.** Anything unmarked stays offline. That zero-egress
  default is load-bearing and must never weaken.
* **an ambient offline switch does not suppress the marker.** A marked test
  must still RUN, and still observe connected mode, when the environment
  exports ``TRAIGENT_OFFLINE`` / ``TRAIGENT_OFFLINE_MODE``. Four CI lanes
  (publish, release-review, sonarcloud, tests) export ``TRAIGENT_OFFLINE_MODE``
  around ``pytest tests/unit``, and none of them asserts a skip count — so a
  fixture that skipped on an ambient switch would silently convert the entire
  cloud suite from "ran" to "skipped" and still exit 0, including in the gate
  in front of a PyPI publish.
* **an unregistered marker is a hard error.** ``strict_markers`` must stay
  honoured, or a typo'd ``backend_onlne`` degrades to a warning and the test
  runs offline under a connected name.

Note: ``backend_online`` is not permission for real network egress — transports
must still be mocked. Nothing here opens a socket; the probes only read
environment variables.
"""

from __future__ import annotations

import ast
import contextlib
import itertools
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import pytest

from traigent.utils.env_config import is_backend_offline

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_TESTS_ROOT = _REPO_ROOT / "tests"
_ROOT_CONFTEST = _TESTS_ROOT / "conftest.py"
# The directory the deleted carve-out named. Probes are executed from INSIDE it.
_CLOUD_DIR = _TESTS_ROOT / "unit" / "cloud"

_OFFLINE_ENV_NAMES = ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE")

_PROBE_COUNTER = itertools.count()


# --------------------------------------------------------------------------
# Test bodies below are referenced by node id in subprocess runs. Deriving the
# ids from ``__file__`` and ``__name__`` (never spelling them) keeps this file
# rename-safe and move-safe: relocating it cannot leave a stale literal behind
# that silently selects nothing.
# --------------------------------------------------------------------------


def _node_id(func) -> str:
    """Node id of a test defined in this module, derived rather than spelled."""
    relative = _THIS_FILE.relative_to(_REPO_ROOT).as_posix()
    return f"{relative}::{func.__name__}"


def test_unmarked_test_is_offline():
    """Offline-by-default: no marker means offline, whatever the directory."""
    assert os.environ["TRAIGENT_OFFLINE_MODE"] == "true"
    assert is_backend_offline() is True


@pytest.mark.backend_online
def test_marked_test_is_connected():
    """The marker alone flips the fixture — outside tests/unit/cloud/.

    Both spellings must be "false": is_backend_offline() ORs them, so clearing
    only TRAIGENT_OFFLINE_MODE would leave an ambient TRAIGENT_OFFLINE=true in
    force. The two are NOT interchangeable — see #1773.
    """
    assert os.environ["TRAIGENT_OFFLINE_MODE"] == "false"
    assert os.environ["TRAIGENT_OFFLINE"] == "false"
    assert is_backend_offline() is False


def test_this_file_is_outside_the_old_cloud_carve_out():
    """Guards the premise of the two tests above.

    If this file is ever moved into ``tests/unit/cloud/`` the pair stops
    demonstrating anything, because a path carve-out would explain the marked
    test's result just as well as the marker does. (The in-cloud direction is
    covered separately, by the probes below.)
    """
    assert _CLOUD_DIR not in _THIS_FILE.parents


# --------------------------------------------------------------------------
# Subprocess harness
# --------------------------------------------------------------------------


class _Inherit:
    """Sentinel: pass the parent process's value for this variable through."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover — debugging aid only
        return "<inherit>"


INHERIT = _Inherit()

# Every case must declare BOTH offline spellings. ``INHERIT`` keeps whatever
# this process currently has — the operator's export, or the value the autouse
# fixture wrote over it — so the child is never LESS offline than the parent.
# Use it for cases whose expectation holds under any ambient environment.
_INHERIT_OFFLINE: dict[str, str | None | _Inherit] = dict.fromkeys(
    _OFFLINE_ENV_NAMES, INHERIT
)


def _run_pytest(
    args: list[str], *, junit_xml: Path, ambient: dict[str, str | None | _Inherit]
):
    """Run pytest in a child process with a deliberately constructed environment.

    ``ambient`` declares the environment the *operator* is simulated to have
    exported. Every name in ``_OFFLINE_ENV_NAMES`` must appear in it, and each
    value is one of:

    * a string — export exactly that value in the child;
    * ``None`` — the case's whole point is an environment WITHOUT this switch,
      so unset it (deliberately, and only where the case id says so);
    * ``INHERIT`` — pass the parent's value through unchanged.

    The declaration is mandatory rather than defaulted because the earlier form
    blanket-deleted both variables: running this suite under
    ``TRAIGENT_OFFLINE=1`` then launched a child with NEITHER switch, so a case
    labelled "operator exported offline" was in fact establishing the opposite
    environment, and the child auto-loaded plugins, the root conftest and its
    fixtures with a protection the parent had and the child had silently lost.
    """
    unspecified = [name for name in _OFFLINE_ENV_NAMES if name not in ambient]
    assert not unspecified, (
        "each case must state what the child's offline environment is, "
        f"including {unspecified} — an implicit default is how a case ends up "
        "testing an environment other than the one its id claims"
    )

    env = dict(os.environ)
    # pytest APPENDS $PYTEST_ADDOPTS to the command line; `-o addopts=` only
    # clears the *ini* addopts, so without this the parent run's options
    # (-n auto, -m "not performance", ...) leak into the child. The xdist
    # worker variables confuse a nested run the same way.
    for leaked in (
        "PYTEST_ADDOPTS",
        "PYTEST_CURRENT_TEST",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_XDIST_TESTRUNUID",
    ):
        env.pop(leaked, None)
    # No credentials reach the child, and mock-LLM stays on, whatever the case.
    env.pop("TRAIGENT_API_KEY", None)
    env["TRAIGENT_MOCK_LLM"] = "true"
    for name, value in ambient.items():
        if isinstance(value, _Inherit):
            continue
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "--junit-xml",
            str(junit_xml),
            "--tb=short",
            "-q",
            *args,
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _outcomes(junit_xml: Path) -> dict[str, str]:
    """Per-test outcomes from the child's JUnit report.

    Parsed rather than scraped from ``-q`` text so "1 passed, 1 skipped" cannot
    be mistaken for the *wrong* test having skipped.
    """
    report = ElementTree.parse(junit_xml)  # noqa: S314 — pytest's own output
    outcomes: dict[str, str] = {}
    for case in report.iter("testcase"):
        name = case.get("name") or "<unnamed>"
        if case.find("skipped") is not None:
            outcomes[name] = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcomes[name] = "failed"
        else:
            outcomes[name] = "passed"
    return outcomes


@contextlib.contextmanager
def _probe_module(directory: Path, source: str):
    """Write a throwaway probe module into ``directory`` and yield its path.

    The name starts with ``_`` so a concurrent directory scan can never collect
    it; pytest still collects it when the path is passed explicitly, because
    initial paths bypass the ``python_files`` patterns (verified on 9.0.2).
    """
    path = directory / f"_offline_probe_{os.getpid()}_{next(_PROBE_COUNTER)}.py"
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        for cached in (directory / "__pycache__").glob(f"{path.stem}.*"):
            cached.unlink(missing_ok=True)


_FLAG_PROBE = '''
"""Throwaway probe written by tests/unit/test_offline_fixture_contract.py."""

import os

import pytest

from traigent.utils.env_config import is_backend_offline


def test_probe_unmarked_is_offline():
    assert os.environ["TRAIGENT_OFFLINE_MODE"] == "true"
    assert is_backend_offline() is True


@pytest.mark.backend_online
def test_probe_marked_is_connected():
    assert os.environ["TRAIGENT_OFFLINE_MODE"] == "false"
    assert os.environ["TRAIGENT_OFFLINE"] == "false"
    assert is_backend_offline() is False
'''

_TYPO_MARKER_PROBE = '''
"""Throwaway probe written by tests/unit/test_offline_fixture_contract.py."""

import pytest


@pytest.mark.backend_onlne  # deliberate typo — must be a collection error
def test_probe_with_typoed_marker():
    pass
'''


# --------------------------------------------------------------------------
# The marker decides — proven where the carve-out used to live
# --------------------------------------------------------------------------


def test_marker_not_directory_decides_offline_inside_the_cloud_directory(tmp_path):
    """Executes the property inside ``tests/unit/cloud/`` itself.

    The two flag tests at the top of this file sit OUTSIDE that directory, so
    on their own nothing checks what happens INSIDE it — the one place a
    reinstated carve-out would bite. Here a marked and an unmarked probe share
    a file and a directory, deep inside the old carve-out, and must still reach
    opposite outcomes.

    This fails if directory-based semantics return by ANY spelling: a
    ``tests/unit/cloud/conftest.py`` that clears offline mode, a collection
    hook that adds ``backend_online`` from ``item.fspath``, ``Path(...).parts``
    checks, string concatenation — none of them can satisfy both probes at
    once, because the two differ only by the marker.

    The child inherits this process's offline environment: both probes expect
    the same outcome under any ambient value, so there is nothing to gain from
    stripping a switch the parent had.
    """
    with _probe_module(_CLOUD_DIR, _FLAG_PROBE) as probe:
        junit = tmp_path / "probe.xml"
        result = _run_pytest(
            [str(probe)], junit_xml=junit, ambient=dict(_INHERIT_OFFLINE)
        )
        outcomes = _outcomes(junit) if junit.exists() else {}

    assert outcomes == {
        "test_probe_unmarked_is_offline": "passed",
        "test_probe_marked_is_connected": "passed",
    }, (
        "offline mode inside tests/unit/cloud/ is not decided by the marker "
        f"alone:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )
    assert result.returncode == 0


# --------------------------------------------------------------------------
# An ambient offline switch must not suppress the marker
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ambient",
    [
        pytest.param(
            {"TRAIGENT_OFFLINE": None, "TRAIGENT_OFFLINE_MODE": None},
            id="no-ambient-switch",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": "0", "TRAIGENT_OFFLINE_MODE": None},
            id="ambient-offline-explicitly-falsy",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": "1", "TRAIGENT_OFFLINE_MODE": None},
            id="ambient-offline-1",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": "true", "TRAIGENT_OFFLINE_MODE": None},
            id="ambient-offline-true",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": None, "TRAIGENT_OFFLINE_MODE": "true"},
            id="ci-lane-shape-offline-mode-true",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": "true", "TRAIGENT_OFFLINE_MODE": "false"},
            id="ambient-split-spellings",
        ),
        pytest.param(
            {"TRAIGENT_OFFLINE": "1", "TRAIGENT_OFFLINE_MODE": "true"},
            id="ambient-both-spellings-true",
        ),
    ],
)
def test_ambient_offline_switch_does_not_suppress_the_marker(ambient, tmp_path):
    """A marked test RUNS, and observes connected mode, under any ambient value.

    ``ci-lane-shape-offline-mode-true`` is the exact environment of four real
    workflows — publish.yml, release-review.yml, sonarcloud.yml and tests.yml
    all export ``TRAIGENT_OFFLINE_MODE`` around ``pytest tests/unit``. A
    fixture that skipped there would turn the whole cloud suite from ~1983
    passed into ~1985 skipped and still exit 0, in the gate immediately before
    a PyPI publish; no lane asserts a skip count, so nothing would catch it.

    Skipping buys no safety to trade for that: ``backend_online`` never
    authorised real egress (its registered description requires mocked
    transports), so the mocks — not the env var — are the egress barrier for
    these tests, and an ambient "no real network" switch is already satisfied.

    ``test_marked_test_is_connected`` asserts BOTH spellings read "false" in
    process, so a green result here is positive proof that the fixture cleared
    the ambient switch rather than merely not skipping on it.

    Run in a subprocess: the autouse fixture has already applied by the time a
    test body runs, so the ambient environment must be set before pytest
    starts. Only the two flag tests are selected, so this cannot recurse.
    """
    junit = tmp_path / "flags.xml"
    result = _run_pytest(
        [
            _node_id(test_unmarked_test_is_offline),
            _node_id(test_marked_test_is_connected),
        ],
        junit_xml=junit,
        ambient=ambient,
    )
    assert junit.exists(), (
        f"child pytest produced no report under {ambient}:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )

    assert _outcomes(junit) == {
        # Offline-by-default never depends on the ambient environment...
        test_unmarked_test_is_offline.__name__: "passed",
        # ...and neither does connected mode. "skipped" here is the release-gate
        # regression: coverage silently lost while the run stays green.
        test_marked_test_is_connected.__name__: "passed",
    }, (
        f"a backend_online test did not run in connected mode under ambient "
        f"{ambient}:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )
    assert result.returncode == 0


# --------------------------------------------------------------------------
# Static bans — the mechanisms that could reinstate location dependence
# --------------------------------------------------------------------------

# Attribute/name accesses that expose a test's location to the fixture.
_PATH_DERIVED_ATTRS = frozenset({"fspath", "nodeid", "path", "location", "__file__"})


class _OfflineWrite(NamedTuple):
    """One write to an offline-mode env var found in a source tree.

    ``value`` is ``None`` when it is not a string literal (computed, or a
    deletion) and therefore cannot be proven to be tightening. ``key_node_id``
    identifies the AST node naming the variable, so the fail-closed sweep can
    tell classified mentions from unclassified ones without relying on line
    numbers (a multi-line call reports the line of its opening paren).
    """

    name: str
    value: str | None
    lineno: int
    key_node_id: int


def _offline_env_writes(tree: ast.AST) -> list[_OfflineWrite]:
    """Every write to an offline-mode env var in ``tree``."""
    writes: list[_OfflineWrite] = []
    for node in ast.walk(tree):
        # monkeypatch.setenv/delenv, os.environ.setdefault, os.putenv
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr in {"setenv", "delenv", "setdefault", "putenv"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in _OFFLINE_ENV_NAMES
            ):
                value = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    value = node.args[1].value
                writes.append(
                    _OfflineWrite(
                        node.args[0].value, value, node.lineno, id(node.args[0])
                    )
                )
            # patch.dict("os.environ", {"TRAIGENT_OFFLINE": ...})
            if node.func.attr == "dict":
                for argument in node.args:
                    if not isinstance(argument, ast.Dict):
                        continue
                    for key, value_node in zip(
                        argument.keys, argument.values, strict=True
                    ):
                        if (
                            isinstance(key, ast.Constant)
                            and key.value in _OFFLINE_ENV_NAMES
                        ):
                            writes.append(
                                _OfflineWrite(
                                    key.value,
                                    value_node.value
                                    if isinstance(value_node, ast.Constant)
                                    else None,
                                    key.lineno,
                                    id(key),
                                )
                            )
        # os.environ["TRAIGENT_OFFLINE"] = ... / del os.environ[...]
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value in _OFFLINE_ENV_NAMES
            ):
                assigned = getattr(node, "value", None)
                writes.append(
                    _OfflineWrite(
                        target.slice.value,
                        assigned.value if isinstance(assigned, ast.Constant) else None,
                        node.lineno,
                        id(target.slice),
                    )
                )
    return writes


def _is_tightening(value: str | None) -> bool:
    """True when the written value can only make the suite MORE offline."""
    return isinstance(value, str) and value.strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def test_no_subdirectory_conftest_relaxes_offline_mode():
    """A subdirectory ``conftest.py`` IS a path-derived condition.

    Its scope is a directory, so anything it does to offline mode is decided by
    where a test file happens to live — the dependence #2033 removed. This is
    also the cheapest evasion of a source grep: a new
    ``tests/unit/cloud/conftest.py`` with an autouse fixture setting both
    spellings to "false" reinstates the carve-out exactly, without ever naming
    the old literal.

    Only the *relaxing* direction is banned. Tightening a subtree to offline
    (``TRAIGENT_OFFLINE_MODE=true``) merely reasserts the suite-wide default and
    cannot make a test claim connectivity it does not have.
    """
    offenders: list[str] = []
    for conftest in sorted(_TESTS_ROOT.rglob("conftest.py")):
        if conftest == _ROOT_CONFTEST:
            continue
        tree = ast.parse(conftest.read_text(encoding="utf-8"), filename=str(conftest))
        relative = conftest.relative_to(_REPO_ROOT).as_posix()
        writes = _offline_env_writes(tree)
        for write in writes:
            if not _is_tightening(write.value):
                offenders.append(
                    f"{relative}:{write.lineno} writes {write.name}={write.value!r}"
                )
        # Fail closed: any other mention of these names in a subdirectory
        # conftest is unclassified, so treat it as an escape hatch.
        classified = {write.key_node_id for write in writes}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value in _OFFLINE_ENV_NAMES
                and id(node) not in classified
            ):
                offenders.append(
                    f"{relative}:{node.lineno} references {node.value} "
                    "outside a recognised tightening write"
                )

    assert not offenders, (
        "offline mode must be decided by @pytest.mark.backend_online in the "
        "root tests/conftest.py, not by which directory a test lives in "
        "(#2033). Offending subdirectory conftest(s):\n  " + "\n  ".join(offenders)
    )


def _resolve(node: ast.AST, scope: dict[str, ast.AST], seen: set[int]) -> list[ast.AST]:
    """Expand ``node`` into every AST node it transitively depends on.

    Local names are substituted with the expression assigned to them, and calls
    to module-level helpers are substituted with those helpers' bodies. This is
    what makes the check spelling-independent: ``"tests/unit/cloud/" in p``,
    ``p.split("/")[-2] == "cloud"`` and ``Path(p).parts`` all resolve back to
    the same ``node.fspath`` read.
    """
    collected: list[ast.AST] = []
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for child in ast.walk(current):
            collected.append(child)
            referenced: ast.AST | None = None
            if isinstance(child, ast.Name):
                referenced = scope.get(child.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                referenced = scope.get(child.func.id)
            if referenced is not None and id(referenced) not in seen:
                stack.append(referenced)
    return collected


def _is_path_derived(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr in _PATH_DERIVED_ATTRS:
            return True
        if isinstance(node, ast.Name) and node.id in _PATH_DERIVED_ATTRS:
            return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "/" in node.value
            and ("tests" in node.value or node.value.startswith("/"))
        ):
            return True
    return False


def _is_env_gated(nodes: list[ast.AST]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            return True
    return False


def _split_disjuncts(
    expr: ast.AST, scope: dict[str, ast.AST], depth: int = 0
) -> list[ast.AST]:
    """Break a guard into independently-sufficient conditions.

    ``a or b`` is checked branch by branch, because either branch alone can
    flip offline mode: a bare directory carve-out bolted on as a third disjunct
    beside the env-gated live clause would otherwise hide behind that clause's
    ``os.getenv``. ``and`` is deliberately NOT split — ``path AND env gate`` is
    the one permitted shape.
    """
    if depth > 6:
        return [expr]
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        branches: list[ast.AST] = []
        for value in expr.values:
            branches.extend(_split_disjuncts(value, scope, depth + 1))
        return branches
    if isinstance(expr, ast.Name):
        bound = scope.get(expr.id)
        if isinstance(bound, ast.expr):
            return _split_disjuncts(bound, scope, depth + 1)
    return [expr]


def test_root_conftest_never_decides_offline_mode_from_a_path_alone():
    """Mechanism proof for the one conftest that legitimately writes these vars.

    Replaces an earlier ``"tests/unit/cloud" not in source`` grep, which was a
    check on spelling: the same carve-out survives as ``"tests/unit/" +
    "cloud"``, ``Path(p).parts[-1] == "cloud"``, or ``p.split("/")[-2]``. Here
    every guard around an offline-mode write is resolved back through local
    assignments and module-level helpers to what it actually reads, and a guard
    that reads the test's LOCATION must also read the ENVIRONMENT.

    That is exactly the shape of the surviving live-contract clause (env gate
    AND known live module), and exactly not the shape of a bare directory
    carve-out.
    """
    tree = ast.parse(_ROOT_CONFTEST.read_text(encoding="utf-8"))
    module_scope: dict[str, ast.AST] = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_scope[target.id] = node.value

    offenders: list[str] = []
    checked_guards = 0
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _offline_env_writes(function):
            continue

        scope = dict(module_scope)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        scope[target.id] = node.value

        guards: list[ast.AST] = []
        for node in ast.walk(function):
            if isinstance(node, (ast.If, ast.IfExp)) and _offline_env_writes(node):
                guards.append(node.test)
        # The written VALUE can encode the decision too, not just the guard.
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and _offline_env_writes(node):
                guards.extend(node.args)

        for guard in guards:
            for branch in _split_disjuncts(guard, scope):
                checked_guards += 1
                resolved = _resolve(branch, scope, set())
                if _is_path_derived(resolved) and not _is_env_gated(resolved):
                    offenders.append(
                        f"tests/conftest.py:{getattr(branch, 'lineno', '?')} in "
                        f"{function.name}(): an offline-mode write is decided by "
                        "the test's location with no environment gate"
                    )

    assert checked_guards, "found no offline-mode decision to check — test is inert"
    assert not offenders, (
        "offline mode must not depend on where a test file lives (#2033). A "
        "path-derived condition is only permitted when it is also env-gated "
        "(the live-contract lane). Offenders:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# The marker registry itself
# --------------------------------------------------------------------------


def test_unregistered_marker_is_a_collection_error(tmp_path):
    """``strict_markers`` must stay honoured — it is version-gated and silent.

    Without it a typo'd ``@pytest.mark.backend_onlne`` is only a
    ``PytestUnknownMarkWarning``: the test still runs, offline, under a name
    that says connected. That is the same silent-no-op class #2033 removed, and
    it is precisely how the ``--strict-markers``-in-addopts form degraded
    unnoticed — nothing asserted the option was in effect.
    """
    with _probe_module(_TESTS_ROOT / "unit", _TYPO_MARKER_PROBE) as probe:
        result = _run_pytest(
            [str(probe)],
            junit_xml=tmp_path / "typo.xml",
            ambient=dict(_INHERIT_OFFLINE),
        )

    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        "an unregistered marker did not fail the run — strict_markers is no "
        f"longer in effect:\n{combined[-4000:]}"
    )
    assert "backend_onlne" in combined and "markers" in combined, combined[-4000:]
