"""Characterization tests for traigent.identity.agent_build.survey_agent_sources.

Collaborators are replaced with scripted doubles so the tests can pin the
call order and counts, the helper-file limit and name rules, the gap list and
the result fields.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import traigent.identity.agent_build as agent_build
from traigent.identity.agent_build import AgentSourceSurvey, survey_agent_sources


def _agent() -> str:
    return "agent"


class _Script:
    """Scripted collaborators plus an ordered event trace."""

    def __init__(
        self,
        *,
        source_file: Any = "/proj/agent.py",
        revision: tuple[Any, Any, Any] | None = None,
        revision_gaps: tuple[str, ...] = (),
        loaded: dict[str, Path] | None = None,
    ) -> None:
        self.events: list[Any] = []
        self.source_file = source_file
        self.revision = revision or (None, set(), None)
        self.revision_gaps = revision_gaps
        self.loaded = loaded if loaded is not None else {}
        self.gaps_seen: list[Any] = []

    def getsourcefile(self, obj: Any) -> Any:
        self.events.append(("getsourcefile", obj))
        if isinstance(self.source_file, BaseException):
            raise self.source_file
        return self.source_file

    def code_revision_of(self, source_file: Any, gaps: list[str]) -> Any:
        self.events.append(("code_revision_of", source_file, list(gaps)))
        self.gaps_seen.append(gaps)
        gaps.extend(self.revision_gaps)
        return self.revision

    def loaded_project_files(self, project_root: Path) -> dict[str, Path]:
        self.events.append(("loaded_project_files", project_root))
        return self.loaded


@contextmanager
def _patched(script: _Script, **extra: Any) -> Iterator[None]:
    with (
        patch.object(agent_build.inspect, "getsourcefile", script.getsourcefile),
        patch.object(agent_build, "_code_revision_of", script.code_revision_of),
        patch.object(agent_build, "_loaded_project_files", script.loaded_project_files),
    ):
        with ExitStack() as stack:
            for attr, value in extra.items():
                stack.enter_context(patch.object(agent_build, attr, value))
            yield


def _survey(script: _Script, func: Any = _agent, **extra: Any) -> AgentSourceSurvey:
    with _patched(script, **extra):
        return survey_agent_sources(func)


def test_unknown_source_file_gives_empty_survey() -> None:
    script = _Script(source_file=None, revision_gaps=("entry_source_file_unknown",))
    survey = _survey(script)
    assert survey == AgentSourceSurvey(
        entry_file=None,
        project_root=None,
        commit=None,
        dirty=False,
        dirty_paths=frozenset(),
        helper_files={},
        gaps=("entry_source_file_unknown",),
    )
    assert [e[0] for e in script.events] == ["getsourcefile", "code_revision_of"]
    assert script.events[1][1:] == (None, [])


def test_getsourcefile_type_error_means_unknown_source() -> None:
    script = _Script(source_file=TypeError("builtin"))
    survey = _survey(script)
    assert survey.entry_file is None
    assert script.events[1] == ("code_revision_of", None, [])


def test_other_getsourcefile_errors_propagate() -> None:
    script = _Script(source_file=OSError("disk"))
    with pytest.raises(OSError, match="disk"):
        _survey(script)
    assert [e[0] for e in script.events] == ["getsourcefile"]


def test_partial_is_unwrapped_and_func_attribute_preferred() -> None:
    class _Bound:
        def __init__(self) -> None:
            self.__func__ = _agent

    script = _Script()
    _survey(script, functools.partial(_agent))
    assert script.events[0] == ("getsourcefile", _agent)

    script = _Script()
    bound = _Bound()
    with patch.object(agent_build, "_innermost", lambda f: f):
        _survey(script, bound)
    assert script.events[0] == ("getsourcefile", _agent)


def test_without_project_root_no_files_are_enumerated() -> None:
    script = _Script(
        revision=(None, {"dirty.py"}, None), loaded={"x.py": Path("/proj/x.py")}
    )
    survey = _survey(script)
    assert survey.helper_files == {}
    assert survey.gaps == ()
    assert survey.dirty_paths == frozenset({"dirty.py"})
    assert [e[0] for e in script.events] == ["getsourcefile", "code_revision_of"]


def test_entry_file_is_resolved_and_commit_fields_are_copied() -> None:
    script = _Script(
        source_file="/proj/../proj/agent.py",
        revision=({"commit": "c" * 40, "dirty": True}, {"a.py"}, Path("/proj")),
        loaded={"a.py": Path("/proj/a.py")},
    )
    survey = _survey(script)
    assert survey.entry_file == Path("/proj/agent.py").resolve()
    assert survey.project_root == Path("/proj")
    assert survey.commit == "c" * 40
    assert survey.dirty is True
    assert survey.dirty_paths == frozenset({"a.py"})
    assert survey.helper_files == {"a.py": Path("/proj/a.py")}
    assert survey.gaps == ()
    assert script.events[1] == ("code_revision_of", "/proj/../proj/agent.py", [])
    assert script.events[2] == ("loaded_project_files", Path("/proj"))


@pytest.mark.parametrize("dirty", [False, 0, None, ""])
def test_dirty_is_a_bool_of_the_revision_flag(dirty: Any) -> None:
    script = _Script(revision=({"commit": "d" * 40, "dirty": dirty}, set(), Path("/p")))
    survey = _survey(script)
    assert survey.dirty is False
    assert survey.commit == "d" * 40


def test_empty_revision_mapping_has_no_commit_and_is_not_dirty() -> None:
    script = _Script(revision=({}, set(), Path("/p")))
    survey = _survey(script)
    assert survey.commit is None
    assert survey.dirty is False


def test_empty_revision_mapping_missing_keys_do_not_raise() -> None:
    # An empty mapping is falsy, so neither key is read.
    script = _Script(revision=({}, set(), None))
    survey = _survey(script)
    assert (survey.commit, survey.dirty) == (None, False)


def test_revision_without_commit_key_raises_key_error() -> None:
    script = _Script(revision=({"dirty": False}, set(), Path("/p")))
    with pytest.raises(KeyError, match="commit"):
        _survey(script)


def test_files_are_visited_in_sorted_name_order() -> None:
    loaded = {
        "zeta.py": Path("/p/zeta.py"),
        "alpha.py": Path("/p/alpha.py"),
        "mid/beta.py": Path("/p/mid/beta.py"),
    }
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script)
    assert list(survey.helper_files) == ["alpha.py", "mid/beta.py", "zeta.py"]


def test_limit_reached_adds_gap_and_stops_at_the_boundary() -> None:
    loaded = {f"m{i}.py": Path(f"/p/m{i}.py") for i in range(4)}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=2)
    assert list(survey.helper_files) == ["m0.py", "m1.py"]
    assert survey.gaps == ("helper_modules_over_limit",)


def test_exactly_at_the_limit_is_not_a_gap() -> None:
    loaded = {f"m{i}.py": Path(f"/p/m{i}.py") for i in range(2)}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=2)
    assert list(survey.helper_files) == ["m0.py", "m1.py"]
    assert survey.gaps == ()


def test_limit_is_checked_before_the_name_rule() -> None:
    loaded = {"a.py": Path("/p/a.py"), "b b": Path("/p/b b")}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=1)
    assert list(survey.helper_files) == ["a.py"]
    assert survey.gaps == ("helper_modules_over_limit",)


def test_zero_limit_stops_before_the_first_file() -> None:
    script = _Script(
        revision=(None, set(), Path("/p")), loaded={"a.py": Path("/p/a.py")}
    )
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=0)
    assert survey.helper_files == {}
    assert survey.gaps == ("helper_modules_over_limit",)


def test_unrepresentable_names_are_skipped_and_do_not_count_toward_the_limit() -> None:
    loaded = {
        "a.py": Path("/p/a.py"),
        "bad name.py": Path("/p/bad name.py"),
        "": Path("/p/empty"),
        "x" * 129: Path("/p/long"),
        "c.py": Path("/p/c.py"),
    }
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=2)
    assert list(survey.helper_files) == ["a.py", "c.py"]
    # sorted order is "", "a.py", "bad name.py", "c.py", "xxx...": the long name
    # is reached with the limit already filled, so the limit gap wins over the
    # name gap for it.
    assert survey.gaps == (
        "helper_module_name_not_representable",
        "helper_modules_over_limit",
    )


def test_name_rule_uses_fullmatch_once_per_visited_file() -> None:
    calls: list[str] = []

    class _Regex:
        def fullmatch(self, text: str) -> object | None:
            calls.append(text)
            return object() if text != "no" else None

    loaded = {"a": Path("/p/a"), "no": Path("/p/no"), "z": Path("/p/z")}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _ASSET_NAME_RE=_Regex())
    assert calls == ["a", "no", "z"]
    assert list(survey.helper_files) == ["a", "z"]


def test_regex_semantics_accept_the_documented_alphabet() -> None:
    assert (
        agent_build._ASSET_NAME_RE.pattern
        == re.compile(r"[A-Za-z0-9_.:/-]{1,128}").pattern
    )
    loaded = {"pkg/mod-1.py": Path("/p/m"), "A:b_c.d": Path("/p/n")}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    assert set(_survey(script).helper_files) == {"pkg/mod-1.py", "A:b_c.d"}


def test_dirty_file_outside_helpers_adds_gap() -> None:
    script = _Script(
        revision=(
            {"commit": "c" * 40, "dirty": True},
            {"a.py", "other.py"},
            Path("/p"),
        ),
        loaded={"a.py": Path("/p/a.py")},
    )
    survey = _survey(script)
    assert survey.gaps == ("dirty_files_outside_manifest",)


def test_dirty_files_all_covered_add_no_gap() -> None:
    script = _Script(
        revision=({"commit": "c" * 40, "dirty": True}, {"a.py"}, Path("/p")),
        loaded={"a.py": Path("/p/a.py")},
    )
    assert _survey(script).gaps == ()


def test_dirty_file_dropped_for_name_is_outside_the_manifest() -> None:
    script = _Script(
        revision=({"commit": "c" * 40, "dirty": True}, {"bad name"}, Path("/p")),
        loaded={"bad name": Path("/p/bad")},
    )
    assert _survey(script).gaps == (
        "dirty_files_outside_manifest",
        "helper_module_name_not_representable",
    )


def test_dirty_difference_is_taken_once_and_dirty_paths_copied_once() -> None:
    counts = {"sub": 0, "iter": 0}

    class _Paths(set[str]):
        def __sub__(self, other: Any) -> set[str]:
            counts["sub"] += 1
            return set(self) - set(other)

        def __iter__(self) -> Iterator[str]:
            counts["iter"] += 1
            return super().__iter__()

    script = _Script(
        revision=(None, _Paths({"a.py"}), Path("/p")),
        loaded={"a.py": Path("/p/a.py")},
    )
    survey = _survey(script)
    assert counts["sub"] == 1
    assert survey.dirty_paths == frozenset({"a.py"})


def test_gaps_from_collaborator_and_loop_are_merged_deduplicated_and_sorted() -> None:
    script = _Script(
        revision=(None, {"x"}, Path("/p")),
        revision_gaps=("zz_gap", "no_code_revision", "zz_gap"),
        loaded={"bad name": Path("/p/b"), "also bad": Path("/p/c")},
    )
    survey = _survey(script)
    assert survey.gaps == (
        "dirty_files_outside_manifest",
        "helper_module_name_not_representable",
        "no_code_revision",
        "zz_gap",
    )


def test_collaborators_receive_the_same_gap_list_and_are_called_once() -> None:
    script = _Script(revision=(None, set(), Path("/p")), loaded={"a": Path("/p/a")})
    _survey(script)
    assert [e[0] for e in script.events] == [
        "getsourcefile",
        "code_revision_of",
        "loaded_project_files",
    ]
    assert len(script.gaps_seen) == 1


def test_loaded_files_failure_propagates() -> None:
    script = _Script(revision=(None, set(), Path("/p")))

    def _boom(root: Path) -> dict[str, Path]:
        raise RuntimeError("enumeration failed")

    with _patched(script), patch.object(agent_build, "_loaded_project_files", _boom):
        with pytest.raises(RuntimeError, match="enumeration failed"):
            survey_agent_sources(_agent)


def test_revision_lookup_failure_propagates_before_enumeration() -> None:
    script = _Script()

    def _boom(source_file: Any, gaps: list[str]) -> Any:
        raise RuntimeError("revision failed")

    with _patched(script), patch.object(agent_build, "_code_revision_of", _boom):
        with pytest.raises(RuntimeError, match="revision failed"):
            survey_agent_sources(_agent)
    assert [e[0] for e in script.events] == ["getsourcefile"]


def test_empty_source_file_string_is_unknown_entry_file() -> None:
    script = _Script(source_file="", revision=(None, set(), Path("/p")))
    survey = _survey(script)
    assert survey.entry_file is None
    assert survey.project_root == Path("/p")


def test_source_file_with_false_truth_is_treated_as_missing() -> None:
    class _Name(str):
        def __bool__(self) -> bool:
            return False

    script = _Script(source_file=_Name("/p/agent.py"), revision=(None, set(), None))
    assert _survey(script).entry_file is None


def test_limit_gap_is_appended_once_and_the_loop_stops() -> None:
    loaded = {f"m{i}.py": Path(f"/p/m{i}.py") for i in range(5)}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    survey = _survey(script, _MAX_ASSETS_PER_CATEGORY=1)
    assert list(survey.helper_files) == ["m0.py"]
    assert script.gaps_seen[0] == ["helper_modules_over_limit"]


def test_name_gap_is_appended_once_per_unrepresentable_file() -> None:
    loaded = {"a b": Path("/p/1"), "c d": Path("/p/2")}
    script = _Script(revision=(None, set(), Path("/p")), loaded=loaded)
    _survey(script)
    assert script.gaps_seen[0] == ["helper_module_name_not_representable"] * 2
