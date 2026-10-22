"""The runtime-tvl skip guard cannot be satisfied by a skipped or missing compiler case."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "assert_tvl_tests_ran", REPO_ROOT / "scripts" / "ci" / "assert_tvl_tests_ran.py"
)
assert _spec and _spec.loader
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

BACKEND_REASON = guard.BACKEND_SKIP_REASON


def _case(module: str, name: str, outcome: str = "pass", reason: str = "") -> str:
    cls = f"tests.unit.runtime.{module}"
    if outcome == "skip":
        return (
            f'<testcase classname="{cls}" name="{name}">'
            f'<skipped message="{reason}"/></testcase>'
        )
    if outcome == "fail":
        return f'<testcase classname="{cls}" name="{name}"><failure/></testcase>'
    return f'<testcase classname="{cls}" name="{name}"/>'


def _required_passing() -> list[str]:
    return [_case(m, n) for m, n in guard.REQUIRED]


BACKEND_CASE = ("test_canary", "test_tvl_route_is_accepted_by_the_real_backend_parser")


def _run(tmp_path: Path, cases: list[str]) -> int:
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
    return guard.main(str(path))


def test_all_required_pass_with_the_documented_backend_skip(tmp_path: Path) -> None:
    cases = _required_passing() + [_case(*BACKEND_CASE, "skip", BACKEND_REASON)]
    assert _run(tmp_path, cases) == 0


def test_all_required_pass_without_any_skip(tmp_path: Path) -> None:
    assert _run(tmp_path, _required_passing() + [_case(*BACKEND_CASE)]) == 0


def test_skipped_compiler_case_with_a_metadata_only_pass_is_red(tmp_path: Path) -> None:
    cases = [
        _case("test_real_compiler", "test_token_map_format_is_the_compilers_contract"),
        _case(
            "test_real_compiler",
            "test_real_compiler_resolves_tokens[decorator]",
            "skip",
            "tvl-spec not installed",
        ),
    ]
    assert _run(tmp_path, cases) == 1


def test_metadata_only_pass_without_compiler_cases_is_red(tmp_path: Path) -> None:
    only = [
        _case("test_real_compiler", "test_token_map_format_is_the_compilers_contract")
    ]
    assert _run(tmp_path, only) == 1


def test_no_compiler_case_at_all_is_red(tmp_path: Path) -> None:
    assert _run(tmp_path, [_case("test_other", "test_x")]) == 1


def test_empty_report_is_red(tmp_path: Path) -> None:
    assert _run(tmp_path, []) == 1


def test_unexpected_skip_is_red(tmp_path: Path) -> None:
    cases = _required_passing() + [
        _case("test_other", "test_x", "skip", "needs TRAIGENT_BACKEND_SRC")
    ]
    assert _run(tmp_path, cases) == 1


def test_backend_skip_with_the_wrong_reason_is_red(tmp_path: Path) -> None:
    cases = _required_passing() + [_case(*BACKEND_CASE, "skip", "tvl missing")]
    assert _run(tmp_path, cases) == 1


def test_a_failed_case_is_red(tmp_path: Path) -> None:
    cases = _required_passing() + [_case("test_other", "test_x", "fail")]
    assert _run(tmp_path, cases) == 1


def test_adding_unrelated_passing_tests_does_not_break_the_guard(
    tmp_path: Path,
) -> None:
    extra = [_case("test_new", f"test_{i}") for i in range(5)]
    assert _run(tmp_path, _required_passing() + extra) == 0


def test_required_names_and_parameter_ids_exist_in_the_test_files() -> None:
    for module, name in (*guard.REQUIRED, BACKEND_CASE):
        text = (REPO_ROOT / "tests" / "unit" / "runtime" / f"{module}.py").read_text()
        base, _, param = name.partition("[")
        assert re.search(rf"^def {base}\(", text, re.M), (module, name)
        if param:
            assert f'"{param.rstrip("]")}"' in text, (module, name)


def test_declared_parameters_are_all_required() -> None:
    text = (REPO_ROOT / "tests/unit/runtime/test_real_compiler.py").read_text()
    ids = re.search(r"ids=\[([^\]]*)\]", text)
    assert ids
    declared = re.findall(r'"([^"]+)"', ids.group(1))
    required = {n for m, n in guard.REQUIRED if m == "test_real_compiler"}
    for pid in declared:
        assert f"test_real_compiler_resolves_tokens[{pid}]" in required


def test_skip_reason_matches_the_conftest_text() -> None:
    from tests.unit.runtime.conftest import BACKEND_SKIP_REASON

    assert guard.BACKEND_SKIP_REASON == BACKEND_SKIP_REASON


def test_reason_that_only_mentions_the_variable_is_red(tmp_path: Path) -> None:
    reason = "compiler import failed while TRAIGENT_BACKEND_SRC was configured"
    cases = _required_passing() + [_case(*BACKEND_CASE, "skip", reason)]
    assert _run(tmp_path, cases) == 1


def test_missing_parameter_case_is_red(tmp_path: Path) -> None:
    cases = [c for c in _required_passing() if "resolves_tokens[fragment]" not in c]
    assert len(cases) == len(guard.REQUIRED) - 1
    assert _run(tmp_path, cases) == 1


FILTER = re.compile(
    r"set_output_bool runtime_tvl_changed has_match '([^']+)'"
    r"",
)


def _filter() -> re.Pattern[str]:
    workflow = (REPO_ROOT / ".github" / "workflows" / "pr-gate.yml").read_text()
    match = FILTER.search(workflow)
    assert match
    return re.compile(match.group(1))


@pytest.mark.parametrize(
    "path",
    [
        "traigent/__init__.py",
        "traigent/core/orchestrator.py",
        "traigent/runtime/model.py",
        "tests/conftest.py",
        "tests/fixtures/rate_limit_fixtures.py",
        "tests/unit/runtime/test_canary.py",
        "scripts/ci/install_sdk_requirements.sh",
        "scripts/ci/assert_tvl_tests_ran.py",
        "pyproject.toml",
        ".github/workflows/pr-gate.yml",
    ],
)
def test_compiler_job_inputs_trigger_the_classifier(path: str) -> None:
    assert _filter().search(path)


@pytest.mark.parametrize(
    "path", ["docs/readme.md", "tests/unit/core/test_x.py", "scripts/ci/other.py"]
)
def test_unrelated_paths_do_not_trigger_the_classifier(path: str) -> None:
    assert not _filter().search(path)
