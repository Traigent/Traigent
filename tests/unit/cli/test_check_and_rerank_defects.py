"""Regression tests for the ``traigent check`` / ``results rerank`` defect group.

Covers Traigent#2797, #2800, #2802, #2803, #2804, #2824, #2825, #2826, #2827,
#2829 and items 1, 2, 5, 6 and 7 of the #2831 umbrella. Everything runs offline
in mock mode; no provider is called.
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import click
import pytest
from click.testing import CliRunner

from traigent.cli.function_discovery import discover_optimized_functions
from traigent.cli.main import _display_optimization_result, _parse_weights, cli
from traigent.cli.optimization_validator import OptimizationValidator
from traigent.cli.validation_types import OptimizedFunction, ValidationResult

DATASET = (
    '{"input": {"q": "a"}, "output": "a"}\n'
    '{"input": {"q": "b"}, "output": "b"}\n'
    '{"input": {"q": "c"}, "output": "c"}\n'
    '{"input": {"q": "d"}, "output": "d"}\n'
)

SCORER = """
def exact(output, expected, **_):
    text = output["text"] if isinstance(output, dict) else output
    return 1.0 if text == expected else 0.0
"""


@pytest.fixture(autouse=True)
def _offline_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ds.jsonl").write_text(DATASET)
    for name in [m for m in sys.modules if m.startswith(("mypkg", "dcmod"))]:
        sys.modules.pop(name, None)
    sys.modules.pop("user_module", None)


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


def _agent(
    *,
    default_config: str = "",
    objectives: str = '["accuracy"]',
    body: str = "return traigent.with_usage(q, total_cost=0.01)",
    extra_decorator: str = "",
) -> str:
    return "import traigent\n" + SCORER + f"""
@traigent.optimize(configuration_space={{"style": ["short", "long"]}},
                   {default_config}
                   eval_dataset="ds.jsonl", objectives={objectives},
                   scoring_function=exact, offline=True{extra_decorator})
def answer(q: str) -> str:
    cfg = traigent.get_config()
    {body}
"""


def _invoke(*args: str):
    return CliRunner().invoke(cli, list(args))


# ---------------------------------------------------------------- loading ---


def test_check_loads_module_inside_package_by_path_2800(tmp_path: Path) -> None:
    _write(tmp_path, "mypkg/__init__.py", "")
    _write(tmp_path, "mypkg/helper.py", 'PROMPT = "hi"\n')
    run = _write(
        tmp_path,
        "mypkg/run.py",
        "from .helper import PROMPT\n" + _agent(),
    )
    found = discover_optimized_functions(str(run))
    assert [f.name for f in found] == ["answer"]


def test_check_loads_module_by_dotted_name_2800(tmp_path: Path) -> None:
    _write(tmp_path, "mypkg/__init__.py", "")
    _write(tmp_path, "mypkg/helper.py", 'PROMPT = "hi"\n')
    _write(tmp_path, "mypkg/run.py", "from .helper import PROMPT\n" + _agent())
    result = _invoke("check", "mypkg.run", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "answer" in result.output


def test_dotted_name_outside_workspace_is_rejected_2800() -> None:
    with pytest.raises(ValueError):
        discover_optimized_functions("json.tool")


def test_relative_import_without_package_explains_the_fix_2800(tmp_path: Path) -> None:
    from traigent.utils.module_loading import load_user_module

    path = _write(tmp_path, "loose.py", "from .helper import PROMPT\n")
    with pytest.raises(ImportError, match="__init__.py"):
        load_user_module(path)


def test_failed_module_load_leaves_no_registration_2825(tmp_path: Path) -> None:
    from traigent.utils.module_loading import load_user_module

    path = _write(tmp_path, "boom.py", "raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError):
        load_user_module(path, "dcmod_boom")
    assert "dcmod_boom" not in sys.modules


DATACLASS_MODULE = (
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n"
    "@dataclass\nclass Cfg:\n    y: int = 0\n" + _agent()
)


def test_check_loads_module_with_future_annotations_dataclass_2825(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "dc2.py", DATACLASS_MODULE)
    result = _invoke("check", "dc2.py", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Error during validation" not in result.output


def test_optimize_loads_module_with_future_annotations_dataclass_2825(
    tmp_path: Path,
) -> None:
    from traigent.cli.main import _load_user_module

    path = _write(tmp_path, "dc3.py", DATACLASS_MODULE)
    assert _load_user_module(path) is not None


def test_mcp_loader_handles_package_relative_imports_2800(tmp_path: Path) -> None:
    from traigent.mcp import tools

    _write(tmp_path, "mypkg/__init__.py", "")
    _write(tmp_path, "mypkg/helper.py", 'PROMPT = "hi"\n')
    run = _write(tmp_path, "mypkg/run.py", "from .helper import PROMPT\n" + _agent())
    module = tools._load_user_module(run)
    assert [n for n, _ in tools._find_optimizable_functions(module)] == ["answer"]


# -------------------------------------------------------------- discovery ---


def test_alias_and_import_of_one_function_is_one_entry_2826(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", _agent())
    runner = _write(
        tmp_path,
        "runner.py",
        "from good import answer\nfrom good import answer as answer_alias\n",
    )
    found = discover_optimized_functions(str(runner))
    assert len(found) == 1


def test_plain_class_with_optimize_method_is_ignored_2826(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "cls.py",
        """
        from traigent.core.optimized_function import OptimizedFunction
        class MyOpt:
            def optimize(self):
                ...
        """,
    )
    assert discover_optimized_functions(str(path)) == []


def test_unknown_function_name_exits_nonzero_and_names_it_2826(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", _agent())
    result = _invoke("check", "good.py", "--functions", "answer,typo", "--dry-run")
    assert result.exit_code != 0, result.output
    assert "typo" in result.output


def test_all_miss_lists_available_functions_2826(tmp_path: Path) -> None:
    _write(tmp_path, "good.py", _agent())
    result = _invoke("check", "good.py", "--functions", "answr", "--dry-run")
    assert result.exit_code != 0
    assert "answr" in result.output
    assert "answer" in result.output.replace("answr", "")


# ---------------------------------------------------------------- dry run ---


def test_dry_run_with_declared_default_config_prints_no_warning_2797(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path, "a.py", _agent(default_config='default_config={"style": "short"},')
    )
    result = _invoke("check", "a.py", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "No default parameters found" not in result.output
    assert "Warning" not in result.output


def test_dry_run_without_baseline_is_an_informational_note_2797(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", _agent())
    result = _invoke("check", "a.py", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "No default parameters found" not in result.output
    assert "empty config" in result.output  # says what the baseline will be
    assert "default_config" in result.output  # and what to add


def test_dry_run_with_blocking_issue_exits_nonzero_2827(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "nods.py",
        """
        import traigent
        @traigent.optimize(configuration_space={"style": ["a", "b"]},
                           objectives=["accuracy"], offline=True)
        def answer(q: str) -> str:
            return q
        """,
    )
    result = _invoke("check", "nods.py", "--dry-run")
    assert result.exit_code != 0, result.output
    assert "ready for validation" not in result.output
    assert "Missing evaluation dataset" in result.output


def test_dry_run_on_clean_function_exits_zero_2827(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", _agent())
    result = _invoke("check", "a.py", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "ready for validation" in result.output


def test_classification_keeps_advisories_out_of_blocking_2827(tmp_path: Path) -> None:
    from traigent.cli.function_discovery import classify_discovered_functions

    path = _write(tmp_path, "a.py", _agent())
    blocking, notes = classify_discovered_functions(
        discover_optimized_functions(str(path))
    )
    assert blocking == []
    assert len(notes) == 1


# ------------------------------------------------------------- baseline -----


def _func_info(
    default_config: dict | None, default_params: dict | None
) -> OptimizedFunction:
    return OptimizedFunction(
        name="f",
        func=Mock(),
        decorator_config={"default_config": default_config or {}},
        default_params=default_params or {},
        eval_dataset="d.jsonl",
        objectives=["accuracy"],
    )


@pytest.mark.parametrize(
    ("default_config", "default_params", "expected", "source"),
    [
        ({"a": 1}, {}, {"a": 1}, "default_config"),
        ({}, {"b": 2}, {"b": 2}, "signature"),
        ({"a": 1}, {"b": 2}, {"a": 1}, "default_config"),
        ({}, {}, {}, "none"),
    ],
)
def test_baseline_config_precedence_2802(
    default_config: dict, default_params: dict, expected: dict, source: str
) -> None:
    info = _func_info(default_config, default_params)
    assert info.baseline_config == expected
    assert info.baseline_source == source


def test_check_baseline_runs_the_declared_default_config_2802(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "a.py",
        _agent(
            default_config='default_config={"style": "short"},',
            body='return traigent.with_usage(q if cfg.get("style") == "short" else q + q, total_cost=0)',
        ),
    )
    result = _invoke("check", "a.py")
    out = result.output
    assert "Baseline configuration (default_config): {'style': 'short'}" in out
    assert result.exit_code != 0  # baseline == best config: nothing to improve
    baseline_block = out.split("Baseline Metrics (default parameters):")[-1]
    assert baseline_block.split("Optimized Metrics")[0].strip() == "accuracy: 1.000"


# ----------------------------------------------------- zero baseline (#2803) -


def test_zero_baseline_improvement_is_superior_2803() -> None:
    v = OptimizationValidator(threshold_pct=10)
    superior, details = v._compare_results(
        {"accuracy": 0.0}, {"accuracy": 1.0}, ["accuracy"]
    )
    assert superior is True
    assert details["accuracy"] == math.inf


def test_zero_baseline_minimize_regression_stays_blocked_and_visible_2803() -> None:
    v = OptimizationValidator(threshold_pct=10)
    superior, details = v._compare_results(
        {"cost": 0.0}, {"cost": 0.5}, ["cost"], {"cost": "minimize"}
    )
    assert superior is False
    assert details["cost"] == -math.inf  # a regression, not 0.0


def test_nonzero_baseline_unchanged_2803() -> None:
    v = OptimizationValidator(threshold_pct=10)
    superior, details = v._compare_results(
        {"accuracy": 0.5}, {"accuracy": 1.0}, ["accuracy"]
    )
    assert superior is True
    assert details["accuracy"] == pytest.approx(100.0)


def test_report_never_prints_zero_percent_for_a_changed_metric_2803() -> None:
    result = ValidationResult(
        function_name="f",
        baseline_metrics={"accuracy": 0.0},
        optimized_metrics={"accuracy": 1.0},
        is_superior=True,
        improvement_details={"accuracy": math.inf},
        blocking_issues=[],
    )
    report = result.get_detailed_report()
    assert "(= 0.0%)" not in report
    assert "from zero" in report
    assert "+1.000" in report
    assert "inf" not in report.lower()


def test_report_shows_no_percentage_for_uncompared_metrics_2831() -> None:
    result = ValidationResult(
        function_name="f",
        baseline_metrics={"accuracy": 0.5},
        optimized_metrics={"accuracy": 1.0, "duration": 0.2},
        is_superior=True,
        improvement_details={"accuracy": 100.0},
        blocking_issues=[],
    )
    lines = result.get_detailed_report().splitlines()
    assert "  duration: 0.200" in lines


# ------------------------------------------------- threshold / NaN (#2831) --


def test_threshold_is_strict_and_immune_to_float_noise_2831() -> None:
    v = OptimizationValidator(threshold_pct=10)
    acc = ["accuracy"]
    # Both are exactly +10% in decimal; neither exceeds a 10% threshold.
    assert v._compare_results({"accuracy": 0.3}, {"accuracy": 0.33}, acc)[0] is False
    assert v._compare_results({"accuracy": 0.8}, {"accuracy": 0.88}, acc)[0] is False
    assert v._compare_results({"accuracy": 0.8}, {"accuracy": 0.89}, acc)[0] is True
    assert OptimizationValidator(7.0).threshold_pct == 7.0


def test_negative_threshold_is_rejected_2831(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", _agent())
    result = _invoke("check", "a.py", "--threshold", "-50", "--dry-run")
    assert result.exit_code == 2
    with pytest.raises(ValueError):
        OptimizationValidator(-1)


def test_non_finite_metrics_are_a_distinct_blocking_reason_2831() -> None:
    v = OptimizationValidator(threshold_pct=10)
    acc = ["accuracy"]
    assert (
        v._compare_results({"accuracy": 0.5}, {"accuracy": math.inf}, acc)[0] is False
    )
    assert (
        v._compare_results({"accuracy": math.nan}, {"accuracy": 1.0}, acc)[0] is False
    )
    reasons = v._incomparable_objectives({"accuracy": 0.5}, {"accuracy": math.nan}, acc)
    assert reasons and "not a finite number" in reasons[0]


def test_objectives_flag_restricts_the_compared_objectives_2831() -> None:
    from traigent.cli.main import _filter_functions_by_objectives

    both = _func_info({}, {})
    both.objectives = ["accuracy", "cost"]
    other = _func_info({}, {})
    other.objectives = ["latency"]
    selected = _filter_functions_by_objectives([both, other], ["accuracy"])
    assert [f.objectives for f in selected] == [["accuracy"]]
    assert both.objectives == ["accuracy", "cost"]  # the original is untouched


def test_optimize_announces_overwriting_a_saved_result_2831() -> None:
    persistence = Mock()
    persistence.result_exists.return_value = True
    result = Mock(best_score=1.0, best_config={}, trials=[], successful_trials=[])
    result.stop_reason = None
    with patch("traigent.cli.main.console") as console:
        _display_optimization_result("answer", result, persistence, "grid", 2)
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list)
    assert "Replacing the earlier saved result 'answer_grid_2'" in printed


# ------------------------------------------------- end-to-end check (#2824) -


COST_BODY = (
    'return traigent.with_usage(q if cfg.get("style") == "short" else (q + "x" if q in "ab" else q),'
    ' total_cost=0.01 if cfg.get("style") == "short" else 0.02)'
)


def test_cost_objective_is_compared_against_a_measured_baseline_2824(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "cost.py",
        _agent(
            objectives='["accuracy", "cost"]',
            default_config='default_config={"style": "long"},',
            body=COST_BODY,
        ),
    )
    result = _invoke("check", "cost.py")
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_total_cost_objective_is_compared_2824(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tc.py",
        _agent(
            objectives='["accuracy", "total_cost"]',
            default_config='default_config={"style": "long"},',
            body=COST_BODY,
        ),
    )
    result = _invoke("check", "tc.py")
    assert "could not be compared" not in result.output
    assert result.exit_code == 0, result.output


def test_unmeasured_objective_gets_a_specific_message_2824() -> None:
    reasons = OptimizationValidator()._incomparable_objectives(
        {"accuracy": 0.5}, {"accuracy": 1.0, "cost": 0.04}, ["accuracy", "cost"]
    )
    assert reasons == [
        "Objective 'cost' could not be compared: it was not measured on the "
        "baseline run"
    ]


# ------------------------------------------------- closing verdict (#2804) --


def test_validation_error_is_not_reported_as_no_improvement_2804(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "err.py",
        _agent(body='raise RuntimeError("agent bug: tool registry not initialised")'),
    )
    result = _invoke("check", "err.py")
    out = result.output
    assert result.exit_code == 1
    assert "does not improve" not in out.split("Validation Summary")[-1]
    assert "Validation could not complete" in out
    assert "--no-verify" not in out


def test_real_no_improvement_still_blocks_with_its_verdict_2804(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "same.py",
        _agent(body="return traigent.with_usage(q, total_cost=0.01)"),
    )
    result = _invoke("check", "same.py")
    assert result.exit_code == 1
    assert "does not improve over default parameters" in result.output
    assert "--no-verify" not in result.output


# --------------------------------------------------------- rerank (#2829) ---


@pytest.fixture
def stored_result() -> Mock:
    trial = Mock()
    trial.trial_id = "t1"
    trial.config = {"temperature": 0.5}
    trial.metrics = {"accuracy": 0.85, "cost": 0.01}
    trial.status = "completed"
    trial.is_successful = True
    result = Mock()
    result.trials = [trial]
    result.successful_trials = [trial]
    result.best_config = {"temperature": 0.5}
    result.best_score = 0.85
    result.best_metrics = {"accuracy": 0.85}
    result.objectives = ["accuracy", "cost"]
    result.metadata = {}
    result.algorithm = "grid"
    return result


@pytest.mark.parametrize(
    "weights",
    [
        "bogus",
        "nosuchmetric=1",
        "accuracy=0",
        "accuracy=0,cost=0",
        "accuracy=-1",
        "accuracy=nan",
        "accuracy=inf",
        "a=1,a=3",
        "accuracy=",
    ],
)
def test_rerank_exits_nonzero_on_invalid_input_2829(
    weights: str, stored_result: Mock
) -> None:
    with patch("traigent.cli.main.PersistenceManager") as manager:
        manager.return_value.load_result.return_value = stored_result
        result = _invoke("results", "rerank", "run", "-w", weights)
    assert result.exit_code != 0, result.output


def test_rerank_valid_weights_unchanged_2829(stored_result: Mock) -> None:
    with patch("traigent.cli.main.PersistenceManager") as manager:
        manager.return_value.load_result.return_value = stored_result
        result = _invoke("results", "rerank", "run", "-w", "accuracy=0.8,cost=0.2")
    assert result.exit_code == 0, result.output


def test_parse_weights_names_the_offending_token_2829() -> None:
    with pytest.raises(click.ClickException, match="accuracy"):
        _parse_weights("accuracy=nan")
    with pytest.raises(click.ClickException, match="Duplicate weight for 'a'"):
        _parse_weights("a=1,a=3")
    assert _parse_weights("a=1,b=3") == {"a": 0.25, "b": 0.75}


def test_rerank_allows_one_zero_weight_when_another_is_positive_2829(
    stored_result: Mock,
) -> None:
    assert _parse_weights("accuracy=1,cost=0") == {"accuracy": 1.0, "cost": 0.0}


def test_rerank_with_all_trials_ineligible_is_an_outcome_not_an_error_2829(
    stored_result: Mock,
) -> None:
    with (
        patch("traigent.cli.main.PersistenceManager") as manager,
        patch("traigent.cli.main._score_trials_for_rerank", return_value=[]),
    ):
        manager.return_value.load_result.return_value = stored_result
        result = _invoke("results", "rerank", "run", "-w", "accuracy=1")
    assert result.exit_code == 0, result.output
    assert "eligible" in result.output


def test_zero_baseline_noise_does_not_pass_the_threshold_2803() -> None:
    v = OptimizationValidator(threshold_pct=10)
    superior, details = v._compare_results(
        {"accuracy": 0.0}, {"accuracy": 0.0001}, ["accuracy"]
    )
    assert superior is False
    assert math.isfinite(details["accuracy"])


def test_crashing_baseline_is_blocked_not_a_zero_baseline_2803(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "broken.py",
        _agent(body='return traigent.with_usage(cfg["style"], total_cost=0)'),
    )
    result = _invoke("check", "broken.py")
    assert result.exit_code != 0, result.output
    assert "PASSED" not in result.output
    assert "Baseline run failed" in result.output
    assert "first error:" in result.output


def test_same_package_name_in_two_projects_loads_each_file_2800(
    tmp_path: Path,
) -> None:
    for project in ("a", "b"):
        _write(tmp_path, f"{project}/agents/__init__.py", "")
        _write(
            tmp_path,
            f"{project}/agents/run.py",
            _agent().replace("def answer", f"def answer_{project}"),
        )
    from traigent.utils.module_loading import find_optimized_functions, load_user_module

    first = load_user_module(tmp_path / "a/agents/run.py")
    second = load_user_module(tmp_path / "b/agents/run.py")
    assert [n for n, _ in find_optimized_functions(first)] == ["answer_a"]
    assert [n for n, _ in find_optimized_functions(second)] == ["answer_b"]
    for name in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
        sys.modules.pop(name, None)


def test_local_cleanup_declined_prompt_is_a_clean_cancel_2830(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "store"))
    result = CliRunner().invoke(cli, ["local", "cleanup", "--days", "30"], input="n\n")
    assert result.exit_code == 1
    assert "nothing was deleted" in result.output
    assert "--yes" in result.output


def test_rerank_rejects_absent_metric_even_with_a_zero_weight_beside_it_2829(
    stored_result: Mock,
) -> None:
    with patch("traigent.cli.main.PersistenceManager") as manager:
        manager.return_value.load_result.return_value = stored_result
        result = _invoke("results", "rerank", "run", "-w", "nosuch=1,cost=0")
    assert result.exit_code != 0, result.output


def test_local_cleanup_without_a_tty_names_yes_2830(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "store"))
    result = CliRunner().invoke(cli, ["local", "cleanup", "--days", "30"], input="")
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert "Error cleaning up sessions" not in result.output


def test_failed_package_load_does_not_poison_the_next_project_2800(
    tmp_path: Path,
) -> None:
    from traigent.utils.module_loading import find_optimized_functions, load_user_module

    _write(tmp_path, "c/agents/__init__.py", "")
    _write(tmp_path, "c/agents/run.py", "raise RuntimeError('boom')\n")
    _write(tmp_path, "d/agents/__init__.py", "")
    _write(tmp_path, "d/agents/run.py", _agent())
    with pytest.raises(RuntimeError):
        load_user_module(tmp_path / "c/agents/run.py")
    module = load_user_module(tmp_path / "d/agents/run.py")
    assert [n for n, _ in find_optimized_functions(module)] == ["answer"]
    for name in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
        sys.modules.pop(name, None)


def test_package_walk_stops_at_world_writable_directory(tmp_path: Path) -> None:
    """A planted ``__init__.py`` in a world-writable ancestor must not be imported."""
    import os

    from traigent.utils.module_loading import package_parts

    shared = tmp_path / "shared"
    pkg = shared / "pkg"
    pkg.mkdir(parents=True)
    (shared / "__init__.py").write_text("raise RuntimeError('planted')\n")
    (pkg / "__init__.py").write_text("")
    mod = pkg / "mod.py"
    mod.write_text("X = 1\n")

    os.chmod(shared, 0o777)
    try:
        root, parts = package_parts(mod)
    finally:
        os.chmod(shared, 0o755)
    assert parts == ["pkg", "mod"]
    assert root == shared

    # control: the same layout with a private ancestor still walks up.
    _, parts = package_parts(mod)
    assert parts == ["shared", "pkg", "mod"]
