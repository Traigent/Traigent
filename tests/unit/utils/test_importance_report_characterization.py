"""Characterization (golden) tests for ParameterImportanceAnalyzer.generate_importance_report."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.utils.importance import ImportanceResult, ParameterImportanceAnalyzer


def _res(name: str, score: float, ci=(0.1, 0.9)) -> ImportanceResult:
    return ImportanceResult(name, score, ci, "m", 10)


def _run(analyzer, variance, assoc, perm=None, n_success=3, boom=None):
    analyzer.analyze_variance_based = lambda trials: variance
    analyzer.analyze_correlation_based = lambda trials: assoc
    if perm is not None:
        analyzer.analyze_permutation_based = lambda trials: perm
    if boom is not None:

        def _boom(trials):
            raise boom

        analyzer.analyze_variance_based = _boom
    opt = SimpleNamespace(
        optimization_id="opt_1",
        trials=[object()] * (n_success + 2),
        successful_trials=[object()] * n_success,
    )
    return analyzer.generate_importance_report(opt)


HEADER = [
    "🔍 Parameter Importance Analysis Report",
    "=" * 50,
    "Optimization ID: opt_1",
    "Objective: accuracy",
    "Total trials: 5",
    "Successful trials: 3",
    "",
]


def test_golden_with_results():
    a = ParameterImportanceAnalyzer("accuracy")
    variance = {"lr": _res("lr", 0.5), "bs": _res("bs", 0.25, (0.0, 0.5))}
    assoc = {"lr": _res("lr", 0.3)}
    report = _run(a, variance, assoc)
    expected = HEADER + [
        "📊 Variance-based Analysis:",
        "-" * 30,
        "  lr             : 0.500 [0.100, 0.900]",
        "  bs             : 0.250 [0.000, 0.500]",
        "",
        "📊 Association-based Analysis:",
        "-" * 30,
        "  lr             : 0.300 [0.100, 0.900]",
        "",
        "💡 Summary & Recommendations:",
        "-" * 30,
        "  Top parameters to focus on:",
        "    • lr: avg importance = 0.400",
        "    • bs: avg importance = 0.250",
        "",
        "  Consider:",
        "    • Fine-tuning the top parameters with more granular values",
        "    • Removing or fixing low-importance parameters",
        "    • Using these insights for future optimization runs",
    ]
    assert report.split("\n") == expected


def test_empty_results_and_no_patterns():
    a = ParameterImportanceAnalyzer()
    report = _run(a, {}, {})
    lines = report.split("\n")
    assert lines.count("  No significant importance detected") == 2
    assert lines[-1] == "  No clear parameter importance patterns detected"
    assert "  Top parameters to focus on:" not in lines


def test_error_in_one_method_is_reported_and_others_continue():
    a = ParameterImportanceAnalyzer()
    report = _run(a, {}, {"lr": _res("lr", 0.2)}, boom=RuntimeError("bad data"))
    lines = report.split("\n")
    assert "  Error in analysis: bad data" in lines
    assert "    • lr: avg importance = 0.200" in lines


def test_all_methods_failing_gives_insufficient_data():
    a = ParameterImportanceAnalyzer()

    def boom(trials):
        raise ValueError("x")

    a.analyze_variance_based = boom
    a.analyze_correlation_based = boom
    opt = SimpleNamespace(optimization_id="o", trials=[], successful_trials=[])
    lines = a.generate_importance_report(opt).split("\n")
    assert lines.count("  Error in analysis: x") == 2
    assert lines[-2:] == [
        "  Insufficient data for reliable importance analysis",
        "  Recommendation: Run more trials for better insights",
    ]


def test_permutation_runs_only_with_twenty_successful_trials():
    a = ParameterImportanceAnalyzer()
    perm = {"lr": _res("lr", 0.9)}
    small = _run(a, {}, {}, perm=perm, n_success=19)
    assert "Permutation-based" not in small
    big = _run(a, {}, {}, perm=perm, n_success=20)
    lines = big.split("\n")
    assert "📊 Permutation-based Analysis:" in lines
    assert "  lr             : 0.900 [0.100, 0.900]" in lines
    assert "    • lr: avg importance = 0.900" in lines


def test_only_top_five_listed_per_method_and_top_three_in_summary():
    a = ParameterImportanceAnalyzer()
    variance = {f"p{i}": _res(f"p{i}", 1.0 - i * 0.1) for i in range(7)}
    lines = _run(a, variance, {}).split("\n")
    listed = [ln for ln in lines if ln.startswith("  p")]
    assert len(listed) == 5
    focus = [ln for ln in lines if ln.startswith("    • p")]
    assert [ln.split(":")[0] for ln in focus] == ["    • p0", "    • p1", "    • p2"]
