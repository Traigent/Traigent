"""Tests for the per-example x per-trial outcome matrix (issue #1838).

Covers:
- building the matrix from an in-memory OptimizationResult (the by-product
  already carried on each trial), including token telemetry projection;
- persistence to the run's artifact directory via the OptimizationLogger;
- the ``load_outcome_matrix`` accessor round-trip;
- the ``OptimizationResult.example_matrix`` ergonomic property.

No LLM calls: results are constructed in memory from fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataclasses import asdict

from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.evaluators.hybrid_api import HybridExampleResult
from traigent.utils.optimization_logger import OptimizationLogger
from traigent.utils.outcome_matrix import (
    OUTCOME_MATRIX_SCHEMA_VERSION,
    build_outcome_matrix,
    load_outcome_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _example_dict(
    example_id: str,
    accuracy: float,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    total_cost: float | None = None,
    success: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    """A real ``ExampleResult.to_dict()`` payload, as stored in trial.metadata.

    Token telemetry uses the evaluator's real per-example metric names
    (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, written by
    ``BaseEvaluator._add_llm_metrics_to_example``).
    """
    metrics: dict[str, Any] = {"accuracy": accuracy}
    if input_tokens is not None:
        metrics["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        metrics["completion_tokens"] = output_tokens
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    if total_cost is not None:
        metrics["total_cost"] = total_cost
    return ExampleResult(
        example_id=example_id,
        input_data={},
        expected_output="e",
        actual_output="a",
        metrics=metrics,
        execution_time=0.5,
        success=success,
        error_message=error,
    ).to_dict()


def _trial(
    trial_id: str,
    config: dict[str, Any],
    examples: list[Any],
    accuracy: float = 0.9,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={"accuracy": accuracy},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"example_results": examples},
    )


def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config if trials else None,
        best_score=0.9,
        optimization_id="opt-1838",
        duration=5.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _two_config_result() -> OptimizationResult:
    trial_a = _trial(
        "trial_0",
        {"model": "gpt-4o"},
        [
            _example_dict(
                "ex-1", 1.0, input_tokens=100, output_tokens=10, total_tokens=110
            ),
            _example_dict("ex-2", 0.0, total_tokens=200, total_cost=0.002),
        ],
    )
    trial_b = _trial(
        "trial_1",
        {"model": "gpt-4o-mini"},
        [
            _example_dict("ex-1", 1.0, total_tokens=90),
            _example_dict("ex-2", 0.0, total_tokens=180),
        ],
    )
    return _result([trial_a, trial_b])


def _make_logger(tmp_path: Path) -> OptimizationLogger:
    return OptimizationLogger(
        experiment_name="test_exp",
        session_id="sess12345678",
        execution_mode="local",
        base_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# build_outcome_matrix
# ---------------------------------------------------------------------------


class TestBuildOutcomeMatrix:
    def test_shape_and_header(self) -> None:
        matrix = build_outcome_matrix(_two_config_result())
        assert matrix["schema_version"] == OUTCOME_MATRIX_SCHEMA_VERSION
        assert matrix["optimization_id"] == "opt-1838"
        assert matrix["algorithm"] == "GridSearchOptimizer"
        assert matrix["objectives"] == ["accuracy"]
        assert matrix["trial_count"] == 2
        assert matrix["example_count"] == 2
        assert "created_at" in matrix

    def test_columns_carry_config_and_hash(self) -> None:
        matrix = build_outcome_matrix(_two_config_result())
        cols = matrix["trials"]
        assert [c["trial_id"] for c in cols] == ["trial_0", "trial_1"]
        assert cols[0]["config"] == {"model": "gpt-4o"}
        # Different configs must hash differently; hash is stable/non-empty.
        assert cols[0]["config_hash"] != cols[1]["config_hash"]
        assert len(cols[0]["config_hash"]) == 12

    def test_cells_keyed_by_trial_with_outcomes(self) -> None:
        matrix = build_outcome_matrix(_two_config_result())
        rows = {row["example_id"]: row["cells"] for row in matrix["examples"]}
        assert set(rows) == {"ex-1", "ex-2"}
        ex1 = rows["ex-1"]
        assert set(ex1) == {"trial_0", "trial_1"}
        assert ex1["trial_0"]["accuracy"] == 1.0
        assert ex1["trial_0"]["success"] is True
        # score is null when the run wrote no ``score`` metric — it is NOT
        # silently faked from accuracy (issue #1838 BUG-B).
        assert ex1["trial_0"]["score"] is None

    def test_token_telemetry_projected(self) -> None:
        matrix = build_outcome_matrix(_two_config_result())
        rows = {row["example_id"]: row["cells"] for row in matrix["examples"]}
        assert rows["ex-1"]["trial_0"]["tokens"] == {
            "input": 100,
            "output": 10,
            "total": 110,
        }
        # ex-2 trial_0 had only a total token count + a cost.
        assert rows["ex-2"]["trial_0"]["tokens"] == {"total": 200}
        assert rows["ex-2"]["trial_0"]["cost_usd"] == 0.002

    def test_missing_tokens_are_null_not_zero(self) -> None:
        trial = _trial("trial_0", {"model": "m"}, [_example_dict("ex-1", 1.0)])
        matrix = build_outcome_matrix(_result([trial]))
        cell = matrix["examples"][0]["cells"]["trial_0"]
        # No fabricated zeros — unknown token telemetry stays null.
        assert cell["tokens"] is None
        assert cell["cost_usd"] is None

    def test_accepts_exampleresult_objects(self) -> None:
        ex = ExampleResult(
            example_id="ex-1",
            input_data={},
            expected_output="a",
            actual_output="a",
            metrics={"accuracy": 1.0, "total_tokens": 42},
            execution_time=0.3,
            success=True,
        )
        trial = _trial("trial_0", {"model": "m"}, [ex])
        matrix = build_outcome_matrix(_result([trial]))
        cell = matrix["examples"][0]["cells"]["trial_0"]
        assert cell["accuracy"] == 1.0
        assert cell["tokens"] == {"total": 42}

    def test_trial_without_examples_contributes_column_only(self) -> None:
        trial = TrialResult(
            trial_id="trial_0",
            config={"model": "m"},
            metrics={"accuracy": 0.5},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={},
        )
        matrix = build_outcome_matrix(_result([trial]))
        assert matrix["trial_count"] == 1
        assert matrix["example_count"] == 0
        assert matrix["examples"] == []

    def test_error_example_carries_message(self) -> None:
        trial = _trial(
            "trial_0",
            {"model": "m"},
            [_example_dict("ex-1", 0.0, success=False, error="boom")],
        )
        matrix = build_outcome_matrix(_result([trial]))
        cell = matrix["examples"][0]["cells"]["trial_0"]
        assert cell["success"] is False
        assert cell["error"] == "boom"

    def test_real_exampleresult_todict_captures_prompt_completion_tokens(self) -> None:
        """A real ExampleResult.to_dict() surfaces prompt/completion tokens.

        Regression for BUG-A: the evaluator writes prompt_tokens/completion_tokens,
        not input_tokens/output_tokens.
        """
        ex = ExampleResult(
            example_id="ex-1",
            input_data={},
            expected_output="a",
            actual_output="a",
            metrics={
                "accuracy": 1.0,
                "prompt_tokens": 120,
                "completion_tokens": 8,
                "total_tokens": 128,
                "total_cost": 0.0004,
            },
            execution_time=0.42,
            success=True,
        ).to_dict()
        trial = _trial("trial_0", {"model": "m"}, [ex])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["tokens"] == {"input": 120, "output": 8, "total": 128}
        assert cell["cost_usd"] == 0.0004
        assert cell["execution_time"] == 0.42
        assert cell["latency_ms"] is None

    def test_input_output_token_aliases_supported(self) -> None:
        """The local-lane input_tokens/output_tokens names also map (BUG-A)."""
        ex = ExampleResult(
            example_id="ex-1",
            input_data={},
            expected_output="a",
            actual_output="a",
            metrics={"accuracy": 1.0, "input_tokens": 50, "output_tokens": 5},
            execution_time=0.1,
            success=True,
        ).to_dict()
        trial = _trial("trial_0", {"model": "m"}, [ex])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["tokens"] == {"input": 50, "output": 5}

    def test_score_metric_is_captured_not_derived(self) -> None:
        """The real per-example ``score`` metric flows into the cell (BUG-B)."""
        ex = _example_dict("ex-1", 1.0)
        ex["metrics"]["score"] = 0.73
        trial = _trial("trial_0", {"model": "m"}, [ex])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["score"] == 0.73
        assert cell["accuracy"] == 1.0

    def test_non_accuracy_metric_flows_through(self) -> None:
        """A run optimizing f1 (no accuracy) still yields a real quality signal.

        Regression for BUG-B: nothing is hardcoded to "accuracy". accuracy is
        null, but the f1 signal survives via ``metrics`` (and ``score`` when the
        evaluator aligned it).
        """
        ex = ExampleResult(
            example_id="ex-1",
            input_data={},
            expected_output="a",
            actual_output="a",
            metrics={"f1": 0.81, "exact_match": 0.0, "score": 0.81},
            execution_time=0.2,
            success=True,
        ).to_dict()
        trial = _trial("trial_0", {"model": "m"}, [ex])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["accuracy"] is None
        assert cell["score"] == 0.81
        assert cell["metrics"]["f1"] == 0.81
        assert cell["metrics"]["exact_match"] == 0.0

    def test_present_zero_tokens_recorded_not_nulled(self) -> None:
        """A present-as-0 token count is recorded 0, not fabricated away (RISK-E)."""
        ex = ExampleResult(
            example_id="ex-1",
            input_data={},
            expected_output="a",
            actual_output="a",
            metrics={"accuracy": 1.0, "total_tokens": 0, "total_cost": 0.0},
            execution_time=0.1,
            success=True,
        ).to_dict()
        trial = _trial("trial_0", {"model": "m"}, [ex])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["tokens"] == {"total": 0}
        assert cell["cost_usd"] == 0.0

    def test_hybrid_example_asdict_success_and_latency(self) -> None:
        """Hybrid results (asdict, no ``success`` key) are marked succeeded.

        Regression for BUG-C: ``HybridExampleResult.success`` is a @property that
        ``asdict`` drops, so success must derive from ``error is None``. Latency
        is captured from ``latency_ms`` (BUG-D).
        """
        hybrid = asdict(
            HybridExampleResult(
                example_id="ex-1",
                actual_output="a",
                expected_output="a",
                metrics={"accuracy": 1.0, "total_tokens": 64},
                cost_usd=0.0009,
                latency_ms=250.0,
                error=None,
            )
        )
        assert "success" not in hybrid  # @property is not serialized by asdict
        trial = _trial("trial_0", {"model": "m"}, [hybrid])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["success"] is True
        assert cell["latency_ms"] == 250.0
        assert cell["execution_time"] is None
        assert cell["cost_usd"] == 0.0009
        assert cell["tokens"] == {"total": 64}
        assert cell["error"] is None

    def test_hybrid_failed_example_marked_unsuccessful(self) -> None:
        """A hybrid example with an error derives success=False (BUG-C)."""
        hybrid = asdict(
            HybridExampleResult(
                example_id="ex-1",
                metrics={},
                error="upstream 500",
            )
        )
        trial = _trial("trial_0", {"model": "m"}, [hybrid])
        cell = build_outcome_matrix(_result([trial]))["examples"][0]["cells"]["trial_0"]
        assert cell["success"] is False
        assert cell["error"] == "upstream 500"


# ---------------------------------------------------------------------------
# Persistence via the logger + accessor round-trip
# ---------------------------------------------------------------------------


class TestPersistAndLoad:
    def test_session_end_writes_matrix_artifact(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_session_end(_two_config_result())
        artifacts = list((log.run_path / "artifacts").glob("outcome_matrix_v*.json"))
        assert len(artifacts) == 1

    def test_load_round_trips_the_matrix(self, tmp_path: Path) -> None:
        result = _two_config_result()
        log = _make_logger(tmp_path)
        log.log_session_end(result)

        loaded = load_outcome_matrix(log.run_path)
        assert loaded is not None
        assert loaded["schema_version"] == OUTCOME_MATRIX_SCHEMA_VERSION
        assert loaded["example_count"] == 2
        assert loaded["trial_count"] == 2
        rows = {row["example_id"]: row["cells"] for row in loaded["examples"]}
        assert rows["ex-1"]["trial_0"]["accuracy"] == 1.0
        assert rows["ex-1"]["trial_0"]["tokens"] == {
            "input": 100,
            "output": 10,
            "total": 110,
        }

    def test_load_accepts_artifacts_dir_directly(self, tmp_path: Path) -> None:
        log = _make_logger(tmp_path)
        log.log_session_end(_two_config_result())
        loaded = load_outcome_matrix(log.run_path / "artifacts")
        assert loaded is not None
        assert loaded["example_count"] == 2

    def test_load_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert load_outcome_matrix(tmp_path) is None

    def test_no_matrix_written_when_no_examples(self, tmp_path: Path) -> None:
        trial = TrialResult(
            trial_id="trial_0",
            config={"model": "m"},
            metrics={"accuracy": 0.5},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={},
        )
        log = _make_logger(tmp_path)
        log.log_session_end(_result([trial]))
        artifacts = list((log.run_path / "artifacts").glob("outcome_matrix_v*.json"))
        assert artifacts == []
        # best_config is still written — matrix skipping is independent.
        assert list((log.run_path / "artifacts").glob("best_config_v*.json"))


# ---------------------------------------------------------------------------
# OptimizationResult.example_matrix property
# ---------------------------------------------------------------------------


class TestExampleMatrixProperty:
    def test_property_matches_builder(self) -> None:
        result = _two_config_result()
        via_property = result.example_matrix
        assert via_property["example_count"] == 2
        assert via_property["trial_count"] == 2
        rows = {row["example_id"]: row["cells"] for row in via_property["examples"]}
        assert rows["ex-2"]["trial_1"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Isolation guard + success derivation edge cases
# ---------------------------------------------------------------------------


class TestIsolationGuard:
    def test_matrix_failure_does_not_break_run_finalization(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # A failure building/persisting the outcome matrix must NOT abort run
        # finalization: best_config (written before) and the rest of
        # log_session_end must still complete. Without the try/except guard,
        # log_session_end would propagate this RuntimeError and this call raises.
        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("matrix build blew up")

        monkeypatch.setattr(OptimizationLogger, "log_outcome_matrix", _boom)
        log = _make_logger(tmp_path)
        log.log_session_end(_two_config_result())  # must not raise
        # best_config still present -> finalization proceeded past the guarded call.
        assert list((log.run_path / "artifacts").glob("best_config_v*.json"))


class TestSuccessDerivation:
    def test_explicit_success_false_with_no_error_stays_false(self) -> None:
        # ExampleResult carries an explicit success bool; success=False with a
        # null error_message must be respected, NOT coerced to True by the
        # error-is-None fallback (which is only for the hybrid asdict shape).
        ex = _example_dict("ex-x", 0.0, success=False, error=None)
        matrix = build_outcome_matrix(_result([_trial("t0", {"m": "x"}, [ex])]))
        cell = matrix["examples"][0]["cells"]["t0"]
        assert cell["success"] is False
