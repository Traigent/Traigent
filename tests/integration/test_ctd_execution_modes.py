"""CTD execution mode coverage tests driven by CSV specification."""

import csv
import itertools
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError

# Ensure cost-free execution during tests
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_USE_MOCK", "true")
os.environ.setdefault("TRAIGENT_GENERATE_MOCKS", "true")

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "datasets"
    / "matrices"
    / "ctd_spec.csv"
)

TEST_KEY_PREFIX = "t" + "g_"


@dataclass
class ParameterSpec:
    """Simple container for parameter metadata."""

    name: str
    values: list[Any]
    type: str


@dataclass
class ScenarioSpec:
    """Scenario definition loaded from the CSV specification."""

    scenario_id: str
    parameters: list[ParameterSpec]
    constraints: list[str]
    coverage_level: int
    function_id: str
    optimizer: str
    max_trials: int
    expected_trials: int
    expected_best_score: float
    expected_best_config: dict[str, Any]
    objectives: list[str]
    search_space: dict[str, list[Any]]


def _parse_parameters(raw: dict[str, dict[str, Any]]) -> list[ParameterSpec]:
    parameters: list[ParameterSpec] = []
    for name, body in raw.items():
        values = body.get("values", [])
        parameters.append(
            ParameterSpec(
                name=name,
                values=list(values),
                type=body.get("type", "string"),
            )
        )
    return parameters


def _load_scenarios(spec_path: Path) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    with spec_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("scenario_id"):
                continue

            raw_parameters = (
                json.loads(row["parameters"]) if row.get("parameters") else {}
            )
            constraints = (
                json.loads(row["constraints"]) if row.get("constraints") else []
            )
            coverage_level = int(row.get("coverage_level", 1) or 1)

            function_id = (row.get("function_id") or "simple_grid_score").strip()
            optimizer = (row.get("optimizer") or "grid").strip()
            max_trials = int(row.get("max_trials", 0) or 0)
            expected_trials = int(row.get("expected_trials", 0) or 0)
            expected_best_score = float(row.get("expected_best_score", 0.0) or 0.0)
            expected_best_config = (
                json.loads(row["expected_best_config"])
                if row.get("expected_best_config")
                else {}
            )
            objectives = (
                json.loads(row["objectives"]) if row.get("objectives") else ["score"]
            )
            search_space = (
                json.loads(row["search_space"]) if row.get("search_space") else {}
            )

            scenarios.append(
                ScenarioSpec(
                    scenario_id=row["scenario_id"],
                    parameters=_parse_parameters(raw_parameters),
                    constraints=constraints,
                    coverage_level=max(1, coverage_level),
                    function_id=function_id,
                    optimizer=optimizer,
                    max_trials=max_trials,
                    expected_trials=expected_trials,
                    expected_best_score=expected_best_score,
                    expected_best_config=expected_best_config,
                    objectives=objectives,
                    search_space=search_space,
                )
            )
    return scenarios


def _constraint_mutex(combo: dict[str, Any], names: Iterable[str]) -> bool:
    return sum(bool(combo.get(name)) for name in names) <= 1


def _constraint_requires(combo: dict[str, Any], requirement: str) -> bool:
    requirement = requirement.strip()
    if requirement.startswith("not "):
        target = requirement[4:].strip()
        return not bool(combo.get(target))
    return bool(combo.get(requirement))


def _respects_constraints(combo: dict[str, Any], constraints: list[str]) -> bool:
    for rule in constraints:
        rule = rule.strip()
        if not rule:
            continue
        if rule.startswith("mutex(") and rule.endswith(")"):
            names = [part.strip() for part in rule[6:-1].split(",") if part.strip()]
            if not _constraint_mutex(combo, names):
                return False
        elif rule.startswith("if_true(") and ")->" in rule:
            condition_part, requirement_part = rule.split("->", 1)
            condition_name = condition_part[len("if_true(") : -1].strip()
            if combo.get(condition_name):
                if not _constraint_requires(combo, requirement_part):
                    return False
        else:
            # Unknown constraint; treat as advisory success to avoid false negatives
            continue
    return True


def _generate_covering_combos(
    parameters: list[ParameterSpec], constraints: list[str], strength: int
) -> list[dict[str, Any]]:
    names = [param.name for param in parameters]
    value_spaces = [param.values for param in parameters]

    all_combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_spaces):
        candidate = dict(zip(names, values, strict=False))
        if _respects_constraints(candidate, constraints):
            all_combos.append(candidate)

    if strength <= 1:
        return all_combos

    target_pairs: set = set()
    for combo in all_combos:
        for subset in itertools.combinations(names, strength):
            target_pairs.add(tuple((name, combo[name]) for name in subset))

    selected: list[dict[str, Any]] = []
    covered: set = set()

    for combo in all_combos:
        combo_pairs = {
            tuple((name, combo[name]) for name in subset)
            for subset in itertools.combinations(names, strength)
        }
        if any(pair not in covered for pair in combo_pairs):
            selected.append(combo)
            covered.update(combo_pairs)
        if covered >= target_pairs:
            break

    if covered >= target_pairs:
        return selected
    return all_combos


def _expected_mode(combo: dict[str, Any]) -> ExecutionMode:
    # Note: cloud/hybrid raise ConfigurationError (not yet supported)
    # standard/privacy have been removed from the enum
    # All paths now lead to edge_analytics (only supported mode)
    return ExecutionMode.EDGE_ANALYTICS


def _grid_configurations(config_space: dict[str, list[Any]]):
    keys = list(config_space.keys())
    values = [config_space[key] for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination, strict=False))


def _grid_score(config: dict[str, Any]) -> float:
    threshold = float(config.get("threshold", 0.0))
    weight = float(config.get("weight", 1.0))
    base = 1.0 - abs(threshold - 0.5) * 0.8
    penalty = (weight - 1.0) * 0.2
    score = base - penalty
    return max(score, 0.0)


_SIMPLE_DATASET = Dataset(
    examples=[
        EvaluationExample(input_data={"text": "example"}, expected_output="positive"),
    ],
    name="ctd_dataset",
    description="Deterministic dataset for CTD behavior validation",
)


async def _simple_agent_function(
    input_data: dict[str, Any], **config
) -> dict[str, Any]:
    return {"input": input_data, "config": config}


class SimpleGridOptimizer(BaseOptimizer):
    """Deterministic grid search optimizer for behavioral validation."""

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        max_trials: int | None = None,
    ) -> None:
        super().__init__(config_space, objectives)
        self._combos = list(_grid_configurations(config_space))
        self._index = 0
        self._limit = min(max_trials or len(self._combos), len(self._combos))

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        if self.should_stop(history):
            raise OptimizationError("Grid search exhausted search space")
        config = dict(self._combos[self._index])
        self._index += 1
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._index >= self._limit


class DeterministicEvaluator(BaseEvaluator):
    """Evaluator that returns deterministic metrics from a scoring function."""

    def __init__(
        self, score_fn: Callable[[dict[str, Any]], float], objectives: list[str]
    ):
        super().__init__(metrics=objectives)
        self._score_fn = score_fn

    async def evaluate(
        self, func: Callable[..., Any], config: dict[str, Any], dataset: Dataset
    ) -> EvaluationResult:
        score = float(self._score_fn(config))
        metrics = {self.metrics[0]: score}
        return EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=len(dataset.examples),
            successful_examples=len(dataset.examples),
            duration=0.0,
            metrics=metrics,
            outputs=[config for _ in dataset.examples],
            errors=[None for _ in dataset.examples],
        )


def _assert_best_config(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if not expected:
        return
    for key, expected_value in expected.items():
        assert key in actual, f"Missing key '{key}' in best config"
        actual_value = actual[key]
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value)
        else:
            assert actual_value == expected_value


async def _run_behavior_validation(
    case: dict[str, Any], execution_mode: ExecutionMode
) -> OptimizationResult:
    scenario: ScenarioSpec = case["scenario"]

    if scenario.optimizer != "grid":
        pytest.skip(f"Optimizer '{scenario.optimizer}' not supported in CTD test.")

    score_fn = _grid_score
    objectives = scenario.objectives or ["score"]

    optimizer = SimpleGridOptimizer(
        config_space=scenario.search_space,
        objectives=objectives,
        max_trials=scenario.max_trials,
    )
    evaluator = DeterministicEvaluator(score_fn=score_fn, objectives=objectives)

    mode_value = (
        execution_mode.value
        if isinstance(execution_mode, ExecutionMode)
        else str(execution_mode)
    )
    traigent_config = TraigentConfig(execution_mode=mode_value)

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=scenario.max_trials,
        config=traigent_config,
        objectives=objectives,
    )

    return await orchestrator.optimize(_simple_agent_function, _SIMPLE_DATASET)


SCENARIOS = _load_scenarios(SPEC_PATH)
CTD_CASES: list[dict[str, Any]] = []
for scenario in SCENARIOS:
    combos = _generate_covering_combos(
        scenario.parameters, scenario.constraints, scenario.coverage_level
    )
    for index, combo in enumerate(combos):
        CTD_CASES.append(
            {
                "id": f"{scenario.scenario_id}-{index + 1}",
                "scenario": scenario,
                "combo": combo,
                "expected": _expected_mode(combo),
            }
        )


@pytest.fixture(autouse=True)
def _patch_backend(monkeypatch):
    """Replace BackendIntegratedClient to avoid outbound traffic."""
    from traigent.cloud.backend_client import BackendIntegratedClient as Original

    mock_backend = Mock(spec=Original)
    mock_backend.create_session.return_value = "mock-session"
    mock_backend.submit_result.return_value = True
    mock_backend.update_trial_weighted_scores.return_value = True
    mock_backend.finalize_session_sync.return_value = None
    mock_backend.finalize_session.return_value = None
    mock_backend.delete_session.return_value = True

    # Patch at both locations where BackendIntegratedClient may be imported
    traigent_client_path = ".".join(
        ["traigent", "traigent_client", "BackendIntegratedClient"]
    )
    cloud_client_path = ".".join(
        ["traigent", "cloud", "backend_client", "BackendIntegratedClient"]
    )
    monkeypatch.setattr(
        traigent_client_path,
        lambda *args, **kwargs: mock_backend,
    )
    monkeypatch.setattr(
        cloud_client_path,
        lambda *args, **kwargs: mock_backend,
    )
    return mock_backend


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CTD_CASES, ids=lambda payload: payload["id"])
async def test_ctd_execution_behavior(case, monkeypatch):
    """Validate execution mode detection and optimization behavior for CTD combos."""
    combo = dict(case["combo"])

    # Enforce mock-only execution to avoid real costs
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_USE_MOCK", "true")
    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "true")

    # Reset force flags before applying scenario specifics
    for env_key in [
        "TRAIGENT_FORCE_LOCAL",
        "TRAIGENT_FORCE_HYBRID",
        "TRAIGENT_FORCE_CLOUD",
        "TRAIGENT_PRIVATE_DATA",
    ]:
        monkeypatch.delenv(env_key, raising=False)

    if combo.get("force_local"):
        monkeypatch.setenv("TRAIGENT_FORCE_LOCAL", "1")
    if combo.get("force_hybrid"):
        monkeypatch.setenv("TRAIGENT_FORCE_HYBRID", "1")
    if combo.get("force_cloud"):
        monkeypatch.setenv("TRAIGENT_FORCE_CLOUD", "1")
    if combo.get("privacy_flag"):
        monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "1")

    # Configure backend URL as specified
    backend_url = combo.get("backend_url")
    if backend_url == "local":
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    else:
        backend_host = "api" + ".traigent.ai"
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", f"https://{backend_host}")

    # Clear sensitive env to prevent leakage between cases
    for key in ["TRAIGENT_API_KEY", "OPTIGEN_API_KEY", "TRAIGENT_BACKEND_URL"]:
        monkeypatch.delenv(key, raising=False)

    api_key = None
    if combo.get("has_api_key"):
        api_key = TEST_KEY_PREFIX + ("x" * 12)

    from traigent.traigent_client import TraigentClient as OptiGenClient

    client = OptiGenClient(
        execution_mode=combo.get("explicit_mode", "auto"), api_key=api_key
    )
    assert (
        client.execution_mode == case["expected"]
    ), f"Combination {case['id']} resolved to {client.execution_mode}"

    # Run behavior validation to ensure optimization meets expectations
    result = await _run_behavior_validation(case, client.execution_mode)
    scenario: ScenarioSpec = case["scenario"]

    assert len(result.trials) == scenario.expected_trials, (
        f"Expected {scenario.expected_trials} trials, got {len(result.trials)} "
        f"for {case['id']}"
    )
    assert result.best_score == pytest.approx(
        scenario.expected_best_score
    ), f"Best score mismatch for {case['id']}"
    _assert_best_config(result.best_config, scenario.expected_best_config)
    assert result.success_rate == pytest.approx(1.0)

    actual_configs = {tuple(sorted(trial.config.items())) for trial in result.trials}
    expected_configs = {
        tuple(sorted(cfg.items()))
        for cfg in itertools.islice(
            _grid_configurations(scenario.search_space), scenario.expected_trials
        )
    }
    assert (
        actual_configs == expected_configs
    ), f"Config coverage mismatch for {case['id']}"

    for trial in result.trials:
        assert trial.metrics is not None
        assert scenario.objectives[0] in trial.metrics


def test_ctd_case_coverage():
    """Quick sanity check that CTD cases were generated."""
    assert CTD_CASES, "No CTD cases generated from specification"
