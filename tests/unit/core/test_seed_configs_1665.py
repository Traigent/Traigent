"""#1665: ``optimize(seed_configs=[...])`` runs user-supplied configs first."""

from __future__ import annotations

import pytest

import traigent
from traigent.core.seed_configs import validate_seed_configs
from traigent.evaluators.base import Dataset, EvaluationExample

SPACE = {"model": ["a", "b", "c"], "top_k": [5, 10, 15], "temp": (0.0, 1.0)}
SEED_1 = {"model": "c", "top_k": 15, "temp": 0.25}
SEED_2 = {"model": "b", "top_k": 5, "temp": 0.5}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _agent(
    default_config=None,
    parallel_trials=None,
    space=None,
    algorithm="random",
    constraints=None,
):
    seen: list[dict] = []
    extra = (
        {}
        if parallel_trials is None
        else {"parallel_config": {"trial_concurrency": parallel_trials}}
    )

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "x")], name="w"),
        objectives=["accuracy"],
        scoring_function=lambda p, e: 1.0,
        configuration_space=space or SPACE,
        constraints=constraints,
        default_config=default_config,
        algorithm=algorithm,
        execution={"offline": True},
        injection_mode="parameter",
        config_param="cfg",
        **extra,
    )
    def agent(text, cfg=None):
        seen.append(dict(cfg))
        return "x"

    return agent, seen


def test_trial_one_is_the_seed_and_seeds_count_toward_max_trials():
    agent, _ = _agent()
    result = agent.optimize_sync(max_trials=4, seed_configs=[SEED_1, SEED_2])

    assert len(result.trials) == 4
    assert result.trials[0].config == SEED_1
    assert result.trials[1].config == SEED_2


def test_seeds_beyond_max_trials_are_truncated_to_the_budget():
    agent, _ = _agent()
    result = agent.optimize_sync(max_trials=1, seed_configs=[SEED_1, SEED_2])

    assert [t.config for t in result.trials] == [SEED_1]


def test_default_config_baseline_runs_before_seeds():
    baseline = {"model": "a", "top_k": 10, "temp": 0.0}
    agent, _ = _agent(default_config=baseline)
    result = agent.optimize_sync(max_trials=3, seed_configs=[SEED_1])

    assert result.trials[0].config == baseline
    assert result.trials[1].config == SEED_1
    assert len(result.trials) == 3


def test_seeds_are_evaluated_first_with_parallel_trials():
    agent, _ = _agent(parallel_trials=2)
    result = agent.optimize_sync(max_trials=5, seed_configs=[SEED_1, SEED_2])

    assert len(result.trials) == 5
    assert {tuple(sorted(t.config.items())) for t in result.trials[:2]} == {
        tuple(sorted(SEED_1.items())),
        tuple(sorted(SEED_2.items())),
    }


def test_no_seeds_leaves_behavior_unchanged():
    agent, _ = _agent()
    result = agent.optimize_sync(max_trials=2)

    assert len(result.trials) == 2


def test_validation_runs_before_any_trial():
    agent, seen = _agent()
    with pytest.raises(ValueError):
        agent.optimize_sync(max_trials=2, seed_configs=[{**SEED_1, "model": "nope"}])
    assert seen == []


def test_unknown_key_error_does_not_echo_the_key_or_value():
    content_key = "Alice_has_diabetes"
    with pytest.raises(ValueError) as exc:
        validate_seed_configs([{**SEED_1, content_key: "x"}], SPACE)
    assert content_key not in str(exc.value)
    assert "Seed config #0" in str(exc.value)


def test_out_of_space_value_error_names_param_but_not_value():
    with pytest.raises(ValueError) as exc:
        validate_seed_configs([{**SEED_1, "model": "Alice_has_diabetes"}], SPACE)
    msg = str(exc.value)
    assert "Alice_has_diabetes" not in msg
    assert "['model']" in msg


@pytest.mark.parametrize(
    "bad",
    [
        [{"model": "a", "top_k": 5}],  # missing declared param
        [{**SEED_1, "temp": 7.0}],  # outside numeric range
        [{**SEED_1, "top_k": True}],  # bool is not an in-space int choice
        [SEED_1, dict(SEED_1)],  # duplicate
        ["not-a-dict"],
        "not-a-list",
    ],
)
def test_invalid_seeds_rejected(bad):
    with pytest.raises(ValueError):
        validate_seed_configs(bad, SPACE)


def test_seeds_require_a_declared_space():
    with pytest.raises(ValueError):
        validate_seed_configs([SEED_1], {})


def test_none_and_empty_mean_no_seeds():
    assert validate_seed_configs(None, SPACE) == []
    assert validate_seed_configs([], SPACE) == []


def test_grid_never_reproposes_a_seed_point():
    agent, seen = _agent(space={"model": ["a", "b", "c"]}, algorithm="grid")
    result = agent.optimize_sync(max_trials=3, seed_configs=[{"model": "b"}])

    models = [t.config["model"] for t in result.trials]
    assert models[0] == "b"
    assert sorted(models) == ["a", "b", "c"]


def test_seed_violating_a_pre_evaluation_constraint_is_rejected():
    agent, seen = _agent(constraints=[lambda cfg: cfg["model"] != "c"])
    with pytest.raises(ValueError, match="Seed config #0 violates"):
        agent.optimize_sync(max_trials=2, seed_configs=[SEED_1])
    assert seen == []


def test_parallel_batch_with_seeds_holds_back_optimizer_suggestions():
    from unittest.mock import MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch._default_config = None
    orch._default_config_used = False
    orch._seed_configs = [dict(SEED_1)]
    orch._constraints_pre_eval = []
    orch.knob_resolver = None
    orch.optimizer = MagicMock()
    orch.optimizer.suggest_next_trial.return_value = dict(SEED_2)

    import asyncio

    configs, used_async = asyncio.run(orch._generate_parallel_configs(MagicMock(), 3))

    assert configs == [SEED_1]
    assert used_async is False
    orch.optimizer.suggest_next_trial.assert_not_called()
    orch.optimizer.mark_config_evaluated.assert_called_once_with(SEED_1)


def test_resume_skips_seeds_already_covered_by_prior_trials():
    from types import SimpleNamespace

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch._seed_configs = [dict(SEED_1), dict(SEED_2)]
    orch._drop_seeds_covered_by_resume(
        [SimpleNamespace(config=dict(SEED_1))], [dict(SEED_2)]
    )

    assert orch._seed_configs == []


@pytest.mark.asyncio
async def test_cloud_guided_path_hands_seeds_to_its_orchestrator(monkeypatch):
    """The cloud-guided (Bayesian) path builds its own orchestrator; the seeds
    must reach it instead of being dropped (terra round 2)."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from traigent.config.types import ExecutionMode, resolve_execution_policy
    from traigent.core.execution_policy_runtime import SOURCE_CLOUD_BRAIN
    from traigent.core.optimization_pipeline import create_traigent_config
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-key-1665")
    policy = resolve_execution_policy(algorithm="auto", source_hint="optimize")
    opt_func = OptimizedFunction(
        func=lambda text, **cfg: "ok",
        configuration_space=SPACE,
        objectives=["accuracy"],
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "x")], name="w"),
        algorithm="auto",
        execution_mode=ExecutionMode.HYBRID,
        execution_policy=policy,
    )
    config = create_traigent_config(
        execution_mode=opt_func.execution_mode,
        local_storage_path=None,
        minimal_logging=False,
        privacy_enabled=False,
        execution_policy=policy,
        result_source=SOURCE_CLOUD_BRAIN,
    )
    orchestrator = MagicMock()
    with (
        patch("traigent.optimizers.interactive_optimizer.InteractiveOptimizer"),
        patch("traigent.cloud.client.TraigentCloudClient"),
        patch.object(
            OptimizedFunction,
            "_build_optimization_orchestrator",
            return_value=orchestrator,
        ),
        patch.object(
            OptimizedFunction,
            "_run_and_finalize_optimization",
            new=AsyncMock(return_value="result"),
        ),
    ):
        out = await opt_func._try_cloud_execution(
            opt_func._load_dataset(None),
            4,
            None,
            SPACE,
            {},
            config,
            {},
            False,
            None,
            True,
            None,
            None,
            seed_configs=[SEED_1],
        )

    assert out == "result"
    orchestrator.set_seed_configs.assert_called_once_with([SEED_1])


@pytest.mark.asyncio
async def test_cached_seed_batch_does_not_end_the_parallel_run():
    """A seed-only batch fully filtered by the cache policy continues, instead
    of ending the run before any optimizer suggestion (terra round 2)."""
    from unittest.mock import MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch._seed_configs = []
    orch._parallel_batch_had_seeds = False

    async def fake_generate(dataset, size):
        orch._parallel_batch_had_seeds = True
        return [dict(SEED_1)], False

    orch._calculate_parallel_batch_caps = MagicMock(return_value=(3, 2, False))
    orch._generate_parallel_configs = fake_generate
    orch._apply_cap_and_prevent_excess = lambda configs, cap: configs
    orch._apply_cache_policy_and_prune = lambda configs, fn, ds: []
    orch.parallel_trials = 2

    step = await orch._run_parallel_batch(
        MagicMock(), MagicMock(), "session", None, 0, 3
    )
    assert step == (0, "continue")

    # Once the seeds are spent, an empty filtered batch ends the run as before.
    orch._parallel_batch_had_seeds = False

    async def fake_generate_plain(dataset, size):
        return [dict(SEED_2)], False

    orch._generate_parallel_configs = fake_generate_plain
    step = await orch._run_parallel_batch(
        MagicMock(), MagicMock(), "session", None, 0, 3
    )
    assert step == (0, "break")
