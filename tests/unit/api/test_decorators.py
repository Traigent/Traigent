"""Comprehensive tests for traigent.api.decorators module."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.api.decorators import optimize
from traigent.api.strategy_presets import (
    MAX_ACCURACY_THEN_CHEAPEST,
    PARETO_FRONTIER,
    QUALITY_FLOOR_MIN_COST,
    VALID_PRESET_NAMES,
    UnknownStrategyPresetError,
)
from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset
from traigent.utils.exceptions import ConfigurationError


def _write_environment_tvl_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "environment_overlay.tvl.yml"
    spec_path.write_text(
        """tvl:
  module: test.environment
tvl_version: "0.9"
tvars:
  - name: retrieval_depth
    type: int
    domain:
      range: [2, 6]
objectives:
  - name: accuracy
    direction: maximize
environments:
  finals_week:
    overrides:
      tvars:
        - name: retrieval_depth
          type: int
          domain:
            range: [3, 8]
""",
        encoding="utf-8",
    )
    return spec_path


class TestOptimizeDecorator:
    """Test suite for the @optimize decorator."""

    def test_decorator_with_config_space(self):
        """Test decorator with configuration space."""

        @optimize(configuration_space={"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})
        def sample_function(x: int, y: float) -> float:
            """Sample function for testing."""
            return x * y

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.__name__ == "sample_function"
        assert sample_function.__doc__ == "Sample function for testing."

    def test_decorator_with_objectives(self):
        """Test decorator with objectives."""

        @optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            objectives=["accuracy", "cost"],
        )
        def llm_function(model: str, temperature: float) -> str:
            return f"Result from {model} with temp {temperature}"

        assert isinstance(llm_function, OptimizedFunction)
        assert llm_function.configuration_space == {
            "model": ["gpt-3.5", "gpt-4"],
            "temperature": [0.0, 0.5, 1.0],
        }

    def test_decorator_with_default_config(self):
        """Test decorator with default configuration."""

        @optimize(
            configuration_space={"batch_size": [16, 32, 64], "lr": [0.001, 0.01, 0.1]},
            default_config={"batch_size": 32, "lr": 0.01},
        )
        def training_function(batch_size: int, lr: float) -> float:
            return batch_size * lr

        assert isinstance(training_function, OptimizedFunction)
        assert training_function.default_config == {"batch_size": 32, "lr": 0.01}

    def test_execution_bundle_passes_hybrid_api_transport(self):
        """Execution bundle should preserve preconfigured transport objects."""
        from traigent.api.decorators import ExecutionOptions

        transport = Mock(name="hybrid_transport")

        @optimize(
            configuration_space={"x": [1, 2]},
            execution=ExecutionOptions(
                execution_mode="hybrid_api",
                hybrid_api_transport=transport,
            ),
        )
        def sample_function(x: int) -> int:
            return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.execution_mode == "hybrid_api"
        assert sample_function.hybrid_api_transport is transport

    def test_execution_bundle_deprecated_cloud_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Deprecated cloud fails closed even when the old env override is set."""
        from traigent.api.decorators import ExecutionOptions

        monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

        with pytest.raises(ConfigurationError, match="fails closed"):

            @optimize(
                configuration_space={"x": [1, 2]},
                execution=ExecutionOptions(
                    execution_mode="cloud",
                    cloud_fallback_policy="auto",
                ),
            )
            def sample_function(x: int) -> int:
                return x

    def test_direct_hybrid_api_transport_runtime_option_is_supported(self):
        """Direct runtime options should accept hybrid_api_transport."""
        transport = Mock(name="hybrid_transport_direct")

        @optimize(
            configuration_space={"x": [1, 2]},
            execution_mode="hybrid_api",
            hybrid_api_transport=transport,
        )
        def sample_function(x: int) -> int:
            return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.hybrid_api_transport is transport

    def test_decorator_execution_mode_registry_matches_runtime(self):
        """Decorator validation accepts the same modes advertised by runtime."""
        import warnings

        from traigent.config.types import accepted_execution_mode_values

        with warnings.catch_warnings(record=True) as local_warnings:
            warnings.simplefilter("always")

            @optimize(configuration_space={"x": [1, 2]}, execution_mode="local")
            def sample_function(x: int) -> int:
                return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.execution_mode == "edge_analytics"
        assert any(issubclass(w.category, DeprecationWarning) for w in local_warnings)

        # "standard" is now a deprecated alias (emits DeprecationWarning, resolves to hybrid)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @optimize(configuration_space={"x": [1, 2]}, execution_mode="standard")
            def deprecated_mode_function(x: int) -> int:
                return x

        assert isinstance(deprecated_mode_function, OptimizedFunction)
        assert deprecated_mode_function.execution_policy.intent.value == "cloud_brain"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

        for mode in accepted_execution_mode_values():

            @optimize(configuration_space={"x": [1, 2]}, execution_mode=mode)
            def accepted_mode_function(x: int) -> int:
                return x

            assert isinstance(accepted_mode_function, OptimizedFunction)

    def test_decorator_max_trials_reaches_optimized_function(self):
        """Regression: max_trials from decorator must reach OptimizedFunction.

        Bug: max_trials went into combined_settings via record_option but was
        never extracted, so OptimizedFunction always got the SDK default.
        """

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            max_trials=10,
        )
        def sample_function(x: int) -> int:
            return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.max_trials == 10, (
            f"max_trials should be 10 (from decorator), got {sample_function.max_trials}"
        )

    def test_decorator_omitted_max_trials_uses_sdk_default(self):
        """Omitting max_trials on the decorator should use the shared default."""

        @optimize(
            configuration_space={"x": [1, 2, 3]},
        )
        def sample_function(x: int) -> int:
            return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.max_trials == 10

    def test_decorator_accepts_algorithm_runtime_default(self):
        """Decorator-level algorithm should become the optimize() default."""

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            algorithm="grid",
        )
        def sample_function(x: int) -> int:
            return x

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.algorithm == "grid"

    def test_decorator_rejects_unknown_strategy_name(self):
        """Non-preset strategy values should list valid presets."""
        with pytest.raises(ValueError) as exc_info:

            @optimize(
                configuration_space={"x": [1, 2]},
                strategy="grid",
            )
            def sample_function(x: int) -> int:
                return x

        assert isinstance(exc_info.value, UnknownStrategyPresetError)
        message = str(exc_info.value)
        assert message == (
            f"Unknown strategy preset 'grid'. Valid presets: "
            f"{', '.join(VALID_PRESET_NAMES)}."
        )
        for preset_name in VALID_PRESET_NAMES:
            assert preset_name in message

    def test_decorator_accepts_strategy_preset(self):
        """Registered strategy names should configure advisory preset metadata."""

        @optimize(
            configuration_space={"model": ["cheap", "accurate"]},
            strategy=MAX_ACCURACY_THEN_CHEAPEST,
            strategy_params={"epsilon": 0.02},
        )
        def sample_function() -> str:
            return "ok"

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.objectives == ["accuracy", "cost"]
        assert sample_function.constraints == []
        assert sample_function.strategy_preset.to_metadata() == {
            "preset_name": MAX_ACCURACY_THEN_CHEAPEST,
            "params": {"epsilon": 0.02},
            "selection_grade": "advisory",
            "selection_rationale": (
                "Selected the lowest-cost completed trial within the preset accuracy band."
            ),
        }

    def test_decorator_accepts_pareto_frontier_strategy_preset(self):
        """Pareto frontier preset should resolve objectives without extra params."""

        @optimize(
            configuration_space={"model": ["cheap", "accurate"]},
            strategy=PARETO_FRONTIER,
            strategy_params={},
        )
        def sample_function() -> str:
            return "ok"

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.objectives == ["accuracy", "cost"]
        assert sample_function.strategy_preset.to_metadata() == {
            "preset_name": PARETO_FRONTIER,
            "params": {},
            "selection_grade": "advisory",
            "selection_rationale": (
                "Selected all completed trials on the advisory accuracy-cost Pareto frontier."
            ),
        }

    def test_decorator_rejects_pareto_frontier_unexpected_params(self):
        with pytest.raises(ValueError, match="does not accept strategy_params"):

            @optimize(
                configuration_space={"model": ["cheap", "accurate"]},
                strategy=PARETO_FRONTIER,
                strategy_params={"epsilon": 0.1},
            )
            def sample_function() -> str:
                return "ok"

    def test_decorator_rejects_strategy_preset_with_objectives(self):
        """Preset business goals should not silently override hand-set objectives."""
        with pytest.raises(ValueError, match="mutually exclusive"):

            @optimize(
                configuration_space={"model": ["cheap", "accurate"]},
                objectives=["accuracy"],
                strategy=QUALITY_FLOOR_MIN_COST,
                strategy_params={"floor": 0.8},
            )
            def sample_function() -> str:
                return "ok"

    def test_decorator_strategy_preset_end_to_end_mock_mode(self, monkeypatch):
        """Preset selection should be exposed without replacing best_config."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        async def evaluator(func, config, example):
            _ = func, example
            if config["model"] == "accurate":
                metrics = {"accuracy": 0.9, "cost": 0.03}
            else:
                metrics = {"accuracy": 0.85, "cost": 0.01}
            return ExampleResult(
                example_id="ex_1",
                input_data={"prompt": "hello"},
                expected_output="ok",
                actual_output="ok",
                metrics=metrics,
                execution_time=0.01,
                success=True,
            )

        @optimize(
            eval_dataset=[{"input": {"prompt": "hello"}, "expected": "ok"}],
            configuration_space={"model": ["accurate", "cheap"]},
            custom_evaluator=evaluator,
            algorithm="grid",
            max_trials=2,
            strategy=MAX_ACCURACY_THEN_CHEAPEST,
            strategy_params={"epsilon": 0.05},
        )
        def sample_function() -> str:
            return "ok"

        result = sample_function.optimize_sync()

        assert result.preset_selection is not None
        assert result.preset_selection.selection_grade == "advisory"
        assert result.preset_selection.selected_config == {"model": "cheap"}
        assert result.metadata["strategy_preset"]["preset_name"] == (
            MAX_ACCURACY_THEN_CHEAPEST
        )
        assert result.metadata["strategy_preset"]["selection_grade"] == "advisory"
        assert result.best_config != result.preset_selection.selected_config

    def test_quality_floor_preset_matches_no_preset_trial_outcomes_and_best_config(
        self, monkeypatch, tmp_path
    ):
        """Floor preset selection must not constrain trials or exported best_config."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
        configuration_space = {"model": ["accurate", "good-cheap", "bad-cheap"]}
        metrics_by_model = {
            "accurate": {"accuracy": 0.95, "cost": 0.20},
            "good-cheap": {"accuracy": 0.91, "cost": 0.01},
            "bad-cheap": {"accuracy": 0.70, "cost": 0.001},
        }

        async def evaluator(func, config, example):
            _ = func, example
            return ExampleResult(
                example_id="ex_1",
                input_data={"prompt": "hello"},
                expected_output="ok",
                actual_output="ok",
                metrics=dict(metrics_by_model[config["model"]]),
                execution_time=0.01,
                success=True,
            )

        @optimize(
            eval_dataset=[{"input": {"prompt": "hello"}, "expected": "ok"}],
            configuration_space=configuration_space,
            custom_evaluator=evaluator,
            algorithm="grid",
            max_trials=3,
            objectives=["accuracy", "cost"],
        )
        def no_preset_function() -> str:
            return "ok"

        @optimize(
            eval_dataset=[{"input": {"prompt": "hello"}, "expected": "ok"}],
            configuration_space=configuration_space,
            custom_evaluator=evaluator,
            algorithm="grid",
            max_trials=3,
            strategy=QUALITY_FLOOR_MIN_COST,
            strategy_params={"floor": 0.9},
        )
        def floor_preset_function() -> str:
            return "ok"

        no_preset_result = no_preset_function.optimize_sync()
        floor_result = floor_preset_function.optimize_sync()

        def trial_outcomes(result):
            return [
                (dict(trial.config), trial.status, trial.error_message)
                for trial in result.trials
            ]

        assert trial_outcomes(floor_result) == trial_outcomes(no_preset_result)
        assert floor_result.best_config == no_preset_result.best_config
        assert floor_preset_function.current_config == no_preset_result.best_config
        assert floor_result.preset_selection is not None
        assert floor_result.preset_selection.selected_config == {"model": "good-cheap"}
        assert floor_result.preset_selection.selected_config != floor_result.best_config

        export_path = floor_preset_function.export_config(
            tmp_path / "floor-preset-config.json",
            format="slim",
        )
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        assert exported["config"] == no_preset_result.best_config
        assert exported["config"] != floor_result.preset_selection.selected_config

    def test_decorated_function_execution(self):
        """Test that decorated function can still be called normally."""

        @optimize(configuration_space={"multiplier": [1, 2, 3, 4, 5]})
        def multiply_function(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Function should still work normally
        result = multiply_function(10, multiplier=3)
        assert result == 30

    def test_decorator_preserves_signature(self):
        """Test that decorator creates callable optimized function."""

        @optimize(configuration_space={"param": ["a", "b", "c"]})
        def complex_function(arg1: str, arg2: int = 5, *args, **kwargs) -> str:
            return f"{arg1}-{arg2}-{len(args)}-{len(kwargs)}"

        # Should be callable and maintain name/doc
        assert callable(complex_function)
        assert complex_function.__name__ == "complex_function"
        assert isinstance(complex_function, OptimizedFunction)

        # Should still work when called
        result = complex_function("test", 10, "extra", key="value")
        assert "test-10-1-1" in result

    def test_decorator_with_cloud_execution_mode_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Deprecated cloud mode raises before decorator construction."""
        monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

        with pytest.raises(ConfigurationError, match="fails closed"):

            @optimize(
                configuration_space={"model": ["claude", "gpt-4"]},
                execution_mode="cloud",
            )
            def ai_function(model: str) -> str:
                return f"Using {model}"

    def test_decorator_accepts_cost_limit_runtime_override(self):
        """cost_limit should be accepted as a runtime override key."""

        @optimize(
            configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
            cost_limit=5.0,
        )
        def ai_function(prompt: str) -> str:
            return prompt

        assert isinstance(ai_function, OptimizedFunction)
        assert ai_function.kwargs["cost_limit"] == 5.0

    def test_decorator_accepts_metric_limit_runtime_override(self):
        """metric_limit should be accepted with a required metric_name."""

        @optimize(
            configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
            metric_limit=50_000,
            metric_name="total_tokens",
        )
        def ai_function(prompt: str) -> str:
            return prompt

        assert isinstance(ai_function, OptimizedFunction)
        assert ai_function.kwargs["metric_limit"] == 50_000
        assert ai_function.kwargs["metric_name"] == "total_tokens"

    def test_decorator_with_auto_optimize(self):
        """Test decorator with auto optimization enabled."""

        with pytest.raises(TypeError, match="auto_optimize"):

            @optimize(
                configuration_space={"threshold": [0.1, 0.5, 0.9]},
                auto_optimize=True,
            )
            def threshold_function(value: float, threshold: float = 0.5) -> bool:
                return value > threshold

    def test_decorator_without_config_space(self):
        """Test decorator without configuration space raises ValueError."""

        with pytest.raises(ValueError, match="Configuration space cannot be empty"):

            @optimize()
            def no_config_function():
                return "test"

    def test_multiple_decorated_functions(self):
        """Test that multiple functions can be decorated independently."""

        @optimize(configuration_space={"param1": [1, 2, 3]})
        def function1(param1: int) -> int:
            return param1 * 2

        @optimize(configuration_space={"param2": ["a", "b", "c"]})
        def function2(param2: str) -> str:
            return param2.upper()

        assert isinstance(function1, OptimizedFunction)
        assert isinstance(function2, OptimizedFunction)
        assert function1 != function2

    def test_decorator_with_constraints(self):
        """Test decorator with constraint functions."""

        def constraint_func(config):
            return config["x"] < config["y"]

        @optimize(
            configuration_space={"x": [1, 2, 3], "y": [2, 3, 4]},
            constraints=[constraint_func],
        )
        def constrained_function(x: int, y: int) -> int:
            return x + y

        assert isinstance(constrained_function, OptimizedFunction)

    def test_decorator_loads_tvl_spec(self):
        """Supplying tvl_spec hydrates the configuration automatically."""

        spec_path = Path("tests/fixtures/tvl/hello_tvl.yml")

        @optimize(tvl_spec=spec_path)
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert "model" in tvl_wrapped.configuration_space
        assert tvl_wrapped.configuration_space["max_tokens"] == [256, 384, 512]

    def test_decorator_wires_evaluation_set_dataset(self, tmp_path):
        """TVL 0.9 evaluation_set.dataset populates eval_dataset when omitted."""
        spec_path = tmp_path / "evalset.tvl.yml"
        spec_path.write_text("""tvl:
  module: test.evalset
tvl_version: "0.9"
evaluation_set:
  dataset: test.jsonl
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: accuracy
    direction: maximize
""")

        @optimize(tvl_spec=spec_path)
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.eval_dataset == "test.jsonl"

    def test_decorator_does_not_override_explicit_eval_dataset(self, tmp_path):
        """Explicit eval_dataset beats evaluation_set in a TVL spec."""
        spec_path = tmp_path / "evalset_override.tvl.yml"
        spec_path.write_text("""tvl:
  module: test.evalset.override
tvl_version: "0.9"
evaluation_set:
  dataset: spec.jsonl
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4"]
objectives:
  - name: accuracy
    direction: maximize
""")

        @optimize(tvl_spec=spec_path, eval_dataset="user.jsonl")
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.eval_dataset == "user.jsonl"

    def test_decorator_applies_tvl_environment_overlay(self, tmp_path):
        spec_path = _write_environment_tvl_spec(tmp_path)

        @optimize(tvl_spec=spec_path, tvl_environment="finals_week")
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.configuration_space["retrieval_depth"] == (3, 8)

    def test_decorator_accepts_structured_tvl_options_model(self, tmp_path):
        from traigent.tvl.options import TVLOptions

        spec_path = _write_environment_tvl_spec(tmp_path)

        @optimize(tvl=TVLOptions(spec_path=str(spec_path), environment="finals_week"))
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.configuration_space["retrieval_depth"] == (3, 8)

    def test_decorator_accepts_structured_tvl_options_dict(self, tmp_path):
        spec_path = _write_environment_tvl_spec(tmp_path)

        @optimize(tvl={"spec_path": str(spec_path), "environment": "finals_week"})
        def tvl_wrapped(question: str) -> str:
            return question

        assert isinstance(tvl_wrapped, OptimizedFunction)
        assert tvl_wrapped.configuration_space["retrieval_depth"] == (3, 8)

    def test_mock_mode_options_round_trip_base_accuracy_and_variance(self):
        from traigent.api.decorators import MockModeOptions

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            mock=MockModeOptions(base_accuracy=0.9, variance=0.05),
        )
        def func_with_mock_round_trip(x):
            return x

        # base_accuracy and variance are inert; this only confirms they survive
        # decorator construction / round-trip into mock_mode_config.
        assert isinstance(func_with_mock_round_trip, OptimizedFunction)
        assert func_with_mock_round_trip.mock_mode_config["base_accuracy"] == 0.9
        assert func_with_mock_round_trip.mock_mode_config["variance"] == 0.05

    def test_decorator_wires_agent_configuration_parameters(self):
        from traigent.api.types import AgentDefinition

        @optimize(
            configuration_space={
                "router_model": ["gpt-4o"],
                "draft_model": ["gpt-4o-mini"],
            },
            agents={
                "router": AgentDefinition(
                    display_name="Router",
                    parameter_keys=["router_model"],
                    measure_ids=["routing_accuracy"],
                )
            },
            agent_prefixes=["draft"],
            agent_measures={
                "router": ["routing_accuracy"],
                "draft": ["draft_accuracy"],
            },
            global_measures=["total_cost"],
        )
        def sample_function() -> str:
            return "ok"

        assert isinstance(sample_function, OptimizedFunction)
        assert sample_function.agents["router"].parameter_keys == ["router_model"]
        assert sample_function.agent_prefixes == ["draft"]
        assert sample_function.agent_measures == {
            "router": ["routing_accuracy"],
            "draft": ["draft_accuracy"],
        }
        assert sample_function.global_measures == ["total_cost"]


class TestOptimizedFunctionIntegration:
    """Integration tests for OptimizedFunction wrapper."""

    def test_optimized_function_has_optimization_methods(self):
        """Test that OptimizedFunction has required optimization methods."""

        @optimize(configuration_space={"param": [1, 2, 3]})
        def test_function(param: int) -> int:
            return param**2

        # Should have optimization-related methods
        assert hasattr(test_function, "optimize")
        assert hasattr(test_function, "run")

        # Methods should be callable
        assert callable(test_function.optimize)
        assert callable(test_function.run)

    def test_function_call_vs_run_method(self):
        """Test difference between direct call and run method."""

        @optimize(configuration_space={"multiplier": [2, 3, 4]})
        def calc_function(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Direct call should work
        direct_result = calc_function(5, multiplier=3)
        assert direct_result == 15

        # Run method should also work
        run_result = calc_function.run(5, multiplier=4)
        assert run_result == 20

    def test_optimization_context_preservation(self):
        """Test that optimization context is preserved across calls."""

        @optimize(
            configuration_space={"mode": ["fast", "accurate"]},
            objectives=["speed", "quality"],
        )
        def contextual_function(data: str, mode: str = "fast") -> dict:
            return {"data": data, "mode": mode, "result": len(data)}

        # Should preserve configuration across multiple calls
        result1 = contextual_function("test1", mode="fast")
        result2 = contextual_function("test2", mode="accurate")

        assert result1["mode"] == "fast"
        assert result2["mode"] == "accurate"

    def test_async_function_decoration(self):
        """Test decoration of async functions."""

        @optimize(configuration_space={"delay": [0.01, 0.02, 0.05]})
        async def async_func(value: int, delay: float = 0.01) -> int:
            await asyncio.sleep(delay)
            return value * 2

        assert isinstance(async_func, OptimizedFunction)
        # Test execution
        result = asyncio.run(async_func(5))
        assert result == 10

    def test_decorator_with_all_parameters(self):
        """Test decorator with all available parameters."""
        mock_dataset = Mock(spec=Dataset)
        mock_constraint = Mock()

        @optimize(
            eval_dataset=mock_dataset,
            objectives=["accuracy", "cost"],
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            default_config={"model": "gpt-3.5"},
            constraints=[mock_constraint],
            injection_mode="parameter",
            config_param="llm_config",
            execution_mode="hybrid",
            auto_override_frameworks=False,
            framework_targets=["openai.OpenAI"],
        )
        def full_function(query: str, llm_config: dict = None) -> str:
            return f"Processing: {query}"

        assert isinstance(full_function, OptimizedFunction)

    def test_dataset_parameter_variations(self):
        """Test different types of dataset parameters."""

        # String dataset
        @optimize(eval_dataset="data.jsonl", configuration_space={"x": [1, 2, 3]})
        def func1(x):
            return x

        assert isinstance(func1, OptimizedFunction)

        # List of strings
        @optimize(
            eval_dataset=["data1.jsonl", "data2.jsonl"],
            configuration_space={"x": [1, 2, 3]},
        )
        def func2(x):
            return x

        assert isinstance(func2, OptimizedFunction)

        # Dataset object
        mock_dataset = Mock(spec=Dataset)

        @optimize(eval_dataset=mock_dataset, configuration_space={"x": [1, 2, 3]})
        def func3(x):
            return x

        assert isinstance(func3, OptimizedFunction)

        # Inline example dicts
        inline_examples = [
            {"input": {"question": "What is 2+2?"}, "expected": "4"},
            {"input": {"question": "Capital of France?"}, "expected": "Paris"},
        ]

        @optimize(eval_dataset=inline_examples, configuration_space={"x": [1, 2, 3]})
        def func4(x):
            return x

        assert isinstance(func4, OptimizedFunction)
        assert func4.eval_dataset == inline_examples

    def test_evaluation_bundle_accepts_inline_examples(self):
        """Evaluation bundle should accept inline datasets in both supported forms."""
        from traigent.api.decorators import EvaluationOptions

        inline_examples = [
            {"input": {"question": "What is 2+2?"}, "expected": "4"},
            {
                "input_data": {"question": "Capital of France?"},
                "expected_output": "Paris",
            },
        ]

        @optimize(
            evaluation={"eval_dataset": inline_examples},
            configuration_space={"x": [1, 2, 3]},
        )
        def func_dict_bundle(x):
            return x

        @optimize(
            evaluation=EvaluationOptions(eval_dataset=inline_examples),
            configuration_space={"x": [1, 2, 3]},
        )
        def func_model_bundle(x):
            return x

        assert isinstance(func_dict_bundle, OptimizedFunction)
        assert isinstance(func_model_bundle, OptimizedFunction)
        assert func_dict_bundle.eval_dataset == inline_examples
        assert func_model_bundle.eval_dataset == inline_examples

    def test_mock_and_legacy_parameter_variations(self):
        """Grouped mock options and legacy bridge inputs should decorate cleanly."""
        from traigent.api.decorators import LegacyOptimizeArgs, MockModeOptions

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            mock=MockModeOptions(enabled=True, override_evaluator=False),
            legacy=LegacyOptimizeArgs(objectives=["accuracy"]),
        )
        def func_with_models(x):
            return x

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            legacy={"objectives": ["accuracy"], "algorithm": "grid"},
        )
        def func_with_legacy_dict(x):
            return x

        assert isinstance(func_with_models, OptimizedFunction)
        assert isinstance(func_with_legacy_dict, OptimizedFunction)
        assert func_with_models.mock_mode_config["override_evaluator"] is False
        assert func_with_legacy_dict.algorithm == "grid"

    def test_injection_modes(self):
        """Test different injection modes."""

        # Context injection (default)
        @optimize(injection_mode="context", configuration_space={"x": [1, 2, 3]})
        def func_context(x):
            return x

        assert isinstance(func_context, OptimizedFunction)

        # Parameter injection
        @optimize(
            injection_mode="parameter",
            config_param="config",
            configuration_space={"x": [1, 2, 3]},
        )
        def func_param(x, config=None):
            return x

        assert isinstance(func_param, OptimizedFunction)

        # Seamless injection (attribute mode was removed in v2.x)
        @optimize(injection_mode="seamless", configuration_space={"x": [1, 2, 3]})
        def func_seamless(x):
            return x

        assert isinstance(func_seamless, OptimizedFunction)

    def test_framework_override_configuration(self):
        """Test framework override configuration."""

        # Default auto_override_frameworks=True
        @optimize(configuration_space={"x": [1, 2, 3]})
        def func1(x):
            return x

        assert isinstance(func1, OptimizedFunction)

        # Disable framework override
        @optimize(auto_override_frameworks=False, configuration_space={"x": [1, 2, 3]})
        def func2(x):
            return x

        assert isinstance(func2, OptimizedFunction)

        # Specific framework targets
        @optimize(
            framework_targets=["openai.OpenAI", "langchain.ChatOpenAI"],
            configuration_space={"x": [1, 2, 3]},
        )
        def func3(x):
            return x

        assert isinstance(func3, OptimizedFunction)

    @patch("traigent.api.decorators.logger")
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""

        @optimize(configuration_space={"x": [1, 2, 3]})
        def test_func(x):
            return x

        # Check debug log for decoration
        mock_logger.debug.assert_any_call(
            "Decorating function test_func with @traigent.optimize"
        )

        # Check info log for creation (message includes experiment_name)
        mock_logger.info.assert_called_with(
            "Created optimizable function: test_func (experiment_name='test_func')"
        )

    def test_decorator_factory_pattern(self):
        """Test that optimize returns a decorator function."""
        decorator = optimize(
            objectives=["accuracy"], configuration_space={"x": [1, 2, 3]}
        )
        assert callable(decorator)

        # Apply the decorator
        def my_func(x):
            return x

        decorated = decorator(my_func)
        assert isinstance(decorated, OptimizedFunction)

    def test_class_method_decoration(self):
        """Test decoration of class methods."""

        class MyClass:
            def __init__(self):
                self.value: int = 2

            @optimize(configuration_space={"x": [2, 3, 4]})
            def method(self, x: int) -> int:
                return x * self.value

        # Static and classmethod don't work well with the decorator
        # because they need to be applied after @optimize

        @optimize(configuration_space={"x": [3, 4, 5]})
        def static_func(x: int) -> int:
            return x * 3

        # Create instance and test
        obj = MyClass()

        # The method is bound, so we test it exists
        assert hasattr(obj, "method")
        assert callable(obj.method)

        # Test the standalone function
        assert isinstance(static_func, OptimizedFunction)
        assert static_func(5) == 15


class TestConstraintNormalization:
    """Tests for constraint normalization in the decorator."""

    def test_decorator_with_boolexpr_constraints(self):
        """Test decorator with BoolExpr constraint objects (not just callables)."""
        from traigent.api.parameter_ranges import Choices, Range

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        # BoolExpr objects should be normalized to callables
        @optimize(
            configuration_space={"temperature": temp, "model": model},
            constraints=[
                temp.lte(0.7),  # BoolExpr
                model.equals("gpt-4"),  # BoolExpr
            ],
        )
        def constrained_func(temperature: float, model: str) -> str:
            return f"{model} at {temperature}"

        assert isinstance(constrained_func, OptimizedFunction)

    def test_decorator_with_constraint_objects(self):
        """Test decorator with Constraint objects using implies()."""
        from traigent.api.constraints import implies
        from traigent.api.parameter_ranges import Choices, Range

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        # Constraint objects should be normalized
        @optimize(
            configuration_space={"temperature": temp, "model": model},
            constraints=[
                implies(model.equals("gpt-4"), temp.lte(0.7)),
            ],
        )
        def constrained_func(temperature: float, model: str) -> str:
            return f"{model} at {temperature}"

        assert isinstance(constrained_func, OptimizedFunction)

    def test_decorator_with_mixed_constraints(self):
        """Test decorator with mixed constraint types."""
        from traigent.api.constraints import implies
        from traigent.api.parameter_ranges import Choices, Range

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        # Mix of Constraint, BoolExpr, and callable
        @optimize(
            configuration_space={"temperature": temp, "model": model},
            constraints=[
                implies(model.equals("gpt-4"), temp.lte(0.7)),  # Constraint
                temp.gte(0.1),  # BoolExpr
                lambda cfg: cfg.get("temperature", 0) < 1.5,  # callable
            ],
        )
        def constrained_func(temperature: float, model: str) -> str:
            return f"{model} at {temperature}"

        assert isinstance(constrained_func, OptimizedFunction)

    def test_decorator_with_configspace_constraints(self):
        """Test decorator with ConfigSpace that has constraints."""
        from traigent.api.config_space import ConfigSpace
        from traigent.api.constraints import implies
        from traigent.api.parameter_ranges import Choices, Range

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        config_space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        @optimize(configuration_space=config_space)
        def constrained_func(temperature: float, model: str) -> str:
            return f"{model} at {temperature}"

        assert isinstance(constrained_func, OptimizedFunction)

    def test_decorator_validates_constraint_scope_at_definition_time(self):
        """Out-of-scope TVAR references should raise during decoration."""
        from traigent.api.constraints import ConstraintScopeError, when
        from traigent.api.parameter_ranges import Choices, Range

        model = Choices(["a", "b"], name="model")
        budget = Range(1.0, 100.0, name="budget")

        with pytest.raises(ConstraintScopeError, match="budget"):

            @optimize(
                configuration_space={"model": model},
                constraints=[when(budget.lte(10)).then(model.equals("a"))],
            )
            def constrained_func(model: str) -> str:
                return model

    def test_scope_validation_accepts_inline_unnamed_parameter_ranges(self):
        """Inline unnamed ranges should still resolve correctly by identity."""
        from traigent.api.constraints import when
        from traigent.api.parameter_ranges import Choices, Range

        model = Choices(["a", "b"])
        temperature = Range(0.0, 2.0)

        @optimize(
            model=model,
            temperature=temperature,
            constraints=[when(model.equals("a")).then(temperature.lte(0.7))],
        )
        def constrained_func(model: str, temperature: float) -> str:
            return f"{model}:{temperature}"

        assert isinstance(constrained_func, OptimizedFunction)

    def test_decorator_rejects_conflicting_constraints(self):
        """Test that decorator raises error for conflicting constraints."""
        from traigent.api.config_space import ConfigSpace
        from traigent.api.constraints import implies
        from traigent.api.parameter_ranges import Choices, Range

        temp = Range(0.0, 2.0, name="temperature")
        model = Choices(["gpt-4", "gpt-3.5"], name="model")

        config_space = ConfigSpace(
            tvars={"temperature": temp, "model": model},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        # Both ConfigSpace constraints and explicit constraints should raise error
        with pytest.raises(TypeError, match="Cannot provide both"):

            @optimize(
                configuration_space=config_space,
                constraints=[temp.gte(0.1)],  # Explicit constraint
            )
            def constrained_func(temperature: float, model: str) -> str:
                return f"{model} at {temperature}"

    def test_decorator_with_custom_evaluator_class(self):
        """Test decorator with custom evaluator class instance."""

        class CustomEvaluator:
            def __call__(self, func, config, example):
                return {"score": 1.0}

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            custom_evaluator=CustomEvaluator(),
        )
        def func_with_evaluator(x: int) -> int:
            return x * 2

        assert isinstance(func_with_evaluator, OptimizedFunction)

    def test_decorator_validates_custom_evaluator_signature(self):
        """Test that custom evaluator signature is validated."""
        from traigent.utils.exceptions import ValidationError

        # Wrong signature - has metric evaluator params instead of (func, config, example)
        def wrong_evaluator(prediction, expected, input_data):
            return 1.0

        with pytest.raises(
            ValidationError, match="custom_evaluator signature mismatch"
        ):

            @optimize(
                configuration_space={"x": [1, 2, 3]},
                custom_evaluator=wrong_evaluator,
            )
            def func_with_bad_evaluator(x: int) -> int:
                return x * 2


class TestRemovedDecoratorCompatibilityOptions:
    """Tests for removed decorator compatibility options."""

    def test_attribute_injection_mode_removed(self):
        """Test that injection_mode='attribute' raises error (removed in v2.x)."""
        from traigent.api.decorators import InjectionOptions

        with pytest.raises((ValueError, Exception), match="removed"):
            InjectionOptions(injection_mode="attribute")

    def test_injection_modes_still_work_without_runtime_option(self):
        """Supported Python injection modes do not require a runtime field."""
        from traigent.api.decorators import InjectionOptions

        for mode in ["context", "seamless"]:

            @optimize(
                configuration_space={"x": [1, 2, 3]},
                injection=InjectionOptions(injection_mode=mode),
            )
            def py_func(x: int) -> int:
                return x

            assert isinstance(py_func, OptimizedFunction)

        @optimize(
            configuration_space={"x": [1, 2, 3]},
            injection=InjectionOptions(injection_mode="parameter"),
        )
        def py_func_with_config(x: int, config=None) -> int:
            return x

        assert isinstance(py_func_with_config, OptimizedFunction)


class TestExperimentName:
    """Tests for experiment_name parameter and TRAIGENT_EXPERIMENT_NAME env var."""

    def test_experiment_name_default_is_func_name(self):
        """When no experiment_name is passed, experiment_name == func.__name__."""

        @optimize(configuration_space={"x": [1, 2]})
        def my_pipeline(x: int) -> int:
            return x

        assert my_pipeline.experiment_name == "my_pipeline"

    def test_experiment_name_override(self):
        """Explicit experiment_name is used instead of func.__name__."""

        @optimize(
            configuration_space={"x": [1, 2]},
            experiment_name="Amir txt2sql v1 (ACL 0.8, 0.15, 0.05)",
        )
        def my_pipeline(x: int) -> int:
            return x

        assert my_pipeline.experiment_name == "Amir txt2sql v1 (ACL 0.8, 0.15, 0.05)"
        # __name__ still reflects the real function name
        assert my_pipeline.__name__ == "my_pipeline"

    def test_experiment_name_env_var(self, monkeypatch):
        """TRAIGENT_EXPERIMENT_NAME env var is used when no explicit name is passed."""
        monkeypatch.setenv("TRAIGENT_EXPERIMENT_NAME", "env_experiment")

        @optimize(configuration_space={"x": [1, 2]})
        def my_pipeline(x: int) -> int:
            return x

        assert my_pipeline.experiment_name == "env_experiment"

    def test_experiment_name_param_beats_env_var(self, monkeypatch):
        """Explicit experiment_name takes precedence over TRAIGENT_EXPERIMENT_NAME."""
        monkeypatch.setenv("TRAIGENT_EXPERIMENT_NAME", "env_experiment")

        @optimize(
            configuration_space={"x": [1, 2]},
            experiment_name="explicit name",
        )
        def my_pipeline(x: int) -> int:
            return x

        assert my_pipeline.experiment_name == "explicit name"

    def test_experiment_name_env_var_cleared(self, monkeypatch):
        """After env var is removed, falls back to func.__name__."""
        monkeypatch.delenv("TRAIGENT_EXPERIMENT_NAME", raising=False)

        @optimize(configuration_space={"x": [1, 2]})
        def my_pipeline(x: int) -> int:
            return x

        assert my_pipeline.experiment_name == "my_pipeline"

    def test_experiment_name_stored_on_optimized_function(self):
        """_experiment_name is stored on the OptimizedFunction instance."""

        @optimize(
            configuration_space={"x": [1, 2]},
            experiment_name="stored name",
        )
        def my_func(x: int) -> int:
            return x

        assert my_func._experiment_name == "stored name"

    def test_experiment_name_none_stores_none(self):
        """When experiment_name=None (default), _experiment_name is None."""

        @optimize(configuration_space={"x": [1, 2]})
        def my_func(x: int) -> int:
            return x

        assert my_func._experiment_name is None
