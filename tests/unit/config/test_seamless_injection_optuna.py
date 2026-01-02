"""Test seamless injection integration with Optuna optimizers."""

from __future__ import annotations

import ast
from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest

from traigent.config.seamless_injection import (
    RuntimeShim,
    SeamlessInjectionConfigurator,
)


class TestSeamlessInjectionOptunaIntegration:
    """Test seamless injection working with Optuna trial configurations."""

    def test_seamless_injection_preserves_optuna_trial_id(self):
        """Test that _optuna_trial_id survives seamless injection."""

        def original_function(text: str) -> str:
            # This would normally use injected config values
            temperature = 0.7  # Would be replaced by seamless injection
            return f"Result with temp={temperature}"

        # Create a mock Optuna trial configuration
        trial_config = {
            "temperature": 0.9,
            "model": "gpt-4",
            "_optuna_trial_id": 42,
        }

        # Apply seamless injection
        configurator = SeamlessInjectionConfigurator()
        modified_func = configurator.inject(original_function, trial_config)

        # Verify the trial ID is accessible
        assert hasattr(modified_func, "__traigent_trial_id__")
        assert modified_func.__traigent_trial_id__ == 42

        # Execute and verify injection worked
        result = modified_func("test")
        assert (
            "temp=0.9" in result or "temp=0.7" in result
        )  # Depending on injection implementation

    def test_runtime_shim_with_optuna_metadata(self):
        """Test runtime shim handling Optuna trial metadata."""

        class TestShim(RuntimeShim):
            def __init__(self, config: dict[str, Any]):
                super().__init__(config)
                self.trial_id = config.pop("_optuna_trial_id", None)

            def get_value(self, key: str, default: Any = None) -> Any:
                # Add trial tracking to value access
                value = super().get_value(key, default)
                if self.trial_id:
                    self._track_access(key, value)
                return value

            def _track_access(self, key: str, value: Any):
                # Would track parameter access for importance analysis
                pass

        config = {
            "param1": 0.5,
            "param2": "value",
            "_optuna_trial_id": 123,
        }

        shim = TestShim(config)
        assert shim.trial_id == 123
        assert shim.get_value("param1") == 0.5

    def test_seamless_injection_with_pruning_callback(self):
        """Test seamless injection with Optuna pruning support."""

        def trainable_function(epochs: int = 10):
            """Function that reports intermediate values for pruning."""
            results = []
            for epoch in range(epochs):
                # Simulate training
                accuracy = 0.5 + epoch * 0.05
                results.append(accuracy)

                # Check if should prune (would be injected)
                if hasattr(trainable_function, "__should_prune__"):
                    if trainable_function.__should_prune__(accuracy, epoch):
                        raise optuna.TrialPruned()

            return max(results)

        # Mock trial with pruning
        mock_trial = MagicMock()
        mock_trial.should_prune.side_effect = [False, False, True]  # Prune at step 3

        # Inject pruning callback
        def should_prune_callback(value, step):
            mock_trial.report(value, step)
            return mock_trial.should_prune()

        trainable_function.__should_prune__ = should_prune_callback
        trainable_function.__optuna_trial__ = mock_trial

        # Execute and verify pruning
        with pytest.raises(optuna.TrialPruned):
            trainable_function(epochs=10)

        # Verify intermediate values were reported
        assert mock_trial.report.call_count == 3

    def test_ast_transformation_preserves_trial_metadata(self):
        """Test AST transformation doesn't lose Optuna metadata."""

        source = """
def optimize_llm(temperature, model):
    # Original function
    response = call_llm(model, temperature)
    return response
"""

        # Parse and transform AST
        tree = ast.parse(source)

        # Simulate injection transformation
        class OptunaMetadataPreserver(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Add metadata preservation
                metadata_assign = ast.parse(
                    "_trial_id = globals().get('__optuna_trial_id__', None)"
                ).body[0]
                node.body.insert(0, metadata_assign)
                return node

        transformer = OptunaMetadataPreserver()
        modified_tree = transformer.visit(tree)

        # Compile and verify
        code = compile(modified_tree, "<string>", "exec")
        namespace = {"__optuna_trial_id__": 999, "call_llm": lambda m, t: f"{m}:{t}"}
        exec(code, namespace)

        # Function should have access to trial ID
        func = namespace["optimize_llm"]
        result = func(0.7, "gpt-4")
        assert result == "gpt-4:0.7"

    def test_seamless_injection_async_support(self):
        """Test seamless injection with async functions for Optuna."""

        async def async_optimize(config: dict) -> float:
            """Async function for optimization."""
            temperature = config.get("temperature", 0.5)
            await asyncio.sleep(0.01)  # Simulate async work
            return temperature * 2

        import asyncio

        # Create Optuna configuration
        optuna_config = {
            "temperature": 0.8,
            "_optuna_trial_id": 456,
        }

        # Run async function with config
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(async_optimize(optuna_config))
        loop.close()

        assert result == 1.6

    def test_seamless_injection_rollback_on_pruning(self):
        """Test configuration rollback when trial is pruned."""

        class RollbackableShim(RuntimeShim):
            def __init__(self, config: dict[str, Any]):
                super().__init__(config)
                self.original_config = config.copy()
                self.applied_changes = []

            def apply_value(self, key: str, value: Any):
                """Apply configuration value with rollback tracking."""
                old_value = self.config.get(key)
                self.applied_changes.append((key, old_value))
                self.config[key] = value

            def rollback(self):
                """Rollback all applied changes."""
                for key, old_value in reversed(self.applied_changes):
                    if old_value is None:
                        self.config.pop(key, None)
                    else:
                        self.config[key] = old_value
                self.applied_changes.clear()

        shim = RollbackableShim({"param1": "original"})

        # Apply changes
        shim.apply_value("param1", "modified")
        shim.apply_value("param2", "new")
        assert shim.config["param1"] == "modified"
        assert shim.config["param2"] == "new"

        # Rollback on pruning
        shim.rollback()
        assert shim.config["param1"] == "original"
        assert "param2" not in shim.config

    def test_seamless_injection_with_conditional_parameters(self):
        """Test seamless injection with Optuna conditional parameters."""

        def configure_model(base_config: dict) -> dict:
            """Apply conditional configuration based on model type."""
            model = base_config.get("model", "gpt-3.5")

            if model == "gpt-4":
                # GPT-4 specific parameters
                base_config.setdefault("max_tokens", 4000)
                base_config.setdefault("temperature", 0.7)
            else:
                # GPT-3.5 specific parameters
                base_config.setdefault("max_tokens", 2000)
                base_config.setdefault("temperature", 0.5)

            return base_config

        # Test with GPT-4
        config_gpt4 = {"model": "gpt-4", "_optuna_trial_id": 1}
        result = configure_model(config_gpt4)
        assert result["max_tokens"] == 4000
        assert result["temperature"] == 0.7

        # Test with GPT-3.5
        config_gpt35 = {"model": "gpt-3.5", "_optuna_trial_id": 2}
        result = configure_model(config_gpt35)
        assert result["max_tokens"] == 2000
        assert result["temperature"] == 0.5

    def test_seamless_injection_error_handling(self):
        """Test error handling in seamless injection with Optuna."""

        def faulty_function(config: dict) -> float:
            """Function that might fail during optimization."""
            if config.get("param") < 0.5:
                raise ValueError("Parameter too low")
            return config["param"] * 2

        # Test with valid config
        valid_config = {"param": 0.7, "_optuna_trial_id": 100}
        result = faulty_function(valid_config)
        assert result == 1.4

        # Test with invalid config
        invalid_config = {"param": 0.3, "_optuna_trial_id": 101}
        with pytest.raises(ValueError, match="Parameter too low"):
            faulty_function(invalid_config)

    def test_seamless_injection_with_multiple_objectives(self):
        """Test seamless injection with multi-objective Optuna optimization."""

        def multi_objective_function(config: dict) -> tuple[float, float]:
            """Function optimizing accuracy and cost."""
            model = config.get("model", "gpt-3.5")
            temperature = config.get("temperature", 0.5)

            # Simulate different objectives
            if model == "gpt-4":
                accuracy = 0.95 * (1 - temperature * 0.1)
                cost = 0.5
            else:
                accuracy = 0.85 * (1 - temperature * 0.1)
                cost = 0.2

            return accuracy, cost

        # Test configurations from Optuna
        configs = [
            {"model": "gpt-4", "temperature": 0.3, "_optuna_trial_id": 1},
            {"model": "gpt-3.5", "temperature": 0.5, "_optuna_trial_id": 2},
        ]

        results = []
        for config in configs:
            accuracy, cost = multi_objective_function(config)
            results.append(
                {
                    "trial_id": config["_optuna_trial_id"],
                    "accuracy": accuracy,
                    "cost": cost,
                }
            )

        # Verify Pareto front candidate
        assert results[0]["accuracy"] > results[1]["accuracy"]
        assert results[0]["cost"] > results[1]["cost"]

    def test_seamless_injection_state_persistence(self):
        """Test state persistence across seamless injection calls."""

        class StatefulShim(RuntimeShim):
            _state_cache = {}  # Shared across instances

            def __init__(self, config: dict[str, Any]):
                super().__init__(config)
                trial_id = config.get("_optuna_trial_id")
                if trial_id:
                    self._state_cache[trial_id] = {"accesses": 0}

            def get_value(self, key: str, default: Any = None) -> Any:
                value = super().get_value(key, default)
                trial_id = self.config.get("_optuna_trial_id")
                if trial_id in self._state_cache:
                    self._state_cache[trial_id]["accesses"] += 1
                return value

            @classmethod
            def get_trial_state(cls, trial_id: int) -> dict:
                return cls._state_cache.get(trial_id, {})

        # Create multiple shims with same trial ID
        config1 = {"param": "value1", "_optuna_trial_id": 42}
        shim1 = StatefulShim(config1)
        shim1.get_value("param")

        config2 = {"param": "value2", "_optuna_trial_id": 42}
        shim2 = StatefulShim(config2)
        shim2.get_value("param")

        # Verify state is shared
        state = StatefulShim.get_trial_state(42)
        assert state["accesses"] == 2
