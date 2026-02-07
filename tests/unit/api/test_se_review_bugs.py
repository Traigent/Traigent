"""Tests exposing bugs identified in Senior SE Review (prancy-seeking-taco plan).

Each test asserts the CORRECT expected behavior. They FAIL now because
the bugs exist. Once a bug is fixed, its test(s) will start passing.

Bugs covered:
- Issue 1: Silent constraint failures (missing params pass validation)
- Issue 2: ID-based variable mapping breaks across object boundaries
- Issue 6: Thread-unsafe sample counter (outside lock)
- Issue 10: Constraints not enforced before trial execution
"""

from __future__ import annotations

import copy
import pickle
import textwrap

from traigent.api.config_space import ConfigSpace
from traigent.api.constraints import (
    Condition,
    implies,
)
from traigent.api.parameter_ranges import Choices, Range

# ---------------------------------------------------------------------------
# Issue 1: Silent Constraint Failures
# ---------------------------------------------------------------------------


class TestSilentConstraintFailures:
    """Constraints silently return True when a referenced parameter is missing.

    BUG: Condition.evaluate_config() returns True when the parameter is not
    found in the config dict, effectively disabling the constraint silently.

    Location: traigent/api/constraints.py:286-288
    """

    def test_missing_param_should_not_satisfy_condition(self) -> None:
        """'temperature <= 0.7' must NOT pass when temperature is absent."""
        temp = Range(0.0, 2.0, name="temperature")
        condition = Condition(_tvar=temp, operator="<=", value=0.7)

        var_names = {id(temp): "temperature"}
        config_without_temp = {"model": "gpt-4"}  # temperature missing!

        result = condition.evaluate_config(config_without_temp, var_names)

        # CORRECT behavior: missing param should NOT satisfy the condition
        assert (
            result is False
        ), "Missing parameter 'temperature' should not satisfy 'temp <= 0.7'"

    def test_implication_should_fail_when_then_param_missing(self) -> None:
        """implies(model=='gpt-4', temp<=0.7) must fail when temp is missing."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))
        var_names = {id(model): "model", id(temp): "temperature"}

        config = {"model": "gpt-4"}  # temperature missing!

        result = constraint.evaluate(config, var_names)

        # CORRECT behavior: when model=gpt-4, temperature MUST be present
        # and <= 0.7. Missing temperature should fail the constraint.
        assert (
            result is False
        ), "Implication should fail when 'then' parameter is missing from config"

    def test_stale_var_names_should_raise_or_fail(self) -> None:
        """A stale var_names mapping must not silently bypass constraints."""
        temp = Range(0.0, 2.0, name="temperature")
        condition = Condition(_tvar=temp, operator="<=", value=0.7)

        wrong_id = id(object())
        var_names = {wrong_id: "temperature"}

        config = {"temperature": 1.5}  # Violates <= 0.7!

        result = condition.evaluate_config(config, var_names)

        # CORRECT behavior: temp=1.5 must NOT pass 'temp <= 0.7'
        assert result is False, (
            "temperature=1.5 must not pass 'temperature <= 0.7' "
            "just because var_names mapping is stale"
        )

    def test_configspace_validate_should_reject_missing_constrained_param(self) -> None:
        """ConfigSpace.validate() must reject config missing constrained param."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"model": model, "temperature": temp},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        config = {"model": "gpt-4"}  # temperature missing!
        result = space.validate(config)

        # CORRECT behavior: validation should catch the missing parameter
        assert result.is_valid is False, (
            "ConfigSpace.validate() should reject config missing "
            "a constrained parameter (temperature)"
        )


# ---------------------------------------------------------------------------
# Issue 2: ID-Based Variable Mapping Breaks Across Object Boundaries
# ---------------------------------------------------------------------------


class TestIdBasedMappingFragility:
    """Variable identity via id() breaks across serialization boundaries.

    BUG: ConfigSpace.var_names uses id(tvar) for identity mapping. When
    constraints reference tvar objects from a different scope, id() doesn't match.

    Location: traigent/api/config_space.py:295
    """

    def test_pickle_roundtrip_should_preserve_var_names_identity(self) -> None:
        """After pickle round-trip, var_names ids should still work."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")

        space = ConfigSpace(
            tvars={"model": model, "temperature": temp},
            constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        )

        var_names_before = space.var_names
        restored = pickle.loads(pickle.dumps(space))
        var_names_after = restored.var_names

        # CORRECT behavior: the mapped NAMES should be the same
        # (even if the mechanism changes from id-based to name-based)
        assert set(var_names_before.values()) == set(
            var_names_after.values()
        ), "Parameter names should be preserved after pickle"

        # And constraints should still work after pickle
        config_bad = {"model": "gpt-4", "temperature": 1.5}
        result = restored.constraints[0].evaluate(config_bad, var_names_after)
        assert (
            result is False
        ), "Constraint should still reject temp=1.5 for gpt-4 after pickle"

    def test_cross_space_constraint_should_still_evaluate(self) -> None:
        """Constraints should work when evaluated with equivalent var_names."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")
        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))

        # Different ConfigSpace with identical parameters
        model2 = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp2 = Range(0.0, 2.0, name="temperature")
        space2 = ConfigSpace(
            tvars={"model": model2, "temperature": temp2},
            constraints=[],
        )

        config_bad = {"model": "gpt-4", "temperature": 1.5}

        # CORRECT behavior: constraint should still reject even with
        # a different ConfigSpace's var_names (same parameter names)
        result = constraint.evaluate(config_bad, space2.var_names)
        assert result is False, (
            "Constraint should reject temp=1.5 even when evaluated "
            "with a different ConfigSpace's var_names"
        )

    def test_deepcopy_constraint_should_still_work(self) -> None:
        """Deepcopied constraints should work with original var_names."""
        model = Choices(["gpt-4", "gpt-3.5"], name="model")
        temp = Range(0.0, 2.0, name="temperature")
        constraint = implies(model.equals("gpt-4"), temp.lte(0.7))

        original_var_names = {id(model): "model", id(temp): "temperature"}
        config_bad = {"model": "gpt-4", "temperature": 1.5}

        assert constraint.evaluate(config_bad, original_var_names) is False

        copied_constraint = copy.deepcopy(constraint)
        result = copied_constraint.evaluate(config_bad, original_var_names)

        # CORRECT behavior: deepcopied constraint should still reject
        assert (
            result is False
        ), "Deepcopied constraint should still reject temp=1.5 for gpt-4"


# ---------------------------------------------------------------------------
# Issue 6: Thread-Unsafe Sample Counter
# ---------------------------------------------------------------------------


class TestThreadUnsafeSampleCounter:
    """_consumed_examples is updated outside the _state_lock.

    BUG: _register_examples_attempted() is called outside the _state_lock
    in _handle_trial_result(), making concurrent updates unsafe.

    Location: traigent/core/orchestrator.py:800 (outside lock at line 938)
    """

    def test_consumed_examples_should_be_inside_state_lock(self) -> None:
        """_register_examples_attempted must be called inside _state_lock."""
        import ast
        import inspect

        from traigent.core.orchestrator import OptimizationOrchestrator

        source = inspect.getsource(OptimizationOrchestrator._handle_trial_result)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Check if _register_examples_attempted is inside an AsyncWith block
        found_in_lock = False

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncWith):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "_register_examples_attempted"
                    ):
                        found_in_lock = True

        # CORRECT behavior: _register_examples_attempted SHOULD be inside the lock
        assert found_in_lock is True, (
            "_register_examples_attempted must be called inside _state_lock "
            "to prevent race conditions on _consumed_examples"
        )

    def test_remaining_budget_should_use_lock(self) -> None:
        """_remaining_sample_budget() should acquire lock before reading counter."""
        import inspect

        from traigent.core.orchestrator import OptimizationOrchestrator

        source = inspect.getsource(OptimizationOrchestrator._remaining_sample_budget)

        assert "_consumed_examples" in source, "Method reads _consumed_examples"

        # CORRECT behavior: should use _state_lock when reading shared state
        assert "_state_lock" in source, (
            "_remaining_sample_budget should acquire _state_lock "
            "before reading _consumed_examples"
        )


# ---------------------------------------------------------------------------
# Issue 10: Constraints Not Enforced Before Trial Execution
# ---------------------------------------------------------------------------


class TestNoPreTrialConstraintEnforcement:
    """Optuna can suggest configurations that violate constraints.

    BUG: No ConstraintAwareSampler or pre-trial validation exists.
    Constraints only checked AFTER expensive trial execution.
    """

    def test_constraint_aware_sampler_should_exist(self) -> None:
        """A pre-trial constraint filtering mechanism should exist."""
        import importlib

        constraint_sampler_found = False

        modules_to_check = [
            "traigent.core.orchestrator",
            "traigent.core.orchestrator_helpers",
            "traigent.optimizers",
        ]

        for mod_name in modules_to_check:
            try:
                mod = importlib.import_module(mod_name)
                for attr_name in dir(mod):
                    if (
                        "constraintaware" in attr_name.lower()
                        or "constraint_sampler" in attr_name.lower()
                        or "pre_trial_valid" in attr_name.lower()
                    ):
                        constraint_sampler_found = True
            except ImportError:
                pass

        # CORRECT behavior: a constraint-aware sampler or pre-trial
        # validation mechanism should exist
        assert constraint_sampler_found is True, (
            "A ConstraintAwareSampler or pre-trial validation mechanism "
            "should exist to prevent wasting budget on invalid configs"
        )

    def test_orchestrator_should_validate_config_before_trial(self) -> None:
        """Orchestrator should validate configs before executing trials."""
        import inspect

        from traigent.core.orchestrator import OptimizationOrchestrator

        source = inspect.getsource(OptimizationOrchestrator)

        has_pre_trial_validate = (
            "validate(config" in source
            or "config_space.validate" in source
            or "space.validate" in source
        )

        # CORRECT behavior: orchestrator should validate config before trial
        assert has_pre_trial_validate is True, (
            "Orchestrator should call validate() on configs before "
            "executing trials to prevent wasting budget"
        )
