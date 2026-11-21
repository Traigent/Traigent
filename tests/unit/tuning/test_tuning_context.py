"""Module context derived from an optimized function (no TVL toolchain needed)."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.api.parameter_ranges import Choices
from traigent.tuning.context import (
    DEFAULT_SNAPSHOT_ID,
    _objective,
    _tvar,
    derive_module_context,
    merge_context,
)

from .conftest import tvl_free


@tvl_free
class TestTvar:
    def test_a_list_of_strings_is_an_enum_of_str(self) -> None:
        assert _tvar("model", ["a", "b"]) == {
            "name": "model",
            "type": "enum[str]",
            "domain": ["a", "b"],
        }

    def test_a_list_of_ints_is_an_enum_of_int(self) -> None:
        assert _tvar("k", [1, 2, 3])["type"] == "enum[int]"

    def test_a_mixed_numeric_list_is_an_enum_of_float(self) -> None:
        assert _tvar("t", [0.1, 1])["type"] == "enum[float]"

    def test_booleans_are_not_numbers(self) -> None:
        # a bool list is neither str nor numeric: the ConfigSpace fallback cannot
        # express it either, so the variable is left out for the TVL lint to report
        assert _tvar("flag", [True, False]) is None

    def test_a_mixed_type_list_is_left_out(self) -> None:
        assert _tvar("x", ["a", 1]) is None

    def test_an_int_pair_tuple_is_an_int_range(self) -> None:
        assert _tvar("n", (1, 10)) == {
            "name": "n",
            "type": "int",
            "domain": {"range": [1, 10]},
        }

    def test_a_float_pair_tuple_is_a_float_range(self) -> None:
        assert _tvar("p", (0.0, 1.0))["type"] == "float"

    def test_a_two_element_list_is_an_enum_not_a_range(self) -> None:
        assert _tvar("pair", [1, 2])["type"] == "enum[int]"

    def test_a_parameter_range_object_uses_the_config_space_export(self) -> None:
        tvar = _tvar("model", Choices(["x", "y"]))
        assert tvar is not None
        assert tvar["name"] == "model"
        assert "x_traigent_parameter_range" not in tvar

    def test_an_unsupported_value_is_left_out(self) -> None:
        assert _tvar("weird", object()) is None


@tvl_free
class TestObjective:
    def test_a_cost_like_name_minimizes(self) -> None:
        assert _objective("cost") == {"name": "cost", "direction": "minimize"}
        assert _objective("Avg_Latency")["direction"] == "minimize"

    def test_a_quality_name_maximizes(self) -> None:
        assert _objective("accuracy") == {"name": "accuracy", "direction": "maximize"}

    def test_an_object_with_an_orientation_keeps_it(self) -> None:
        item = SimpleNamespace(name="score", orientation="MINIMIZE")
        assert _objective(item) == {"name": "score", "direction": "minimize"}

    def test_an_object_without_an_orientation_falls_back_to_the_name_hint(self) -> None:
        assert _objective(SimpleNamespace(name="error_rate"))["direction"] == "minimize"
        assert _objective(SimpleNamespace(name="f1"))["direction"] == "maximize"

    def test_a_nameless_object_is_left_out(self) -> None:
        assert _objective(SimpleNamespace(name=3)) is None
        assert _objective(object()) is None


def _fn(**attrs: object) -> SimpleNamespace:
    return SimpleNamespace(**attrs)


@tvl_free
class TestDeriveModuleContext:
    def test_a_function_without_a_configuration_space_has_no_context(self) -> None:
        assert derive_module_context(_fn()) is None
        assert derive_module_context(_fn(configuration_space={})) is None
        assert derive_module_context(_fn(configuration_space=["a"])) is None

    def test_blocks_are_derived_from_the_function(self) -> None:
        ctx = derive_module_context(
            _fn(
                __name__="router",
                configuration_space={"model": ["a", "b"]},
                objectives=["accuracy", "cost"],
                eval_dataset="data/eval.jsonl",
            )
        )
        assert ctx is not None
        assert ctx["tvl"] == {"module": "traigent.router"}
        assert ctx["environment"] == {"snapshot_id": DEFAULT_SNAPSHOT_ID}
        assert ctx["evaluation_set"] == {"dataset": "data/eval.jsonl"}
        assert [t["name"] for t in ctx["tvars"]] == ["model"]
        assert ctx["objectives"] == [
            {"name": "accuracy", "direction": "maximize"},
            {"name": "cost", "direction": "minimize"},
        ]
        assert ctx["promotion_policy"]["min_effect"] == {"accuracy": 0.02, "cost": 0.02}

    def test_defaults_when_the_function_declares_little(self) -> None:
        ctx = derive_module_context(_fn(configuration_space={"m": ["a"]}))
        assert ctx is not None
        assert ctx["tvl"] == {"module": "traigent.agent"}
        assert ctx["evaluation_set"] == {"dataset": "evaluation_set"}
        assert ctx["objectives"] == [{"name": "accuracy", "direction": "maximize"}]

    def test_explicit_arguments_win(self) -> None:
        ctx = derive_module_context(
            _fn(configuration_space={"m": ["a"]}, eval_dataset="from-fn"),
            module_name="custom",
            snapshot_id="2026-11-20T00:00:00Z",
            dataset="explicit",
        )
        assert ctx is not None
        assert ctx["tvl"] == {"module": "traigent.custom"}
        assert ctx["environment"] == {"snapshot_id": "2026-11-20T00:00:00Z"}
        assert ctx["evaluation_set"] == {"dataset": "explicit"}

    def test_a_non_string_dataset_falls_back_to_the_placeholder(self) -> None:
        ctx = derive_module_context(
            _fn(configuration_space={"m": ["a"]}, eval_dataset=[{"input": 1}])
        )
        assert ctx is not None
        assert ctx["evaluation_set"] == {"dataset": "evaluation_set"}

    def test_unsupported_variables_and_objectives_are_dropped(self) -> None:
        ctx = derive_module_context(
            _fn(
                configuration_space={"ok": ["a"], "bad": object()},
                objectives=["accuracy", object()],
            )
        )
        assert ctx is not None
        assert [t["name"] for t in ctx["tvars"]] == ["ok"]
        assert [o["name"] for o in ctx["objectives"]] == ["accuracy"]


@tvl_free
class TestMergeContext:
    def test_nothing_on_either_side_is_none(self) -> None:
        assert merge_context(None, None) is None

    def test_the_override_wins_per_block(self) -> None:
        base = {"a": 1, "env": {"snapshot_id": "x"}}
        merged = merge_context(base, {"env": {"snapshot_id": "y"}, "b": 2})
        assert merged == {"a": 1, "env": {"snapshot_id": "y"}, "b": 2}

    def test_the_inputs_are_not_mutated_or_aliased(self) -> None:
        base = {"env": {"snapshot_id": "x"}}
        override = {"extra": {"k": [1]}}
        merged = merge_context(base, override)
        merged["env"]["snapshot_id"] = "changed"
        merged["extra"]["k"].append(2)
        assert base == {"env": {"snapshot_id": "x"}}
        assert override == {"extra": {"k": [1]}}

    def test_one_side_only_is_copied(self) -> None:
        assert merge_context(None, {"a": 1}) == {"a": 1}
        assert merge_context({"a": 1}, None) == {"a": 1}
