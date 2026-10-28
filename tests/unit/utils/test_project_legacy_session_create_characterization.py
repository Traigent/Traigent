"""Characterization of the legacy session-create projection, pinned before extraction.

Full-result assertions for the config / dataset / metadata sections, the order in which
they are processed, the exact collaborator call sequence and the exception boundaries.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as egress
from traigent.utils.optimization_only_egress import (
    MetadataContractViolation,
    project_legacy_session_create,
)

# sha256("My Dataset")[:16] and sha256("opt: a b")[:16], computed independently.
_TOKEN_MY_DATASET = "dn_2b326e1bcc05f7d8"
_TOKEN_FUNCTION = "fn_c0186a9ab8ffac46"


def _project(data: Any) -> Any:
    return project_legacy_session_create(data)


class _Trace:
    """Records _run and _Ctx calls and delegates to the real implementations."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runs: list[tuple[Any, Any, Any]] = []
        self.ctxs: list[tuple[Any, ...]] = []
        self.ctx_objects: list[Any] = []
        real_run = egress._run
        real_ctx: Any = egress._Ctx
        trace = self

        def recording_ctx(*args: Any) -> Any:
            trace.ctxs.append(args)
            ctx = real_ctx(*args)
            trace.ctx_objects.append(ctx)
            return ctx

        def recording_run(spec: Any, value: Any, ctx: Any, **kwargs: Any) -> Any:
            trace.runs.append((spec, value, ctx))
            return real_run(spec, value, ctx, **kwargs)

        monkeypatch.setattr(egress, "_run", recording_run)
        monkeypatch.setattr(egress, "_Ctx", recording_ctx)


# --------------------------------------------------------------------------- inputs


@pytest.mark.parametrize("bad", [None, [], (), "text", 5, set()])
def test_non_dict_input_is_refused(bad: Any) -> None:
    with pytest.raises(MetadataContractViolation):
        _project(bad)


def test_empty_input_projects_to_empty_dict() -> None:
    assert _project({}) == {}


def test_dict_subclass_input_is_accepted_and_copied() -> None:
    data = OrderedDict([("problem_statement", "x")])
    out = _project(data)
    assert out == {"problem_statement": "optimization"}
    assert type(out) is dict
    assert data == OrderedDict([("problem_statement", "x")])


def test_input_is_not_mutated() -> None:
    data = {
        "metadata_contract_version": 1,
        "problem_statement": "x",
        "optimization_config": {"objectives": ["acc"], "max_trials": 3},
        "dataset": {"name": "My Dataset", "description": "d", "size": 3},
        "metadata": {"optimization_id": "o-1"},
    }
    before = {
        "metadata_contract_version": 1,
        "problem_statement": "x",
        "optimization_config": {"objectives": ["acc"], "max_trials": 3},
        "dataset": {"name": "My Dataset", "description": "d", "size": 3},
        "metadata": {"optimization_id": "o-1"},
    }
    _project(data)
    assert data == before


# ------------------------------------------------------------- contract version


@pytest.mark.parametrize("bad", [0, 2, True, False, "1", 1.5, None, [1]])
def test_wrong_contract_version_is_refused(bad: Any) -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"metadata_contract_version": bad})


def test_correct_contract_version_is_removed() -> None:
    assert _project({"metadata_contract_version": 1, "problem_statement": "p"}) == {
        "problem_statement": "optimization"
    }


def test_version_float_equal_to_one_is_accepted() -> None:
    assert _project({"metadata_contract_version": 1.0}) == {}


def test_version_checked_before_any_section_is_read() -> None:
    reads: list[str] = []

    class Config(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            reads.append(key)
            return super().get(key, default)

    with pytest.raises(MetadataContractViolation):
        _project({"metadata_contract_version": 2, "optimization_config": Config()})
    assert reads == []


# -------------------------------------------------------------- problem statement


@pytest.mark.parametrize("value", ["free customer text", "", None, 5, ["a"]])
def test_problem_statement_is_replaced_whenever_present(value: Any) -> None:
    assert _project({"problem_statement": value}) == {
        "problem_statement": "optimization"
    }


def test_problem_statement_is_not_added_when_absent() -> None:
    assert "problem_statement" not in _project({"dataset": {"size": 1}})


# --------------------------------------------------------------- optimization config


def test_config_is_projected_and_unknown_keys_dropped() -> None:
    out = _project(
        {
            "optimization_config": {
                "max_trials": 5,
                "algorithm": "grid",
                "objectives": ["accuracy", {"metric": "my_metric", "direction": "max"}],
                "unknown_key": "free text",
            }
        }
    )
    assert out == {
        "optimization_config": {
            "algorithm": "grid",
            "max_trials": 5,
            "objectives": ["accuracy", {"metric": "my_metric"}],
        }
    }


def test_config_optional_malformed_field_is_dropped() -> None:
    out = _project({"optimization_config": {"objectives": ["acc"], "timeout": -1}})
    assert out == {"optimization_config": {"objectives": ["acc"]}}


def test_config_dict_objectives_form() -> None:
    out = _project(
        {
            "optimization_config": {
                "objectives": {"accuracy": "maximize", "cost": "minimize"}
            }
        }
    )
    assert out == {
        "optimization_config": {
            "objectives": {"accuracy": "maximize", "cost": "minimize"}
        }
    }


@pytest.mark.parametrize(
    "objectives",
    [
        ["bad name!"],
        {"bad name!": "maximize"},
        [{"metric": "has space"}],
        [{"name": "x" * 65}],
        ["ok", "bad/slash"],
    ],
)
def test_declared_metric_name_that_breaks_the_key_pattern_is_refused(
    objectives: Any,
) -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"optimization_config": {"objectives": objectives}})


@pytest.mark.parametrize("name", ["a", "a" * 64, "Acc_1.x%-y"])
def test_declared_metric_name_boundaries_are_accepted(name: str) -> None:
    out = _project({"optimization_config": {"objectives": [name]}})
    assert out == {"optimization_config": {"objectives": [name]}}


def test_non_string_declared_names_are_not_validated_or_declared() -> None:
    out = _project({"optimization_config": {"objectives": [5, None]}})
    assert out == {"optimization_config": {"objectives": []}}


def test_config_without_objectives_is_projected() -> None:
    assert _project({"optimization_config": {"max_trials": 2}}) == {
        "optimization_config": {"max_trials": 2}
    }


def test_config_with_wrong_contract_version_is_refused() -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"optimization_config": {"metadata_contract_version": 7}})


def test_config_over_the_size_cap_is_refused() -> None:
    objectives = [
        {
            "metric": f"m{i}".ljust(64, "x"),
            "name": f"n{i}".ljust(64, "y"),
            "unit": "u" * 32,
        }
        for i in range(256)
    ]
    with pytest.raises(MetadataContractViolation):
        _project({"optimization_config": {"objectives": objectives}})


def test_config_just_under_the_size_cap_is_accepted() -> None:
    objectives = [
        {"metric": f"m{i}".ljust(64, "x"), "name": f"n{i}".ljust(64, "y")}
        for i in range(100)
    ]
    out = _project({"optimization_config": {"objectives": objectives}})
    assert out == {"optimization_config": {"objectives": objectives}}


@pytest.mark.parametrize("value", [[1, 2], "text", 5, None])
def test_non_dict_config_is_passed_through_untouched(value: Any) -> None:
    assert _project({"optimization_config": value}) == {"optimization_config": value}


def test_config_get_is_read_exactly_twice_for_objectives_with_second_result_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class Config(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            calls.append((key, default))
            return ["first_name"] if len(calls) == 1 else ["second_name"]

    trace = _Trace(monkeypatch)
    _project({"optimization_config": Config(max_trials=1)})
    assert calls == [("objectives", None), ("objectives", None)]
    assert trace.ctxs == [(), (["second_name"],)]
    assert trace.ctx_objects[1].allowed_metrics == (
        trace.ctx_objects[0].allowed_metrics | {"second_name"}
    )


def test_first_objectives_read_is_the_validated_one() -> None:
    calls: list[str] = []

    class Config(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            calls.append(key)
            return ["bad name!"] if len(calls) == 1 else ["fine"]

    with pytest.raises(MetadataContractViolation):
        _project({"optimization_config": Config()})
    assert calls == ["objectives"]


def test_raising_config_get_propagates_and_later_sections_are_untouched() -> None:
    class Config(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("get failed")

    class Dataset(dict[str, Any]):
        touched = False

        def keys(self) -> Any:
            Dataset.touched = True
            return super().keys()

    with pytest.raises(RuntimeError, match="get failed"):
        _project({"optimization_config": Config(), "dataset": Dataset(size=1)})
    assert Dataset.touched is False


def test_raise_once_config_get_second_read_failure_propagates() -> None:
    calls: list[int] = []

    class Config(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("second read failed")
            return ["acc"]

    with pytest.raises(RuntimeError, match="second read failed"):
        _project({"optimization_config": Config()})
    assert len(calls) == 2


def test_collaborator_sequence_with_all_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace(monkeypatch)
    config = {"objectives": ["acc"]}
    metadata = {"optimization_id": "o-1"}
    _project(
        {
            "optimization_config": config,
            "dataset": {"size": 1},
            "metadata": metadata,
        }
    )
    assert [(run[0], run[1]) for run in trace.runs] == [
        (egress._LEGACY_OPTIMIZATION_CONFIG, config),
        (egress._SESSION_METADATA, metadata),
    ]
    assert trace.ctxs == [(), (["acc"],)]
    # both sections run with the context created after the declared names were read
    assert trace.runs[0][2] is trace.ctx_objects[1]
    assert trace.runs[1][2] is trace.ctx_objects[1]


def test_metadata_uses_default_context_when_config_is_not_a_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace(monkeypatch)
    _project({"optimization_config": "text", "metadata": {"optimization_id": "o-1"}})
    assert trace.ctxs == [()]
    assert len(trace.runs) == 1
    assert trace.runs[0][2] is trace.ctx_objects[0]


# ------------------------------------------------------------------------ dataset


def test_dataset_name_token_description_dropped_and_other_keys_kept() -> None:
    out = _project(
        {
            "dataset": {
                "name": "My Dataset",
                "size": 3,
                "description": "free text",
                "extra": {"a": 1},
            }
        }
    )
    assert out == {"dataset": {"name": _TOKEN_MY_DATASET, "size": 3, "extra": {"a": 1}}}


def test_dataset_identifier_shaped_name_is_kept_verbatim() -> None:
    assert _project({"dataset": {"name": "set_1.v2"}}) == {
        "dataset": {"name": "set_1.v2"}
    }


@pytest.mark.parametrize("name", ["", "   ", None, 5, ["x"], b"bytes", 1.5, True])
def test_dataset_unusable_name_is_dropped(name: Any) -> None:
    assert _project({"dataset": {"name": name, "size": 2}}) == {"dataset": {"size": 2}}


def test_dataset_without_name_gets_none() -> None:
    assert _project({"dataset": {"description": "d", "size": 2}}) == {
        "dataset": {"size": 2}
    }


def test_dataset_empty_dict_stays_empty() -> None:
    assert _project({"dataset": {}}) == {"dataset": {}}


@pytest.mark.parametrize("value", ["free text", [1], 5, None])
def test_non_dict_dataset_is_passed_through_untouched(value: Any) -> None:
    assert _project({"dataset": value}) == {"dataset": value}


def test_dataset_is_copied_so_input_keeps_its_name_and_description() -> None:
    dataset = {"name": "My Dataset", "description": "d"}
    out = _project({"dataset": dataset})
    assert dataset == {"name": "My Dataset", "description": "d"}
    assert out["dataset"] is not dataset


def test_dataset_dict_subclass_is_converted_before_membership_checks() -> None:
    class Hostile(dict[str, Any]):
        def __contains__(self, key: object) -> bool:
            return False

    out = _project({"dataset": Hostile(name="My Dataset", description="d", size=1)})
    assert out == {"dataset": {"name": _TOKEN_MY_DATASET, "size": 1}}
    assert type(out["dataset"]) is dict


def test_dataset_name_str_subclass_with_false_bool_is_still_tokenized() -> None:
    class FalsyName(str):
        def __bool__(self) -> bool:
            return False

    out = _project({"dataset": {"name": FalsyName("My Dataset")}})
    assert out == {"dataset": {"name": _TOKEN_MY_DATASET}}


def test_dataset_name_with_raising_strip_propagates() -> None:
    class Boom(str):
        def strip(self, *args: Any) -> str:
            raise RuntimeError("strip failed")

    with pytest.raises(RuntimeError, match="strip failed"):
        _project({"dataset": {"name": Boom("My Dataset")}})


def test_dataset_name_strip_returning_empty_drops_the_name() -> None:
    class EmptyStrip(str):
        def strip(self, *args: Any) -> str:
            return ""

    assert _project({"dataset": {"name": EmptyStrip("My Dataset")}}) == {"dataset": {}}


def test_dataset_name_is_resolved_through_the_module_level_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefixes: list[str] = []
    real = egress.s_tok

    def spy(prefix: str) -> Any:
        prefixes.append(prefix)
        return real(prefix)

    monkeypatch.setattr(egress, "s_tok", spy)
    _project({"dataset": {"name": "My Dataset"}})
    assert prefixes == ["dn"]
    prefixes.clear()
    _project({"dataset": {"size": 1}})
    assert prefixes == []


def test_dataset_tokenizer_is_not_resolved_when_name_absent_or_dataset_not_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefixes: list[str] = []

    def spy(prefix: str) -> Any:
        prefixes.append(prefix)
        raise AssertionError("must not be called")

    monkeypatch.setattr(egress, "s_tok", spy)
    assert _project({"dataset": {"size": 1}}) == {"dataset": {"size": 1}}
    assert _project({"dataset": "text"}) == {"dataset": "text"}
    assert prefixes == []


def test_dataset_name_drop_is_the_only_swallowed_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(prefix: str) -> Any:
        def spec(value: Any, ctx: Any, depth: int) -> Any:
            raise ValueError("not a drop")

        return spec

    monkeypatch.setattr(egress, "s_tok", failing)
    with pytest.raises(ValueError, match="not a drop"):
        _project({"dataset": {"name": "My Dataset"}})


def test_dataset_tokenizer_receives_context_and_depth_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, Any, Any]] = []

    def factory(prefix: str) -> Any:
        def spec(value: Any, ctx: Any, depth: int) -> Any:
            seen.append((value, ctx, depth))
            return "tokenized"

        return spec

    trace = _Trace(monkeypatch)
    monkeypatch.setattr(egress, "s_tok", factory)
    out = _project(
        {"optimization_config": {"objectives": ["acc"]}, "dataset": {"name": "n"}}
    )
    assert out["dataset"] == {"name": "tokenized"}
    assert seen == [("n", trace.ctx_objects[1], 0)]


# ----------------------------------------------------------------------- metadata


def test_metadata_is_projected_and_unknown_keys_dropped() -> None:
    out = _project(
        {
            "metadata": {
                "optimization_id": "o-1",
                "function_name": "opt: a b",
                "unknown_key": "free text",
            }
        }
    )
    assert out == {
        "metadata": {"optimization_id": "o-1", "function_name": _TOKEN_FUNCTION}
    }


def test_metadata_none_is_kept_as_none() -> None:
    assert _project({"metadata": None}) == {"metadata": None}


@pytest.mark.parametrize("value", [5, "text", [1], False, 0, ""])
def test_metadata_that_is_not_a_dict_is_refused(value: Any) -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"metadata": value})


def test_empty_dict_metadata_is_projected_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace(monkeypatch)
    assert _project({"metadata": {}}) == {"metadata": {}}
    assert len(trace.runs) == 1


def test_metadata_strict_field_malformed_is_refused() -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"metadata": {"privacy_mode": "yes"}})


def test_metadata_wrong_contract_version_is_refused() -> None:
    with pytest.raises(MetadataContractViolation):
        _project({"metadata": {"metadata_contract_version": 3}})


# ------------------------------------------------------------------------ ordering


def test_config_failure_wins_over_later_section_failures() -> None:
    class Boom(str):
        def strip(self, *args: Any) -> str:
            raise RuntimeError("late failure")

    with pytest.raises(MetadataContractViolation):
        _project(
            {
                "optimization_config": {"objectives": ["bad name!"]},
                "dataset": {"name": Boom("n")},
                "metadata": 5,
            }
        )


def test_dataset_failure_wins_over_metadata_failure() -> None:
    class Boom(str):
        def strip(self, *args: Any) -> str:
            raise RuntimeError("dataset failure")

    with pytest.raises(RuntimeError, match="dataset failure"):
        _project({"dataset": {"name": Boom("n")}, "metadata": 5})


def test_dataset_is_processed_before_metadata_is_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_run = egress._run

    def recording_run(spec: Any, value: Any, ctx: Any, **kwargs: Any) -> Any:
        events.append("run")
        return real_run(spec, value, ctx, **kwargs)

    real_tok = egress.s_tok

    def recording_tok(prefix: str) -> Any:
        events.append("tok")
        return real_tok(prefix)

    monkeypatch.setattr(egress, "_run", recording_run)
    monkeypatch.setattr(egress, "s_tok", recording_tok)
    _project(
        {
            "optimization_config": {"objectives": ["acc"]},
            "dataset": {"name": "My Dataset"},
            "metadata": {"optimization_id": "o-1"},
        }
    )
    assert events == ["run", "tok", "run"]


def test_result_key_order_follows_input_order() -> None:
    out = _project(
        {
            "metadata": {"optimization_id": "o-1"},
            "dataset": {"size": 1},
            "problem_statement": "x",
            "optimization_config": {"max_trials": 1},
        }
    )
    assert list(out) == [
        "metadata",
        "dataset",
        "problem_statement",
        "optimization_config",
    ]
