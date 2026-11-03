"""Golden characterization for ``build_outcome_matrix`` assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from traigent.utils import outcome_matrix as om


def _build(result: Any) -> tuple[dict[str, Any], Any]:
    with (
        patch.object(om, "_extract_cell", lambda ex: om._get(ex, "v")),
        patch.object(om, "logger") as log,
    ):
        out = om.build_outcome_matrix(result)
    out.pop("created_at")
    return out, log


def _trial(tid: Any, examples: Any, config: Any = None, metadata: Any = ...) -> Any:
    if metadata is ...:
        metadata = {"example_results": examples}
    return SimpleNamespace(trial_id=tid, config=config, metadata=metadata)


def test_empty_and_missing_trials() -> None:
    for result in (
        SimpleNamespace(),
        SimpleNamespace(trials=None),
        SimpleNamespace(trials=[]),
    ):
        out, log = _build(result)
        assert out["trial_count"] == 0 and out["example_count"] == 0
        assert out["trials"] == [] and out["examples"] == []
        assert out["optimization_id"] is None and out["objectives"] == []
        log.warning.assert_not_called()


def test_rows_in_first_seen_order_and_cells_keyed_by_trial() -> None:
    t0 = _trial("a", [{"example_id": "x", "v": 1}, {"example_id": 7, "v": 2}])
    t1 = _trial("b", [{"example_id": "y", "v": 3}, {"example_id": "x", "v": 4}])
    out, log = _build(
        SimpleNamespace(
            trials=[t0, t1], optimization_id="o", algorithm="al", objectives=("acc",)
        )
    )
    assert [e["example_id"] for e in out["examples"]] == ["x", "7", "y"]
    assert out["examples"][0]["cells"] == {"a": 1, "b": 4}
    assert out["examples"][1]["cells"] == {"a": 2}
    assert out["examples"][2]["cells"] == {"b": 3}
    assert (out["trial_count"], out["example_count"]) == (2, 3)
    assert (out["optimization_id"], out["algorithm"], out["objectives"]) == (
        "o",
        "al",
        ["acc"],
    )
    log.warning.assert_not_called()


def test_columns_fallback_ids_configs_and_hashes() -> None:
    trials = [_trial(None, None, {"k": 1}), _trial("", None, None), _trial("z", None)]
    out, _ = _build(SimpleNamespace(trials=trials))
    cols = out["trials"]
    assert [c["trial_id"] for c in cols] == ["trial_0", "trial_1", "z"]
    assert [c["index"] for c in cols] == [0, 1, 2]
    assert cols[0]["config"] == {"k": 1} and cols[1]["config"] == {}
    assert cols[1]["config_hash"] == cols[2]["config_hash"] == om._config_hash({})
    assert out["examples"] == []


def test_non_list_example_results_and_non_dict_metadata_contribute_no_cells() -> None:
    trials = [
        _trial("a", "not-a-list"),
        _trial("b", None),
        _trial("c", ({"example_id": "x", "v": 1},)),  # tuple is not a list
        _trial("d", None, metadata=["example_results"]),
        _trial("e", None, metadata=None),
        _trial("f", [{"example_id": "x", "v": 9}]),
    ]
    out, _ = _build(SimpleNamespace(trials=trials))
    assert out["trial_count"] == 6
    assert out["examples"] == [{"example_id": "x", "cells": {"f": 9}}]


def test_examples_without_id_skipped_objects_supported_and_falsey_ids_kept() -> None:
    rows = [
        {"v": 1},
        {"example_id": None, "v": 2},
        SimpleNamespace(example_id="o", v=3),
        {"example_id": 0, "v": 4},
        {"example_id": "", "v": 5},
        SimpleNamespace(v=6),
    ]
    out, _ = _build(SimpleNamespace(trials=[_trial("a", rows)]))
    assert [(e["example_id"], e["cells"]) for e in out["examples"]] == [
        ("o", {"a": 3}),
        ("0", {"a": 4}),
        ("", {"a": 5}),
    ]


def test_duplicate_ids_in_one_trial_last_wins_and_merge_warning_is_content_free() -> (
    None
):
    t0 = _trial("secret-a", [{"example_id": "x", "v": 1}])
    t1 = _trial(
        "secret-b",
        [
            {"example_id": "x", "v": 2},
            {"example_id": "x", "v": 3},
            {"example_id": "x", "v": 4},
            {"example_id": "y", "v": 5},
            {"example_id": "y", "v": 6},
        ],
    )
    out, log = _build(SimpleNamespace(trials=[t0, t1]))
    assert out["examples"][0]["cells"] == {"secret-a": 1, "secret-b": 4}
    assert out["examples"][1]["cells"] == {"secret-b": 6}
    log.warning.assert_called_once()
    args = log.warning.call_args.args
    assert args[1] == "column 1: 3 merged"
    assert "secret" not in " ".join(str(a) for a in args)


def test_merge_counted_per_column_in_insertion_order() -> None:
    t0 = _trial("a", [{"example_id": "x", "v": 1}, {"example_id": "x", "v": 2}])
    t1 = _trial("b", [{"example_id": "x", "v": 3}])
    t2 = _trial("c", [{"example_id": "x", "v": 4}, {"example_id": "x", "v": 5}])
    _, log = _build(SimpleNamespace(trials=[t0, t1, t2]))
    assert log.warning.call_args.args[1] == "column 0: 1 merged, column 2: 1 merged"


def test_same_trial_id_in_two_columns_shares_cells_and_counts_as_merge() -> None:
    t0 = _trial("same", [{"example_id": "x", "v": 1}])
    t1 = _trial("same", [{"example_id": "x", "v": 2}])
    out, log = _build(SimpleNamespace(trials=[t0, t1]))
    assert out["examples"][0]["cells"] == {"same": 2}
    assert log.warning.call_args.args[1] == "column 1: 1 merged"
