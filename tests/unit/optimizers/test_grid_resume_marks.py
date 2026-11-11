"""#3271: resuming a grid run must not rebuild the remaining grid per mark."""

from __future__ import annotations

import random
from typing import Any

from traigent.optimizers.grid import GridSearchOptimizer


def _old_mark(points: list[dict], cursor: int, config: dict) -> list[dict]:
    """Verbatim pre-#3271 algorithm, kept as the behavioural reference."""
    keep = list(points[:cursor])
    for point in points[cursor:]:
        if any(k not in config or config[k] != v for k, v in point.items()):
            keep.append(point)
    return keep


def _drain(opt: GridSearchOptimizer) -> list[dict]:
    out = []
    while not opt.should_stop([]):
        out.append(opt.suggest_next_trial([]))
    return out


SPACE = {"a": [0, 1, 2, 3], "b": ["x", "y", "z"], "model": ["m1", "m2"]}


def test_marks_match_old_algorithm_over_random_sequences() -> None:
    rng = random.Random(3271)
    for _ in range(60):
        opt = GridSearchOptimizer(SPACE, ["accuracy"])
        ref = [dict(p) for p in opt._grid_points]
        cursor = 0
        expected: list[dict] = []
        got: list[dict] = []
        for _ in range(rng.randint(1, 30)):
            action = rng.random()
            if action < 0.7:
                cfg: dict[str, Any] = {
                    "a": rng.choice([0, 1, 2, 3, 9]),
                    "b": rng.choice(["x", "y", "z", "q"]),
                    "model": rng.choice(["m1", "m2"]),
                }
                if rng.random() < 0.2:
                    cfg["extra"] = 1
                if rng.random() < 0.1:
                    del cfg[rng.choice(list(cfg))]
                opt.mark_config_evaluated(cfg)
                ref = _old_mark(ref, cursor, cfg)
            elif cursor < len(ref):
                got.append(opt.suggest_next_trial([]))
                expected.append(ref[cursor])
                cursor += 1
            assert opt.total_combinations == len(ref)
            assert opt.should_stop([]) == (cursor >= len(ref))
        got.extend(_drain(opt))
        expected.extend(ref[cursor:])
        assert got == expected


def test_equal_but_differently_typed_values_match_like_old() -> None:
    opt = GridSearchOptimizer({"a": [1, 2], "b": [True, False]}, ["accuracy"])
    ref = [dict(p) for p in opt._grid_points]
    cfg = {"a": 1.0, "b": 1}
    opt.mark_config_evaluated(cfg)
    assert opt.get_remaining_combinations() == _old_mark(ref, 0, cfg)


def test_unhashable_and_nan_values_match_old() -> None:
    nan = float("nan")
    space = {"a": [[1], [2]], "b": [0.5, 1.0]}
    opt = GridSearchOptimizer(space, ["accuracy"])
    ref = [dict(p) for p in opt._grid_points]
    for cfg in ({"a": [1], "b": 1.0}, {"a": [2], "b": nan}):
        opt.mark_config_evaluated(cfg)
        ref = _old_mark(ref, 0, cfg)
    assert opt.get_remaining_combinations() == ref


def test_reset_applies_marks_at_the_old_cursor() -> None:
    opt = GridSearchOptimizer({"a": [0, 1, 2]}, ["accuracy"])
    opt.suggest_next_trial([])
    opt.mark_config_evaluated({"a": 0})  # already suggested: stays in the grid
    opt.mark_config_evaluated({"a": 2})
    opt.reset()
    assert _drain(opt) == [{"a": 0}, {"a": 1}]


def test_many_marks_rebuild_the_grid_once() -> None:
    space = {"a": list(range(50)), "b": list(range(20))}
    opt = GridSearchOptimizer(space, ["accuracy"])
    calls = 0
    original = opt._apply_pending_resume_marks

    def counting() -> None:
        nonlocal calls
        calls += 1
        original()

    opt._apply_pending_resume_marks = counting  # type: ignore[method-assign]
    for a in range(50):
        for b in range(10):
            opt.mark_config_evaluated({"a": a, "b": b})
    assert calls == 0
    assert opt.total_combinations == 500
    assert calls == 1
    assert len(_drain(opt)) == 500


def test_shallow_copy_does_not_share_pending_marks() -> None:
    import copy

    opt = GridSearchOptimizer({"a": [0, 1, 2]}, ["accuracy"])
    opt.mark_config_evaluated({"a": 0})
    clone = copy.copy(opt)
    clone.mark_config_evaluated({"a": 1})
    assert _drain(opt) == [{"a": 1}, {"a": 2}]
    assert _drain(clone) == [{"a": 2}]


def test_pickle_roundtrip_and_legacy_state() -> None:
    import pickle

    opt = GridSearchOptimizer({"a": [0, 1, 2]}, ["accuracy"])
    opt.mark_config_evaluated({"a": 1})
    clone = pickle.loads(pickle.dumps(opt))
    assert _drain(clone) == [{"a": 0}, {"a": 2}]

    legacy = opt.__getstate__()
    legacy["_grid_points"] = legacy.pop("_grid_store")
    legacy.pop("_pending_resume_marks")
    legacy.pop("_pending_slow_marks")
    fresh = GridSearchOptimizer.__new__(GridSearchOptimizer)
    fresh.__setstate__(legacy)
    assert fresh.get_remaining_combinations() == [{"a": 0}, {"a": 2}]


def test_mutating_a_marked_config_afterwards_does_not_undo_the_mark() -> None:
    opt = GridSearchOptimizer({"a": [[1], [2]]}, ["accuracy"])
    cfg = {"a": [1]}
    opt.mark_config_evaluated(cfg)
    cfg["a"].append(9)
    assert opt.get_remaining_combinations() == [{"a": [2]}]


def test_list_and_tuple_values_stay_distinct() -> None:
    opt = GridSearchOptimizer({"a": [[1], (1,)]}, ["accuracy"])
    ref = [dict(p) for p in opt._grid_points]
    opt.mark_config_evaluated({"a": (1,)})
    assert opt.get_remaining_combinations() == _old_mark(ref, 0, {"a": (1,)})


def test_list_valued_grid_resumes_correctly_at_scale() -> None:
    n = 200
    opt = GridSearchOptimizer({"a": [[i] for i in range(n)]}, ["accuracy"])
    for i in range(n - 1):
        opt.mark_config_evaluated({"a": [i]})
    assert opt.get_remaining_combinations() == [{"a": [n - 1]}]


def test_subclass_points_with_extra_fields_match_on_every_field() -> None:
    class Extra(GridSearchOptimizer):
        def _generate_grid(self):
            return [{"a": 0, "extra": 1}, {"a": 0, "extra": 2}]

    opt = Extra({"a": [0]}, ["accuracy"])
    ref = [dict(p) for p in opt._grid_points]
    cfg = {"a": 0, "extra": 1}
    opt.mark_config_evaluated(cfg)
    assert opt.get_remaining_combinations() == _old_mark(ref, 0, cfg)
