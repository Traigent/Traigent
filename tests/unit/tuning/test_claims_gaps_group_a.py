"""SDK-129 group A regressions for the tuning loop.

Checks in this module:
- an approver mapping is refused rather than reduced to its names (g7);
- a missing head file is refused with E_HEAD_DIVERGED before the optimizer runs (sol1);
- an actual tuple does not match an expected list of the same items (sol3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.tuning.errors import TuningError
from traigent.tuning.evidence import _accuracy

from .test_controller import (
    FakeOptimizer,
)
from .test_controller import _holdout as _kit_holdout  # noqa: F401 - _holdout is an autouse fixture the kit needs
from .test_controller import at, controller, e_ref


def test_g7_approvers_mapping_is_refused_not_reduced_to_its_names() -> None:
    from traigent.tuning.authority import authority_from_config

    with pytest.raises(TuningError):
        authority_from_config({"approvers": {"alice": {"role": "x"}}})
    authority, _ = authority_from_config({"approvers": ["alice"]})
    assert authority is not None and authority.authorized("alice")


def test_sol1_missing_head_is_refused_before_the_marker(tmp_path: Path) -> None:
    opt = FakeOptimizer()
    c = controller(tmp_path, opt)
    c.approve_optimization(e_ref("E1"), "approver-1", at(10))
    (tmp_path / "store" / "head.json").unlink()
    with pytest.raises(TuningError) as err:
        c.step(at(10))
    assert err.value.code == "E_HEAD_DIVERGED"
    assert opt.calls == []


def test_sol3_tuple_never_equals_list() -> None:
    assert _accuracy([{"output": [1]}], [(True, (1,))]) == [0.0]
    assert _accuracy([{"output": [1]}], [(True, [1])]) == [1.0]
