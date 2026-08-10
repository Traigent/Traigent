from __future__ import annotations

import pytest

from traigent.generation import SkillTrainOptions


def test_skill_train_options_defaults() -> None:
    options = SkillTrainOptions()

    assert options.epochs == 4
    assert options.rollout_batch == 40
    assert options.selection_split == 0.2
    assert options.test_split == 0.0
    assert options.privacy_mode is True
    assert options.edit_budget_floor == 2
    assert options.edit_budget_schedule == "cosine"
    assert options.rejected_buffer is True
    assert options.rejected_buffer_max == 50
    assert options.slow_update is True
    assert options.slow_update_probe_size == 20
    assert options.meta_skill is True
    assert options.write_artifacts is True


def test_skill_train_options_bounds() -> None:
    with pytest.raises(ValueError):
        SkillTrainOptions(epochs=0)
    with pytest.raises(ValueError):
        SkillTrainOptions(edit_budget=17)
    with pytest.raises(ValueError):
        SkillTrainOptions(selection_split=0.0)
    with pytest.raises(ValueError):
        SkillTrainOptions(test_split=0.5)
    with pytest.raises(ValueError):
        SkillTrainOptions(rejected_buffer_max=501)
    with pytest.raises(ValueError):
        SkillTrainOptions(slow_update_probe_size=3)
    with pytest.raises(ValueError):
        SkillTrainOptions(edit_budget_schedule="linear")


def test_skill_train_options_forbid_extra() -> None:
    with pytest.raises(ValueError):
        SkillTrainOptions(unknown=True)
