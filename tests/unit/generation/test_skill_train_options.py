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


def test_skill_train_options_bounds() -> None:
    with pytest.raises(ValueError):
        SkillTrainOptions(epochs=0)
    with pytest.raises(ValueError):
        SkillTrainOptions(edit_budget=17)
    with pytest.raises(ValueError):
        SkillTrainOptions(selection_split=0.0)
    with pytest.raises(ValueError):
        SkillTrainOptions(test_split=0.5)


def test_skill_train_options_forbid_extra() -> None:
    with pytest.raises(ValueError):
        SkillTrainOptions(unknown=True)
