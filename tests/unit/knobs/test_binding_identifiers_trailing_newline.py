"""Knob and binding identifiers match the ASCII contract exactly (sibling of Traigent#3006).

``re.match`` with ``$`` accepts a single trailing newline, so ``"x\\n"`` passed
as an identifier. The #3006 fix moved the measure/TVAR/kwarg sites to
``fullmatch``; these knob-binding sites were left on ``match``.
"""

from __future__ import annotations

import pytest

from traigent.knobs.bindings import Ref


@pytest.mark.parametrize("knob", ["x\n", "my.knob\n"])
def test_ref_rejects_identifier_with_trailing_newline(knob: str) -> None:
    with pytest.raises(ValueError, match="not a valid identifier"):
        Ref(knob=knob)


def test_ref_accepts_dotted_identifier() -> None:
    assert Ref(knob="model.name").knob == "model.name"


def test_knob_rejects_name_with_trailing_newline() -> None:
    from traigent.knobs.bindings import Fixed, Knob

    with pytest.raises(ValueError, match="not a valid identifier"):
        Knob(name="x\n", binding=Fixed(value=1))
