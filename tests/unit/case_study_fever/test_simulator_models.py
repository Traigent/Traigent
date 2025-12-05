import pytest

from paper_experiments.case_study_fever import simulator


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4.1", "gpt-4o"),
        ("gpt-4.1-mini", "gpt-4o-mini"),
        ("gpt-4.1-nano", "haiku-3.5"),
    ],
)
def test_model_aliases(model: str, expected: str) -> None:
    assert simulator._ensure_known_model(model) == expected


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        simulator._ensure_known_model("unknown-model")
