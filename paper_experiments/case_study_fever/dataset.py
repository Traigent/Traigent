"""Small in-repo FEVER-style dataset used for regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Iterable

from traigent.evaluators.base import Dataset, EvaluationExample

__all__ = ["load_case_study_dataset"]


@dataclass(frozen=True)
class _RawExample:
    claim: str
    verdict: str
    metadata: dict[str, Any]


_RAW_EXAMPLES: Final[list[_RawExample]] = [
    _RawExample(
        claim="The Eiffel Tower is located in Berlin, Germany.",
        verdict="REFUTES",
        metadata={
            "page": "Eiffel_Tower",
            "line": 12,
            "evidence_text": (
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
            ),
        },
    ),
    _RawExample(
        claim="Marie Curie earned Nobel Prizes in both physics and chemistry.",
        verdict="SUPPORTS",
        metadata={
            "page": "Marie_Curie",
            "line": 28,
            "evidence_text": (
                "She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two scientific fields."
            ),
        },
    ),
    _RawExample(
        claim="The Great Wall of China can be seen unaided from the Moon.",
        verdict="REFUTES",
        metadata={
            "page": "Great_Wall_of_China",
            "line": 91,
            "evidence_text": (
                "Astronauts have confirmed that the wall is not visible to the naked eye from lunar distance."
            ),
        },
    ),
    _RawExample(
        claim="Mount Everest sits on the border between Nepal and China.",
        verdict="SUPPORTS",
        metadata={
            "page": "Mount_Everest",
            "line": 5,
            "evidence_text": "Its summit point lies on the international border separating Nepal and the Tibetan Autonomous Region of China.",
        },
    ),
]

_DATASET: Dataset | None = None


def _build_examples(raw_examples: Iterable[_RawExample]) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    for raw in raw_examples:
        metadata = dict(raw.metadata)
        metadata.setdefault("case_study", "fever")
        metadata.setdefault("verdict", raw.verdict)
        examples.append(
            EvaluationExample(
                input_data={"claim": raw.claim},
                expected_output=raw.verdict,
                metadata=metadata,
            )
        )
    return examples


def load_case_study_dataset(*, force_refresh: bool = False) -> Dataset:
    """Return a cached dataset containing a handful of FEVER claims."""

    global _DATASET

    if _DATASET is None or force_refresh:
        _DATASET = Dataset(
            examples=_build_examples(_RAW_EXAMPLES),
            name="fever_case_study",
            description="Synthetic FEVER-style claims bundled with the repo for regression tests.",
            metadata={"task": "fact_verification"},
        )

    return _DATASET
