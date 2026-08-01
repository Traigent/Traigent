"""Synthesized model labels must never enter cold-start candidates."""

from __future__ import annotations

from traigent.evaluators.base import EvaluationExample
from traigent.generation.models import GuidanceAction

from traigent.generation.coldstart.contracts import ParameterSpec, SystemSpec
from traigent.generation.coldstart.generators import SynthesizedInputGenerator


class _SyntheticExamples:
    def synthesize(
        self,
        seed_examples: object,
        action: GuidanceAction,
        count: int,
    ) -> list[EvaluationExample]:
        return [
            EvaluationExample(
                input_data={"question": "new input"},
                expected_output="MODEL_PROPOSED_GOLD_MUST_BE_DROPPED",
            )
        ]


def test_synthesized_input_generator_drops_every_model_proposed_expected_output() -> (
    None
):
    generator = SynthesizedInputGenerator(
        _SyntheticExamples(),  # type: ignore[arg-type]
        [
            EvaluationExample(
                input_data={"question": "seed"}, expected_output="seed gold"
            )
        ],
    )
    system = SystemSpec(
        "answer",
        "example",
        (ParameterSpec("question", "str", True),),
        "str",
        (),
        "a" * 64,
    )

    candidates = generator.propose(system, count=1, seed=9)

    assert candidates[0].inputs == {"question": "new input"}
    assert candidates[0].ground_truth is None
