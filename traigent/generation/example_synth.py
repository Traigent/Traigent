"""Client-side benchmark example synthesis.

Builds an action-keyed meta-prompt from LOCALLY-held seed examples, calls the
user's own LLM, parses + validates the result, and returns new
``EvaluationExample`` objects tagged ``metadata.synthetic`` so the loop can grow
the dataset. No seed content leaves the client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from traigent.evaluators.base import EvaluationExample

from .llm_provider import RewriteLLM, resolve_rewrite_llm
from .models import GuidanceAction
from .options import DatasetGrowthOptions
from .validators import _example_key, dedupe_example_keys, extract_json_block

_ACTION_DIRECTIVE = {
    GuidanceAction.GENERATE_SIMILAR: (
        "Generate new examples similar in form and difficulty to the seeds."
    ),
    GuidanceAction.GENERATE_HARDER: (
        "Generate new examples that are HARDER variations of the seeds — more "
        "edge cases, trickier inputs — while staying on the same task."
    ),
    GuidanceAction.DIVERSIFY_AROUND: (
        "Generate new examples that DIVERSIFY around the seeds: different angles, "
        "phrasings, and underrepresented cases of the same task."
    ),
}


def _build_meta_prompt(
    seeds: Sequence[EvaluationExample], action: GuidanceAction, count: int
) -> str:
    directive = _ACTION_DIRECTIVE.get(
        action, _ACTION_DIRECTIVE[GuidanceAction.GENERATE_SIMILAR]
    )
    lines = [
        "You are expanding an evaluation dataset for an LLM task.",
        directive,
        "",
        "Seed examples (input / expected_output):",
    ]
    for seed in seeds[:20]:
        lines.append(f"- input: {seed.input_data!r}")
        lines.append(f"  expected_output: {seed.expected_output!r}")
    lines += [
        "",
        f"Return ONLY a JSON array of {count} objects, each with an 'input' object "
        "(same keys as the seeds) and an 'expected_output'. No commentary. Do not "
        "include any instruction that would override system behavior.",
    ]
    return "\n".join(lines)


class ExampleSynthesizer:
    """Synthesize new evaluation examples with the user's own LLM."""

    def __init__(
        self,
        llm: RewriteLLM | Callable[[str], str],
        options: DatasetGrowthOptions | None = None,
    ) -> None:
        # Accept a RewriteLLM, a constructed client, or a bare fn(prompt) -> str.
        self._llm = resolve_rewrite_llm(llm)
        self._options = options or DatasetGrowthOptions()

    def synthesize(
        self,
        seed_examples: Sequence[EvaluationExample],
        action: GuidanceAction,
        count: int | None = None,
        *,
        seed_ids: Sequence[str] = (),
        existing: Sequence[EvaluationExample] = (),
    ) -> list[EvaluationExample]:
        """Return validated, de-duplicated synthetic examples tagged as synthetic."""
        n = count or self._options.examples_per_round
        meta_prompt = _build_meta_prompt(seed_examples, action, n)
        raw = self._llm.complete(meta_prompt)

        parsed = extract_json_block(raw)
        pairs: list[tuple[Any, Any]] = []
        if isinstance(parsed, list):
            for obj in parsed:
                if isinstance(obj, dict) and "input" in obj:
                    pairs.append((obj.get("input"), obj.get("expected_output")))

        existing_keys = {
            _example_key(e.input_data, e.expected_output) for e in existing
        }
        existing_keys.update(
            _example_key(s.input_data, s.expected_output) for s in seed_examples
        )
        novel = dedupe_example_keys(pairs, existing_keys)[:n]

        out: list[EvaluationExample] = []
        for input_data, expected_output in novel:
            normalized_input = (
                input_data if isinstance(input_data, dict) else {"input": input_data}
            )
            out.append(
                EvaluationExample(
                    input_data=normalized_input,
                    expected_output=expected_output,
                    metadata={
                        "synthetic": True,
                        "seed_ids": list(seed_ids),
                        "action": action.value,
                    },
                )
            )
        return out


__all__ = ["ExampleSynthesizer"]
