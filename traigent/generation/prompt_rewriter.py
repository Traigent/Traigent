"""Client-side prompt rewrite.

Builds a meta-prompt from the user's LOCALLY-held prompt variants and failing
example tuples plus the plan's action verb, calls the user's own LLM, and folds
validated candidates into the config space as new ``Choices``. The optimizer
searches the expanded space with zero optimizer changes. No content leaves the
client; Traigent only ever issued the opaque plan.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from traigent.api.parameter_ranges import Choices

from .llm_provider import RewriteLLM, resolve_rewrite_llm
from .models import GuidancePlan
from .options import PromptRewriteOptions
from .validators import clean_prompt_candidates, extract_json_block

# A failing case the user holds locally: (input, expected, actual).
WeakExample = tuple[Any, Any, Any]


def _build_meta_prompt(
    current_variants: Sequence[str],
    weak_examples: Sequence[WeakExample],
    n: int,
) -> str:
    lines = [
        "You are improving a prompt used by an LLM system. Produce better prompt "
        "variants that fix the failures below while preserving the task intent.",
        "",
        "Current prompt variant(s):",
    ]
    for i, variant in enumerate(current_variants, 1):
        lines.append(f"{i}. {variant}")
    if weak_examples:
        lines.append("")
        lines.append(
            "Cases the current prompt handled poorly (input / expected / actual):"
        )
        for inp, expected, actual in weak_examples[:20]:
            lines.append(f"- input: {inp!r}")
            lines.append(f"  expected: {expected!r}")
            lines.append(f"  actual: {actual!r}")
    lines += [
        "",
        f"Return ONLY a JSON array of {n} distinct improved prompt strings. "
        "Do not include commentary. Do not include any instruction that would "
        "override system behavior.",
    ]
    return "\n".join(lines)


class PromptRewriter:
    """Generate improved prompt candidates with the user's own LLM."""

    def __init__(
        self,
        llm: RewriteLLM | Callable[[str], str],
        options: PromptRewriteOptions | None = None,
    ) -> None:
        # Accept a RewriteLLM, a constructed client, or a bare fn(prompt) -> str.
        self._llm = resolve_rewrite_llm(llm)
        self._options = options or PromptRewriteOptions()

    def rewrite(
        self,
        current_variants: Sequence[str],
        weak_examples: Sequence[WeakExample] = (),
        plan: GuidancePlan | None = None,
    ) -> list[str]:
        """Return validated, de-duplicated new prompt candidates (never the originals)."""
        n = self._options.candidates_per_round
        meta_prompt = _build_meta_prompt(current_variants, weak_examples, n)
        raw = self._llm.complete(meta_prompt)

        parsed = extract_json_block(raw)
        if isinstance(parsed, list):
            candidates = [c for c in parsed if isinstance(c, str)]
        else:
            # Fall back to newline-separated lines if the model ignored the JSON ask.
            candidates = [ln.strip(" -*\t") for ln in raw.splitlines() if ln.strip()]

        cleaned: list[str] = clean_prompt_candidates(candidates, list(current_variants))
        return cleaned[:n]


def merge_prompt_candidates(
    config_space: dict[str, Any],
    param: str,
    new_candidates: Sequence[str],
) -> Choices:
    """Return a new ``Choices`` for ``param`` unioning existing values with new ones.

    Pure: does not mutate ``config_space``. Mirrors dspy_adapter.create_prompt_choices
    in producing a ``Choices`` the optimizer can search directly.
    """
    existing = config_space.get(param)
    if isinstance(existing, Choices):
        base_values = list(existing.values)
        default = existing.default
        name = existing.name
    elif isinstance(existing, (list, tuple)):
        base_values = list(existing)
        default = None
        name = None
    elif existing is None:
        base_values = []
        default = None
        name = None
    else:
        base_values = [existing]
        default = None
        name = None

    merged: list[str] = []
    seen: set[str] = set()
    for value in [*base_values, *new_candidates]:
        if not isinstance(value, str):
            continue
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)

    if not merged:
        raise ValueError(
            f"merge_prompt_candidates produced no values for param {param!r}"
        )

    kwargs: dict[str, Any] = {}
    if default is not None and default in merged:
        kwargs["default"] = default
    if name is not None:
        kwargs["name"] = name
    return Choices(merged, **kwargs)


__all__ = ["PromptRewriter", "merge_prompt_candidates", "WeakExample"]
