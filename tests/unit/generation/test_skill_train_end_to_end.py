"""End-to-end behaviour of the public ``train_skill`` entry point.

Why this file exists: every other ``train_skill`` test passes
``optimizer_llm=lambda prompt: json.dumps({"edits": []})``, and the only
``Reflector.analyze`` test asserts the empty-list case. So no test in the suite
had ever observed the public entry point change a document, and the whole
generative seam -- model reply -> ``EditOp`` -> applied edit -> gate verdict --
could regress to ``return []`` with the suite still green.

These two tests close that gap from opposite directions:

* ``test_train_skill_applies_a_beneficial_edit`` proves an improvement is
  accepted and the document really changes.
* ``test_train_skill_rejects_a_harmful_edit`` is the negative control: a
  strictly worse document must be refused and the original kept. Without it the
  first test would still pass against a gate that accepts everything.

Both drive the real edit protocol from ``skill_train/edits.py``
(``op`` in append/insert_after/replace/delete, plus ``target``/``content``) --
not a paraphrase of it -- so a change to that protocol fails here.
"""

from __future__ import annotations

import json

import pytest

import traigent
from traigent.api.parameter_ranges import TextDocument

_NEGATIVE_MARKERS = (
    "cold",
    "awful",
    "never",
    "terrible",
    "worst",
    "bland",
    "rude",
    "slow",
    "dirty",
    "overpriced",
)

# The doc under training only helps if it pins the output format; the stub agent
# below obeys exactly one instruction, so accuracy is a real function of the text.
_FORMAT_INSTRUCTION = "one lowercase word"
_WEAK_DOC = "Classify."
_STRONG_DOC = f"Answer with exactly {_FORMAT_INSTRUCTION}: positive or negative."
_HARMFUL_DOC = "Write a long flowery paragraph about the meal."

# split_dataset needs >= 5 selection examples and selection_split defaults to 0.2,
# so anything under 25 rows raises before training starts.
_ROWS = 30


def _dataset() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(_ROWS // 2):
        marker = _NEGATIVE_MARKERS[index % len(_NEGATIVE_MARKERS)]
        rows.append(
            {
                "input": {"text": f"the food was {marker} tonight"},
                "expected_output": "negative",
            }
        )
        rows.append(
            {
                "input": {"text": f"absolutely wonderful meal number {index}"},
                "expected_output": "positive",
            }
        )
    return rows


def _agent(text: str) -> str:
    document = traigent.get_config().get("system_prompt", _WEAK_DOC)
    label = "negative" if any(m in text for m in _NEGATIVE_MARKERS) else "positive"
    if _FORMAT_INSTRUCTION in document:
        return label
    return f"The sentiment is {label}!"


def _optimized(initial_document: str):
    return traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"system_prompt": TextDocument(initial_document)},
        execution_mode="local",
        max_trials=4,
    )(_agent)


def _replace_edit(target: str, content: str) -> str:
    """A reply in the real reflector protocol (see skill_train/edits.py)."""
    return json.dumps(
        {
            "edits": [
                {
                    "op": "replace",
                    "target": target,
                    "content": content,
                    "rationale": "test edit",
                }
            ]
        }
    )


def test_train_skill_applies_a_beneficial_edit() -> None:
    trained = _optimized(_WEAK_DOC).train_skill(
        document=_WEAK_DOC,
        optimizer_llm=lambda prompt: _replace_edit(_WEAK_DOC, _STRONG_DOC),
        doc_param="system_prompt",
    )

    assert trained.best_document == _STRONG_DOC
    assert trained.best_document != _WEAK_DOC

    accepted = trained.summary["accept_history"]
    assert accepted, "a strictly better document must be accepted"
    assert accepted[0]["op"] == "replace"
    assert accepted[0]["status"] == "applied"
    # The gate must justify acceptance with a measured gain, not accept blindly.
    assert accepted[0]["selection_delta"] > 0


def test_train_skill_rejects_a_harmful_edit() -> None:
    """Negative control: the selection gate must refuse a regression."""
    trained = _optimized(_STRONG_DOC).train_skill(
        document=_STRONG_DOC,
        optimizer_llm=lambda prompt: _replace_edit(_STRONG_DOC, _HARMFUL_DOC),
        doc_param="system_prompt",
    )

    assert trained.best_document == _STRONG_DOC
    assert trained.best_document != _HARMFUL_DOC
    assert not trained.summary["accept_history"]

    rejected = trained.summary["reject_history"]
    assert rejected, "a strictly worse document must be rejected"
    assert rejected[0]["status"] == "rejected_gate"
    assert rejected[0]["selection_delta"] < 0


def test_reflection_prompt_carries_rollout_content() -> None:
    """The reflector must actually see failures -- otherwise it cannot improve anything.

    This also pins the documented privacy semantics of ``train_skill``: rollout
    inputs and expected outputs DO reach the caller-supplied ``optimizer_llm``.
    """
    seen: list[str] = []

    def recording_llm(prompt: str) -> str:
        seen.append(prompt)
        return _replace_edit(_WEAK_DOC, _STRONG_DOC)

    _optimized(_WEAK_DOC).train_skill(
        document=_WEAK_DOC,
        optimizer_llm=recording_llm,
        doc_param="system_prompt",
    )

    assert seen, "the optimizer LLM must be called"
    combined = "\n".join(seen)
    assert "the food was" in combined
    assert "negative" in combined


def test_train_skill_requires_enough_examples_for_a_selection_split() -> None:
    """Fewer than 25 rows cannot yield the 5 selection examples the gate needs."""
    too_small = _dataset()[:6]
    fn = traigent.optimize(
        eval_dataset=too_small,
        objectives=["accuracy"],
        configuration_space={"system_prompt": TextDocument(_WEAK_DOC)},
        execution_mode="local",
        max_trials=2,
    )(_agent)

    with pytest.raises(ValueError, match="selection split requires at least 5") as exc:
        fn.train_skill(
            document=_WEAK_DOC,
            optimizer_llm=lambda prompt: _replace_edit(_WEAK_DOC, _STRONG_DOC),
            doc_param="system_prompt",
        )

    # The caller controls the dataset size, not the derived selection count, so
    # the message must name the input to change and the target to reach.
    message = str(exc.value)
    assert "total=6" in message
    assert "selection_fraction=" in message
    assert "Provide at least 25 examples" in message
