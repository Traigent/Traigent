"""Characterization for JudgeConfigDTO.from_benchmark_payload."""

from __future__ import annotations

import pytest

from traigent.evaluation.dtos import JudgeConfigDTO


class _LoggingDict(dict):
    """dict that records every ``.get`` key and every item read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log: list[str] = []

    def get(self, key, default=None):
        self.log.append(f"get:{key}")
        return super().get(key, default)

    def __getitem__(self, key):
        self.log.append(f"item:{key}")
        return super().__getitem__(key)


def _as_tuple(dto):
    if dto is None:
        return None
    return (
        dto.instructions,
        dto.model_id,
        dto.context_type,
        dto.context_source,
        dto.parameters,
        dto.scoring_rubric,
        dto.response_format_instructions,
        dto.max_budget,
    )


def test_falsey_payload_returns_none():
    assert JudgeConfigDTO.from_benchmark_payload(None) is None
    assert JudgeConfigDTO.from_benchmark_payload({}) is None


def test_minimal_payload_defaults():
    assert _as_tuple(JudgeConfigDTO.from_benchmark_payload({"x": 1})) == (
        "",
        "",
        "none",
        None,
        {"model_id": ""},
        None,
        None,
        None,
    )


def test_explicit_parameters_win_over_model_parameters():
    dto = JudgeConfigDTO.from_benchmark_payload(
        {"parameters": {"a": 1}, "modelParameters": [{"temperature": 0.5}]}
    )
    assert dto.parameters == {"a": 1, "model_id": ""}


def test_model_parameters_aliases_and_zero_values():
    dto = JudgeConfigDTO.from_benchmark_payload(
        {
            "modelParameters": [
                {
                    "temperature": 0,
                    "maxTokens": 10,
                    "top_p": 0.0,
                    "topP": 0.9,
                    "topK": 4,
                    "frequencyPenalty": 0.1,
                    "presence_penalty": 0.2,
                    "repetitionPenalty": 1.1,
                    "stopSequences": ["x"],
                },
                {"temperature": 9},
            ]
        }
    )
    assert dto.parameters == {
        "model_id": "",
        "temperature": 0,
        "max_tokens": 10,
        "top_p": 0.0,
        "top_k": 4,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "stop_sequences": ["x"],
    }


@pytest.mark.parametrize(
    "model_parameters",
    [[], "abc", {"temperature": 1}, ["notadict"], [None]],
)
def test_unusable_model_parameters_yield_empty(model_parameters):
    dto = JudgeConfigDTO.from_benchmark_payload(
        {"modelParameters": model_parameters, "instructions": "i"}
    )
    assert dto.parameters == {"model_id": ""}


def test_parameters_none_with_falsey_model_parameters_is_empty():
    dto = JudgeConfigDTO.from_benchmark_payload(
        {"parameters": None, "modelParameters": []}
    )
    assert dto.parameters == {"model_id": ""}


def test_model_and_context_fallbacks():
    dto = JudgeConfigDTO.from_benchmark_payload(
        {
            "instructions": 5,
            "model": "m",
            "context": {"type": "doc", "source": "s"},
            "scoring_rubric": {"k": 1},
            "response_format_instructions": "fmt",
            "max_budget": "2.5",
        }
    )
    assert _as_tuple(dto) == (
        "5",
        "m",
        "doc",
        "s",
        {"model_id": "m"},
        {"k": 1},
        "fmt",
        2.5,
    )


def test_primary_keys_beat_fallbacks_and_empty_context():
    dto = JudgeConfigDTO.from_benchmark_payload(
        {
            "model_id": "mid",
            "model": "m",
            "context_type": "ct",
            "context_source": "cs",
            "context": {"type": "doc", "source": "s"},
            "scoring_rubric": {},
            "max_budget": 0,
        }
    )
    assert _as_tuple(dto) == ("", "mid", "ct", "cs", {"model_id": "mid"}, {}, None, 0.0)
    dto = JudgeConfigDTO.from_benchmark_payload(
        {"model_id": "", "model": "m", "context_type": "", "context": None}
    )
    assert dto.model_id == "m"
    assert dto.context_type == "none"
    assert dto.context_source is None


def test_candidate_dict_subclass_get_order():
    candidate = _LoggingDict(maxTokens=3)
    payload = _LoggingDict(modelParameters=[candidate])
    dto = JudgeConfigDTO.from_benchmark_payload(payload)
    assert dto.parameters["max_tokens"] == 3
    assert candidate.log == [
        "get:temperature",
        "get:max_tokens",
        "get:maxTokens",
        "get:top_p",
        "get:topP",
        "get:top_k",
        "get:topK",
        "get:frequency_penalty",
        "get:frequencyPenalty",
        "get:presence_penalty",
        "get:presencePenalty",
        "get:repetition_penalty",
        "get:repetitionPenalty",
        "get:stop_sequences",
        "get:stopSequences",
    ]


def test_payload_read_sequence_with_model_parameters():
    payload = _LoggingDict(
        modelParameters=[{"temperature": 1}], max_budget=1, scoring_rubric={"a": 1}
    )
    JudgeConfigDTO.from_benchmark_payload(payload)
    assert payload.log == [
        "get:parameters",
        "get:modelParameters",
        "get:modelParameters",
        "get:instructions",
        "get:model_id",
        "get:model",
        "get:context_type",
        "get:context",
        "get:context_source",
        "get:context",
        "get:scoring_rubric",
        "get:scoring_rubric",
        "get:response_format_instructions",
        "get:max_budget",
        "item:max_budget",
    ]


def test_payload_read_sequence_with_explicit_parameters():
    payload = _LoggingDict(parameters={"a": 1}, modelParameters=[{"temperature": 1}])
    JudgeConfigDTO.from_benchmark_payload(payload)
    assert payload.log[:2] == ["get:parameters", "get:instructions"]
