"""Characterization of langchain handler _extract_response_model."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from traigent.integrations.langchain.handler import _extract_response_model as ex


def test_nothing_found():
    assert ex(NS()) is None
    assert ex(object()) is None


def test_llm_output_model_name_beats_model_and_everything():
    r = NS(
        llm_output={"model_name": "a", "model": "b"},
        response_metadata={"model": "c"},
        model="d",
    )
    assert ex(r) == "a"


def test_llm_output_falls_back_to_model_when_model_name_falsy():
    assert ex(NS(llm_output={"model_name": "", "model": "b"})) == "b"
    assert ex(NS(llm_output={"model_name": None, "model": "b"})) == "b"


def test_non_str_or_empty_llm_output_falls_through_to_next_source():
    r = NS(llm_output={"model_name": 5}, response_metadata={"model": "rm"})
    assert ex(r) == "rm"
    r = NS(llm_output={"model_name": 7, "model": "ignored"}, model="m")
    # truthy non-str model_name short-circuits the `or`, so "ignored" is not used
    assert ex(r) == "m"


@pytest.mark.parametrize("llm_output", [None, [], "s", 3])
def test_non_dict_llm_output_ignored(llm_output):
    assert ex(NS(llm_output=llm_output, model="m")) == "m"


def test_response_metadata_then_model_attr():
    assert ex(NS(response_metadata={"model_name": "n", "model": "x"})) == "n"
    assert ex(NS(response_metadata={"model": "x"})) == "x"
    assert ex(NS(response_metadata="no", model="m")) == "m"
    assert ex(NS(model="")) is None
    assert ex(NS(model=3)) is None


def test_generation_message_metadata_before_generation_itself():
    gen = NS(
        message=NS(response_metadata={"model": "from-message"}, model="msg-attr"),
        response_metadata={"model": "from-gen"},
        model="gen-attr",
    )
    assert ex(NS(generations=[[gen]])) == "from-message"


def test_generation_candidate_order_and_flattening():
    g1 = NS(message=NS(response_metadata={}, model=""), response_metadata={"model_name": "g1"})
    g2 = NS(response_metadata=None, model="g2")
    assert ex(NS(generations=[[g1], g2])) == "g1"
    assert ex(NS(generations=[[NS(), NS(model="")], g2])) == "g2"


def test_generation_message_attr_model_when_no_metadata():
    gen = NS(message=NS(model="mm"))
    assert ex(NS(generations=[gen])) == "mm"


@pytest.mark.parametrize("gens", [None, (), "x", {}])
def test_non_list_generations_ignored(gens):
    assert ex(NS(generations=gens)) is None


def test_read_order_and_counts_of_attribute_access():
    reads = []

    class R:
        def __getattr__(self, name):
            reads.append(name)
            return None

    assert ex(R()) is None
    assert reads == ["llm_output", "response_metadata", "model", "generations"]
