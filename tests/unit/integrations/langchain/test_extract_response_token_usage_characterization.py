"""Characterization of langchain handler _extract_response_token_usage precedence."""

from __future__ import annotations

from types import SimpleNamespace as NS

from traigent.integrations.langchain.handler import _extract_response_token_usage as ex


def _u(i: int, o: int) -> dict[str, int]:
    return {"input_tokens": i, "output_tokens": o}


def test_nothing_found_is_zero_triple():
    assert ex(NS()) == (0, 0, 0)
    assert ex(object()) == (0, 0, 0)
    assert ex(NS(llm_output={"token_usage": {}}, usage_metadata=_u(0, 0))) == (0, 0, 0)


def test_llm_output_keys_in_priority_order():
    out = {"usage_metadata": _u(7, 7), "usage": _u(2, 2), "token_usage": _u(1, 1)}
    assert ex(NS(llm_output=out)) == (1, 1, 2)
    assert ex(NS(llm_output={"usage_metadata": _u(7, 7), "usage": _u(2, 2)})) == (
        2,
        2,
        4,
    )


def test_llm_output_beats_response_attrs_and_metadata():
    r = NS(
        llm_output={"usage": _u(1, 2)},
        usage_metadata=_u(9, 9),
        response_metadata={"usage": _u(8, 8)},
    )
    assert ex(r) == (1, 2, 3)


def test_non_dict_llm_output_is_ignored():
    assert ex(NS(llm_output="x", usage_metadata=_u(3, 4))) == (3, 4, 7)


def test_response_usage_metadata_beats_usage_beats_metadata():
    assert ex(NS(usage_metadata=_u(1, 1), usage=_u(2, 2))) == (1, 1, 2)
    assert ex(NS(usage=_u(2, 2), response_metadata={"usage": _u(5, 5)})) == (2, 2, 4)
    assert ex(NS(usage_metadata=None, usage=_u(2, 3))) == (2, 3, 5)


def test_response_metadata_dict_only_and_key_order():
    r = NS(response_metadata={"usage_metadata": _u(6, 6), "token_usage": _u(1, 4)})
    assert ex(r) == (1, 4, 5)
    assert ex(NS(response_metadata="nope")) == (0, 0, 0)


def test_explicit_total_wins():
    r = NS(usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 10})
    assert ex(r) == (1, 2, 10)


def test_generation_message_usage_then_generation_usage_then_metadata():
    gen = NS(
        message=NS(usage_metadata=_u(1, 1)),
        usage_metadata=_u(9, 9),
    )
    assert ex(NS(generations=[[gen]])) == (1, 1, 2)
    gen2 = NS(message=NS(), usage=_u(2, 2), usage_metadata=_u(9, 9))
    assert ex(NS(generations=[[gen2]])) == (9, 9, 18)
    gen3 = NS(message=NS(response_metadata={"token_usage": _u(4, 4)}), usage=_u(7, 7))
    assert ex(NS(generations=[gen3])) == (4, 4, 8)
    gen4 = NS(message=None, response_metadata={"usage": _u(3, 3)})
    assert ex(NS(generations=[[gen4]])) == (3, 3, 6)


def test_generations_scanned_in_order_and_first_hit_wins():
    empty = NS(message=NS())
    hit = NS(message=NS(usage=_u(2, 5)))
    later = NS(message=NS(usage=_u(9, 9)))
    assert ex(NS(generations=[[empty, hit], [later]])) == (2, 5, 7)


def test_non_list_generations_ignored_and_response_level_wins_over_generations():
    gen = NS(message=NS(usage=_u(2, 2)))
    assert ex(NS(generations=(gen,))) == (0, 0, 0)
    assert ex(NS(usage=_u(1, 1), generations=[gen])) == (1, 1, 2)
