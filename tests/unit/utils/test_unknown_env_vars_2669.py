"""#2669: warn (never fail) on unrecognised TRAIGENT_* environment variables."""

from __future__ import annotations

import warnings

import pytest

from traigent.utils.env_audit import (
    find_unknown_traigent_env_vars,
    known_traigent_env_names,
    warn_unknown_traigent_env_vars,
)


def test_known_set_is_derived_from_code():
    known = known_traigent_env_names()
    assert "TRAIGENT_RUN_COST_LIMIT" in known
    assert "TRAIGENT_OFFLINE_MODE" in known


def test_misspelt_name_found_with_suggestion():
    found = find_unknown_traigent_env_vars(
        {"TRAIGENT_RUN_COST_LIMT": "0.5", "TRAIGENT_OFFLINE": "1", "HOME": "/x"}
    )
    assert found == {"TRAIGENT_RUN_COST_LIMT": "TRAIGENT_RUN_COST_LIMIT"}


def test_unknown_without_close_match_has_no_suggestion():
    found = find_unknown_traigent_env_vars({"TRAIGENT_ZZZZ_QQQQ_WWWW": "x"})
    assert found == {"TRAIGENT_ZZZZ_QQQQ_WWWW": None}


def test_results_folder_typo_suggests_real_name():
    found = find_unknown_traigent_env_vars({"TRAIGENT_RESULT_FOLDER": "./r"})
    assert found["TRAIGENT_RESULT_FOLDER"] == "TRAIGENT_RESULTS_FOLDER"


def test_warning_lists_names_never_values():
    env = {"TRAIGENT_RUN_COST_LIMT": "super-secret-value-123"}
    with pytest.warns(UserWarning) as rec:
        warn_unknown_traigent_env_vars(env)
    text = str(rec[0].message)
    assert "TRAIGENT_RUN_COST_LIMT" in text
    assert "did you mean TRAIGENT_RUN_COST_LIMIT" in text
    assert "super-secret-value-123" not in text


def test_no_warning_when_all_known():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_unknown_traigent_env_vars({"TRAIGENT_RUN_COST_LIMIT": "1"})
