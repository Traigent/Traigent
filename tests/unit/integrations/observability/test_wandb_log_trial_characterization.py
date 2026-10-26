"""Characterization tests pinning ``TraigentWandBTracker.log_trial``."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from traigent.api.types import TrialStatus
from traigent.integrations.observability import wandb as wandb_module

MOD = "traigent.integrations.observability.wandb"


class _Obj:
    def __str__(self):
        return "OBJ"


class _FakeWandb:
    def __init__(self, log_error=None, save_error=None):
        self.logged = []
        self.saved = []
        self.log_error = log_error
        self.save_error = save_error

    def log(self, data, step=None):
        if self.log_error:
            raise self.log_error
        self.logged.append((data, step))

    def save(self, path):
        if self.save_error:
            raise self.save_error
        self.saved.append((path, json.loads(Path(path).read_text())))


class _BadIso:
    def isoformat(self):
        raise RuntimeError("no iso")


def trial(**over):
    base = dict(
        trial_id="t1",
        status=TrialStatus.COMPLETED,
        duration=1.5,
        config={
            "model": "m",
            "n": 3,
            "flag": True,
            "none": None,
            "obj": _Obj(),
            "l": [1],
        },
        metrics={"acc": 0.5, "i": 2, "s": "text", "b": True, "n": None},
        timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    base.update(over)
    return SimpleNamespace(**base)


# name -> (trial kwargs, trial_number, step, wandb kwargs, run?)
CASES = {
    "full": ({}, 1, None, {}, True),
    "custom_step": ({}, 4, 9, {}, True),
    "no_run": ({}, 1, None, {}, False),
    "non_mapping_config": ({"config": "oops"}, 1, None, {}, True),
    "empty_list_config": ({"config": []}, 1, None, {}, True),
    "none_config": ({"config": None}, 1, None, {}, True),
    "non_mapping_metrics": ({"metrics": [1]}, 1, None, {}, True),
    "empty_dict_metrics": ({"metrics": {}}, 1, None, {}, True),
    "zero_metrics_warns": ({"metrics": 0}, 1, None, {}, True),
    "no_timestamp": ({"timestamp": None}, 1, None, {}, True),
    "timestamp_without_iso": ({"timestamp": "2025"}, 1, None, {}, True),
    "timestamp_iso_raises": ({"timestamp": _BadIso()}, 1, None, {}, True),
    "log_fails": ({}, 2, None, {"log_error": RuntimeError("x")}, True),
    "save_fails": ({}, 2, None, {"save_error": RuntimeError("y")}, True),
    "float_trial_number_artifact": ({}, 3.0, None, {}, True),
    "bad_trial_number": ({}, "x", None, {}, True),
    "failed_status": ({"status": TrialStatus.FAILED}, 1, None, {}, True),
}


def _snapshot(name, tmp_path, caplog, monkeypatch):
    kwargs, number, step, wb_kwargs, has_run = CASES[name]
    fake = _FakeWandb(**wb_kwargs)
    monkeypatch.setattr(wandb_module, "wandb", fake)
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    monkeypatch.chdir(tmp_path)
    tracker = wandb_module.TraigentWandBTracker()
    tracker.current_run = SimpleNamespace(id="r") if has_run else None
    caplog.clear()
    try:
        with caplog.at_level(logging.DEBUG, logger=MOD):
            tracker.log_trial(trial(**kwargs), number, step)
    except Exception as exc:  # noqa: BLE001 - the type is the pin
        outcome = type(exc).__name__
    else:
        outcome = "ok"
    logs = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == MOD]
    return {
        "outcome": outcome,
        "logged": repr(fake.logged),
        "saved": repr(fake.saved),
        "logs": repr(logs),
        "leftover": sorted(os.listdir(tmp_path)),
    }


GOLDEN: dict = {
    "full": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, 'trial_1/config/flag': "
        "True, 'trial_1/config/none': None, 'trial_1/config/obj': 'OBJ', "
        "'trial_1/config/l': '[1]', 'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_1/metrics/i': 2, 'metrics/i': 2, 'trial_1/metrics/b': True, "
        "'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, 'config': {'model': "
        "'m', 'n': 3, 'flag': True, 'none': None, 'obj': 'OBJ', 'l': '[1]'}, 'metrics': "
        "{'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, 'status': 'completed', "
        "'timestamp': '2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), ('WARNING', "
        "\"Skipping non-numeric metric 'n' for trial t1\")]",
        "leftover": [],
    },
    "custom_step": {
        "outcome": "ok",
        "logged": "[({'trial_4/status': 'completed', 'trial_4/duration': 1.5, "
        "'trial_4/config/model': 'm', 'trial_4/config/n': 3, "
        "'trial_4/config/flag': True, 'trial_4/config/none': None, "
        "'trial_4/config/obj': 'OBJ', 'trial_4/config/l': '[1]', "
        "'trial_4/metrics/acc': 0.5, 'metrics/acc': 0.5, 'trial_4/metrics/i': "
        "2, 'metrics/i': 2, 'trial_4/metrics/b': True, 'metrics/b': True}, 9)]",
        "saved": "[('trial_4.json', {'trial_id': 't1', 'trial_number': 4, 'config': "
        "{'model': 'm', 'n': 3, 'flag': True, 'none': None, 'obj': 'OBJ', 'l': "
        "'[1]'}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': '2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\")]",
        "leftover": [],
    },
    "no_run": {
        "outcome": "ok",
        "logged": "[]",
        "saved": "[]",
        "logs": "[('WARNING', 'No active W&B run. Call start_optimization_run first.')]",
        "leftover": [],
    },
    "non_mapping_config": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_1/metrics/i': 2, 'metrics/i': 2, 'trial_1/metrics/b': "
        "True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, "
        "'duration': 1.5, 'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', 'Trial t1 config is not a mapping; skipping config "
        "logging'), ('WARNING', \"Skipping non-numeric metric 's' for "
        "trial t1\"), ('WARNING', \"Skipping non-numeric metric 'n' for "
        'trial t1")]',
        "leftover": [],
    },
    "empty_list_config": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_1/metrics/i': 2, 'metrics/i': 2, 'trial_1/metrics/b': "
        "True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, 'config': "
        "{}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', 'Trial t1 config is not a mapping; skipping config "
        "logging'), ('WARNING', \"Skipping non-numeric metric 's' for "
        "trial t1\"), ('WARNING', \"Skipping non-numeric metric 'n' for "
        'trial t1")]',
        "leftover": [],
    },
    "none_config": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, 'trial_1/metrics/i': "
        "2, 'metrics/i': 2, 'trial_1/metrics/b': True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, 'config': {}, "
        "'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, 'status': "
        "'completed', 'timestamp': '2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\")]",
        "leftover": [],
    },
    "non_mapping_metrics": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]'}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {'model': 'm', 'n': 3, 'flag': True, 'none': None, "
        "'obj': 'OBJ', 'l': '[1]'}, 'metrics': {}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', 'Trial t1 metrics is not a mapping; skipping metric "
        "logging')]",
        "leftover": [],
    },
    "empty_dict_metrics": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]'}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {'model': 'm', 'n': 3, 'flag': True, 'none': None, "
        "'obj': 'OBJ', 'l': '[1]'}, 'metrics': {}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[]",
        "leftover": [],
    },
    "zero_metrics_warns": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]'}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {'model': 'm', 'n': 3, 'flag': True, 'none': None, "
        "'obj': 'OBJ', 'l': '[1]'}, 'metrics': {}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', 'Trial t1 metrics is not a mapping; skipping metric "
        "logging')]",
        "leftover": [],
    },
    "no_timestamp": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]', "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, 'trial_1/metrics/i': "
        "2, 'metrics/i': 2, 'trial_1/metrics/b': True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, 'config': "
        "{'model': 'm', 'n': 3, 'flag': True, 'none': None, 'obj': 'OBJ', 'l': "
        "'[1]'}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': None})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\")]",
        "leftover": [],
    },
    "timestamp_without_iso": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]', "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_1/metrics/i': 2, 'metrics/i': 2, 'trial_1/metrics/b': "
        "True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {'model': 'm', 'n': 3, 'flag': True, 'none': None, "
        "'obj': 'OBJ', 'l': '[1]'}, 'metrics': {'acc': 0.5, 'i': 2, "
        "'b': True}, 'duration': 1.5, 'status': 'completed', "
        "'timestamp': None})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial "
        "t1\"), ('WARNING', \"Skipping non-numeric metric 'n' for "
        'trial t1")]',
        "leftover": [],
    },
    "timestamp_iso_raises": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'completed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]', "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_1/metrics/i': 2, 'metrics/i': 2, 'trial_1/metrics/b': "
        "True, 'metrics/b': True}, 1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, "
        "'config': {'model': 'm', 'n': 3, 'flag': True, 'none': None, "
        "'obj': 'OBJ', 'l': '[1]'}, 'metrics': {'acc': 0.5, 'i': 2, "
        "'b': True}, 'duration': 1.5, 'status': 'completed', "
        "'timestamp': None})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial "
        "t1\"), ('WARNING', \"Skipping non-numeric metric 'n' for "
        'trial t1")]',
        "leftover": [],
    },
    "log_fails": {
        "outcome": "ok",
        "logged": "[]",
        "saved": "[('trial_2.json', {'trial_id': 't1', 'trial_number': 2, 'config': "
        "{'model': 'm', 'n': 3, 'flag': True, 'none': None, 'obj': 'OBJ', 'l': "
        "'[1]'}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': '2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\"), "
        "('WARNING', 'Failed to log trial data to W&B: x')]",
        "leftover": [],
    },
    "save_fails": {
        "outcome": "ok",
        "logged": "[({'trial_2/status': 'completed', 'trial_2/duration': 1.5, "
        "'trial_2/config/model': 'm', 'trial_2/config/n': 3, "
        "'trial_2/config/flag': True, 'trial_2/config/none': None, "
        "'trial_2/config/obj': 'OBJ', 'trial_2/config/l': '[1]', "
        "'trial_2/metrics/acc': 0.5, 'metrics/acc': 0.5, 'trial_2/metrics/i': 2, "
        "'metrics/i': 2, 'trial_2/metrics/b': True, 'metrics/b': True}, 2)]",
        "saved": "[]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\"), "
        "('WARNING', 'Failed to save trial artifact to W&B: y')]",
        "leftover": [],
    },
    "float_trial_number_artifact": {
        "outcome": "ok",
        "logged": "[({'trial_3.0/status': 'completed', "
        "'trial_3.0/duration': 1.5, 'trial_3.0/config/model': "
        "'m', 'trial_3.0/config/n': 3, 'trial_3.0/config/flag': "
        "True, 'trial_3.0/config/none': None, "
        "'trial_3.0/config/obj': 'OBJ', 'trial_3.0/config/l': "
        "'[1]', 'trial_3.0/metrics/acc': 0.5, 'metrics/acc': "
        "0.5, 'trial_3.0/metrics/i': 2, 'metrics/i': 2, "
        "'trial_3.0/metrics/b': True, 'metrics/b': True}, 3.0)]",
        "saved": "[('trial_3.json', {'trial_id': 't1', 'trial_number': "
        "3.0, 'config': {'model': 'm', 'n': 3, 'flag': True, "
        "'none': None, 'obj': 'OBJ', 'l': '[1]'}, 'metrics': "
        "{'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'completed', 'timestamp': "
        "'2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for "
        "trial t1\"), ('WARNING', \"Skipping non-numeric metric "
        "'n' for trial t1\")]",
        "leftover": [],
    },
    "bad_trial_number": {
        "outcome": "ok",
        "logged": "[({'trial_x/status': 'completed', 'trial_x/duration': 1.5, "
        "'trial_x/config/model': 'm', 'trial_x/config/n': 3, "
        "'trial_x/config/flag': True, 'trial_x/config/none': None, "
        "'trial_x/config/obj': 'OBJ', 'trial_x/config/l': '[1]', "
        "'trial_x/metrics/acc': 0.5, 'metrics/acc': 0.5, "
        "'trial_x/metrics/i': 2, 'metrics/i': 2, 'trial_x/metrics/b': "
        "True, 'metrics/b': True}, 'x')]",
        "saved": "[]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\"), "
        "('WARNING', \"Failed to write trial artifact for t1: invalid "
        "literal for int() with base 10: 'x'\")]",
        "leftover": [],
    },
    "failed_status": {
        "outcome": "ok",
        "logged": "[({'trial_1/status': 'failed', 'trial_1/duration': 1.5, "
        "'trial_1/config/model': 'm', 'trial_1/config/n': 3, "
        "'trial_1/config/flag': True, 'trial_1/config/none': None, "
        "'trial_1/config/obj': 'OBJ', 'trial_1/config/l': '[1]', "
        "'trial_1/metrics/acc': 0.5, 'metrics/acc': 0.5, 'trial_1/metrics/i': "
        "2, 'metrics/i': 2, 'trial_1/metrics/b': True, 'metrics/b': True}, "
        "1)]",
        "saved": "[('trial_1.json', {'trial_id': 't1', 'trial_number': 1, 'config': "
        "{'model': 'm', 'n': 3, 'flag': True, 'none': None, 'obj': 'OBJ', 'l': "
        "'[1]'}, 'metrics': {'acc': 0.5, 'i': 2, 'b': True}, 'duration': 1.5, "
        "'status': 'failed', 'timestamp': '2025-01-02T03:04:05+00:00'})]",
        "logs": "[('WARNING', \"Skipping non-numeric metric 's' for trial t1\"), "
        "('WARNING', \"Skipping non-numeric metric 'n' for trial t1\")]",
        "leftover": [],
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_log_trial(name, tmp_path, caplog, monkeypatch):
    assert _snapshot(name, tmp_path, caplog, monkeypatch) == GOLDEN[name]
