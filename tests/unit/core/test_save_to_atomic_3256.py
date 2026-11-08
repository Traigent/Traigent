"""#3256: ``save_to`` must not truncate the previous good file on a failed write."""

from __future__ import annotations

import copy
import json
import os

import pytest

from traigent.api.types import OptimizationResult
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.optimization_result_persistence import _SENTINELS


def _manager(tmp_path) -> ConfigStateManager:
    def _fn(x):
        return x

    return ConfigStateManager(
        func=_fn,
        default_config={"model": "cheap"},
        local_storage_path=str(tmp_path / "store"),
        configuration_space={"model": ["cheap", "smart"]},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
    )


def _save(tmp_path, path):
    saver = _manager(tmp_path)
    saver._optimization_results = OptimizationResult(**copy.deepcopy(_SENTINELS))
    saver.save_optimization_results(str(path))


def test_crash_mid_write_keeps_previous_result(tmp_path, monkeypatch):
    path = tmp_path / "out.json"
    _save(tmp_path, path)
    good = path.read_bytes()
    json.loads(good)

    def _crash(*_a, **_k):
        raise OSError(28, "No space left on device")

    # the crash happens after the file was opened, i.e. mid-write
    monkeypatch.setattr(os, "fsync", _crash)
    with pytest.raises(Exception):  # noqa: B017 - refusal type is not the point
        _save(tmp_path, path)
    monkeypatch.undo()

    assert path.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


def test_serialization_error_mid_dump_keeps_previous_result(tmp_path, monkeypatch):
    path = tmp_path / "out.json"
    _save(tmp_path, path)
    good = path.read_bytes()

    import traigent.core.config_state_manager as csm

    real_dump = csm.json.dump

    def _partial_dump(obj, f, **kw):
        f.write('{"half": ')
        raise RuntimeError("boom")

    monkeypatch.setattr(csm.json, "dump", _partial_dump)
    with pytest.raises(Exception):  # noqa: B017
        _save(tmp_path, path)
    monkeypatch.setattr(csm.json, "dump", real_dump)

    assert path.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_overwrite_keeps_restrictive_permissions(tmp_path):
    path = tmp_path / "out.json"
    _save(tmp_path, path)
    path.chmod(0o600)
    _save(tmp_path, path)
    assert (path.stat().st_mode & 0o777) == 0o600
