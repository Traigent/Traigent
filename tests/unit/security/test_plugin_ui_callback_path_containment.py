"""Path-containment regression tests for traigent-ui optimization callbacks.

Aikido SAST "Potential file inclusion via reading file" flagged the
``InvocationLoggingCallback`` (invocations/metrics writes) and
``ComprehensiveTrialLoggingCallback`` (optimization_log / llm_usage_summary
writes): ``run_id`` and ``problem_id`` were joined onto ``base_dir`` without
validation, so an identifier such as ``../../x`` or ``/abs`` steered writes
outside the log root. These tests pin the fail-closed behaviour.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "plugins" / "traigent-ui" / "traigent_ui" / "optimization_callbacks.py"
)


@pytest.fixture(scope="module")
def callbacks():
    spec = importlib.util.spec_from_file_location(
        "_test_plugin_ui_optimization_callbacks", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# "ABS" is replaced with an absolute path under tmp_path so a regression
# never writes outside the test sandbox.
TRAVERSAL_IDS = ["../escape", "../../escape", "a/../../escape", "ABS", ""]


def _abs(value: str, tmp_path: Path) -> str:
    return str(tmp_path / "abs-escape") if value == "ABS" else value


@pytest.mark.parametrize("run_id", TRAVERSAL_IDS)
def test_invocation_callback_rejects_run_id_escaping_base(callbacks, tmp_path, run_id):
    base = tmp_path / "nest" / "logs"
    run_id = _abs(run_id, tmp_path)
    with pytest.raises(PathTraversalError):
        callbacks.InvocationLoggingCallback(run_id=run_id, base_dir=base)
    # Nothing was created outside the base directory.
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "nest" / "escape").exists()
    assert not (tmp_path / "abs-escape").exists()


def test_invocation_callback_accepts_normal_run_id(callbacks, tmp_path):
    base = tmp_path / "logs"
    cb = callbacks.InvocationLoggingCallback(run_id="run_123", base_dir=base)
    assert cb.log_dir == (base / "run_123").resolve()
    assert cb.log_dir.is_dir()
    assert cb.invocations_file.parent == cb.log_dir
    assert cb.metrics_file.parent == cb.log_dir


def test_invocation_callback_rejects_symlink_escape(callbacks, tmp_path):
    base = tmp_path / "logs"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathTraversalError):
        callbacks.InvocationLoggingCallback(run_id="link", base_dir=base)


@pytest.mark.parametrize(
    ("problem_id", "run_id"),
    [
        ("../escape", "run"),
        ("problem", "../../escape"),
        ("ABS", "run"),
        ("problem", "ABS"),
        ("", "run"),
    ],
)
def test_comprehensive_callback_rejects_ids_escaping_base(
    callbacks, tmp_path, problem_id, run_id
):
    with pytest.raises(PathTraversalError):
        callbacks.ComprehensiveTrialLoggingCallback(
            run_id=_abs(run_id, tmp_path),
            problem_id=_abs(problem_id, tmp_path),
            base_dir=tmp_path / "logs",
        )


def test_comprehensive_callback_writes_inside_base(callbacks, tmp_path):
    base = tmp_path / "logs"
    cb = callbacks.ComprehensiveTrialLoggingCallback(
        run_id="run_1", problem_id="problem_a", base_dir=base
    )
    cb._save_comprehensive_logs()
    log_dir = (base / "problem_a" / "run_1").resolve()
    assert cb.log_dir == log_dir
    assert (log_dir / "optimization_log.json").is_file()
    assert (log_dir / "llm_usage_summary.json").is_file()


# --- DeferredLoggingCallback / EnhancedDeferredLoggingCallback ------------

ID_PAIR_CLASSES = ["DeferredLoggingCallback", "EnhancedDeferredLoggingCallback"]
ESCAPING_PAIRS = [
    ("../escape", "run"),
    ("problem", "../../escape"),
    ("a/../../escape", "run"),
    ("ABS", "run"),
    ("problem", "ABS"),
    ("problem", ""),
]


@pytest.mark.parametrize("cls_name", ID_PAIR_CLASSES)
@pytest.mark.parametrize(("problem_id", "run_id"), ESCAPING_PAIRS)
def test_deferred_callbacks_reject_ids_escaping_base(
    callbacks, tmp_path, cls_name, problem_id, run_id
):
    with pytest.raises(PathTraversalError):
        getattr(callbacks, cls_name)(
            run_id=_abs(run_id, tmp_path),
            problem_id=_abs(problem_id, tmp_path),
            base_dir=tmp_path / "nest" / "logs",
        )
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "nest" / "escape").exists()
    assert not (tmp_path / "abs-escape").exists()


@pytest.mark.parametrize("cls_name", ID_PAIR_CLASSES)
def test_deferred_callbacks_reject_symlink_escape(callbacks, tmp_path, cls_name):
    base = tmp_path / "logs"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathTraversalError):
        getattr(callbacks, cls_name)(run_id="run", problem_id="link", base_dir=base)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("cls_name", "save"),
    [
        ("DeferredLoggingCallback", "_save_logs"),
        ("EnhancedDeferredLoggingCallback", "_save_enhanced_logs"),
    ],
)
def test_deferred_callbacks_write_inside_base(callbacks, tmp_path, cls_name, save):
    base = tmp_path / "logs"
    cb = getattr(callbacks, cls_name)(
        run_id="run_1", problem_id="problem_a", base_dir=base
    )
    getattr(cb, save)()
    log_dir = (base / "problem_a" / "run_1").resolve()
    assert cb.log_dir == log_dir
    assert (log_dir / "optimization_log.json").is_file()


def test_deferred_callback_none_problem_id_still_works(callbacks, tmp_path):
    base = tmp_path / "logs"
    cb = callbacks.DeferredLoggingCallback(run_id="run_1", base_dir=base)
    assert cb.problem_id == "unknown_problem"
    cb._save_logs()
    log_dir = (base / "unknown_problem" / "run_1").resolve()
    assert cb.log_dir == log_dir
    assert (log_dir / "optimization_log.json").is_file()
