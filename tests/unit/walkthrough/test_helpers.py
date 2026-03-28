from __future__ import annotations

from pathlib import Path

import pytest

from walkthrough.utils.helpers import maybe_run_mock_example


def _create_example_tree(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    real_dir = tmp_path / "walkthrough" / "real"
    mock_dir = tmp_path / "walkthrough" / "mock"
    real_dir.mkdir(parents=True)
    mock_dir.mkdir(parents=True)
    example_path = real_dir / "01_tuning_qa.py"
    example_path.write_text("print('real')\n")
    (mock_dir / "01_tuning_qa.py").write_text("print('mock')\n")
    return example_path


def test_real_examples_fail_without_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        maybe_run_mock_example(str(example_path))

    message = str(exc_info.value)
    assert "OPENAI_API_KEY environment variable is required" in message
    assert "python walkthrough/real/01_tuning_qa.py" in message
    assert "python walkthrough/mock/01_tuning_qa.py" in message


def test_real_examples_reject_mock_fallback_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    with pytest.raises(SystemExit) as exc_info:
        maybe_run_mock_example(str(example_path))

    message = str(exc_info.value)
    assert "do not fall back to mock mode" in message
    assert "python walkthrough/mock/01_tuning_qa.py" in message


def test_real_examples_continue_when_provider_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example_path = _create_example_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    maybe_run_mock_example(str(example_path))
