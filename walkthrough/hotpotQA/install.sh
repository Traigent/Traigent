#!/usr/bin/env bash

set -euo pipefail

if [[ ${TRACE_INSTALL:-0} == 1 ]]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_VENV_DIR="$SCRIPT_DIR/.venv"
ENV_MODE="dedicated"
ACTIVE_ENV_PATH="$DEFAULT_VENV_DIR"
PYTHON_CREATION_BIN=""
PYTHON_EXEC=""
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
HOTPOT_SAMPLE_SLICE="${HOTPOT_SAMPLE_SLICE:-validation[:50]}"
DATASETS_DIR="$ROOT_DIR/paper_experiments/case_study_rag/datasets"
SAMPLE_DATASET_PATH="$DATASETS_DIR/hotpotqa_dev_subset.jsonl"
FULL_DATASET_PATH="$DATASETS_DIR/hotpotqa_distractor_dev.jsonl"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

log() {
    printf '\033[36m[TraiGent HotpotQA]\033[0m %s\n' "$1"
}

warn() {
    printf '\033[33m[TraiGent HotpotQA]\033[0m %s\n' "$1"
}

error_exit() {
    printf '\033[31m[TraiGent HotpotQA] ERROR:\033[0m %s\n' "$1" >&2
    exit 1
}

choose_environment() {
    local detected_env="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"

    if [[ -n "$detected_env" ]]; then
        read -rp "Detected active environment at $detected_env. Use this environment? (y/N) Make sure it is activated before selecting yes: " reply
        if [[ $reply =~ ^[Yy]$ ]]; then
            ENV_MODE="existing"
            ACTIVE_ENV_PATH="$detected_env"
            log "Using currently activated environment at $ACTIVE_ENV_PATH."
            return
        fi
        log "Proceeding with dedicated walkthrough environment at $DEFAULT_VENV_DIR."
        return
    fi

    read -rp "No active virtual environment detected. Create/use dedicated walkthrough environment at $DEFAULT_VENV_DIR? (Y/n). If you prefer an existing environment, answer 'n', activate it (e.g., 'source path/bin/activate'), then rerun this script: " reply
    if [[ $reply =~ ^[Nn]$ ]]; then
        log "Please activate your preferred environment first, then rerun ./install.sh."
        exit 0
    fi
    log "Creating or reusing dedicated walkthrough environment at $DEFAULT_VENV_DIR."
}

verify_python() {
    local interpreter="$1"
    local required_major=3
    local required_minor=10

    if ! command -v "$interpreter" >/dev/null 2>&1; then
        error_exit "Interpreter '$interpreter' not found. Activate your environment or install Python ${required_major}.${required_minor}+."
    fi

    if ! "$interpreter" -c "import sys; sys.exit(0 if sys.version_info >= ($required_major, $required_minor) else 1)" >/dev/null 2>&1; then
        local version
        version="$("$interpreter" -V 2>&1 || true)"
        error_exit "Python ${required_major}.${required_minor}+ is required (found $version)."
    fi

    log "Using interpreter: $("$interpreter" -c 'import sys; print(sys.executable)')"
}

find_python_candidate() {
    local candidates=("python3" "python")
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
            PYTHON_CREATION_BIN="$candidate"
            log "Python interpreter detected for venv creation: $("$candidate" -c 'import sys; print(sys.executable)')"
            return
        fi
    done
    error_exit "Python 3.10+ is required to create the walkthrough environment."
}

ensure_virtualenv() {
    if [[ "$ENV_MODE" != "dedicated" ]]; then
        return
    fi

    if [[ -d "$DEFAULT_VENV_DIR" ]]; then
        log "Virtual environment already exists at $DEFAULT_VENV_DIR (skipping creation)."
    else
        log "Creating virtual environment at $DEFAULT_VENV_DIR"
        "$PYTHON_CREATION_BIN" -m venv "$DEFAULT_VENV_DIR"
    fi

    if [[ ! -x "$DEFAULT_VENV_DIR/bin/python" ]]; then
        error_exit "Virtual environment is missing a python executable at $DEFAULT_VENV_DIR/bin/python."
    fi
    ACTIVE_ENV_PATH="$DEFAULT_VENV_DIR"
}

install_dependencies() {
    local python_exec="$PYTHON_EXEC"

    if [[ "$FORCE_REINSTALL" != "1" ]] && "$python_exec" -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('traigent') else 1)" >/dev/null 2>&1; then
        log "TraiGent is already installed in the selected environment (set FORCE_REINSTALL=1 to force reinstall)."
        return
    fi

    log "Installing TraiGent dependencies (this may take a few minutes)..."
    log "Equivalent manual commands:"
    log "  pip install -r \"$REQUIREMENTS_FILE\""
    log "  pip install -e \"$ROOT_DIR\""
    "$python_exec" -m pip install --upgrade pip setuptools wheel
    "$python_exec" -m pip install -r "$REQUIREMENTS_FILE"
    "$python_exec" -m pip install -e "$ROOT_DIR"
    log "Dependencies installed successfully."
}

ensure_dataset() {
    mkdir -p "$DATASETS_DIR"

    if [[ -f "$FULL_DATASET_PATH" ]]; then
        log "Full HotpotQA dataset already present at $FULL_DATASET_PATH."
        return
    fi

    if [[ -f "$SAMPLE_DATASET_PATH" ]]; then
        log "Sample HotpotQA dataset already present at $SAMPLE_DATASET_PATH."
        return
    fi

    local python_exec="$PYTHON_EXEC"

    log "HotpotQA dataset not found. Generating a sample slice ($HOTPOT_SAMPLE_SLICE)."

    if ! "$python_exec" -m pip show datasets >/dev/null 2>&1; then
        log "Installing Hugging Face 'datasets' package for sample generation..."
        "$python_exec" -m pip install "datasets>=2.18.0"
    fi

    HOTPOT_OUTPUT_PATH="$SAMPLE_DATASET_PATH" \
    HOTPOT_SAMPLE_SLICE="$HOTPOT_SAMPLE_SLICE" \
    "$python_exec" <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

target = Path(os.environ["HOTPOT_OUTPUT_PATH"])
slice_expr = os.environ.get("HOTPOT_SAMPLE_SLICE", "validation[:50]")

target.parent.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("hotpot_qa", "distractor", split=slice_expr)

with target.open("w", encoding="utf-8") as handle:
    for row in dataset:
        paragraphs = []
        for title, sentences in row.get("context", []):
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue
            paragraph = f"{title}: {' '.join(sentences)}"
            paragraphs.append(paragraph)

        record = {
            "input": {
                "question": row.get("question", ""),
                "context": paragraphs[:10],
            },
            "output": row.get("answer", ""),
            "metadata": {
                "difficulty": row.get("level", "unknown"),
                "answer_aliases": row.get("answer_aliases", []),
                "supporting_facts": row.get("supporting_facts", []),
                "fallback_answer": row.get("answer", "I am not sure."),
                "topic": row.get("type", "unknown"),
            },
        }
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")

print(f"Generated sample HotpotQA dataset at {target}")
PY
    log "Sample HotpotQA dataset generated at $SAMPLE_DATASET_PATH"
}

summarize() {
    if [[ "$ENV_MODE" == "dedicated" ]]; then
        cat <<EOF

Setup complete! Next steps:

1. Activate the virtual environment:
   source "$ACTIVE_ENV_PATH/bin/activate"

2. Run the HotpotQA walkthrough:
   "$SCRIPT_DIR/run_demo.sh"

Need to reinstall manually somewhere else? Run:
pip install -r "$REQUIREMENTS_FILE"
pip install -e "$ROOT_DIR"

Use FORCE_REINSTALL=1 ./install.sh to force dependency reinstall.
Adjust HOTPOT_SAMPLE_SLICE="validation[:20]" ./install.sh to change sample size.
EOF
    else
        cat <<EOF

Setup complete! The script used your active environment at $ACTIVE_ENV_PATH.

Feel free to run the HotpotQA walkthrough:
"$SCRIPT_DIR/run_demo.sh"

Need to mirror this setup elsewhere? Run:
pip install -r "$REQUIREMENTS_FILE"
pip install -e "$ROOT_DIR"

Use FORCE_REINSTALL=1 ./install.sh to force dependency reinstall.
Adjust HOTPOT_SAMPLE_SLICE="validation[:20]" ./install.sh to change sample size.
EOF
    fi
}

choose_environment

if [[ "$ENV_MODE" == "dedicated" ]]; then
    find_python_candidate
    ensure_virtualenv
    PYTHON_EXEC="$ACTIVE_ENV_PATH/bin/python"
    verify_python "$PYTHON_EXEC"
else
    verify_python "python"
    PYTHON_EXEC="python"
fi

install_dependencies
ensure_dataset
summarize
