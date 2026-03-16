#!/usr/bin/env bash
# Interactive launcher for Traigent demo scenarios.

set -u
IFS=$'\n\t'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables from .env if present.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# Activate repo virtual environment when not already active.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# NOTE: Paper experiments (FEVER, HotpotQA) moved to TraigentDemo repository
# See: https://github.com/Traigent/TraigentDemo

scenarios=(
  "RAG Optimization – RAG tuner"
  "Prompt A/B tradeoff"
  "Structured JSON extractor"
)

descriptions=(
  "Compare RAG vs. no-RAG and top_k choices on a simple QA task."
  "Grid-test prompt variants, models, and temperatures with parallel trials."
  "Enforce JSON structure while optimizing a custom json_score metric."
)

commands=(
  "python examples/core/rag-optimization/run.py"
  "python examples/core/prompt-ab-test/run.py"
  "python examples/core/structured-output-json/run.py"
)

real_flags=(
  ""
  ""
  ""
)

dataset_paths=(
  "${PROJECT_ROOT}/examples/datasets/rag-optimization/evaluation_set.jsonl"
  "${PROJECT_ROOT}/examples/datasets/prompt-ab-test/evaluation_set.jsonl"
  "${PROJECT_ROOT}/examples/datasets/structured-output-json/evaluation_set.jsonl"
)

snippet_paths=(
  "${PROJECT_ROOT}/examples/core/rag-optimization/run.py"
  "${PROJECT_ROOT}/examples/core/prompt-ab-test/run.py"
  "${PROJECT_ROOT}/examples/core/structured-output-json/run.py"
)

print_menu() {
  echo "Traigent Demo Scenarios"
  echo "-----------------------"
  local idx
  for idx in "${!scenarios[@]}"; do
    printf "  %d) %s\n     %s\n" "$((idx + 1))" "${scenarios[idx]}" "${descriptions[idx]}"
  done
  echo "  q) Exit"
  echo
}

scenario_detail() {
  case "$1" in
    0)
      cat <<'EOF'
Explore how retrieval depth influences accuracy and cost on a lightweight QA workload.
The optimizer toggles RAG on/off, adjusts top_k, and keeps temperatures fixed so you
can see the effect of injecting context versus flying without knowledge retrieval.
EOF
      ;;
    1)
      cat <<'EOF'
Run a structured A/B test that sweeps prompt variants, models, and temperatures in
parallel. Useful when a team is debating copy tweaks or model swaps and wants a
single accuracy leaderboard plus cost and latency telemetry.
EOF
      ;;
    2)
      cat <<'EOF'
Demonstrates seamless injection with a custom json_score metric so you can measure
field-level extraction accuracy while nudging the LLM toward particular formatting
guidelines. Great for downstream API integrations that reject malformed payloads.
EOF
      ;;
  esac
  printf '\nDataset URI: file://%s\n' "${dataset_paths[$1]}"
  printf 'Decorator source: file://%s\n' "${snippet_paths[$1]}"
}

scenario_snippet() {
  case "$1" in
    0)
      cat <<'EOF'
@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["cost"],
    configuration_space={
        "model": ["claude-3-haiku-20240307"],
        "temperature": [0.0],
        "use_rag": [True, False],
        "top_k": [1, 2, 3],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    ...
EOF
      ;;
    1)
      cat <<'EOF'
@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "prompt_variant": PROMPT_VARIANTS,
        "model": MODEL_CHOICES,
        "temperature": TEMPERATURE_CHOICES,
    },
    execution_mode="edge_analytics",
    algorithm="grid",
    parallel_config=GLOBAL_PARALLEL_CONFIG,
)
def qa_with_variant(question: str) -> str:
    ...
EOF
      ;;
    2)
      cat <<'EOF'
@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["json_score"],
    configuration_space={
        "temperature": [0.0, 0.2],
        "format_hint": ["strict_json", "relaxed_json"],
        "schema_rigidity": ["strict", "lenient"],
    },
    metric_functions={"json_score": json_score_metric},
    execution_mode="edge_analytics",
    injection_mode="seamless",
    algorithm="grid",
)
def extract_fields(
    text: str,
    temperature: float = 0.0,
    format_hint: str = "strict_json",
    schema_rigidity: str = "strict",
) -> str:
    ...
EOF
      ;;
  esac
}

post_run_navigation() {
  while true; do
    echo
    read -rp "Next action: [b] back to scenarios, [e] exit > " nav_choice
    case "${nav_choice,,}" in
      b)
        return 0
        ;;
      e)
        echo "Goodbye!"
        exit 0
        ;;
      *)
        echo "Please choose 'b' or 'e'."
        ;;
    esac
  done
}

run_selected_scenario() {
  local idx="$1"
  local mode="$2"
  local cmd="${commands[idx]}"
  local extra=""
  if [[ "$mode" == "real" ]]; then
    extra="${real_flags[idx]}"
  fi
  if [[ -n "$extra" ]]; then
    cmd+=" ${extra}"
  fi

  echo
  if [[ "$mode" == "mock" ]]; then
    echo "▶ Running in mock mode (no external API calls)."
    (
      cd "$PROJECT_ROOT" >/dev/null 2>&1
      TRAIGENT_MOCK_LLM=1 bash -lc "$cmd"
    )
  else
    echo "▶ Running in real mode. Ensure provider API keys are configured."
    (
      cd "$PROJECT_ROOT" >/dev/null 2>&1
      TRAIGENT_MOCK_LLM=0 bash -lc "$cmd"
    )
  fi

  local status=$?
  echo
  if [[ "$status" -eq 0 ]]; then
    echo "✅ Demo complete. View results in Traigent web or inspect generated artifacts."
  else
    echo "⚠️  Demo exited with status ${status}. Check logs above for details."
  fi
  post_run_navigation
}

main_loop() {
  while true; do
    print_menu
    read -rp "Select a scenario by number (or q to exit) > " choice
    echo

    case "${choice,,}" in
      q)
        echo "Goodbye!"
        exit 0
        ;;
      *)
        if [[ "$choice" =~ ^[0-9]+$ ]]; then
          local idx=$((choice - 1))
          if (( idx >= 0 && idx < ${#scenarios[@]} )); then
            echo "${scenarios[idx]}"
            echo "----------------------------------------"
            scenario_detail "$idx"
            echo
            echo "Optimized function:"
            echo "-------------------"
            scenario_snippet "$idx"
            echo
            while true; do
              echo "[1] Run mock"
              echo "[2] Run real"
              echo "[3] Back to scenarios"
              echo "[4] Exit"
              read -rp "Choose an action > " action
              echo
              case "$action" in
                1)
                  run_selected_scenario "$idx" "mock"
                  break
                  ;;
                2)
                  run_selected_scenario "$idx" "real"
                  break
                  ;;
                3)
                  break
                  ;;
                4)
                  echo "Goodbye!"
                  exit 0
                  ;;
                *)
                  echo "Please enter 1, 2, 3, or 4."
                  ;;
              esac
            done
          else
            echo "Invalid selection. Please choose a listed scenario."
          fi
        else
          echo "Please enter a number or 'q'."
        fi
        ;;
    esac
    echo
  done
}

main_loop
