#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE_DEFAULT="/home/nimrodbu/Traigent_enterprise/Traigent/walkthrough/examples/real/.env"

if [[ -z "${TRAIGENT_BACKEND_URL:-}" && -z "${TRAIGENT_API_URL:-}" && -f "$ENV_FILE_DEFAULT" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE_DEFAULT"
  set +a
fi

if [[ -z "${TRAIGENT_BACKEND_URL:-}" && -z "${TRAIGENT_API_URL:-}" ]]; then
  echo "[error] Set TRAIGENT_BACKEND_URL or TRAIGENT_API_URL."
  exit 1
fi

if [[ -z "${TRAIGENT_API_KEY:-}" ]]; then
  echo "[error] Set TRAIGENT_API_KEY."
  exit 1
fi

API_BASE="${TRAIGENT_API_URL:-${TRAIGENT_BACKEND_URL%/}/api/v1}"

probe_typed_backend() {
  local response
  response="$(curl -sS -m 10 \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${TRAIGENT_API_KEY}" \
    -H "Authorization: Bearer ${TRAIGENT_API_KEY}" \
    -X POST "${API_BASE%/}/sessions" \
    -d '{"function_name":"showcase_probe","configuration_space":{"temperature":{"type":"categorical","choices":[0,0.2]}},"objectives":["accuracy"],"dataset_metadata":{"size":1},"max_trials":0,"optimization_strategy":{"algorithm":"optuna"}}' || true)"

  if [[ "$response" == *"problem_statement"* && "$response" == *"search_space"* ]]; then
    echo "[error] ${API_BASE%/}/sessions is still serving the legacy contract."
    echo "[error] Point TRAIGENT_BACKEND_URL / TRAIGENT_API_URL at the typed interactive session backend before running this showcase."
    exit 1
  fi

  if [[ "$response" == *"AUTHENTICATION_REQUIRED"* || "$response" == *"Authentication required"* ]]; then
    echo "[error] ${API_BASE%/}/sessions rejected the configured TRAIGENT_API_KEY."
    echo "[error] Update TRAIGENT_API_KEY or point the showcase at a backend that accepts the current key."
    exit 1
  fi

  if [[ "$response" == *"optuna version"* || "$response" == *"INTERNAL_ERROR"* ]]; then
    echo "[error] ${API_BASE%/}/sessions is reachable, but the backend returned an internal typed-session error."
    echo "[error] Current response snippet:"
    echo "$response" | sed -n '1,12p'
    exit 1
  fi
}

PROVIDER_NOTE="OpenAI fallback"
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  PROVIDER_NOTE="OpenRouter free model (${OPENROUTER_FREE_MODEL:-meta-llama/llama-3.3-8b-instruct:free})"
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY."
  exit 1
fi

SECTIONS=(
  "1|Seamless OpenAI + env auth|Backend-guided Optuna with seamless wrapper overrides, env-based Traigent auth, and status/finalize helper evidence.|$ROOT/examples/core/online-showcase/01_seamless_openai_env.mjs"
  "2|Context injection + explicit backend options|Uses getTrialParam() inside the agent prompt and passes backendUrl/apiKey explicitly.|$ROOT/examples/core/online-showcase/02_context_explicit.mjs"
  "3|Parameter injection + custom evaluator|Passes config as the second arg, uses weighted objectives, and custom evaluation metrics.|$ROOT/examples/core/online-showcase/03_parameter_custom_evaluator.mjs"
  "4|Seamless LangChain + session helpers|Runs a wrapped LangChain model through hybrid mode and exercises status/finalize reporting helpers.|$ROOT/examples/core/online-showcase/04_seamless_langchain_session.mjs"
  "5|Constraints + full reporting|Shows hybrid-only conditional params, structural constraints, and full-history reporting.|$ROOT/examples/core/online-showcase/05_constraints_reporting.mjs"
)

echo
echo "Interactive Traigent JS online showcase"
echo "Backend: ${TRAIGENT_BACKEND_URL:-${TRAIGENT_API_URL}}"
echo "Provider: ${PROVIDER_NOTE}"
echo "Trials per section: ${TRAIGENT_SHOWCASE_MAX_TRIALS:-5}"
echo "Examples per section: ${TRAIGENT_SHOWCASE_DATASET_SIZE:-10}"
echo "Delete after section: ${TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION:-0}"
echo

probe_typed_backend

cd "$ROOT"
npm run build:sdk >/dev/null

for entry in "${SECTIONS[@]}"; do
  IFS='|' read -r id title description code_path <<<"$entry"
  echo "================================================================"
  echo "Section ${id}: ${title}"
  echo "${description}"
  echo "Code: ${code_path}"
  echo
  read -r -p "Press Enter to run this section..."
  node "$ROOT/scripts/demo/interactive_online_showcase.mjs" --section "$id"
  echo
  if [[ "$id" != "5" ]]; then
    read -r -p "Press Enter to continue to the next section..."
  fi
done

echo
echo "Showcase complete."
echo "If TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION=0, the sessions are still available on the backend for FE inspection."
