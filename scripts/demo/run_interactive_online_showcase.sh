#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE_DEFAULT="/home/nimrodbu/Traigent_enterprise/Traigent/walkthrough/examples/real/.env"
ENV_FILE="${TRAIGENT_SHOWCASE_ENV_FILE:-$ENV_FILE_DEFAULT}"
USE_ENV_FILE="${TRAIGENT_SHOWCASE_USE_ENV_FILE:-1}"
RESPECT_CURRENT_TRAIGENT_ENV="${TRAIGENT_SHOWCASE_RESPECT_CURRENT_TRAIGENT_ENV:-0}"
STRICT_CREATE_PROBE="${TRAIGENT_SHOWCASE_STRICT_CREATE_PROBE:-0}"

load_demo_env() {
  if [[ "$USE_ENV_FILE" == "0" || ! -f "$ENV_FILE" ]]; then
    return
  fi

  local saved_openai_api_key="${OPENAI_API_KEY:-}"
  local saved_openrouter_api_key="${OPENROUTER_API_KEY:-}"
  local saved_openrouter_free_model="${OPENROUTER_FREE_MODEL:-}"

  if [[ "$RESPECT_CURRENT_TRAIGENT_ENV" != "1" ]]; then
    if [[ -n "${TRAIGENT_BACKEND_URL:-}${TRAIGENT_API_URL:-}${TRAIGENT_API_KEY:-}" ]]; then
      echo "[info] Overriding current TRAIGENT_* environment from ${ENV_FILE} for the local showcase."
    fi
    unset TRAIGENT_BACKEND_URL TRAIGENT_API_URL TRAIGENT_API_KEY
  fi

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  if [[ -n "$saved_openai_api_key" ]]; then
    export OPENAI_API_KEY="$saved_openai_api_key"
  fi
  if [[ -n "$saved_openrouter_api_key" ]]; then
    export OPENROUTER_API_KEY="$saved_openrouter_api_key"
  fi
  if [[ -n "$saved_openrouter_free_model" ]]; then
    export OPENROUTER_FREE_MODEL="$saved_openrouter_free_model"
  fi
}

load_demo_env

if [[ "$USE_ENV_FILE" != "0" && -f "$ENV_FILE" ]]; then
  echo "[info] Using showcase env file: ${ENV_FILE}"
fi

resolve_openrouter_free_model() {
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    return
  fi

  local requested_model="${OPENROUTER_FREE_MODEL:-}"
  local resolved_model
  resolved_model="$(python3 - <<'PY'
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

api_key = os.environ.get("OPENROUTER_API_KEY", "")
requested = os.environ.get("OPENROUTER_FREE_MODEL", "").strip()
preferred = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

if not api_key:
    sys.exit(0)

base_headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://traigent.ai",
    "X-Title": "Traigent JS Interactive Showcase",
}

request = urllib.request.Request("https://openrouter.ai/api/v1/models", headers=base_headers)

try:
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode())
except Exception:
    if requested:
        print(requested)
    sys.exit(0)

available = {
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict) and isinstance(item.get("id"), str)
}
free_models = [model for model in available if model.endswith(":free")]

def can_complete(model: str) -> bool:
    payload = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 24,
        "messages": [
            {
                "role": "system",
                "content": "You are a deterministic evaluator. Reply with the exact uppercase token only. No punctuation, no explanation.",
            },
            {
                "role": "user",
                "content": "Reply with exactly this uppercase token and nothing else: ALPHA",
            },
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={**base_headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False

if requested and requested in available and can_complete(requested):
    print(requested)
    sys.exit(0)

for candidate in preferred:
    if candidate in available and can_complete(candidate):
        print(candidate)
        sys.exit(0)

for candidate in sorted(free_models):
    if can_complete(candidate):
        print(candidate)
        sys.exit(0)

if os.environ.get("OPENAI_API_KEY", "").strip():
    print("__fallback_openai__")
elif requested:
    print(requested)
PY
)"

  if [[ "$resolved_model" == "__fallback_openai__" ]]; then
    echo "[info] No working OpenRouter free model is currently available; falling back to OpenAI."
    unset OPENROUTER_API_KEY OPENROUTER_FREE_MODEL
    return
  fi

  if [[ -n "$resolved_model" ]]; then
    if [[ -n "$requested_model" && "$requested_model" != "$resolved_model" ]]; then
      echo "[info] OPENROUTER_FREE_MODEL=${requested_model} is unavailable; using ${resolved_model}."
    elif [[ -z "$requested_model" ]]; then
      echo "[info] Selected available OpenRouter free model: ${resolved_model}"
    fi
    export OPENROUTER_FREE_MODEL="$resolved_model"
  fi
}

if [[ -z "${TRAIGENT_BACKEND_URL:-}" && -z "${TRAIGENT_API_URL:-}" ]]; then
  echo "[error] Set TRAIGENT_BACKEND_URL or TRAIGENT_API_URL."
  exit 1
fi

if [[ -z "${TRAIGENT_API_KEY:-}" ]]; then
  echo "[error] Set TRAIGENT_API_KEY."
  exit 1
fi

resolve_openrouter_free_model

API_BASE="${TRAIGENT_API_URL:-${TRAIGENT_BACKEND_URL%/}/api/v1}"

probe_typed_backend() {
  SHOWCASE_API_BASE="${API_BASE%/}" SHOWCASE_STRICT_CREATE_PROBE="${STRICT_CREATE_PROBE}" python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

api_base = os.environ["SHOWCASE_API_BASE"].rstrip("/")
api_key = os.environ["TRAIGENT_API_KEY"]
strict_create_probe = os.environ.get("SHOWCASE_STRICT_CREATE_PROBE", "0") == "1"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key,
    "Authorization": f"Bearer {api_key}",
}


def post(path: str, payload: dict, timeout: int = 12):
    request = urllib.request.Request(
        f"{api_base}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def delete(path: str, timeout: int = 12):
    request = urllib.request.Request(
        f"{api_base}{path}",
        headers=headers,
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def fail(title: str, detail: str):
    print(f"[error] {title}")
    print(detail)
    print("[error] No session was created, so nothing will appear in the FE.")
    sys.exit(1)


def extract_session_id(raw: str):
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("session_id") or payload.get("sessionId")


invalid_probe = {
    "function_name": "showcase_probe",
    "configuration_space": {
        "temperature": {"type": "categorical", "choices": [0, 0.2]}
    },
    "objectives": ["accuracy"],
    "dataset_metadata": {"size": 1},
    "max_trials": 0,
    "optimization_strategy": {"algorithm": "optuna"},
}

status, body = post("/sessions", invalid_probe)

if "problem_statement" in body and "search_space" in body:
    fail(
        f"{api_base}/sessions is still serving the legacy contract.",
        "[error] Point TRAIGENT_BACKEND_URL / TRAIGENT_API_URL at the typed interactive session backend before running this showcase.",
    )

if "AUTHENTICATION_REQUIRED" in body or "Authentication required" in body or status == 401:
    fail(
        f"{api_base}/sessions rejected the configured TRAIGENT_API_KEY.",
        "[error] Update TRAIGENT_API_KEY or point the showcase at a backend that accepts the current key.",
    )

if not strict_create_probe:
    if status >= 500 or "INTERNAL_ERROR" in body or "optuna version" in body:
        snippet = "\n".join(body.splitlines()[:12])
        fail(
            f"{api_base}/sessions is reachable, but the backend validation probe hit an internal error.",
            f"[error] Current response snippet:\n{snippet}",
        )
    session_id = extract_session_id(body)
    if session_id:
        delete(f"/sessions/{session_id}?cascade=true", timeout=10)
    sys.exit(0)

create_probe = dict(invalid_probe)
create_probe["max_trials"] = 1

status, body = post("/sessions", create_probe, timeout=20)

if status >= 500 or "INTERNAL_ERROR" in body or "optuna version" in body:
    snippet = "\n".join(body.splitlines()[:12])
    fail(
        f"{api_base}/sessions is reachable, but the backend cannot create typed sessions.",
        f"[error] Current response snippet:\n{snippet}",
    )

if status >= 400:
    snippet = "\n".join(body.splitlines()[:12])
    fail(
        f"{api_base}/sessions rejected the create probe with HTTP {status}.",
        f"[error] Current response snippet:\n{snippet}",
    )

session_id = None
try:
    payload = json.loads(body)
    session_id = payload.get("session_id") or payload.get("sessionId")
except Exception:
    session_id = None

if session_id:
    delete(f"/sessions/{session_id}?cascade=true", timeout=10)
PY
}

run_section_with_retry() {
  local id="$1"
  local attempt=1
  local max_attempts=3
  local output_file
  output_file="$(mktemp)"
  trap 'rm -f "$output_file"' RETURN

  while true; do
    node "$ROOT/scripts/demo/interactive_online_showcase.mjs" --section "$id" >"$output_file"

    local parsed
    parsed="$(
      python3 - "$output_file" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    try:
        payload = json.load(handle)
    except Exception:
        print("")
        raise SystemExit(0)

result = payload.get("result") if isinstance(payload, dict) else None
if not isinstance(result, dict):
    print("|")
    raise SystemExit(0)

stop_reason = result.get("stopReason")
if stop_reason != "error":
    print(f"{stop_reason or ''}|")
    raise SystemExit(0)

message = result.get("errorMessage")
if not isinstance(message, str) or "rate_limit_exceeded" not in message:
    print(f"{stop_reason or ''}|")
    raise SystemExit(0)

match = re.search(r'"retry_after"\s*:\s*(\d+)', message)
print(f"{stop_reason or ''}|{match.group(1) if match else '15'}")
PY
    )"
    local stop_reason="${parsed%%|*}"
    local retry_after="${parsed#*|}"

    if [[ -z "$retry_after" || "$attempt" -ge "$max_attempts" ]]; then
      cat "$output_file"
      if [[ "$stop_reason" == "error" ]]; then
        return 1
      fi
      return 0
    fi

    echo
    echo "[warn] Backend session rate limit hit for section ${id}. Retrying in ${retry_after}s (attempt ${attempt}/${max_attempts})."
    sleep "$retry_after"
    attempt=$((attempt + 1))
  done
}

PROVIDER_NOTE="OpenAI fallback"
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  PROVIDER_NOTE="OpenRouter free model (${OPENROUTER_FREE_MODEL:-nvidia/nemotron-3-super-120b-a12b:free})"
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
echo "Delete after section: ${TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION:-0} (handled by the section runner)"
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
  run_section_with_retry "$id"
  echo
  if [[ "$id" != "5" ]]; then
    read -r -p "Press Enter to continue to the next section..."
  fi
done

echo
echo "Showcase complete."
echo "If TRAIGENT_SHOWCASE_DELETE_AFTER_SECTION=0, the sessions are still available on the backend for FE inspection."
