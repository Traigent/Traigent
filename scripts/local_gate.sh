#!/usr/bin/env bash
#
# local_gate.sh — run the locally-checkable CI/policy gates BEFORE pushing.
#
# Why: coding agents (and humans) routinely miss Traigent policy and only find
# out when the cloud PR goes red — ruff-format reds, the validation-spine
# spine-trail gate, SonarQube. This mirrors those gates locally so the failure
# is caught in seconds, not after a push + a full CI round-trip.
#
# Runs (fast → slow):
#   0. freshness preflight                 (origin/develop + origin/main)
#   1. ruff check + ruff format --check    (mirrors the SDK Required PR Gate
#                                            'preflight' job in pr-gate.yml)
#   2. pytest smoke tier                   (bounded local pre-push unit smoke
#                                            when SDK code/tests changed)
#   3. spine preflight                     (mirrors 'spine-trail present')
#   4. SonarQube quality gate              (main-bound branches only; see below)
#
# NOTE on the linter — CHANGED FILES, not the whole tree. This repo's pre-commit
# uses black + isort, but the REQUIRED cloud check (pr-gate.yml 'preflight')
# runs `ruff check` + `ruff format --check` on the PR's CHANGED .py files only
# (`git diff … -- '*.py' | xargs ruff …`). The tree carries pre-existing ruff
# drift (hundreds of files); checking the whole tree would block every push on
# unrelated debt and tempt a wide auto-fix sweep — both anti-patterns. So this
# gate mirrors CI exactly: it ruff-checks only what THIS branch changes vs its
# base (develop, or main for release/hotfix), plus staged/unstaged/untracked .py
# so an about-to-be-pushed edit is seen. Fix reds with `ruff format <files>` (or
# `make format-ruff`, which formats the whole source tree).
#
# Usage:
#   scripts/local_gate.sh            # auto: freshness+ruff+pytest+spine; sonar if main-bound
#   LOCAL_GATE_SONAR=1 scripts/local_gate.sh   # force the sonar step
#   LOCAL_GATE_SKIP=pytest,sonar scripts/local_gate.sh
#   LOCAL_GATE_FULL_UNIT=1 scripts/local_gate.sh   # opt into the full tests/unit tier
#   LOCAL_GATE_PYTEST_WORKERS=8 scripts/local_gate.sh   # full-unit opt-in only; default: 4
#   LOCAL_GATE_STRICT_FRESHNESS=1 scripts/local_gate.sh   # fail on any remote drift
# It is also installed as a git pre-push hook (see `make install-hooks`).
#
# SonarQube: required for main-bound branches (release/*, hotfix/*). The cloud
#   'SonarQube Quality Gate' check (sonarqube-local.yml) is REQUIRED on main and
#   optional on develop. Set SONAR_HOST_URL (your persistent SonarQube) +
#   SONAR_TOKEN. Needs Docker (or a local sonar-scanner). It runs with
#   -Dsonar.qualitygate.wait=true so a failing gate fails the push.
#
# Bypass (discouraged, leaves a paper trail in reflog): git push --no-verify
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)"
FAIL=0
skip() { [[ ",${LOCAL_GATE_SKIP:-}," == *",$1,"* ]]; }

hr() { printf '─%.0s' {1..64}; echo; }
section() { hr; echo "▶ $1"; }

# Base ref for the changed-file diff (mirrors pr-gate.yml's three-dot range):
# release/hotfix branches target main; everything else targets develop.
if [[ "$BRANCH" =~ ^(release|hotfix)/ ]]; then base_branch="main"; else base_branch="develop"; fi
base_ref="origin/$base_branch"

MAIN_BOUND=0
[[ "$BRANCH" =~ ^(release|hotfix)/ ]] && MAIN_BOUND=1
[[ "${LOCAL_GATE_SONAR:-0}" == "1" ]] && MAIN_BOUND=1

# Resolve a ruff entrypoint: PATH ruff, else the venv's, else `python -m ruff`.
RUFF=""
if command -v ruff >/dev/null 2>&1; then RUFF="ruff"
elif [[ -x ".venv/bin/ruff" ]]; then RUFF=".venv/bin/ruff"
elif python3 -c "import ruff" >/dev/null 2>&1; then RUFF="python3 -m ruff"
fi

# Resolve a pytest entrypoint for the local smoke/full-unit tier.
PYTEST=()
if command -v pytest >/dev/null 2>&1; then PYTEST=(pytest)
elif [[ -x ".venv/bin/pytest" ]]; then PYTEST=(.venv/bin/pytest)
elif python3 -c "import pytest" >/dev/null 2>&1; then PYTEST=(python3 -m pytest)
fi

SMOKE_TESTS=(
  tests/unit/test_init_imports.py
  tests/unit/api/test_types.py
  tests/unit/wrapper/test_wrapper_service.py
)

freshness_preflight() {
  section "remote freshness preflight (origin/develop + origin/main)"
  local rc=0 strict=0 ref sha counts ahead behind
  [[ "$MAIN_BOUND" == "1" || "${LOCAL_GATE_STRICT_FRESHNESS:-0}" == "1" ]] && strict=1

  echo "  • branch=${BRANCH}; base=${base_branch}; strict_freshness=${strict}"
  for remote_branch in develop main; do
    ref="origin/${remote_branch}"
    if ! git rev-parse --verify --quiet "$ref" >/dev/null; then
      if [[ "$ref" == "$base_ref" || "$strict" == "1" ]]; then
        echo "  ❌ required remote ref $ref is missing; fetch origin before trusting changed-file gates"
        rc=1
      else
        echo "  ⚠️  remote ref $ref is missing; not used for this ${base_branch}-bound diff"
      fi
      continue
    fi

    sha="$(git rev-parse --short=12 "$ref")"
    if counts="$(git rev-list --left-right --count "HEAD...$ref" 2>/dev/null)"; then
      read -r ahead behind <<< "$counts"
      echo "  • $ref@$sha (HEAD ahead $ahead / behind $behind)"
      if [[ "$behind" =~ ^[0-9]+$ && "$behind" -gt 0 ]]; then
        if [[ "$strict" == "1" ]] || { [[ "$ref" == "origin/main" ]] && [[ "$MAIN_BOUND" == "1" ]]; }; then
          echo "    ❌ HEAD is behind $ref; refresh/rebase before a main-bound gate"
          rc=1
        else
          echo "    ⚠️  HEAD is behind $ref; develop-bound local results may differ from hosted CI"
        fi
      fi
    else
      if [[ "$strict" == "1" || "$ref" == "$base_ref" ]]; then
        echo "  ❌ unable to compare HEAD with $ref; changed-file diff would be untrusted"
        rc=1
      else
        echo "  ⚠️  unable to compare HEAD with $ref"
      fi
    fi
  done

  return "$rc"
}

if ! freshness_preflight; then FAIL=1; fi

# Collect the .py files THIS branch changes (committed vs merge-base) plus
# staged/unstaged/untracked, so an about-to-be-pushed edit is covered.
changed_files() {
  local pathspec=("$@")
  { if git rev-parse --verify --quiet "$base_ref" >/dev/null; then
      mb="$(git merge-base "$base_ref" HEAD 2>/dev/null || true)"
      if [[ -n "$mb" ]]; then
        git diff --name-only --diff-filter=ACMRT "${mb}...HEAD" -- "${pathspec[@]}"
      else
        git diff --name-only --diff-filter=ACMRT "$base_ref" HEAD -- "${pathspec[@]}"
      fi
    fi
    git diff --name-only --diff-filter=ACMRT HEAD -- "${pathspec[@]}"
    git ls-files --others --exclude-standard -- "${pathspec[@]}"
  } | sort -u
}

# ── 1. ruff (changed-file format-check + lint) ────────────────────────────
section "ruff check + format --check on CHANGED files (SDK Required PR Gate: preflight)"
mapfile -t CHANGED < <(changed_files '*.py' | while read -r f; do [[ -f "$f" ]] && echo "$f"; done)
if [[ "${#CHANGED[@]}" -eq 0 ]]; then
  echo "  ✅ no changed .py files vs $base_ref (ruff no-op)"
elif [[ -n "$RUFF" ]]; then
  echo "  • checking ${#CHANGED[@]} changed .py file(s) vs $base_ref"
  if $RUFF format --check "${CHANGED[@]}"; then echo "  ✅ ruff format clean"
  else echo "  ❌ ruff format would reformat changed files — run 'ruff format <files>'"; FAIL=1; fi
  if $RUFF check "${CHANGED[@]}"; then echo "  ✅ ruff check clean"
  else echo "  ❌ ruff check found issues — run 'ruff check --fix <files>'"; FAIL=1; fi
else
  echo "  ❌ ruff is not installed, but ${#CHANGED[@]} changed .py file(s) require the SDK Ruff gate"
  echo "     Install dev deps (pip install -e '.[all,dev]') or install ruff, then re-run."
  FAIL=1
fi

# ── 2. pytest smoke/full-unit tier ────────────────────────────────────────
sdk_code_changed() {
  [[ "$1" =~ ^(traigent|traigent_validation|tests|scripts|walkthrough)/ ]] || \
    [[ "$1" =~ ^(Makefile|\.pre-commit-config\.yaml|pyproject\.toml|requirements[^/]*\.txt|\.github/workflows/pr-gate\.yml)$ ]]
}

if ! skip pytest; then
  section "pytest smoke tests on code changes (local pre-push SDK unit guard)"
  mapfile -t CODE_CHANGED < <(changed_files | while read -r f; do sdk_code_changed "$f" && echo "$f"; done)
  if [[ "${#CODE_CHANGED[@]}" -eq 0 ]]; then
    echo "  ✅ no SDK code/test/gate files changed vs $base_ref (pytest tier not required)"
  elif [[ "${#PYTEST[@]}" -gt 0 ]]; then
    # Keep ambient user-site pytest plugins from changing the CI-shaped unit run.
    # The repo depends on pytest-asyncio, not pytest-asyncio-cooperative; the
    # latter can reorder xdist reports and produce internal errors locally.
    pytest_args=(-p no:asyncio-cooperative -q --tb=short)
    if [[ "${LOCAL_GATE_FULL_UNIT:-0}" == "1" ]]; then
      pytest_workers="${LOCAL_GATE_PYTEST_WORKERS:-4}"
      pytest_args+=(tests/unit)
      if [[ -n "$pytest_workers" ]]; then
        pytest_args+=(-n "$pytest_workers" --dist loadgroup)
      fi
      echo "  • LOCAL_GATE_FULL_UNIT=1; running optional full unit tier"
    else
      pytest_args+=("${SMOKE_TESTS[@]}" -n 0)
      echo "  • bounded smoke tier selected; set LOCAL_GATE_FULL_UNIT=1 for full tests/unit"
    fi
    echo "  • ${#CODE_CHANGED[@]} code/test/gate file(s) changed; running ${PYTEST[*]} ${pytest_args[*]}"
    if PYTHONPATH=. "${PYTEST[@]}" "${pytest_args[@]}"; then echo "  ✅ pytest tier clean"
    else echo "  ❌ pytest tier failed"; FAIL=1; fi
  else
    echo "  ❌ pytest is not installed; install dev deps (pip install -e '.[all,dev]') before pushing code changes"
    FAIL=1
  fi
else
  hr; echo "ℹ️  pytest tier skipped via LOCAL_GATE_SKIP=pytest"
fi

# ── 3. spine preflight ────────────────────────────────────────────────────
section "spine preflight (spine-trail present)"
if ! python3 scripts/ci/spine_preflight.py; then FAIL=1; fi

# ── 4. SonarQube quality gate (main-bound only) ──────────────────────────
if [[ "$MAIN_BOUND" == "1" ]] && ! skip sonar; then
  section "SonarQube quality gate (main-bound: $BRANCH)"
  have_creds=0
  { [[ -n "${SONAR_TOKEN:-}" ]] || [[ -f .env.sonar ]]; } && have_creds=1
  if [[ "$have_creds" == "0" ]]; then
    echo "  ❌ main-bound push requires the SonarQube gate, but no SONAR_TOKEN/.env.sonar found."
    echo "     Set SONAR_HOST_URL + SONAR_TOKEN (your persistent SonarQube) and re-run,"
    echo "     or run 'make sonar-scan' manually, then push. (bypass: git push --no-verify)"
    FAIL=1
  elif ! command -v docker >/dev/null 2>&1 && ! command -v sonar-scanner >/dev/null 2>&1; then
    echo "  ❌ need Docker or sonar-scanner for the local SonarQube gate (main-bound)."
    FAIL=1
  else
    host="${SONAR_HOST_URL:-http://localhost:9000}"
    echo "  🔎 scanning against $host with qualitygate.wait=true …"
    if command -v docker >/dev/null 2>&1; then
      net=""; [[ "$host" == *localhost* || "$host" == *127.0.0.1* ]] && net="--network=host"
      docker run --rm $net -e SONAR_HOST_URL="$host" -e SONAR_TOKEN \
        -v "$PWD":/usr/src sonarsource/sonar-scanner-cli:latest \
        -Dsonar.projectKey=Traigent_Traigent \
        -Dsonar.qualitygate.wait=true -Dsonar.qualitygate.timeout=300
    else
      sonar-scanner -Dsonar.host.url="$host" \
        -Dsonar.projectKey=Traigent_Traigent \
        -Dsonar.qualitygate.wait=true -Dsonar.qualitygate.timeout=300
    fi
    if [[ $? -eq 0 ]]; then echo "  ✅ SonarQube quality gate passed"
    else echo "  ❌ SonarQube quality gate FAILED — fix locally before the main PR"; FAIL=1; fi
  fi
else
  hr; echo "ℹ️  SonarQube step skipped (not a release/hotfix branch). For main-bound"
  echo "   work set LOCAL_GATE_SONAR=1 or push a release/* branch."
fi

hr
if [[ "$FAIL" == "0" ]]; then
  echo "✅ local gate PASSED — safe to push"
else
  echo "❌ local gate FAILED — fix the items above (or 'git push --no-verify' to bypass)"
fi
exit "$FAIL"
