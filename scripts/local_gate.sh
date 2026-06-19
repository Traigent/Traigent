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
#   1. ruff check + ruff format --check    (mirrors the SDK Required PR Gate
#                                            'preflight' job in pr-gate.yml)
#   2. spine preflight                     (mirrors 'spine-trail present')
#   3. SonarQube quality gate              (main-bound branches only; see below)
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
#   scripts/local_gate.sh            # auto: ruff+spine always; sonar if main-bound
#   LOCAL_GATE_SONAR=1 scripts/local_gate.sh   # force the sonar step
#   LOCAL_GATE_SKIP=sonar scripts/local_gate.sh
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

# Resolve a ruff entrypoint: PATH ruff, else the venv's, else `python -m ruff`.
RUFF=""
if command -v ruff >/dev/null 2>&1; then RUFF="ruff"
elif [[ -x ".venv/bin/ruff" ]]; then RUFF=".venv/bin/ruff"
elif python3 -c "import ruff" >/dev/null 2>&1; then RUFF="python3 -m ruff"
fi

# Base ref for the changed-file diff (mirrors pr-gate.yml's three-dot range):
# release/hotfix branches target main; everything else targets develop.
if [[ "$BRANCH" =~ ^(release|hotfix)/ ]]; then base_branch="main"; else base_branch="develop"; fi
BASE_REF=""
for ref in "origin/$base_branch" "$base_branch"; do
  if git rev-parse --verify --quiet "$ref" >/dev/null; then BASE_REF="$ref"; break; fi
done

# Collect the .py files THIS branch changes (committed vs merge-base) plus
# staged/unstaged/untracked, so an about-to-be-pushed edit is covered.
changed_py() {
  { if [[ -n "$BASE_REF" ]]; then
      mb="$(git merge-base "$BASE_REF" HEAD 2>/dev/null || echo "$BASE_REF")"
      git diff --name-only --diff-filter=ACMRT "${mb}...HEAD" -- '*.py'
    fi
    git diff --name-only --diff-filter=ACMRT HEAD -- '*.py'
    git ls-files --others --exclude-standard -- '*.py'
  } | sort -u
}

# ── 1. ruff (changed-file format-check + lint) ────────────────────────────
section "ruff check + format --check on CHANGED files (SDK Required PR Gate: preflight)"
if [[ -n "$RUFF" ]]; then
  mapfile -t CHANGED < <(changed_py | while read -r f; do [[ -f "$f" ]] && echo "$f"; done)
  if [[ "${#CHANGED[@]}" -eq 0 ]]; then
    echo "  ✅ no changed .py files vs ${BASE_REF:-$base_branch} (nothing to check)"
  else
    echo "  • checking ${#CHANGED[@]} changed .py file(s) vs ${BASE_REF:-$base_branch}"
    if $RUFF format --check "${CHANGED[@]}"; then echo "  ✅ ruff format clean"
    else echo "  ❌ ruff format would reformat changed files — run 'ruff format <files>'"; FAIL=1; fi
    if $RUFF check "${CHANGED[@]}"; then echo "  ✅ ruff check clean"
    else echo "  ❌ ruff check found issues — run 'ruff check --fix <files>'"; FAIL=1; fi
  fi
else
  echo "  ⚠️  ruff not installed (pip install ruff); skipping — CI will still run it"
fi

# ── 2. spine preflight ────────────────────────────────────────────────────
section "spine preflight (spine-trail present)"
if ! python3 scripts/ci/spine_preflight.py; then FAIL=1; fi

# ── 3. SonarQube quality gate (main-bound only) ──────────────────────────
MAIN_BOUND=0
[[ "$BRANCH" =~ ^(release|hotfix)/ ]] && MAIN_BOUND=1
[[ "${LOCAL_GATE_SONAR:-0}" == "1" ]] && MAIN_BOUND=1
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
