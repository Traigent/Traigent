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
#   0.5 generated schema types drift       (traigent/generated/schema_types.py
#                                            vs TraigentSchema; the FIRST real
#                                            gate — skips locally with a warning
#                                            when no TraigentSchema checkout is
#                                            found, REQUIRED and fail-closed in CI)
#   1. ruff check + ruff format --check    (mirrors the SDK Required PR Gate
#                                            'preflight' job in pr-gate.yml)
#   2. pytest collection + smoke tier      (whole-suite import/collection guard,
#                                            then bounded local pre-push unit smoke
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

# Git exports GIT_DIR, GIT_INDEX_FILE, GIT_WORK_TREE and friends into hooks. Any
# test that builds a temp repo would then act on THIS repo ("remote origin already
# exists"). Run pytest with git's own list of repository-local variables cleared.
GIT_ENV_CLEAR=(env)
while IFS= read -r _gitvar; do
  [[ -n "$_gitvar" ]] && GIT_ENV_CLEAR+=(-u "$_gitvar")
done < <(git rev-parse --local-env-vars 2>/dev/null)
unset _gitvar

# Resolve a ruff entrypoint: the repo venv's first (it carries the pinned
# version; the pre-push hook does not activate the venv), else PATH ruff, else
# `python -m ruff`.
RUFF=""
if [[ -x ".venv/bin/ruff" ]]; then RUFF=".venv/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then RUFF="ruff"
elif python3 -c "import ruff" >/dev/null 2>&1; then RUFF="python3 -m ruff"
fi

# Resolve a pytest entrypoint for the local smoke/full-unit tier. The repo venv
# wins over PATH: pyproject addopts needs pytest-xdist, which a user-level or
# system pytest usually lacks (#2474).
PYTEST=()
PYTEST_PY=""
if [[ -x ".venv/bin/pytest" ]]; then PYTEST=(.venv/bin/pytest); PYTEST_PY=".venv/bin/python"
elif command -v pytest >/dev/null 2>&1; then PYTEST=(pytest)
elif python3 -c "import pytest" >/dev/null 2>&1; then PYTEST=(python3 -m pytest); PYTEST_PY="python3"
fi
echo "  resolved tools: ruff=${RUFF:-none} pytest=${PYTEST[*]:-none}"

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
        if [[ "$BRANCH" =~ ^hotfix/ && "$ref" == "origin/develop" && "${LOCAL_GATE_STRICT_FRESHNESS:-0}" != "1" ]]; then
          # A hotfix is cut from main, so it is always behind develop; the
          # develop check cannot apply. Still required: not behind origin/main.
          echo "    ℹ️  HEAD is behind $ref; expected for a hotfix cut from main (not enforced; origin/main freshness still is)"
        elif [[ "$strict" == "1" ]] || { [[ "$ref" == "origin/main" ]] && [[ "$MAIN_BOUND" == "1" ]]; }; then
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

# ── 0.5. generated schema types drift (FIRST real gate, mirrors traigent-js's
#         test:ci where check:schema-types runs before anything else) ───────
if ! skip schema-types; then
  section "generated schema types drift (traigent/generated/schema_types.py)"
  schema_repo="${TRAIGENT_SCHEMA_REPO:-}"
  if [[ -z "$schema_repo" ]]; then
    default_schema_repo="$(cd "$REPO_ROOT/.." 2>/dev/null && pwd)/TraigentSchema"
    [[ -d "$default_schema_repo" ]] && schema_repo="$default_schema_repo"
  fi
  if [[ -z "$schema_repo" || ! -d "$schema_repo" ]]; then
    echo "  ⚠️  no TraigentSchema checkout found (set TRAIGENT_SCHEMA_REPO or place one at"
    echo "     ../TraigentSchema); skipping locally. This check is REQUIRED and fail-closed in CI."
    schema_repo=""
  else
    echo "  • TRAIGENT_SCHEMA_REPO=$schema_repo"
    # The gate must test against the schema build CI uses (#2469):
    # scripts/ci/schema-pin.txt is the single source of truth. If the checkout
    # is not at the pin, export the pinned commit to a temp dir (no mutation of
    # the checkout); if the pin is unreadable or absent, fail closed.
    pin_sha="$(grep -oE '@[0-9a-f]{40}$' scripts/ci/schema-pin.txt 2>/dev/null | tr -d '@' | head -n1)"
    if [[ -z "$pin_sha" ]]; then
      echo "  ❌ cannot read the TraigentSchema pin from scripts/ci/schema-pin.txt"
      FAIL=1
      schema_repo=""
    elif git -C "$schema_repo" cat-file -e "${pin_sha}^{commit}" 2>/dev/null; then
      # Always export: the generator reads working-tree files, so a dirty or
      # untracked checkout must never influence the verdict.
      schema_pin_dir="$(mktemp -d)"
      trap 'rm -rf "$schema_pin_dir"' EXIT
      if git -C "$schema_repo" archive "$pin_sha" | tar -x -C "$schema_pin_dir"; then
        echo "  • using TraigentSchema@${pin_sha:0:12} exported to a temp dir (checkout working tree is ignored)"
        schema_repo="$schema_pin_dir"
      else
        echo "  ❌ could not export TraigentSchema@${pin_sha:0:12} from the checkout"
        FAIL=1
        schema_repo=""
      fi
    else
      echo "  ❌ TraigentSchema checkout lacks pinned commit ${pin_sha:0:12} (run 'git fetch' there)"
      FAIL=1
      schema_repo=""
    fi
  fi
  if [[ -n "$schema_repo" ]]; then
    py_bin="python3"; [[ -x ".venv/bin/python" ]] && py_bin=".venv/bin/python"
    if TRAIGENT_SCHEMA_REPO="$schema_repo" "$py_bin" scripts/generate_schema_types.py --check; then
      echo "  ✅ generated schema types are current"
    else
      echo "  ❌ traigent/generated/schema_types.py is stale — run 'make generate-schema-types'"
      FAIL=1
    fi
  fi
fi

# Collect the .py files THIS branch changes (committed vs merge-base) plus
# staged/unstaged/untracked, so an about-to-be-pushed edit is covered.
changed_files() {
  local pathspec=("$@")
  local out mb all=""
  if git rev-parse --verify --quiet "$base_ref" >/dev/null; then
    mb="$(git merge-base "$base_ref" HEAD 2>/dev/null)" || { echo "git merge-base $base_ref HEAD failed" >&2; return 1; }
    out="$(git diff --name-only --diff-filter=ACMRT "${mb}...HEAD" -- "${pathspec[@]}")" || { echo "git diff vs merge-base failed" >&2; return 1; }
    all+="$out"$'\n'
  fi
  out="$(git diff --name-only --diff-filter=ACMRT HEAD -- "${pathspec[@]}")" || { echo "git diff HEAD failed" >&2; return 1; }
  all+="$out"$'\n'
  out="$(git ls-files --others --exclude-standard -- "${pathspec[@]}")" || { echo "git ls-files failed" >&2; return 1; }
  all+="$out"
  printf '%s\n' "$all" | sort -u
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
  section "pytest collection + smoke tests on code changes (local pre-push SDK guard)"
  mapfile -t CODE_CHANGED < <(changed_files | while read -r f; do sdk_code_changed "$f" && echo "$f"; done)
  if [[ "${#CODE_CHANGED[@]}" -eq 0 ]]; then
    echo "  ✅ no SDK code/test/gate files changed vs $base_ref (pytest tier not required)"
  elif [[ "${#PYTEST[@]}" -gt 0 ]]; then
    # Keep ambient user-site pytest plugins from changing the CI-shaped unit run.
    # The repo depends on pytest-asyncio, not pytest-asyncio-cooperative; the
    # latter can reorder xdist reports and produce internal errors locally.
    if [[ -n "$PYTEST_PY" ]] && ! "$PYTEST_PY" -c "import xdist" >/dev/null 2>&1; then
      echo "  ❌ pytest interpreter '$PYTEST_PY' cannot import pytest-xdist (required by pyproject addopts)"
      echo "     install the dev extra: pip install -e '.[dev]' (in the repo .venv)"
      FAIL=1
    fi
    collection_args=(--collect-only -q tests/ -o addopts= -p no:cacheprovider -rs)
    echo "  • ${#CODE_CHANGED[@]} code/test/gate file(s) changed; first verifying whole-suite collection"
    echo "  • TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true PYTHONPATH=. ${PYTEST[*]} ${collection_args[*]}"
    if TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true PYTHONPATH=. \
      "${GIT_ENV_CLEAR[@]}" "${PYTEST[@]}" "${collection_args[@]}"; then
      echo "  ✅ whole test suite collects"
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
      echo "  • running ${PYTEST[*]} ${pytest_args[*]}"
      if PYTHONPATH=. "${GIT_ENV_CLEAR[@]}" "${PYTEST[@]}" "${pytest_args[@]}"; then echo "  ✅ pytest tier clean"
      else echo "  ❌ pytest tier failed"; FAIL=1; fi
    else
      echo "  ❌ whole-suite collection failed; smoke/full-unit tier not run"
      FAIL=1
    fi
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

# ── Sonar coverage (full unit suite) ──────────────────────────────────────
# The scanner only reads sonar.python.coverage.reportPaths (coverage.xml); it
# never runs tests. This step runs only for main-bound pushes (the Sonar step
# below), and measures coverage with the FULL tests/unit suite, the same
# command the hosted coverage lane runs. A changed-scope subset under-reports
# new-code coverage (a release push once read ~22% from a subset where the full
# suite measures ~90%), because the lines a subset never executes count as
# uncovered. Develop-bound pushes never reach this step and keep the cheap
# smoke tier above (docs/ci-policy.md). The 80% new-code threshold is unchanged.
# Workers default to 4 (LOCAL_GATE_COVERAGE_WORKERS=0 forces a serial run).
sonar_prepare_coverage() {
  rm -f coverage.xml
  local rc=0 workers="${LOCAL_GATE_COVERAGE_WORKERS:-4}" home_dir rcfile
  if [[ "${#PYTEST[@]}" -eq 0 ]]; then
    echo "  ❌ pytest unavailable: cannot measure coverage for the main-bound scan"; FAIL=1; return 0
  fi
  # Fail closed on missing plugins: a silent serial or un-timed run would
  # either take hours or hang the push.
  local help_out; help_out="$(timeout -k 10 120 "${PYTEST[@]}" --help 2>/dev/null)"
  local -a par=(-n 0)
  if [[ "$workers" != "0" ]]; then par=(-n "$workers" --dist loadgroup); fi
  if ! grep -q -- '--timeout' <<<"$help_out" || ! grep -q -- '--cov' <<<"$help_out" \
     || { [[ "$workers" != "0" ]] && ! grep -q -- '--dist' <<<"$help_out"; }; then
    echo "  ❌ pytest-timeout, pytest-cov and (unless LOCAL_GATE_COVERAGE_WORKERS=0) pytest-xdist are required"
    echo "     for the main-bound coverage run: pip install pytest-timeout pytest-cov pytest-xdist"
    FAIL=1; return 0
  fi
  # Fresh HOME: a developer's real ~/.traigent can hold tens of thousands of
  # session files that make local runs hang, and a few tests assert the
  # default ~/.traigent location, so redirect HOME rather than the results dir.
  home_dir="$(mktemp -d "${TMPDIR:-/tmp}/lg-home.XXXXXX")"
  # relative_files => coverage.xml paths resolve inside the scanner container too.
  rcfile="$(mktemp "${TMPDIR:-/tmp}/lg-covrc.XXXXXX")"
  printf '[run]\nrelative_files = True\nsource = traigent\nomit =\n    */tests/*\n    */test_*\n    */demos/*\n    */docs/*\n' >"$rcfile"
  local pid=""
  trap '[[ -n "$pid" ]] && { kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; rm -rf "$home_dir" "$rcfile"; trap - INT TERM; exit 130' INT TERM
  echo "  • coverage with the FULL tests/unit suite (${par[*]}); this takes a while"
  # Background + wait so an interrupt reaches the trap immediately and the
  # whole run (timeout -> pytest) is stopped and reaped before cleanup.
  TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true HOME="$home_dir" PYTHONPATH=. \
    "${GIT_ENV_CLEAR[@]}" timeout -k 60 "${LOCAL_GATE_COVERAGE_TIMEOUT:-3600}" "${PYTEST[@]}" -p no:asyncio-cooperative -p no:cacheprovider -q --tb=short \
    "${par[@]}" --timeout=30 \
    --cov=traigent --cov-config="$rcfile" --cov-report=xml:coverage.xml tests/unit/ &
  pid=$!
  wait "$pid" || rc=$?
  pid=""
  rm -f "$rcfile"; rm -rf "$home_dir"; trap - INT TERM
  if [[ ! -s coverage.xml ]]; then
    echo "  ❌ coverage.xml was not produced (pytest rc=$rc): coverage NOT MEASURED"; FAIL=1; return 0
  fi
  # pyproject sets no coverage fail-under, so a nonzero pytest rc is a real
  # test failure (or collection error): fail the gate rather than scan a
  # partial report.
  if [[ "$rc" -ne 0 ]]; then
    echo "  ❌ unit suite FAILED (pytest rc=$rc): fix it; coverage.xml is PARTIAL"
    FAIL=1
  else
    echo "  ✅ coverage.xml written ($(wc -c <coverage.xml) bytes)"
  fi
}

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
    # Release version from pyproject.toml [project].version, so the server's
    # PREVIOUS_VERSION new-code period compares against the last analysed
    # release instead of the stale version pinned in sonar-project.properties.
    sonar_version="$(awk '/^\[project\]/{p=1;next} /^\[/{p=0} p&&/^version[[:space:]]*=/{gsub(/^[^"]*"|".*$/,"");print;exit}' pyproject.toml)"
    if [[ -z "$sonar_version" ]]; then
      echo "  ❌ could not read [project].version from pyproject.toml for sonar.projectVersion"
      FAIL=1
    else
    sonar_prepare_coverage
    echo "  🔎 scanning against $host (sonar.projectVersion=$sonar_version) with qualitygate.wait=true …"
    if command -v docker >/dev/null 2>&1; then
      net=""; [[ "$host" == *localhost* || "$host" == *127.0.0.1* ]] && net="--network=host"
      # Mount the repo at the SAME absolute path inside the container so the
      # host paths recorded in coverage.xml resolve (otherwise Sonar logs
      # "Cannot resolve N file paths" and new coverage reads 0.0).
      #
      # In a linked worktree `.git` is a file pointing at <common-dir>/worktrees/<n>,
      # outside the mount, so JGit fails with "Unable to open Git repository" and
      # SCM blame (new-code detection) is lost. Mount the git common dir at the
      # same absolute path (read-only) so the pointer resolves.
      git_mount=()
      common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
      if [[ -n "$common_dir" && -d "$common_dir" && "$common_dir" != "$PWD/.git" ]]; then
        git_mount=(-v "$common_dir":"$common_dir":ro)
      fi
      docker run --rm $net -e SONAR_HOST_URL="$host" -e SONAR_TOKEN \
        -v "$PWD":"$PWD" ${git_mount[@]+"${git_mount[@]}"} -w "$PWD" sonarsource/sonar-scanner-cli:latest \
        -Dsonar.projectBaseDir="$PWD" \
        -Dsonar.projectKey=Traigent_Traigent \
        -Dsonar.projectVersion="$sonar_version" \
        -Dsonar.qualitygate.wait=true -Dsonar.qualitygate.timeout=300
    else
      sonar-scanner -Dsonar.host.url="$host" \
        -Dsonar.projectKey=Traigent_Traigent \
        -Dsonar.projectVersion="$sonar_version" \
        -Dsonar.qualitygate.wait=true -Dsonar.qualitygate.timeout=300
    fi
    if [[ $? -eq 0 ]]; then echo "  ✅ SonarQube quality gate passed"
    else echo "  ❌ SonarQube quality gate FAILED — fix locally before the main PR"; FAIL=1; fi
    fi
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
