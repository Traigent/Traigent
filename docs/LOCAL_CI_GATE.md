# Local CI gate — catch policy/CI failures before you push

Coding agents (and humans) repeatedly miss Traigent policy and only discover it
when the cloud PR goes **red** — a wasted push + a full CI round-trip per miss.
Most of those checks are runnable **locally in seconds**. This repo ships a
single local gate that mirrors them so the failure is caught before the push.

## TL;DR

```bash
make install-hooks   # once per clone: installs pre-commit + commit-msg + pre-push hooks
make local-gate      # before every push: run the gate manually (agents: do this)
```

`make local-gate` (a.k.a. `scripts/local_gate.sh`) exits non-zero on anything CI
would reject. Fix what it reports, then push.

## What it runs

| Step | Mirrors cloud check | Notes |
|------|---------------------|-------|
| `ruff check` + `ruff format --check` on **changed `.py` files** | `SDK Required PR Gate` → `preflight` (`.github/workflows/pr-gate.yml`) | The #1 source of avoidable develop reds. Scoped to what your branch changes vs its base (like CI), not the whole tree. Fix with `ruff format <files>`. |
| `scripts/ci/spine_preflight.py` | `spine-trail present` (`.github/workflows/spine-trail-gate.yml`) | Reminds you to add a `Spine-Trail:`/`Spine:` line to the PR body (CI reads the **body**, so this is a warning, not a hard block). |
| SonarQube quality gate | `SonarQube Quality Gate` (`.github/workflows/sonarqube-local.yml`, **required on `main`**) | **Only for `release/*` / `hotfix/*` branches** (or `LOCAL_GATE_SONAR=1`). Runs `sonar-scanner -Dsonar.qualitygate.wait=true` against your SonarQube. |

### Why ruff (not black)?

This repo's **pre-commit** formats with `black` + `isort`, but the **required
cloud check** (`pr-gate.yml` → `preflight`) runs `ruff check` +
`ruff format --check` on the PR's **changed `.py` files**. That required check
is what turns a PR red, so the gate mirrors **ruff**, and — crucially — only on
the files your branch changes vs its base, exactly like CI. (The tree carries
pre-existing ruff drift; checking the whole tree would block every push on
unrelated debt and tempt a wide auto-fix sweep, both anti-patterns.) ruff-format
and black agree on this codebase today; if they ever diverge on a file,
ruff-format wins in the gate because that's what CI enforces. Fix reds with
`ruff format <files>` (or `make format-ruff` to format the whole source tree).

## The spine breadcrumb (why CI can fail late)

Every product PR on this core repo needs a validation-spine mark in its **body**
— CI's `spine-trail present` check fails the PR otherwise. You can't satisfy it
*after* the fact without re-editing the PR. Create the breadcrumb up front:

```bash
python3 ../tools/spine-trail/spine_trail.py get-or-create \
  --repo Traigent --branch "$(git branch --show-current)" --base-branch develop \
  --kind <bug_fix|feature|change> --source-type <type> --source-ref "<ref>" \
  --changed-paths "<files>"
# then add ONE of these lines to the PR body:
#   Spine-Trail: st_xxxxxxxxxxxx     (a Tier-0 WorkIntent)
#   Spine: cs_xxxxxxxx               (a promoted ChangeSession)
#   Spine: none (reason: <why ungoverned>)
```

> **Note:** unlike TraigentBackend, this repo has **no** `require-spine-session`
> gate and **no** `.github/spine-policy-surface-globs.txt` policy-surface file
> (its `validation-spine-pr.yml` security scan is advisory / `continue-on-error`).
> So the gate does **not** hard-block on a missing `Spine-Session`. If a
> policy-surface globs file is ever added here, `spine_preflight.py` picks it up
> automatically and starts enforcing a `Spine-Session:` on policy-surface diffs,
> matching the TraigentBackend gate — no script change needed.

## SonarQube locally before a main PR

For `release/*` / `hotfix/*` branches the gate runs the SonarQube quality gate
locally and **blocks the push if it fails** (mirroring the `SonarQube Quality
Gate` check that is required on `main`). Configure once:

```bash
export SONAR_HOST_URL="https://<your-persistent-sonarqube>"   # or http://localhost:9000
export SONAR_TOKEN="<token>"        # or keep a .env.sonar entry
```

It runs the Dockerized `sonar-scanner` (or a local one) with
`-Dsonar.qualitygate.wait=true` and the repo's `Traigent_Traigent` project key,
so you get the same pass/fail verdict the required check produces — before you
push. If you don't have SonarQube access, run it another way or bypass
consciously (`git push --no-verify`) and have a reviewer who can run it.

## Bypassing

`git push --no-verify` skips the pre-push hook (leaves a reflog trail). Use only
when you've verified another way — the gate exists to stop avoidable cloud reds,
not to block you.
