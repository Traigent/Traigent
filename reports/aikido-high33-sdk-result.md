# Aikido high33 — Traigent SDK (develop)

CSV slice: `aikido-filtered-high33.csv` (5 rows)

| Issue ID | Type | Finding | Disposition |
|----------|------|---------|-------------|
| 285161241, 285161101 | open_source | chromadb 1.5.9 CVE-2026-45829 in uv.lock | **Accepted risk** — chromadb is opt-in via `[chroma]` extra only; pyproject documents no patched release; `.aikidoignore` entry added |
| 113974724, 113974722, 113974723 | leaked_secret | credentials.py generic rule | **False positive** — SecureString memory wipe (`= 0`, memset); `.aikidoignore` + inline `aikido-ignore` comments |

Branch: `codex/aikido-sdk-high33` → `develop`
