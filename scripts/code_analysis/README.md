# Code Analysis Toolkit

This directory contains the repository inventory and analysis utilities Codex generated to help map the codebase. The toolkit is designed to run entirely from source without external graph-layout libraries and to degrade gracefully when optional tooling (e.g., `pytest`, `ruff`) is unavailable on the host.

## Components

- `analysis_utils.py` – shared helpers (SLOC counting, AST parsing, quantiles, subprocess wrapper, CODEOWNERS lookup).
- `inventory.py` – builds `inventory.csv` with per-module metadata: language, SLOC, file size, last-modified timestamp, CODEOWNERS (if any), test coverage %, and Ruff lint error counts.
- `dependency_graph.py` – walks Python imports to create a static dependency graph, exporting both GraphML (`deps.graphml`) and a PNG preview (`deps.png`). No external layout tool is required; the renderer keeps output ≤4k×4k px.
- `metrics.py` – computes deeper metrics per module: cyclomatic/cognitive complexity aggregates, fan-in/out, public symbol counts, function length quantiles, and inferred test counts (based on tests importing the module).
- `run_analysis.py` – orchestration entry point that coordinates lint/coverage runs, executes the above analyses, and writes timestamped report bundles under `reports/code_analysis/`.
- `viz_atlas.py` – Phase 1 “Repository Atlas”: hierarchical module tree, SCC DAG overview, per-package split views, and hub neighbourhood cards.
- `viz_clusters.py` – Phase 2 “Clustered Capability Views”: Louvain communities, per-cluster PNGs, markdown summaries, and `clusters.json` metadata.
- `viz_layers.py` – Phase 2 “Layered Dependency Views”: layer-coloured component DAG plus a `violations.csv` report for upward dependencies.
- `risk_score.py` – Phase 3 “Risk & Priority Heatmap”: aggregates metrics, lint, clones, churn, coverage, and ownership to produce `risk_heatmap.csv` and `top20.md`.
- `viz_heatmap.py` – renders `risk_heatmap.csv` into paginated PNG heat tables for quick scanning.

## Running the Toolkit

All entry points accept command-line flags so they can be run independently, but most users will just call:

```bash
python -m scripts.code_analysis.run_analysis
```

Useful options:

- `--tag my_label` – store results in `reports/code_analysis/my_label/` instead of a timestamped folder.
- `--skip-coverage` / `--skip-lint` / `--skip-owners` – bypass expensive steps when pytest, Ruff, or CODEOWNERS data are unavailable.
- `--source-root`, `--tests-root`, `--output-root` – override default paths if the project layout changes.

> **Tip:** Ensure your virtualenv is active (`python -m venv venv && source venv/bin/activate`) and run `make install-smart` so `pytest` and `ruff` are on PATH. Otherwise the coverage run will log a warning (`coverage.log`) and coverage percentages will be blank.

## Report Outputs

Every run creates a subdirectory inside `reports/code_analysis/` with the following files:

- `inventory.csv` – canonical module inventory. Open in a spreadsheet or load with pandas to sort/filter by SLOC, file size, owners, coverage, or lint findings.
- `metrics.csv` – per-module metrics. Key columns:
  - `cyclomatic_total` / `cognitive_total` and their averages – gauge complexity hotspots.
  - `fan_in` / `fan_out` – identify highly shared modules or dependency magnets.
  - `function_length_p25/p50/p75` – understand distribution of function sizes.
  - `test_count` – approximate number of discovered tests touching the module.
- `deps.graphml` – static import graph suitable for Gephi, yEd, or Neo4j Bloom. Node IDs are fully-qualified module names (prefixed with `src.`); edge weights reflect import frequency.
- `deps.png` – quick visual of the graph (circular layout). Use the GraphML for precise inspection.
- `lint.json` – raw Ruff output (`--exit-zero`) so you can inspect per-file lint issues.
- `coverage.log` – only present when the coverage step fails; contains the command, return code, and stdout/stderr. If coverage succeeds, a `coverage.xml` file is written instead.
- `SUMMARY.txt` – human-readable pointers to the generated artifacts.

## Interpreting Results

1. **Inventory first:** sort `inventory.csv` by `sloc` or `lint_error_count` to surface large or noisy modules.
2. **Complexity & coupling:** `metrics.csv` highlights cyclomatic/cognitive outliers and modules with extreme fan-in/out.
3. **Graph inspection:** load `deps.graphml` into a graph viewer to see clustering; the PNG offers a quick glance but is lossy.
4. **Coverage & lint follow-up:** if `coverage.log` reports missing `pytest`, install dependencies and rerun. For lint, parse `lint.json` for file-level details.

## Script-Level Usage

You can call the individual modules directly if you need a single artifact:

```bash
# Generate only the dependency graph
python -m scripts.code_analysis.dependency_graph \
  --graphml reports/code_analysis/manual/deps.graphml \
  --png reports/code_analysis/manual/deps.png

# Regenerate metrics without re-running lint/coverage
python -m scripts.code_analysis.metrics --output reports/code_analysis/manual/metrics.csv

# Phase 1 atlas from an existing analysis bundle
python -m scripts.code_analysis.viz_atlas \
  --in reports/code_analysis/rerun_smoke \
  --out reports/viz/atlas

# Phase 2 clustered views and layer audit
python -m scripts.code_analysis.viz_clusters \
  --graph reports/code_analysis/rerun_smoke/deps.graphml \
  --out reports/viz/clusters
python -m scripts.code_analysis.viz_layers \
  --graph reports/code_analysis/rerun_smoke/deps.graphml \
  --out reports/viz/layers

# Phase 3 risk scoring and heatmap
python -m scripts.code_analysis.risk_score \
  --metrics reports/code_analysis/rerun_smoke/metrics.csv \
  --lint reports/code_analysis/rerun_smoke/lint.json \
  --coverage reports/code_analysis/rerun_smoke/coverage.log \
  --clones reports/clones/jscpd.json \
  --churn reports/churn/churn.csv \
  --owners CODEOWNERS \
  --out reports/risk
python -m scripts.code_analysis.viz_heatmap \
  --csv reports/risk/risk_heatmap.csv \
  --out reports/risk/risk_heatmap.png

# Phase 4 refactoring proposals
python -m scripts.refactor.suggest_capabilities \
  --clusters reports/viz/clusters/clusters.json \
  --metrics reports/code_analysis/rerun_smoke/metrics.csv \
  --out reports/refactor/capabilities.json
python -m scripts.refactor.find_shared_libs \
  --clones reports/clones/jscpd.json \
  --out reports/refactor/shared_lib_candidates.md
python -m scripts.refactor.shape_facades \
  --capabilities reports/refactor/capabilities.json \
  --out reports/refactor/abstractions_proposal.md
python -m scripts.refactor.extraction_plan \
  --proposal reports/refactor/abstractions_proposal.md \
  --out reports/refactor/extraction_plan.md
```

Each script exposes `--project-root`, `--source-root`, and (where relevant) `--tests-root` arguments; check `--help` for the full list.

## Troubleshooting

- **“No such file or directory: 'pytest'”** – activate the virtualenv/ install dependencies, then rerun without `--skip-coverage`.
- **Large PNGs** – rendering scales with module count; if 4k×4k is still too big, reuse the GraphML in a dedicated graph tool.
- **Atlas/cluster images too big?** – the generated HTML now wraps each graphic in a scrollable frame so you can pan at native resolution. Adjust `image_max_edge_px` in `config/viz.yaml` (e.g., 2200) if you prefer smaller exports.
- **Need to tweak risk weighting?** – edit `config/risk.yaml` to rebalance the z-score weights or the owner penalty, then rerun `risk_score.py`.
- **Missing CODEOWNERS data** – supply a `CODEOWNERS` file or pass `--skip-owners` to silence owner lookup.

Feel free to extend the toolkit by adding new scripts here; they inherit shared functionality from `analysis_utils.py`.
