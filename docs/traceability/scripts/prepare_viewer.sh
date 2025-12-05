#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="${ROOT_DIR}/.traceability_venv"
REQUIREMENTS_FILE="${ROOT_DIR}/docs/traceability/scripts/requirements.txt"
GRAPH_EXPORT="${ROOT_DIR}/tools/traceability/export_trace_graph.py"
GRAPH_JSON="${ROOT_DIR}/docs/traceability/reports/graph.json"

echo "==> Creating virtual environment at ${ENV_DIR}"
python -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

echo "==> Installing Python requirements"
pip install --upgrade pip
pip install -r "${REQUIREMENTS_FILE}"

echo "==> Exporting graph JSON"
python "${GRAPH_EXPORT}"

echo "==> Writing Cytoscape HTML viewer to docs/traceability/reports/viewer.html"
cat > "${ROOT_DIR}/docs/traceability/reports/viewer.html" <<'EOF'
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TraiGent Traceability Graph</title>
  <style>
    html, body { margin: 0; padding: 0; width: 100%; height: 100%; }
    #cy { width: 100%; height: 100%; }
  </style>
  <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
</head>
<body>
  <div id="cy"></div>
  <script>
    async function loadGraph() {
      const resp = await fetch('graph.json');
      const data = await resp.json();
      const elements = [];
      const gaps = new Set();

      // Mark gaps: code nodes without concept/func mapping
      for (const n of data.nodes) {
        if (n.type === 'code_file' && !n.concept_id && !n.functionalities) {
          gaps.add(n.id);
        }
      }

      for (const n of data.nodes) {
        elements.push({ data: n });
      }
      for (const e of data.edges) {
        elements.push({ data: e });
      }

      const cy = cytoscape({
        container: document.getElementById('cy'),
        elements,
        layout: { name: 'cose', idealEdgeLength: 120, nodeRepulsion: 8000 },
        style: [
          {
            selector: 'node',
            style: {
              'label': 'data(label)',
              'font-size': 8,
              'background-color': '#8da0cb',
              'border-width': 1,
              'border-color': '#4b4b4b',
              'text-wrap': 'wrap',
              'text-max-width': 120
            }
          },
          { selector: 'node[type=\"requirement\"]', style: { 'shape': 'round-rectangle', 'background-color': '#e78ac3' } },
          { selector: 'node[type=\"functionality\"]', style: { 'shape': 'round-rectangle', 'background-color': '#66c2a5' } },
          { selector: 'node[type=\"concept\"]', style: { 'shape': 'hexagon', 'background-color': '#fc8d62' } },
          { selector: 'node[type=\"sync\"]', style: { 'shape': 'diamond', 'background-color': '#ffd92f' } },
          { selector: 'node[type=\"code_file\"]', style: { 'shape': 'rectangle', 'background-color': '#a6d854' } },
          { selector: 'node[type=\"code_symbol\"]', style: { 'shape': 'ellipse', 'background-color': '#b3b3b3' } },
          { selector: `node[?id]`, style: { 'opacity': 0.95 } },
          {
            selector: 'edge',
            style: {
              'width': 1,
              'line-color': '#999',
              'target-arrow-color': '#999',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'font-size': 7,
              'label': 'data(type)'
            }
          },
          { selector: 'edge[type=\"satisfied_by\"]', style: { 'line-color': '#4daf4a', 'target-arrow-color': '#4daf4a' } },
          { selector: 'edge[type=\"realized_by\"]', style: { 'line-color': '#377eb8', 'target-arrow-color': '#377eb8' } },
          { selector: 'edge[type=\"implemented_by\"]', style: { 'line-color': '#984ea3', 'target-arrow-color': '#984ea3' } },
          { selector: 'edge[type=\"contains\"]', style: { 'line-color': '#666', 'target-arrow-color': '#666', 'line-style': 'dotted' } },
          { selector: 'edge[type=\"touches\"], edge[type=\"invokes\"]', style: { 'line-style': 'dashed' } },
          { selector: 'edge[status=\"proposed\"]', style: { 'line-style': 'dashed', 'line-color': '#ff7f00', 'target-arrow-color': '#ff7f00' } },
          { selector: 'edge[status=\"confirmed\"]', style: { 'line-style': 'solid', 'line-color': '#4daf4a', 'target-arrow-color': '#4daf4a' } },
          { selector: `node[id @*= \"code:\"]`, style: { 'border-width': 2, 'border-color': '#444' } },
          { selector: `node`, style: { 'z-index': 1 } },
        ]
      });

      // Highlight gaps
      cy.nodes().forEach(n => {
        if (gaps.has(n.id())) {
          n.style({
            'background-color': '#fbb4ae',
            'border-color': '#e41a1c',
            'border-width': 2
          });
        }
      });
    }
    loadGraph();
  </script>
</body>
</html>
EOF

echo "==> Done. To view:"
echo "    source ${ENV_DIR}/bin/activate"
echo "    python -m http.server 9000 -d docs/traceability/reports"
echo "    # then open http://localhost:9000/viewer.html"
