const MONACO_BASE = "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min";

function loadMonaco() {
  if (window.monaco?.editor) {
    return Promise.resolve(window.monaco);
  }

  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-monaco-loader]');
    if (existing) {
      existing.addEventListener("load", () => resolve(window.monaco));
      existing.addEventListener("error", reject);
      return;
    }

    const loader = document.createElement("script");
    loader.dataset.monacoLoader = "true";
    loader.src = `${MONACO_BASE}/vs/loader.js`;
    loader.async = true;
    loader.onload = () => {
      if (!window.require) {
        reject(new Error("Monaco loader did not expose require."));
        return;
      }
      window.require.config({ paths: { vs: `${MONACO_BASE}/vs` } });
      window.require(["vs/editor/editor.main"], () => resolve(window.monaco));
    };
    loader.onerror = reject;
    document.body.appendChild(loader);
  });
}

function extractOptimizeFunction(source) {
  const lines = source.split(/\r?\n/);
  const decoratorIndex = lines.findIndex((line) => line.trim().startsWith("@") && line.includes("optimize"));
  if (decoratorIndex === -1) {
    return null;
  }

  const snippet = [];
  let defIndent = null;
  let bodyStarted = false;
  let inSignature = false;
  let parenDepth = 0;

  for (let i = decoratorIndex; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    snippet.push(line);

    if (defIndent === null && (trimmed.startsWith("def ") || trimmed.startsWith("async def "))) {
      defIndent = line.match(/^\s*/)[0].length;
      inSignature = true;
      const opens = (line.match(/\(/g) || []).length;
      const closes = (line.match(/\)/g) || []).length;
      parenDepth += opens - closes;
      if (parenDepth <= 0 && line.includes(":")) {
        inSignature = false;
      }
      continue;
    }

    if (defIndent !== null && trimmed && !trimmed.startsWith("#")) {
      if (inSignature) {
        const opens = (line.match(/\(/g) || []).length;
        const closes = (line.match(/\)/g) || []).length;
        parenDepth += opens - closes;
        if (parenDepth <= 0 && line.includes(":")) {
          inSignature = false;
        }
      }

      const indent = line.match(/^\s*/)[0].length;
      if (indent > defIndent && !inSignature) {
        bodyStarted = true;
      }
      if (!inSignature && bodyStarted && indent <= defIndent && i > decoratorIndex && !trimmed.startsWith("@")) {
        snippet.pop();
        break;
      }
    }
  }

  while (snippet.length && snippet[snippet.length - 1].trim() === "") {
    snippet.pop();
  }

  return snippet.join("\n");
}

async function hydrateViewer(container) {
  const sourcePath = container.dataset.optimizeSource || "run.py";
  const readOnly = container.dataset.optimizeReadonly !== "false";
  const height = parseInt(container.dataset.optimizeHeight || "320", 10);

  container.classList.add("optimize-viewer");
  container.style.minHeight = `${height}px`;
  container.innerHTML = `<div class="optimize-viewer__loading">Loading optimization target…</div>`;

  let response;
  try {
    response = await fetch(sourcePath);
  } catch (error) {
    container.innerHTML = `<div class="optimize-viewer__error">Failed to fetch ${sourcePath}: ${error}</div>`;
    return;
  }

  if (!response.ok) {
    container.innerHTML = `<div class="optimize-viewer__error">Could not load ${sourcePath} (status ${response.status}).</div>`;
    return;
  }

  const source = await response.text();
  const snippet = extractOptimizeFunction(source);
  if (!snippet) {
    container.innerHTML = `<div class="optimize-viewer__error">No @optimize function found in ${sourcePath}.</div>`;
    return;
  }

  try {
    const monaco = await loadMonaco();
    container.innerHTML = "";
    const editor = monaco.editor.create(container, {
      value: snippet,
      language: "python",
      readOnly,
      automaticLayout: true,
      minimap: { enabled: false },
      wordWrap: "on",
      theme: "vs-dark",
      fontSize: 13,
      lineNumbers: "on",
      scrollBeyondLastLine: false,
    });

    // Resize observer to keep editor responsive inside flex layouts.
    const resizeObserver = new ResizeObserver(() => editor.layout());
    resizeObserver.observe(container);
  } catch (error) {
    container.innerHTML = `<div class="optimize-viewer__error">Failed to initialise editor: ${error}</div>`;
  }
}

function injectStylesOnce() {
  if (document.getElementById("optimize-viewer-styles")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "optimize-viewer-styles";
  style.textContent = `
    .optimize-viewer {
      border: 1px solid var(--card-border, #1f2933);
      border-radius: 8px;
      overflow: hidden;
      background: #1e1e1e;
      position: relative;
    }
    .optimize-viewer__loading,
    .optimize-viewer__error {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      padding: 24px;
      text-align: center;
      background: rgba(15, 23, 42, 0.6);
      color: var(--text-secondary, #94a3b8);
      font-size: 14px;
    }
    .optimize-viewer__error {
      color: #f87171;
      background: rgba(127, 29, 29, 0.3);
    }
  `;
  document.head.appendChild(style);
}

document.addEventListener("DOMContentLoaded", () => {
  const viewers = document.querySelectorAll("[data-optimize-source]");
  if (!viewers.length) return;
  injectStylesOnce();
  viewers.forEach((container) => {
    hydrateViewer(container);
  });
});
