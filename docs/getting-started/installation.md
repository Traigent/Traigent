# Traigent SDK Installation Guide

Release-ready, minimal install steps for the SDK and examples.

## Recommended Installs

### Recommended path (uv)

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[recommended]"        # Recommended bundle: integrations, analytics, bayesian, visualization, hybrid, pydanticai
```

> **Don't have uv?** Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Alternative path (pip)

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[recommended]"           # Same recommended bundle, slower resolver
```

### PyPI vs source installs

`0.10.0` release validation uses the source install from this repository. Use a published package only if your team separately validates that artifact for your environment.

## Extras (from `pyproject.toml`)

| Extra | What's included | Install example |
| --- | --- | --- |
| **`recommended`** | **All user-facing features (integrations + analytics + bayesian + visualization + hybrid + pydanticai)** | **`uv pip install -e ".[recommended]"`** |
| `integrations` | LangChain, OpenAI, Anthropic, MLflow, W&B | `uv pip install -e ".[integrations]"` |
| `analytics` | numpy, pandas, matplotlib | `uv pip install -e ".[analytics]"` |
| `bayesian` | Optuna + sklearn/scipy | `uv pip install -e ".[bayesian]"` |
| `visualization` | matplotlib, plotly | `uv pip install -e ".[visualization]"` |
| `hybrid` | HTTP/2 transport plus MCP-backed hybrid integrations | `uv pip install -e ".[hybrid]"` |
| `security` | FastAPI, JWT, cryptography, Redis | `uv pip install -e ".[security]"` |
| `test` | pytest + tooling | `uv pip install -e ".[test]"` |
| `dev` | Linters + tests | `uv pip install -e ".[dev]"` |
| `docs` | MkDocs tooling | `uv pip install -e ".[docs]"` |
| `tracing` | OpenTelemetry SDK | `uv pip install -e ".[tracing]"` |
| `all` / `enterprise` | Everything above | `uv pip install -e ".[all]"` |

## Common Scenarios

- **Run examples (no API keys):**

  ```bash
  uv pip install -e ".[recommended]"
  export TRAIGENT_MOCK_LLM=true
  python examples/core/rag-optimization/run.py
  ```

- **Develop/contribute:**

  ```bash
  uv pip install -e ".[recommended,dev]"
  TRAIGENT_MOCK_LLM=true pytest tests/ -q
  ```

- **Interactive UI & advanced examples:**

  See the [TraigentDemo](https://github.com/Traigent/TraigentDemo) repository for Streamlit UI tools, use cases, and research examples.

- **Full bundle for team environments:**

  ```bash
  uv pip install -e ".[all]"
  ```

## Verify the install

```bash
python - <<'PY'
import traigent
print("Traigent version:", traigent.get_version_info()["version"])
PY
```

## Troubleshooting (quick fixes)

- **`ModuleNotFoundError: langchain`** — install recommended extras: `uv pip install -e ".[recommended]"`.
- **Missing API keys** — copy `.env.example` to `.env` and set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (skip if using `TRAIGENT_MOCK_LLM=true`). On Ubuntu desktop, you can also store keys in GNOME Keyring and export them with `secret-tool lookup ...` before running examples (see `docs/guides/secrets_management.md`).
- **Virtualenv confusion** — recreate: `deactivate; rm -rf .venv; uv venv --python 3.12; source .venv/bin/activate; uv pip install -e ".[recommended]"`.

---

Need help? Open an issue on GitHub or ping the team in the repository discussions.
