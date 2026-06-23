# Traigent SDK Installation Guide

Release-ready, minimal install steps for the SDK and examples.

## Recommended Installs

### Fast path (pip)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "traigent[recommended]"       # Recommended bundle: integrations, analytics, bayesian, visualization, hybrid, pydanticai
```

### Fast path (uv)

Use this path if your environment already uses `uv`. It installs the same
published package and extras; `pip` remains fully supported. Replace `3.11` with
any supported Python 3.11-3.13 interpreter available in your environment.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "traigent[recommended]"    # Same recommended bundle as the pip path
```

### PyPI vs source installs

For Traigent SDK 0.12.0, prefer the published PyPI package: `pip install "traigent[recommended]"`. Use a source checkout only for development or when validating unreleased changes.

License: Traigent SDK is dual-licensed under AGPL-3.0-only OR LicenseRef-Traigent-Commercial.

## Extras (from `pyproject.toml`)

| Extra | What's included | Install example |
| --- | --- | --- |
| **`recommended`** | **All user-facing features (integrations + analytics + bayesian + visualization + hybrid + pydanticai)** | **`pip install "traigent[recommended]"`** |
| `integrations` | LangChain, OpenAI, Anthropic, MLflow, W&B | `pip install -e ".[integrations]"` |
| `analytics` | numpy, pandas, matplotlib | `pip install -e ".[analytics]"` |
| `bayesian` | scikit-learn + scipy | `pip install -e ".[bayesian]"` |
| `visualization` | matplotlib, plotly | `pip install -e ".[visualization]"` |
| `hybrid` | HTTP/2 transport plus MCP-backed hybrid integrations | `pip install -e ".[hybrid]"` |
| `cloud` | Reserved dependencies for future remote execution; not required for portal-tracked `hybrid` runs | `pip install -e ".[cloud]"` |
| `security` | FastAPI, JWT, cryptography, Redis | `pip install -e ".[security]"` |
| `test` | pytest + tooling | `pip install -e ".[test]"` |
| `dev` | Linters + tests | `pip install -e ".[dev]"` |
| `docs` | MkDocs tooling | `pip install -e ".[docs]"` |
| `tracing` | OpenTelemetry SDK | `pip install -e ".[tracing]"` |
| `all` / `enterprise` | Everything above | `pip install -e ".[all]"` |

## Common Scenarios

- **Run the packaged quickstart with pip (no API keys):**

  ```bash
  pip install "traigent[recommended]"
  traigent quickstart
  ```

- **Run the packaged quickstart with uv (no API keys):**

  ```bash
  uv pip install "traigent[recommended]"
  traigent quickstart
  ```

- **Develop/contribute:**

  ```bash
  pip install -e ".[recommended,dev]"
  TRAIGENT_MOCK_LLM=true pytest tests/ -q
  ```

- **Interactive UI & advanced examples:**

  See the [TraigentDemo](https://github.com/Traigent/TraigentDemo) repository for Streamlit UI tools, use cases, and research examples.

- **Full bundle for team environments:**

  ```bash
  pip install -e ".[all]"
  ```

## Verify the install

```bash
python - <<'PY'
import traigent
print("Traigent version:", traigent.get_version_info()["version"])
PY
```

## Troubleshooting (quick fixes)

- **`ModuleNotFoundError: langchain`** — install recommended extras: `pip install "traigent[recommended]"`.
- **Missing API keys** — copy `.env.example` to `.env` and set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or call `traigent.testing.enable_mock_mode_for_quickstart()` in local tutorial code. On Ubuntu desktop, you can also store keys in GNOME Keyring and export them with `secret-tool lookup ...` before running examples (see `docs/guides/secrets_management.md`).
- **Virtualenv confusion** — recreate: `deactivate; rm -rf .venv; python -m venv .venv; source .venv/bin/activate; pip install "traigent[recommended]"`.

---

Need help? Open an issue on GitHub or ping the team in the repository discussions.
