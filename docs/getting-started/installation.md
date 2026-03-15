# Traigent SDK Installation Guide

Release-ready, minimal install steps for the SDK and examples.

## Recommended Installs

### Fast path (pip)

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[integrations]"          # Core + LangChain/OpenAI/Anthropic
```

### Faster path (uv)

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
uv venv && source .venv/bin/activate
uv pip install -e ".[integrations]"       # Same extras, faster resolver
```

### PyPI vs source installs

For the latest changes, install from source (GitHub). If you're on a pinned release, you can use the corresponding PyPI package and extras once available for your environment.

## Extras (from `pyproject.toml`)

| Extra | What's included | Install example |
| --- | --- | --- |
| `analytics` | numpy, pandas, matplotlib | `pip install -e ".[analytics]"` |
| `bayesian` | Optuna + sklearn/scipy | `pip install -e ".[bayesian]"` |
| `integrations` | LangChain, OpenAI, Anthropic, MLflow, W&B | `pip install -e ".[integrations]"` |
| `security` | FastAPI, JWT, cryptography, Redis | `pip install -e ".[security]"` |
| `visualization` | matplotlib, plotly | `pip install -e ".[visualization]"` |
| `test` | pytest + tooling | `pip install -e ".[test]"` |
| `dev` | Linters + tests | `pip install -e ".[dev]"` |
| `docs` | MkDocs tooling | `pip install -e ".[docs]"` |
| `tracing` | OpenTelemetry SDK | `pip install -e ".[tracing]"` |
| `all` / `enterprise` | Everything above | `pip install -e ".[all]"` |

## Common Scenarios

- **Run examples (no API keys):**

  ```bash
  pip install -e ".[integrations]"
  export TRAIGENT_MOCK_LLM=true
  python examples/core/hello-world/run.py
  ```

- **Develop/contribute:**

  ```bash
  pip install -e ".[dev,integrations,analytics]"
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

- **`ModuleNotFoundError: langchain`** — install integrations: `pip install -e ".[integrations]"`.
- **Missing API keys** — copy `.env.example` to `.env` and set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (skip if using `TRAIGENT_MOCK_LLM=true`). On Ubuntu desktop, you can also store keys in GNOME Keyring and run via `python3 tools/keyring_run.py` (see `docs/guides/secrets_management.md`).
- **Virtualenv confusion** — recreate: `deactivate; rm -rf .venv; python -m venv .venv; source .venv/bin/activate; pip install -e ".[integrations]"`.

---

Need help? Open an issue on GitHub or ping the team in the repository discussions.
