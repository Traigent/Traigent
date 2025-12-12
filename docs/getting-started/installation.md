# TraiGent SDK Installation Guide

Release-ready, minimal install steps for the SDK and examples.

## 🚀 Recommended Installs

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

### Not on PyPI yet

TraiGent is currently distributed from GitHub. Keep using the commands above until the first PyPI release lands.

## 📦 Extras (from `pyproject.toml`)

| Extra | What's included | Install example |
| --- | --- | --- |
| `analytics` | numpy, pandas, matplotlib | `pip install -e ".[analytics]"` |
| `bayesian` | Optuna + sklearn/scipy | `pip install -e ".[bayesian]"` |
| `integrations` | LangChain, OpenAI, Anthropic, MLflow, W&B | `pip install -e ".[integrations]"` |
| `security` | FastAPI, JWT, cryptography, Redis | `pip install -e ".[security]"` |
| `visualization` | matplotlib, plotly | `pip install -e ".[visualization]"` |
| `playground` | Streamlit control center | `pip install -e ".[playground]"` |
| `examples` | Full example/demo deps | `pip install -e ".[examples]"` |
| `test` | pytest + tooling | `pip install -e ".[test]"` |
| `dev` | Linters + tests | `pip install -e ".[dev]"` |
| `all` / `enterprise` | Everything above | `pip install -e ".[all]"` |

## 📚 Common Scenarios

- **Run quickstart examples (no API keys):**

  ```bash
  pip install -e ".[examples]"
  export TRAIGENT_MOCK_MODE=true
  python examples/quickstart/01_simple_qa.py
  ```

- **Develop/contribute:**

  ```bash
  pip install -e ".[dev,integrations,analytics]"
  pytest tests/ -q
  ```

- **Playground UI:**

  ```bash
  pip install -e ".[playground]"
  streamlit run playground/traigent_control_center.py
  ```

- **Full bundle for team environments:**

  ```bash
  pip install -e ".[all]"
  ```

## ✅ Verify the install

```bash
python - <<'PY'
import traigent
print("TraiGent version:", traigent.get_version_info()["version"])
PY
```

## 🐛 Troubleshooting (quick fixes)

- **`ModuleNotFoundError: langchain`** — install integrations: `pip install -e ".[integrations]"`.
- **Missing API keys** — copy `.env.example` to `.env` and set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (skip if using `TRAIGENT_MOCK_MODE=true`).
- **Virtualenv confusion** — recreate: `deactivate; rm -rf .venv; python -m venv .venv; source .venv/bin/activate; pip install -e ".[integrations]"`.

---

Need help? Open an issue on GitHub or ping the team in the repository discussions.
