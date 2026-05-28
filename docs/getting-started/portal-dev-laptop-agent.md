# Portal-dev laptop agent setup

Use this when running Traigent from this laptop against the dev backend.

## Backend URL

Use the API host, not the portal host:

```bash
export TRAIGENT_BACKEND_URL=https://api-dev.traigent.ai
export TRAIGENT_API_URL=https://api-dev.traigent.ai
export TRAIGENT_CLOUD_API_URL=https://api-dev.traigent.ai
```

`portal-dev.traigent.ai` is the browser application. SDK and agent code should call
`api-dev.traigent.ai`.

## API key

Use a key that validates against `POST /api/v1/keys/validate`. Do not paste keys into shell
history. Store them through the SDK credential helper or a hidden prompt:

```bash
python - <<'PY'
import getpass
import json
from pathlib import Path

path = Path.home() / ".traigent" / "credentials.json"
path.parent.mkdir(mode=0o700, exist_ok=True)

data = {
    "backend_url": "https://api-dev.traigent.ai",
    "api_key": getpass.getpass("Traigent API key: "),
}
path.write_text(json.dumps(data, indent=2) + "\n")
path.chmod(0o600)
print(f"Wrote {path}")
PY
```

## Smoke checks

```bash
traigent auth whoami "$(python - <<'PY'
import json
from pathlib import Path
print(json.loads((Path.home() / ".traigent" / "credentials.json").read_text())["api_key"])
PY
)"
```

If a newly minted key is listed in the UI but `whoami` or `/keys/validate` returns 401,
mint a new key after the backend fix that verifies generated keys before returning them.
Do not keep retrying an unpersisted one-time key; it cannot be recovered after creation.
