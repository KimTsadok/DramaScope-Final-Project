"""Small environment-file helpers shared by pipeline integrations."""

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs without overwriting existing environment values."""
    env_path = Path(dotenv_path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
