"""JSON file helpers shared by pipeline entry points."""

import json
from pathlib import Path
from typing import Any, Dict


def load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON file and require an object at its root."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"received {type(data).__name__}."
        )

    return data
