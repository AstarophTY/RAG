import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk and validate object root type."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root in {path}: expected object.")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save a JSON object to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_output_path(save_dir: str, input_path: str, suffix: str) -> Path:
    """Build an output file path derived from the input stem and suffix."""
    input_stem = Path(input_path).stem
    return Path(save_dir) / f"{input_stem}_{suffix}.json"
