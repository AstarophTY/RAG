import os
from pathlib import Path
from .constants import (
    IGNORED_DIRS,
    ALL_EXTENSIONS,
    IGNORED_FILES
)


class Walker:
    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Detect whether a file appears to contain binary content."""
        try:
            with open(path, "rb") as f:
                return b"\x00" in f.read(1024)
        except OSError:
            return False

    @staticmethod
    def walk_dir(path_str: str) -> list[Path]:
        """Return supported, non-empty, non-binary files under a directory."""
        result: list[Path] = []
        root = Path(path_str)

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in IGNORED_DIRS
                and not d.startswith(".")
                and not d.endswith(".egg-info")
            ]

            for filename in filenames:
                path = Path(dirpath) / filename
                suffix = path.suffix.lower()

                if filename in IGNORED_FILES:
                    continue

                if suffix not in ALL_EXTENSIONS:
                    continue

                try:
                    if path.stat().st_size == 0:
                        continue

                    if Walker._is_binary(path):
                        continue

                    result.append(path)

                except OSError:
                    continue

        return result
