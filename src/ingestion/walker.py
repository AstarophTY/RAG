import os
from pathlib import Path
from .constants import (
    IGNORED_DIRS,
    IGNORED_FILENAMES,
    ALL_EXTENSIONS,
)


class Walker:
    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return b"\x00" in f.read(1024)
        except OSError:
            return False

    @staticmethod
    def walk_dir(path_str: str) -> list[Path]:
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

                if filename in IGNORED_FILENAMES:
                    continue

                suffix = path.suffix.lower()
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
