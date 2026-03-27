from pathlib import Path


class Walker:
    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return b"\x00" in f.read(1024)
        except OSError:
            return False
