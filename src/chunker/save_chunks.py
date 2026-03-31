import json
from pathlib import Path
from dataclasses import asdict
from src.chunker.chunk import Chunk


class SaveChunk:
    def __init__(self, base_path: str = "data/processed/chunks"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "code": self.base_path / "code_chunks.json",
            "docs": self.base_path / "docs_chunks.json"
        }

    def save_all(
        self,
        code_chunks: list[Chunk],
        docs_chunks: list[Chunk]
    ) -> None:
        self._save(code_chunks, self.paths["code"])
        self._save(docs_chunks, self.paths["docs"])

    def load_all(self) -> tuple[list[Chunk], list[Chunk]]:
        return (
            self._load(self.paths["code"]),
            self._load(self.paths["docs"])
        )

    def _save(self, chunks: list[Chunk], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                [asdict(c) for c in chunks], f, ensure_ascii=False, indent=2
            )

    def _load(self, path: Path) -> list[Chunk]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [Chunk(**data) for data in json.load(f)]
