import json
from pathlib import Path
from src.chunker.chunk import Chunk


class SaveChunks:
    @staticmethod
    def _save_chunks(chunks: list[Chunk], output_file: str) -> None:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.model_dump() for chunk in chunks]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_chunks(input_file: str) -> list[Chunk]:
        path = Path(input_file)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [Chunk(**chunk) for chunk in data]

    def save_all(
        self,
        code_chunks: list[Chunk],
        docs_chunks: list[Chunk]
    ) -> None:
        self._save_chunks(
            code_chunks,
            "data/processed/chunks/code_chunks.json"
        )
        self._save_chunks(
            docs_chunks,
            "data/processed/chunks/docs_chunks.json"
        )

    def load_all(self) -> tuple[list[Chunk], list[Chunk]]:
        return (
            self._load_chunks("data/processed/chunks/code_chunks.json"),
            self._load_chunks("data/processed/chunks/docs_chunks.json")
        )
