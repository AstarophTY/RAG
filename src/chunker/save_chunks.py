from chunk import Chunk
from pathlib import Path
import json


class SaveChunk:
    @staticmethod
    def _save_chunks(chunks: list[Chunk], output_file: str) -> None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        data = [chunk for chunk in chunks]

        with open(output_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_chunks(input_file: str) -> list[Chunk]:
        with open(input_file, "r") as f:
            data = json.load(f)
        return [Chunk(**chunk) for chunk in data]

    def save_all(
        self,
        code_chunks: list[Chunk],
        docs_chunks: list[Chunk]
    ) -> None:
        self._save_chunks(code_chunks, "data/chunks/code_chunks.json")
        self._save_chunks(docs_chunks, "data/chunks/docs_chunks.json")

    def load_all(self):
        return (
            self._load_chunks("data/chunks/code_chunks.json"),
            self._load_chunks("data/chunks/docs_chunks.json")
        )
