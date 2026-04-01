import re
import json
import bm25s
from pathlib import Path
from src.chunker.chunk import Chunk


class Index:
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9]", " ", text.lower()).split()

    def index(
        self,
        chunks_code: list[Chunk],
        chunks_docs: list[Chunk]
    ) -> tuple[bm25s.BM25, bm25s.BM25]:
        corpus_tokens_code = [
            self._tokenize(chunk.content) for chunk in chunks_code
        ]
        corpus_tokens_docs = [
            self._tokenize(chunk.content) for chunk in chunks_docs
        ]

        index_code = bm25s.BM25()
        index_code.index(corpus_tokens_code)

        index_docs = bm25s.BM25()
        index_docs.index(corpus_tokens_docs)

        code_dir = Path("data/processed/bm25_index/code")
        docs_dir = Path("data/processed/bm25_index/docs")
        code_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)

        index_code.save(str(code_dir))
        index_docs.save(str(docs_dir))

        self._save_metadata(chunks_code, code_dir / "metadata.json")
        self._save_metadata(chunks_docs, docs_dir / "metadata.json")

        return (index_code, index_docs)

    @staticmethod
    def _save_metadata(chunks: list[Chunk], output_file: Path) -> None:
        metadata = [
            {
                "file_path": str(c.file_path),
                "first_character_index": c.first_character_index,
                "last_character_index": c.last_character_index
            }
            for c in chunks
        ]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
