import re
import json
import bm25s
from pathlib import Path
from ..models import MinimalSource


class Retriever:
    def __init__(self, index_base_path: str = "data/processed/bm25_index"):
        self.base_path = Path(index_base_path)
        self.index_code: bm25s.BM25 | None = None
        self.index_docs: bm25s.BM25 | None = None
        self.meta_code: list[dict] = []
        self.meta_docs: list[dict] = []

    def _ensure_loaded(self) -> None:
        if self.index_code is not None and self.index_docs is not None:
            return

        code_dir = self.base_path / "code"
        docs_dir = self.base_path / "docs"
        code_metadata = code_dir / "metadata.json"
        docs_metadata = docs_dir / "metadata.json"

        required_paths = [code_dir, docs_dir, code_metadata, docs_metadata]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "BM25 index files are missing. Run the indexing step first. "
                f"Missing: {', '.join(missing)}"
            )

        self.index_code = bm25s.BM25.load(str(code_dir))
        self.index_docs = bm25s.BM25.load(str(docs_dir))

        with open(code_metadata, "r", encoding="utf-8") as f:
            self.meta_code = json.load(f)

        with open(docs_metadata, "r", encoding="utf-8") as f:
            self.meta_docs = json.load(f)

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-z0-9]", " ", text.lower()).split()

    def retrieve(self, query: str, k: int = 5) -> list[MinimalSource]:
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        self._ensure_loaded()

        if self.index_code is None or self.index_docs is None:
            raise RuntimeError("Indexes are not loaded.")

        tokens = self._tokenize(query)
        if not tokens:
            return []
        queries = [tokens]

        docs_code, scores_code = self.index_code.retrieve(
            queries,
            corpus=self.meta_code,
            k=k
        )
        docs_docs, scores_docs = self.index_docs.retrieve(
            queries,
            corpus=self.meta_docs,
            k=k
        )

        results_code = zip(docs_code[0], scores_code[0])
        results_docs = zip(docs_docs[0], scores_docs[0])

        all_results = []
        for doc, score in results_code:
            all_results.append({"score": float(score), "meta": doc})

        for doc, score in results_docs:
            all_results.append({"score": float(score), "meta": doc})

        all_results.sort(key=lambda x: x["score"], reverse=True)

        final_sources = []
        seen_files = set()

        for item in all_results:
            doc = item["meta"]
            file_path = doc["file_path"]

            if file_path in seen_files:
                continue

            seen_files.add(file_path)

            final_sources.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=doc["first_character_index"],
                    last_character_index=doc["last_character_index"]
                )
            )

            if len(final_sources) == k:
                break

        return final_sources
