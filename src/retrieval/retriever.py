import re
import json
import bm25s
from pathlib import Path
from ..models import MinimalSource
from typing import Any


class Retriever:
    """Retrieve ranked source spans from persisted BM25 artifacts."""

    def __init__(self, index_base_path: str = "data/processed/bm25_index"):
        """Initialize retriever paths and lazy-loaded index placeholders."""
        self.base_path = Path(index_base_path)

        self.index_code: bm25s.BM25 | None = None
        self.index_docs: bm25s.BM25 | None = None
        self.meta_code: list[dict[str, Any]] = []
        self.meta_docs: list[dict[str, Any]] = []

    def _ensure_loaded(self) -> None:
        """Load BM25 indexes and metadata from disk if not already loaded."""
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
        """Convert text to lowercase alphanumeric BM25 tokens."""
        return re.sub(r"[^a-z0-9]", " ", text.lower()).split()

    @staticmethod
    def _query_hints(query: str) -> set[str]:
        """Extract query hints useful for file-path level reranking."""
        stop = {
            "what",
            "which",
            "when",
            "where",
            "how",
            "does",
            "about",
            "configuration",
            "constructor",
            "value",
            "default",
            "parameter",
            "method",
            "function",
            "class",
            "assert",
        }

        hints: set[str] = set()
        hints.update(re.findall(r"[a-z0-9_]{4,}", query.lower()))

        camel_parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+[0-9]*",
            query,
        )
        hints.update(part.lower() for part in camel_parts if len(part) >= 4)

        return {hint for hint in hints if hint not in stop}

    @staticmethod
    def _path_bonus(doc: dict[str, Any], hints: set[str]) -> float:
        """Compute a small reranking bonus from query hints and file path."""
        path_lc = str(doc.get("file_path", "")).lower()
        if not path_lc:
            return 0.0

        bonus = 0.0
        for hint in hints:
            if hint in path_lc:
                bonus += 1.5

        if "/models/" in path_lc and bonus > 0:
            bonus += 0.5

        return bonus

    def _bm25_ranked(
        self,
        query: str,
        tokens: list[str],
        candidate_k: int,
        selected_mode: str,
    ) -> list[dict[str, Any]]:
        """Return ranked metadata docs from BM25 for selected corpus mode."""
        if self.index_code is None or self.index_docs is None:
            return []

        queries = [tokens]
        docs_code: list[list[dict[str, Any]]] = [[]]
        scores_code: list[list[float]] = [[]]
        docs_docs: list[list[dict[str, Any]]] = [[]]
        scores_docs: list[list[float]] = [[]]

        if selected_mode in {"all", "code"}:
            docs_code, scores_code = self.index_code.retrieve(
                queries,
                corpus=self.meta_code,
                k=candidate_k,
            )

        if selected_mode in {"all", "docs"}:
            docs_docs, scores_docs = self.index_docs.retrieve(
                queries,
                corpus=self.meta_docs,
                k=candidate_k,
            )

        hints = self._query_hints(query)
        ranked = [
            (float(score) + self._path_bonus(doc, hints), doc)
            for doc, score in zip(docs_code[0], scores_code[0])
        ]
        ranked.extend(
            (float(score) + self._path_bonus(doc, hints), doc)
            for doc, score in zip(docs_docs[0], scores_docs[0])
        )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [doc for _score, doc in ranked]

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """Validate retrieval mode and normalize it to lowercase."""
        normalized = mode.strip().lower()
        if normalized not in {"all", "code", "docs"}:
            raise ValueError("mode must be one of: all, code, docs.")
        return normalized

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "all",
    ) -> list[MinimalSource]:
        """Retrieve top-k unique source spans for a query.

        Args:
            query: Natural language query text.
            k: Number of sources to return.
            mode: Corpus mode among `all`, `code`, or `docs`.

        Returns:
            Ranked list of minimal source spans.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        selected_mode = self._normalize_mode(mode)

        self._ensure_loaded()

        if self.index_code is None or self.index_docs is None:
            raise RuntimeError("Indexes are not loaded.")

        tokens = self._tokenize(query)
        if not tokens:
            return []
        candidate_k = max(k * 12, 80)
        ranked = self._bm25_ranked(
            query=query,
            tokens=tokens,
            candidate_k=candidate_k,
            selected_mode=selected_mode,
        )

        sources: list[MinimalSource] = []
        seen_spans: set[tuple[str, int, int]] = set()

        for doc in ranked:
            span = (
                str(doc["file_path"]),
                int(doc["first_character_index"]),
                int(doc["last_character_index"]),
            )
            if span in seen_spans:
                continue

            seen_spans.add(span)
            sources.append(
                MinimalSource(
                    file_path=doc["file_path"],
                    first_character_index=doc["first_character_index"],
                    last_character_index=doc["last_character_index"],
                )
            )
            if len(sources) >= k:
                break

        return sources
