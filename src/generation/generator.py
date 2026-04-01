from pathlib import Path
from typing import Any

from src.chunker.chunk import Chunk
from src.chunker.save_chunks import SaveChunks
from src.models import MinimalSource


class Generator:
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        max_new_tokens: int = 180,
        ollama_host: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host
        self._client: Any | None = None
        self._chunk_lookup = self._build_chunk_lookup()

    @staticmethod
    def _build_chunk_lookup() -> dict[tuple[str, int, int], str]:
        storage = SaveChunks()
        code_chunks, docs_chunks = storage.load_all()
        lookup: dict[tuple[str, int, int], str] = {}

        for chunk in [*code_chunks, *docs_chunks]:
            validated = Chunk.model_validate(chunk)
            key = (
                validated.file_path,
                validated.first_character_index,
                validated.last_character_index,
            )
            lookup[key] = validated.content

        return lookup

    def _load_model(self) -> None:
        if self._client is not None:
            return

        import importlib

        ollama_module = importlib.import_module("ollama")
        client_cls = getattr(ollama_module, "Client")

        if self.ollama_host:
            self._client = client_cls(host=self.ollama_host)
        else:
            self._client = client_cls()

        self._client.list()

    def _read_source_text(self, source: MinimalSource) -> str:
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )
        from_cache = self._chunk_lookup.get(key)
        if from_cache is not None:
            return from_cache

        path = Path(source.file_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")

        return text[source.first_character_index:source.last_character_index]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _build_context(
        self,
        sources: list[MinimalSource],
        token_budget: int,
    ) -> str:
        pieces: list[str] = []
        total_tokens = 0

        for source in sources:
            content = self._read_source_text(source)
            snippet = (
                f"File: {source.file_path}\n"
                f"Span: {source.first_character_index}-"
                f"{source.last_character_index}\n"
                f"Content:\n{content}\n"
            )
            token_count = self._estimate_tokens(snippet)
            if total_tokens + token_count > token_budget:
                break
            pieces.append(snippet)
            total_tokens += token_count

        return "\n---\n".join(pieces)

    def answer(self, query: str, sources: list[MinimalSource]) -> str:
        """Generate an answer from query and retrieved sources."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        if not sources:
            return "I could not find supporting sources for this question."

        try:
            self._load_model()
        except Exception as error:
            return (
                "Model loading failed. "
                f"Please verify local model access: {error}"
            )

        if self._client is None:
            return "Model initialization failed."

        max_model_tokens = 4096
        prompt_budget = max(512, max_model_tokens - self.max_new_tokens - 120)
        context_budget = max(320, prompt_budget - 200)
        context = self._build_context(sources, token_budget=context_budget)

        prompt = (
            "You are answering questions about the vLLM codebase.\n"
            "Use only the provided context.\n"
            "If context is insufficient, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {cleaned_query}\n"
            "Answer:"
        )

        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0,
                },
            )
            answer_text = str(response.get("response", ""))
            return answer_text.strip() or "No answer generated."
        except Exception as error:
            return f"Answer generation failed: {error}"
