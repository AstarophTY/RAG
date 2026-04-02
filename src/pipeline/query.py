from src.generation.generator import Generator
from src.models import MinimalSource
from src.retrieval.retriever import Retriever


class QueryService:
    def __init__(self) -> None:
        """Initialize retriever and lazy generator."""
        self.retriever = Retriever()
        self.generator: Generator | None = None

    def _ensure_generator(
        self,
        model_name: str | None,
        max_new_tokens: int,
        ollama_host: str | None,
    ) -> Generator:
        """Create or update generator with the requested model settings."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        if self.generator is None:
            self.generator = Generator(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                ollama_host=ollama_host,
            )
            return self.generator

        if model_name is not None:
            self.generator.set_model(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
        else:
            self.generator.max_new_tokens = max_new_tokens

        if (
            ollama_host is not None
            and self.generator.ollama_host != ollama_host
        ):
            self.generator.ollama_host = ollama_host
            self.generator._client = None

        return self.generator

    def search(
        self,
        query: str,
        k: int,
        mode: str = "all",
    ) -> list[MinimalSource]:
        """Retrieve top-k source spans for one query."""
        text = query.strip()
        if not text:
            raise ValueError("Query cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        return self.retriever.retrieve(text, k=k, mode=mode)

    def answer(
        self,
        query: str,
        k: int,
        mode: str = "all",
        model_name: str | None = None,
        max_new_tokens: int = 120,
        ollama_host: str | None = None,
    ) -> tuple[str, list[MinimalSource]]:
        """Generate one answer with its retrieved supporting sources."""
        sources = self.search(query=query, k=k, mode=mode)
        generator = self._ensure_generator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            ollama_host=ollama_host,
        )
        answer = generator.answer(query.strip(), sources)
        return answer, sources
