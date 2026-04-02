from src.generation.generator import Generator
from src.models import MinimalSource
from src.retrieval.retriever import Retriever


class QueryService:
    def __init__(self) -> None:
        """Initialize retriever and lazy generator."""
        self.retriever = Retriever()
        self.generator: Generator | None = None

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
    ) -> tuple[str, list[MinimalSource]]:
        """Generate one answer with its retrieved supporting sources."""
        sources = self.search(query=query, k=k, mode=mode)
        if self.generator is None:
            self.generator = Generator()
        answer = self.generator.answer(query.strip(), sources)
        return answer, sources
