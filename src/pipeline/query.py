"""Query service for single-question search and answer."""

from src.generation.generator import Generator
from src.models import MinimalSource
from src.retrieval.retriever import Retriever


class QueryService:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator: Generator | None = None

    def search(self, query: str, k: int) -> list[MinimalSource]:
        text = query.strip()
        if not text:
            raise ValueError("Query cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        return self.retriever.retrieve(text, k=k)

    def answer(self, query: str, k: int) -> tuple[str, list[MinimalSource]]:
        sources = self.search(query=query, k=k)
        if self.generator is None:
            self.generator = Generator()
        answer = self.generator.answer(query.strip(), sources)
        return answer, sources
