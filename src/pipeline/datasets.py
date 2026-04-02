from pathlib import Path

from tqdm import tqdm

from src.generation.generator import Generator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.pipeline.helpers import build_output_path, load_json, save_json
from src.retrieval.retriever import Retriever


class DatasetService:
    """Service for dataset search and dataset answer generation."""

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

    @staticmethod
    def _load_dataset(path: Path) -> RagDataset:
        """Load and validate a dataset JSON file."""
        data = load_json(path)
        return RagDataset.model_validate(data)

    @staticmethod
    def _infer_mode_from_dataset_path(dataset_path: str) -> str:
        """Infer retrieval mode from dataset filename."""
        name = Path(dataset_path).name.lower()
        if "dataset_code" in name or "_code" in name:
            return "code"
        if "dataset_docs" in name or "_docs" in name:
            return "docs"
        return "all"

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_dir: str,
    ) -> Path:
        """Run retrieval for each dataset question and save output."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        dataset = self._load_dataset(path)
        mode = self._infer_mode_from_dataset_path(dataset_path)
        rows: list[MinimalSearchResults] = []

        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            sources = self.retriever.retrieve(item.question, k=k, mode=mode)
            rows.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=rows, k=k)
        output_path = build_output_path(
            save_dir=save_dir,
            input_path=dataset_path,
            suffix=f"k{k}_search_results",
        )
        save_json(output_path, output.model_dump())
        return output_path

    def answer_dataset(
        self,
        search_results_path: str,
        save_dir: str,
        model_name: str | None = None,
        max_new_tokens: int = 120,
        ollama_host: str | None = None,
    ) -> Path:
        """Generate answers from saved search results and save output."""
        path = Path(search_results_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Search results file does not exist: {path}"
            )

        data = load_json(path)
        parsed = StudentSearchResults.model_validate(data)

        generator = self._ensure_generator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            ollama_host=ollama_host,
        )

        rows: list[MinimalAnswer] = []
        for item in tqdm(parsed.search_results, desc="Answering dataset"):
            text = generator.answer(item.question, item.retrieved_sources)
            rows.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=text,
                )
            )

        output = StudentSearchResultsAndAnswer(search_results=rows, k=parsed.k)
        output_path = build_output_path(
            save_dir=save_dir,
            input_path=search_results_path,
            suffix="answers",
        )
        save_json(output_path, output.model_dump())
        return output_path
