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
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator: Generator | None = None

    @staticmethod
    def _load_dataset(path: Path) -> RagDataset:
        data = load_json(path)
        return RagDataset.model_validate(data)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_dir: str,
    ) -> Path:
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        dataset = self._load_dataset(path)
        rows: list[MinimalSearchResults] = []

        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            sources = self.retriever.retrieve(item.question, k=k)
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

    def answer_dataset(self, search_results_path: str, save_dir: str) -> Path:
        path = Path(search_results_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Search results file does not exist: {path}"
            )

        data = load_json(path)
        parsed = StudentSearchResults.model_validate(data)

        if self.generator is None:
            self.generator = Generator()

        rows: list[MinimalAnswer] = []
        for item in tqdm(parsed.search_results, desc="Answering dataset"):
            text = self.generator.answer(item.question, item.retrieved_sources)
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
