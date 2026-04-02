import fire
from src.evaluation.evaluator import Evaluator
from src.pipeline.datasets import DatasetService
from src.pipeline.indexing import IndexingService
from src.pipeline.query import QueryService


class Pipeline:
    def __init__(self) -> None:
        self.index_service = IndexingService()
        self.query_service = QueryService()
        self.dataset_service = DatasetService()
        self.evaluator = Evaluator()

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        print(f"Starting indexing for repository: {repo_path}")
        stats = self.index_service.run(
            repo_path=repo_path,
            max_chunk_size=max_chunk_size,
        )

        print(
            "Indexing complete: "
            f"{stats['files']} files scanned, "
            f"{stats['code_chunks']} code chunks, "
            f"{stats['docs_chunks']} docs chunks, "
            f"{stats['skipped']} files skipped, "
            f"{stats['duration']} seconds."
        )

    def search(self, query: str, k: int = 5, mode: str = "all") -> None:
        results = self.query_service.search(query=query, k=k, mode=mode)

        if not results:
            print("No matching sources found.")
            return

        for i, res in enumerate(results, start=1):
            print(
                f"{i}. {res.file_path} "
                "(chars "
                f"{res.first_character_index}-{res.last_character_index})"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        output_path = self.dataset_service.search_dataset(
            dataset_path=dataset_path,
            k=k,
            save_dir=save_directory,
        )
        print(f"Saved search results to: {output_path}")

    def answer(self, query: str, k: int = 5, mode: str = "all") -> None:
        answer, sources = self.query_service.answer(
            query=query,
            k=k,
            mode=mode,
        )

        print("Answer:")
        print(answer)
        print("\nSources:")
        for i, source in enumerate(sources, start=1):
            print(
                f"{i}. {source.file_path} "
                f"[{source.first_character_index}, "
                f"{source.last_character_index}]"
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        output_path = self.dataset_service.answer_dataset(
            search_results_path=student_search_results_path,
            save_dir=save_directory,
        )
        print(f"Saved answers to: {output_path}")

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        report = self.evaluator.evaluate(
            student_answer_path=student_answer_path,
            dataset_path=dataset_path,
            k=k,
        )

        print("Evaluation summary")
        print(f"Questions: {int(report['question_count'])}")
        print(f"Recall@1:  {report['recall@1']:.4f}")
        print(f"Recall@3:  {report['recall@3']:.4f}")
        print(f"Recall@5:  {report['recall@5']:.4f}")
        print(f"Recall@10: {report['recall@10']:.4f}")


if __name__ == "__main__":
    fire.Fire(Pipeline)
