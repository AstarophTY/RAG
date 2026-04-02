import inspect
import sys

import fire
from src.evaluation.evaluator import Evaluator
from src.pipeline.datasets import DatasetService
from src.pipeline.indexing import IndexingService
from src.pipeline.query import QueryService


def _validate_no_extra_flags(argv: list[str]) -> None:
    """Reject unknown flags for Pipeline subcommands before invoking Fire."""
    if not argv:
        return

    command_token = argv[0]
    if command_token.startswith("-"):
        return

    command = command_token.replace("-", "_")
    method = getattr(Pipeline, command, None)
    if method is None or not callable(method):
        return

    signature = inspect.signature(method)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    fire_flags = {
        "help",
        "verbose",
        "interactive",
        "completion",
        "trace",
        "separator",
    }

    unknown: set[str] = set()
    for token in argv[1:]:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        flag = token[2:].split("=", 1)[0].replace("-", "_")
        if flag in fire_flags:
            continue
        if flag not in allowed:
            unknown.add(flag)

    if unknown:
        allowed_list = ", ".join(sorted(allowed))
        unknown_list = ", ".join(sorted(unknown))
        raise SystemExit(
            "Unknown flag(s) for command "
            f"'{command_token}': {unknown_list}. "
            f"Allowed flags: {allowed_list}."
        )


class Pipeline:
    def __init__(self) -> None:
        """Initialize pipeline services for indexing, query, and evaluation."""
        self.index_service = IndexingService()
        self.query_service = QueryService()
        self.dataset_service = DatasetService()
        self.evaluator = Evaluator()

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Run repository indexing and print progress statistics."""
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
        """Search for relevant source spans and print ranked results."""
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
        """Run retrieval on a dataset and save serialized search results."""
        output_path = self.dataset_service.search_dataset(
            dataset_path=dataset_path,
            k=k,
            save_dir=save_directory,
        )
        print(f"Saved search results to: {output_path}")

    def answer(
        self,
        query: str,
        k: int = 5,
        mode: str = "all",
        model_name: str | None = None,
        max_new_tokens: int = 120,
        ollama_host: str | None = None,
    ) -> None:
        """Generate an answer and print supporting sources."""
        answer, sources = self.query_service.answer(
            query=query,
            k=k,
            mode=mode,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            ollama_host=ollama_host,
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
        model_name: str | None = None,
        max_new_tokens: int = 120,
        ollama_host: str | None = None,
    ) -> None:
        """Generate answers from saved retrieval results and persist them."""
        output_path = self.dataset_service.answer_dataset(
            search_results_path=student_search_results_path,
            save_dir=save_directory,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            ollama_host=ollama_host,
        )
        print(f"Saved answers to: {output_path}")

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Evaluate retrieval performance and print recall metrics."""
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
    _validate_no_extra_flags(sys.argv[1:])
    fire.Fire(Pipeline)
