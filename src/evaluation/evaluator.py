import json
from pathlib import Path
from typing import Any

from src.models import MinimalSearchResults, RagDataset, StudentSearchResults


class Evaluator:
    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON root in {path}: expected object.")
        return payload

    @staticmethod
    def _span_overlap_ratio(
        first_start: int,
        first_end: int,
        second_start: int,
        second_end: int,
    ) -> float:
        intersection = max(
            0,
            min(first_end, second_end) - max(first_start, second_start),
        )
        gt_length = max(1, first_end - first_start)
        return intersection / gt_length

    def _is_found(
        self,
        gt_source: dict[str, Any],
        predicted_sources: list[dict[str, Any]],
        threshold: float,
    ) -> bool:
        gt_file = str(gt_source["file_path"])
        gt_start = int(gt_source["first_character_index"])
        gt_end = int(gt_source["last_character_index"])

        for source in predicted_sources:
            if str(source["file_path"]) != gt_file:
                continue
            pred_start = int(source["first_character_index"])
            pred_end = int(source["last_character_index"])
            ratio = self._span_overlap_ratio(
                first_start=gt_start,
                first_end=gt_end,
                second_start=pred_start,
                second_end=pred_end,
            )
            if ratio >= threshold:
                return True
        return False

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
        overlap_threshold: float = 0.05,
    ) -> dict[str, float]:
        """Evaluate recall@1/@3/@5/@10 for student retrieval outputs."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if not 0 < overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be in (0, 1].")

        student_file = Path(student_answer_path)
        dataset_file = Path(dataset_path)
        if not student_file.exists():
            raise FileNotFoundError(
                f"Student file does not exist: {student_file}"
            )
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: {dataset_file}"
            )

        raw_student = self._load_json(student_file)
        raw_dataset = self._load_json(dataset_file)

        student = StudentSearchResults.model_validate(raw_student)
        dataset = RagDataset.model_validate(raw_dataset)

        gt_by_question: dict[str, list[dict[str, Any]]] = {}
        for question in dataset.rag_questions:
            serialized = question.model_dump()
            sources = serialized.get("sources", [])
            if not isinstance(sources, list):
                continue
            gt_by_question[question.question_id] = [
                source for source in sources if isinstance(source, dict)
            ]

        predictions_by_question: dict[str, list[dict[str, Any]]] = {}
        for result in student.search_results:
            parsed_result = MinimalSearchResults.model_validate(result)
            predictions_by_question[parsed_result.question_id] = [
                source.model_dump()
                for source in parsed_result.retrieved_sources
            ]

        cutoffs = [1, 3, 5, 10]
        recall_sums = {cutoff: 0.0 for cutoff in cutoffs}
        evaluated_questions = 0

        for question_id, gt_sources in gt_by_question.items():
            if not gt_sources:
                continue
            predicted = predictions_by_question.get(question_id, [])
            evaluated_questions += 1

            for cutoff in cutoffs:
                top_sources = predicted[: min(cutoff, k)]
                found = 0
                for gt_source in gt_sources:
                    if self._is_found(
                        gt_source,
                        top_sources,
                        overlap_threshold,
                    ):
                        found += 1
                recall_sums[cutoff] += found / len(gt_sources)

        if evaluated_questions == 0:
            raise RuntimeError("No comparable questions found for evaluation.")

        return {
            "question_count": float(evaluated_questions),
            "recall@1": recall_sums[1] / evaluated_questions,
            "recall@3": recall_sums[3] / evaluated_questions,
            "recall@5": recall_sums[5] / evaluated_questions,
            "recall@10": recall_sums[10] / evaluated_questions,
        }
