import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.evaluator import Evaluator


def _source(path: str, start: int, end: int) -> dict[str, Any]:
    return {
        "file_path": path,
        "first_character_index": start,
        "last_character_index": end,
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    file = tmp_path / name
    file.write_text(json.dumps(payload))
    return str(file)


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    return _write(tmp_path, "dataset.json", {"rag_questions": [
        {"question_id": "q1", "question": "Q1?", "answer": "A1",
         "sources": [_source("a.py", 0, 100)]},
        {"question_id": "q2", "question": "Q2?", "answer": "A2",
         "sources": [_source("b.md", 50, 150)]},
    ]})


def test_recall_by_rank(tmp_path: Path, dataset: str) -> None:
    results = _write(tmp_path, "results.json", {"k": 3, "search_results": [
        # q1: the right chunk is ranked first.
        {"question_id": "q1", "question": "Q1?", "retrieved_sources": [
            _source("a.py", 10, 90), _source("x.py", 0, 10)]},
        # q2: the right chunk is ranked third.
        {"question_id": "q2", "question": "Q2?", "retrieved_sources": [
            _source("x.py", 0, 10), _source("b.md", 0, 40),
            _source("b.md", 60, 120)]},
    ]})

    report = Evaluator().evaluate(results, dataset, k=10)

    assert report["question_count"] == 2
    assert report["recall@1"] == pytest.approx(0.5)
    assert report["recall@3"] == pytest.approx(1.0)


def test_wrong_file_is_not_a_match(tmp_path: Path, dataset: str) -> None:
    results = _write(tmp_path, "results.json", {"k": 1, "search_results": [
        {"question_id": "q1", "question": "Q1?",
         "retrieved_sources": [_source("other.py", 0, 100)]},
    ]})

    report = Evaluator().evaluate(results, dataset, k=10)

    assert report["recall@10"] == 0.0


def test_invalid_k_is_rejected(tmp_path: Path, dataset: str) -> None:
    with pytest.raises(ValueError):
        Evaluator().evaluate(dataset, dataset, k=0)
