from pathlib import Path

from src.chunker.chunker_python import PythonChunker
from src.chunker.chunker_text import TextChunker


def test_text_chunks_match_their_offsets() -> None:
    text = "intro\n\n" + "a" * 300 + "b" * 300 + "c" * 300 + "\n\noutro"
    chunks = TextChunker.extract_chunks(text, chunk_size=512, overlap=100)

    assert len(chunks) > 2
    for content, start, end in chunks:
        assert start < end
        assert text[start:end] == content


def test_text_chunks_cover_long_paragraph() -> None:
    paragraph = "".join(chr(ord("a") + i % 26) for i in range(1500))
    chunks = TextChunker.extract_chunks(paragraph, chunk_size=512,
                                        overlap=100)

    assert chunks[0][1] == 0
    assert chunks[-1][2] == len(paragraph)
    for (_, _, prev_end), (_, next_start, _) in zip(chunks, chunks[1:]):
        assert next_start < prev_end


def test_short_paragraphs_are_kept_whole() -> None:
    text = "first paragraph\n\nsecond paragraph"
    chunks = TextChunker.extract_chunks(text)

    assert [c[0] for c in chunks] == ["first paragraph", "second paragraph"]


def test_python_chunker_extracts_functions_and_classes(
    tmp_path: Path,
) -> None:
    source = (
        "import os\n\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "class Box:\n"
        "    def size(self):\n"
        "        return 1\n"
    )
    file = tmp_path / "sample.py"
    file.write_text(source)

    chunks = PythonChunker().extract_chunks(file)
    contents = [c[0] for c in chunks]

    assert contents[0].startswith("def add")
    assert any(c.startswith("class Box") for c in contents)
    assert any(c.startswith("def size") for c in contents)
    for content, start, end in chunks:
        assert source[start:end] == content


def test_python_chunker_ignores_invalid_syntax(tmp_path: Path) -> None:
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n")

    assert PythonChunker().extract_chunks(file) == []
