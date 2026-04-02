from pathlib import Path
from sys import stderr
from tqdm import tqdm
import time

from src.chunker.chunk import Chunk
from src.chunker.chunker_python import PythonChunker
from src.chunker.chunker_text import TextChunker
from src.chunker.save_chunks import SaveChunks
from src.chunker.walker import Walker
from src.index.bm25_index import Index


class IndexingService:
    def __init__(self) -> None:
        """Initialize chunkers, storage, and index writer."""
        self.walker = Walker()
        self.py_chunker = PythonChunker()
        self.text_chunker = TextChunker()
        self.storage = SaveChunks()
        self.indexer = Index()

    @staticmethod
    def _split_large_chunk(
        content: str,
        start: int,
        max_size: int,
        overlap: int,
    ) -> list[tuple[str, int, int]]:
        """Split one large chunk into overlapping subchunks."""
        if len(content) <= max_size:
            end = start + len(content)
            return [(content, start, end)]

        parts: list[tuple[str, int, int]] = []
        local_start = 0

        while local_start < len(content):
            local_end = min(local_start + max_size, len(content))
            text = content[local_start:local_end]
            global_start = start + local_start
            global_end = start + local_end
            parts.append((text, global_start, global_end))

            if local_end >= len(content):
                break

            local_start = local_end - min(overlap, max_size // 4)

        return parts

    def _split_raw_chunks(
        self,
        raw_chunks: list[tuple[str, int, int]],
        max_size: int,
        overlap: int,
    ) -> list[tuple[str, int, int]]:
        """Apply large-chunk splitting to all raw chunks."""
        parts: list[tuple[str, int, int]] = []
        for content, start, _end in raw_chunks:
            parts.extend(
                self._split_large_chunk(
                    content=content,
                    start=start,
                    max_size=max_size,
                    overlap=overlap,
                )
            )
        return parts

    @staticmethod
    def _to_code_chunks(
        file_path: Path,
        raw_chunks: list[tuple[str, int, int]],
    ) -> list[Chunk]:
        """Convert raw tuples to typed code chunk models."""
        return [
            Chunk(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                content=content,
                chunk_type="code",
            )
            for content, start, end in raw_chunks
        ]

    @staticmethod
    def _to_docs_chunks(
        file_path: Path,
        raw_chunks: list[tuple[str, int, int]],
    ) -> list[Chunk]:
        """Convert raw tuples to typed docs chunk models."""
        return [
            Chunk(
                file_path=str(file_path),
                first_character_index=start,
                last_character_index=end,
                content=content,
                chunk_type="docs",
            )
            for content, start, end in raw_chunks
        ]

    def run(self, repo_path: str, max_chunk_size: int) -> dict[str, int]:
        """Run indexing pipeline and return summary statistics."""
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo}")
        if not repo.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repo}"
            )
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0.")

        files = self.walker.walk_dir(str(repo))
        if not files:
            raise RuntimeError("No supported files were found to index.")

        code_chunks: list[Chunk] = []
        docs_chunks: list[Chunk] = []
        skipped = 0

        start_time = time.perf_counter()
        for file_path in tqdm(files, desc="Indexing files", unit="file"):
            try:
                if file_path.suffix.lower() == ".py":
                    raw = self.py_chunker.extract_chunks(file_path)
                    if not raw:
                        try:
                            source_text = file_path.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            source_text = file_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )

                        raw = self.text_chunker.extract_chunks(
                            source_text,
                            chunk_size=max_chunk_size,
                            overlap=min(120, max_chunk_size // 5),
                        )

                    raw = self._split_raw_chunks(
                        raw_chunks=raw,
                        max_size=max_chunk_size,
                        overlap=120,
                    )
                    code_chunks.extend(self._to_code_chunks(file_path, raw))
                    continue

                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = file_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )

                raw = self.text_chunker.extract_chunks(
                    text,
                    chunk_size=max_chunk_size,
                    overlap=min(100, max_chunk_size // 5),
                )
                docs_chunks.extend(self._to_docs_chunks(file_path, raw))
            except OSError as error:
                skipped += 1
                print(
                    f"Skipping unreadable file {file_path}: {error}",
                    file=stderr,
                )
            except Exception as error:
                skipped += 1
                print(
                    "Skipping file due to unexpected error "
                    f"{file_path}: {error}",
                    file=stderr,
                )

        if not code_chunks and not docs_chunks:
            raise RuntimeError("No chunks were produced. Nothing to index.")

        self.storage.save_all(code_chunks, docs_chunks)
        self.indexer.index(code_chunks, docs_chunks)

        end_time = time.perf_counter()

        return {
            "files": len(files),
            "code_chunks": len(code_chunks),
            "docs_chunks": len(docs_chunks),
            "skipped": skipped,
            "duration": int(end_time - start_time),
        }
