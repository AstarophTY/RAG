import ast
from pathlib import Path
from sys import stderr


class PythonChunker:
    def __init__(self) -> None:
        pass

    def _get_line_offsets(self, text: str) -> list[int]:
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def extract_chunks(self, file_path: Path) -> list[tuple[str, int, int]]:
        try:
            source_code: str = file_path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Read error in {file_path}: {e}", file=stderr)
            return []

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            print(f"SyntaxError in {file_path}", file=stderr)
            return []

        line_offsets: list[int] = self._get_line_offsets(source_code)
        chunks: list[tuple[str, int, int]] = []

        for node in ast.walk(tree):
            if isinstance(node, (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                )
            ):
                if hasattr(node, "end_lineno") and node.end_lineno is not None:
                    try:
                        start_idx = (
                            line_offsets[node.lineno - 1] + node.col_offset
                        )
                        end_idx = (
                            line_offsets[node.end_lineno - 1]
                            + (node.end_col_offset or 0)
                        )

                        content = source_code[start_idx:end_idx]
                        chunks.append((content, start_idx, end_idx))
                    except (IndexError, AttributeError):
                        continue

        return sorted(chunks, key=lambda x: x[1])
