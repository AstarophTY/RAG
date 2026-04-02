import re
from pathlib import Path
from typing import Any

from src.chunker.chunk import Chunk
from src.chunker.save_chunks import SaveChunks
from src.models import MinimalSource


class Generator:
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        max_new_tokens: int = 120,
        ollama_host: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host
        self._client: Any | None = None
        self._chunk_lookup = self._build_chunk_lookup()

    @staticmethod
    def _build_chunk_lookup() -> dict[tuple[str, int, int], str]:
        storage = SaveChunks()
        code_chunks, docs_chunks = storage.load_all()
        lookup: dict[tuple[str, int, int], str] = {}
        for chunk in [*code_chunks, *docs_chunks]:
            validated = Chunk.model_validate(chunk)
            key = (
                validated.file_path,
                validated.first_character_index,
                validated.last_character_index,
            )
            lookup[key] = validated.content
        return lookup

    def _load_model(self) -> None:
        if self._client is not None:
            return
        import importlib
        ollama_module = importlib.import_module("ollama")
        client_cls = getattr(ollama_module, "Client")
        self._client = client_cls(host=self.ollama_host) \
            if self.ollama_host else client_cls()
        self._client.list()

    def _read_source_text(self, source: MinimalSource) -> str:
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )
        cached = self._chunk_lookup.get(key)
        if cached is not None:
            return cached

        path = Path(source.file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[source.first_character_index:source.last_character_index]

    @staticmethod
    def _query_keywords(query: str) -> set[str]:
        stop = {
            "what", "which", "when", "where", "how", "with", "from",
            "that", "this", "does", "must", "have", "true", "false",
            "the", "are", "for", "vllm",
        }
        words = re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())
        return {w for w in words if w not in stop}

    def _extract_relevant_lines(self, query: str, content: str) -> str:
        keywords = self._query_keywords(query)
        scored: list[tuple[int, str]] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()

            if len(line) < 10:
                continue
            if line.startswith(("#", '"""', "'''", "import ", "from ")):
                continue

            line_lc = line.lower()
            score = sum(1 for kw in keywords if kw in line_lc)

            if "=" in line and " and " in line_lc:
                score += 2

            if score > 0:
                scored.append((score, line))

        if not scored:
            return content[:600].strip()

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(line for _, line in scored[:6])

    def _fallback_from_context(self, query: str, context: str) -> str:
        keywords = self._query_keywords(query)
        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip() and not line.strip().startswith("[Source")
        ]
        if not lines:
            return ""

        query_lc = query.lower()
        if "supported" in query_lc:
            quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", context)
            values = [a or b for a, b in quoted if (a or b)]
            values.extend(re.findall(r"`([^`]+)`", context))
            values = [v.strip() for v in values if 1 <= len(v.strip()) <= 32]
            unique_values: list[str] = []
            for val in values:
                if val not in unique_values:
                    unique_values.append(val)
            if len(unique_values) >= 2:
                joined = " and ".join(
                    [f'"{v}"' for v in unique_values[:2]]
                )
                return self._enforce_answer_style(
                    f"supported values are {joined}."
                )

        def score_line(line: str) -> int:
            line_lc = line.lower()
            score = sum(1 for kw in keywords if kw in line_lc)
            if " or " in line_lc or " and " in line_lc:
                score += 1
            if "true" in line_lc or "false" in line_lc:
                score += 1
            if "expected" in line_lc:
                score += 2
            if re.search(r"\[[^\]]+\]\[[^\]]+\]", line):
                score -= 2
            return score

        best = max(lines, key=score_line)
        best = re.sub(r"\s+", " ", best).strip()
        if len(best) > 240:
            best = best[:240].rstrip() + "..."
        if best and best[-1] not in {".", "!", "?"}:
            best += "."
        return self._enforce_answer_style(best)

    @staticmethod
    def _is_usable_answer(text: str) -> bool:
        if len(text) < 12 or len(text.split()) < 2:
            return False

        lower = text.lower()
        banned_starts = (
            "i don't know",
            "not enough information",
            "cannot determine",
        )
        if any(lower.startswith(prefix) for prefix in banned_starts):
            return False

        # Évite les sorties markdown du type [text](link) ou [text][ref].
        if re.search(r"\[[^\]]+\]\([^)]+\)", text):
            return False
        if re.search(r"\[[^\]]+\]\[[^\]]+\]", text):
            return False

        return True

    def _build_context(
        self,
        query: str,
        sources: list[MinimalSource],
        max_chars: int = 1200,
    ) -> str:
        pieces: list[str] = []
        total = 0

        for rank, source in enumerate(sources[:3], start=1):
            raw = self._read_source_text(source)
            relevant = self._extract_relevant_lines(query, raw)
            if not relevant:
                continue

            snippet = (
                f"[Source {rank} — {Path(source.file_path).name}]\n"
                f"{relevant}"
            )

            if total + len(snippet) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    snippet = snippet[:remaining].rstrip() + "..."
                else:
                    break

            pieces.append(snippet)
            total += len(snippet)

        return "\n\n".join(pieces)

    @staticmethod
    def _clean_response(text: str) -> str:
        cleaned = text.strip()

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if "<think>" in cleaned:
            cleaned = cleaned.split("<think>", 1)[0]

        cleaned = re.sub(r"^answer\s*:\s*", "", cleaned.strip(), flags=re.I)

        cleaned = re.sub(r"\n{2,}", " ", cleaned)

        cleaned = re.sub(r" {2,}", " ", cleaned).strip()

        if len(cleaned) > 600:
            truncated = cleaned[:600]
            last_dot = truncated.rfind(".")
            if last_dot > 200:
                cleaned = truncated[:last_dot + 1]
            else:
                cleaned = truncated.rstrip() + "..."

        if cleaned and cleaned[-1] not in {".", "!", "?"}:
            cleaned += "."

        return cleaned

    @staticmethod
    def _enforce_answer_style(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        cleaned = re.sub(r"\s+", " ", cleaned)

        # Garder une seule phrase (la première complète)
        sentence_end = re.search(r"[.!?]", cleaned)
        if sentence_end:
            cleaned = cleaned[:sentence_end.end()].strip()

        # Début simple et homogène avec le style du dataset
        if not cleaned.lower().startswith("the "):
            cleaned = f"The {cleaned[0].lower()}{cleaned[1:]}" \
                if len(cleaned) > 1 else f"The {cleaned.lower()}"

        return cleaned

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if hasattr(response, "model_dump"):
            try:
                response = response.model_dump()
            except Exception:
                pass

        if isinstance(response, dict):
            direct = response.get("response")
            if isinstance(direct, str) and direct.strip():
                return direct.strip()

            message = response.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        else:
            for attr in ("response", "content", "text"):
                val = getattr(response, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        return ""

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            "You are a precise assistant answering questions about the "
            "vLLM source code.\n"
            "\n"
            "Rules:\n"
            "- Answer in ONE complete English sentence.\n"
            "- Start with a simple direct lead-in (preferably 'The ...').\n"
            "- Use only information from the context below.\n"
            "- Keep important code identifiers when needed "
            "(e.g., do_not_compile, CompilationLevel.X).\n"
            "- Translate conditions clearly using words like when, or, and.\n"
            "- Do NOT include any reasoning, tags, or metadata.\n"
            "\n"
            "Style example:\n"
            "Question: What are the compilation levels that cause "
            "do_not_compile to be set to True in vLLM's compilation "
            "decorator?\n"
            "Answer: The do_not_compile flag is set to True when the "
            "compilation level is CompilationLevel.NO_COMPILATION or "
            "CompilationLevel.DYNAMO_AS_IS, or when dynamo is not supported, "
            "or when the class is ignored for torch compile.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def answer(self, query: str, sources: list[MinimalSource]) -> str:
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")
        if not sources:
            return "No supporting sources were found for this question."

        try:
            self._load_model()
        except Exception as error:
            return f"Model loading failed: {error}"

        if self._client is None:
            return "Model initialization failed."

        context = self._build_context(query, sources)
        if not context.strip():
            return "Sources were found but their content could not be read."

        prompt = self._build_prompt(query, context)

        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.2,
                },
            )
            text = self._clean_response(self._extract_response_text(response))
            text = self._enforce_answer_style(text)

            if self._is_usable_answer(text):
                return text

        except Exception as error:
            return f"Answer generation failed: {error}"

        try:
            retry_response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1,
                },
            )
            text = self._clean_response(
                self._extract_response_text(retry_response)
            )
            text = self._enforce_answer_style(text)
            if self._is_usable_answer(text):
                return text

        except Exception:
            pass

        fallback = self._fallback_from_context(query, context)
        if self._is_usable_answer(fallback):
            return fallback

        return "The model did not produce a usable answer for this question."
