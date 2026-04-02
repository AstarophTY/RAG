import re
from pathlib import Path
from typing import Any

from src.chunker.chunk import Chunk
from src.chunker.save_chunks import SaveChunks
from src.models import MinimalSource


class Generator:
    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int = 180,
        ollama_host: str | None = None,
    ) -> None:
        """Initialize generator configuration and preload chunk lookup."""
        self.model_name = model_name or "qwen3:0.6b"
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host
        self._client: Any | None = None
        self._chunk_lookup = self._build_chunk_lookup()

    def set_model(
        self,
        model_name: str,
        max_new_tokens: int | None = None,
    ) -> None:
        """Switch the generation model and optionally token budget."""
        new_name = model_name.strip()
        if not new_name:
            raise ValueError("model_name cannot be empty.")

        self.model_name = new_name
        if max_new_tokens is not None:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be greater than 0.")
            self.max_new_tokens = max_new_tokens

    @staticmethod
    def _build_chunk_lookup() -> dict[tuple[str, int, int], str]:
        """Build a dictionary mapping source spans to their chunk content."""
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
        """Lazily create and validate the Ollama client instance."""
        if self._client is not None:
            return
        import importlib
        ollama_module = importlib.import_module("ollama")
        client_cls = getattr(ollama_module, "Client")
        self._client = client_cls(host=self.ollama_host) \
            if self.ollama_host else client_cls()
        self._client.list()

    def _read_source_text(self, source: MinimalSource) -> str:
        """Read source text for a span.

        Prefer cached chunk content when available.
        """
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
        """Extract non-trivial query keywords.

        Common stop words are removed.
        """
        stop = {
            "what", "which", "when", "where", "how", "with", "from",
            "that", "this", "does", "must", "have", "true", "false",
            "the", "are", "for", "vllm", "is", "in", "to", "a", "an",
            "and", "or", "of", "by", "set", "get", "value", "values",
        }
        words = re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())
        return {w for w in words if w not in stop}

    def _extract_relevant_lines(self, query: str, content: str) -> str:
        """Select and rank the most relevant lines from source content."""
        keywords = self._query_keywords(query)
        lines = content.splitlines()
        scored: list[tuple[int, str]] = []

        for raw_line in lines:
            line = raw_line.strip()

            if len(line) < 5:
                continue
            if line.startswith(('"""', "'''", "import ", "from ", "#")):
                continue

            line_lc = line.lower()
            score = 0

            for kw in keywords:
                if kw in line_lc:
                    score += 2

            if "=" in line and "==" not in line:
                score += 1

            if "(" in line and ")" in line:
                score += 1

            if any(
                line_lc.startswith(
                    prefix) for prefix in ("if ", "elif ", "for ", "while ")
            ):
                if any(kw in line_lc for kw in keywords):
                    score += 1

            if score > 0:
                scored.append((score, line))

        if not scored:
            return content[:800].strip()

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(line for _, line in scored[:10])

    @staticmethod
    def _is_usable_answer(text: str) -> bool:
        """Check whether generated text is suitable as a final answer."""
        if len(text) < 12 or len(text.split()) < 4:
            return False

        lower = text.lower()

        banned_starts = (
            "i don't know",
            "i do not know",
            "not enough information",
            "cannot determine",
            "unable to determine",
            "the context does not",
            "there is no information",
        )
        if any(lower.startswith(prefix) for prefix in banned_starts):
            return False

        banned_code_like_starts = (
            "the if ",
            "the for ",
            "the while ",
            "the return ",
            "the elif ",
            "the else",
            "the def ",
            "the class ",
        )
        if any(lower.startswith(prefix) for prefix in banned_code_like_starts):
            return False

        if re.search(r"\[[^\]]+\]\([^)]+\)", text):
            return False
        if re.search(r"\[[^\]]+\]\[[^\]]+\]", text):
            return False

        return True

    def _build_context(
        self,
        query: str,
        sources: list[MinimalSource],
        max_chars: int = 2000,
    ) -> str:
        """Assemble a bounded textual context from top retrieved sources."""
        pieces: list[str] = []
        total = 0

        for rank, source in enumerate(sources[:4], start=1):
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
        """Normalize and truncate raw model output into one clean response."""
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

        if (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in {'"', "'"}
        ):
            cleaned = cleaned[1:-1].strip()

        cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
        cleaned = re.sub(r"([.!?])\s*([.!?])+", r"\1", cleaned)

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
        """Normalize answer formatting while keeping concise detail."""
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        cleaned = re.sub(
            r"^(here(?:'s| is)\s+the\s+answer\s*[:\-]?|"
            r"the\s+answer\s+is\s*[:\-]?|"
            r"it\s+is\s+that\s+|"
            r"in\s+this\s+context,\s*)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(r"\s+", " ", cleaned)

        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]", cleaned))
        if sentence_matches:
            if len(sentence_matches) >= 2:
                cleaned = cleaned[:sentence_matches[1].end()].strip()
            else:
                cleaned = sentence_matches[0].group(0).strip()

        if cleaned and cleaned[-1] not in {".", "!", "?"}:
            cleaned += "."

        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        if cleaned and not re.match(
            r"^(The|When|If|All|Both|This|These)\b",
            cleaned,
        ):
            cleaned = f"The answer is that {cleaned[0].lower() + cleaned[1:]}"

        return cleaned

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from multiple possible model response shapes."""
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

    def _generate(self, prompt: str, options: dict[str, Any]) -> Any:
        """Call the model with reasoning disabled when supported."""
        if self._client is None:
            raise RuntimeError("Model initialization failed.")

        try:
            return self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                think=False,
                options=options,
            )
        except TypeError:
            return self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options=options,
            )

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Build the generation prompt from query and context."""
        return (
            "You are a precise assistant answering questions about the "
            "vLLM source code.\n"
            "\n"
            "Rules:\n"
            "- Answer in 1 to 2 complete English sentences.\n"
            "- Match the exact style of the reference dataset answers: short, "
            "direct, factual, and neutral.\n"
            "- Be specific and factual: include exact values, variable names, "
            "and conditions from the code.\n"
            "- Prefer patterns like: 'The default value ... is ...', "
            "'When ... , ...', or 'The method ... returns ...'.\n"
            "- Use natural language to describe code logic clearly.\n"
            "- Use only information from the context below.\n"
            "- Keep important code identifiers exactly as they appear "
            "(e.g., requires_grad=False, CompilationLevel.NO_COMPILATION).\n"
            "- If the answer involves multiple items, list them clearly.\n"
            "- Do NOT include any reasoning, explanations, tags, or "
            "metadata.\n"
            "- Do NOT invent information not present in the context.\n"
            "\n"
            "Examples:\n"
            "Q: What is the requires_grad value set for parameters in the "
            "process_weights_after_loading method?\n"
            "A: All parameters (B_24, s, B_meta, and workspace) are set "
            "with requires_grad=False in the "
            "process_weights_after_loading method.\n"
            "\n"
            "Q: What compilation levels cause do_not_compile to be True?\n"
            "A: The do_not_compile flag is set to True when the compilation "
            "level is CompilationLevel.NO_COMPILATION or "
            "CompilationLevel.DYNAMO_AS_IS, or when dynamo is not supported.\n"
            "\n"
            "Q: What is the default value of max_model_len?\n"
            "A: The default value of max_model_len is None.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def answer(self, query: str, sources: list[MinimalSource]) -> str:
        """Generate an answer using retrieved supporting sources."""
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")
        if not sources:
            raise RuntimeError(
                "No supporting sources were found for this question."
            )

        try:
            self._load_model()
        except Exception as error:
            raise RuntimeError(f"Model loading failed: {error}")

        if self._client is None:
            raise RuntimeError("Model initialization failed.")

        context = self._build_context(query, sources)
        if not context.strip():
            raise RuntimeError(
                "Sources were found but their content could not be read."
            )

        prompt = self._build_prompt(query, context)

        try:
            response = self._generate(
                prompt=prompt,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            )
            text = self._clean_response(self._extract_response_text(response))
            text = self._enforce_answer_style(text)

            if self._is_usable_answer(text):
                return text

        except Exception as error:
            raise RuntimeError(f"Answer generation failed: {error}")

        try:
            retry_response = self._generate(
                prompt=prompt,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.3,
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

        try:
            final_response = self._generate(
                prompt=prompt,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "repeat_penalty": 1.15,
                },
            )
            text = self._clean_response(
                self._extract_response_text(final_response)
            )
            text = self._enforce_answer_style(text)

            if self._is_usable_answer(text):
                return text

        except Exception:
            pass

        return "The model was unable to generate a \
satisfactory answer for this question."
