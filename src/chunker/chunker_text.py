class TextChunker:
    @staticmethod
    def extract_chunks(
        text: str,
        chunk_size: int = 512,
        overlap: int = 100
    ) -> list[tuple[str, int, int]]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        final_chunks = []
        search_start = 0

        for paragraph in paragraphs:
            p_start_idx = text.find(paragraph, search_start)
            p_end_idx = p_start_idx + len(paragraph)
            search_start = p_end_idx

            if len(paragraph) <= chunk_size:
                final_chunks.append((paragraph, p_start_idx, p_end_idx))
            else:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunk_content = paragraph[start:end]

                    global_start = p_start_idx + start
                    global_end = p_start_idx + len(chunk_content)

                    final_chunks.append((
                        chunk_content,
                        global_start,
                        global_end
                    ))

                    start += chunk_size - overlap

                    if start >= len(paragraph) - overlap:
                        break

        return final_chunks
