class TextChunker:
    @staticmethod
    def extract_chunk(text: str, chunk_size: int = 512, overlap: int = 100):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        final_chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                final_chunks.append(paragraph)
            else:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunk = paragraph[start:end]

                    final_chunks.append(chunk)
                    start += chunk_size - overlap

                    if (start >= len(paragraph) - overlap
                            and len(paragraph) > chunk_size):
                        break

        return final_chunks
