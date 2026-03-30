from pydantic import BaseModel


class Chunk(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str
    chunk_type: str
