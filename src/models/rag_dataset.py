from pydantic import BaseModel
from .answered_question import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
