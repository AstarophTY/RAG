from pydantic import BaseModel
from .minimal import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]
