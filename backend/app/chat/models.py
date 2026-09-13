"""Chat DTOs. Response shape follows the rubric exactly:
answer + sources + retrieval_type + role."""
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    """One source citation. Field names match rubric exactly."""
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: Literal["hybrid_rag", "sql_rag", "rbac_blocked"]
    role: str