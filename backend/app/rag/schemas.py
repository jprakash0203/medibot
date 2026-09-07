"""Pydantic schemas for the ingestion & retrieval pipeline.

Keeping these in one file gives us a single source of truth for the shape
of a chunk as it flows from parser → embedder → Qdrant → retriever.
"""
from typing import Literal

from pydantic import BaseModel, Field

ChunkType = Literal["text", "table", "heading", "code"]


class ChunkMetadata(BaseModel):
    """The 5 required metadata fields per the assignment rubric.

    Stored as Qdrant point payload. Used for:
      - `access_roles`: RBAC filter at retrieval time
      - `collection`: routing / filtering
      - `source_document`, `section_title`: citation display in UI
      - `chunk_type`: analytics / debugging
    """
    source_document: str = Field(..., description="Original filename, e.g. 'drug_formulary.pdf'")
    collection: str = Field(..., description="One of: general, clinical, nursing, billing, equipment")
    access_roles: list[str] = Field(..., min_length=1, description="Roles permitted to see this chunk")
    section_title: str = Field(default="", description="Parent heading path, e.g. 'Amoxicillin > Adult Dosage'")
    chunk_type: ChunkType = Field(default="text")


class Chunk(BaseModel):
    """A single chunk ready to be embedded and stored."""
    text: str = Field(..., min_length=1, description="Chunk text WITH section title prepended")
    metadata: ChunkMetadata