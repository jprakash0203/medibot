"""Qdrant vector store wrapper for MediBot.

One collection stores dense + sparse vectors as **named vectors** on each
point. This is what enables Qdrant-native hybrid search in Phase 3 —
we don't run two queries and merge in Python.
"""
from typing import Iterable
from uuid import uuid5, NAMESPACE_URL

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.rag.schemas import Chunk

# Names for the two vectors on each point.
# The retrieval code in Phase 3 will reference these by name.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorStore:
    """Owns the medibot_chunks collection.

    Responsibilities:
      - Ensure the collection exists with correct schema (dense + sparse named vectors)
      - Upsert chunks with all required metadata
      - Provide deterministic IDs so re-runs update rather than duplicate
    """

    def __init__(self, client: QdrantClient, collection_name: str, dense_dim: int):
        self.client = client
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            logger.info(f"Collection '{self.collection_name}' already exists.")
            return

        logger.info(f"Creating collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            # Dense vectors — one named vector called "dense".
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            # Sparse vectors — one named sparse vector called "sparse".
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict],
        batch_size: int = 64,
    ) -> None:
        """Upsert chunks with both dense and sparse vectors + payload.

        Uses deterministic UUIDs derived from source_document + chunk index
        so re-ingesting the same document *updates* points rather than
        creating duplicates. This is what makes the script idempotent.
        """
        assert len(chunks) == len(dense_vectors) == len(sparse_vectors), \
            "Chunks, dense vectors, and sparse vectors must be same length"

        points = []
        for i, (chunk, dvec, svec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            # Deterministic ID: same source + chunk position → same UUID.
            point_id = str(uuid5(NAMESPACE_URL, f"{chunk.metadata.source_document}::{i}"))

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dvec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=svec["indices"],
                            values=svec["values"],
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        **chunk.metadata.model_dump(),
                    },
                )
            )

        # Batch upserts to avoid hitting request size limits.
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)
            logger.info(f"  Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    def count(self) -> int:
        """Return total number of points in the collection."""
        return self.client.count(collection_name=self.collection_name, exact=True).count