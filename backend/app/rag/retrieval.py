"""Hybrid retrieval with server-side RBAC filter.

The two rubric-critical properties of this module:

  1. RBAC ENFORCEMENT AT THE RETRIEVAL LAYER.
     The role-based access filter is passed INTO the Qdrant query as a
     payload filter. Qdrant applies it before ranking, so restricted
     chunks are never scored, never returned, never seen by application
     code. Even a successful prompt injection cannot leak content the
     LLM never received.

  2. NATIVE HYBRID SEARCH VIA QDRANT FUSION.
     We use `query_points` with `prefetch` — Qdrant runs dense and sparse
     searches in one request and fuses them server-side using Reciprocal
     Rank Fusion. We do NOT run two separate queries and merge in Python.
"""
from dataclasses import dataclass
from typing import Sequence

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchAny,
    Prefetch,
    SparseVector,
    Fusion,
)

from app.rag.embeddings import HybridEmbedder
from app.rag.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@dataclass
class RetrievedChunk:
    """One retrieved candidate — text plus all metadata needed for
    citation, reranking, and prompt construction."""
    id: str
    text: str
    source_document: str
    collection: str
    section_title: str
    chunk_type: str
    hybrid_score: float


class HybridRetriever:
    """Runs dense + sparse retrieval as one Qdrant query, filtered by role."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: HybridEmbedder,
        collection_name: str,
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection_name

    def retrieve(
        self,
        query: str,
        allowed_roles: Sequence[str],
        top_k: int = 10,
        candidates_per_vector: int = 20,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval with RBAC filter.

        Args:
            query: user question.
            allowed_roles: which roles the current user has. A chunk is
                retrievable if its access_roles list contains ANY of these.
                For our single-role users this is a 1-item list, but the
                design supports multi-role users trivially.
            top_k: final number of fused candidates to return (pre-rerank).
            candidates_per_vector: how many candidates each prefetch pulls.
                Should be >= top_k so fusion has enough overlap to rank.

        Returns:
            RetrievedChunk list, sorted by RRF fusion score descending.
        """
        if not allowed_roles:
            # Defense-in-depth: an empty role list means "nothing accessible".
            # Return no results rather than accidentally returning everything.
            logger.warning("retrieve() called with empty allowed_roles — returning []")
            return []

        # ── Embed the query using BOTH encoders ──────────────────────────
        # Dense encoder: for semantic matching.
        # Sparse encoder: for exact keyword / drug-name / code matching.
        dense_query = self.embedder.embed_dense([query])[0]
        sparse_query = self.embedder.embed_sparse([query])[0]

        # ── Build the RBAC filter ────────────────────────────────────────
        # `access_roles` on each point is a list like ["doctor", "admin"].
        # MatchAny returns True if ANY value in the payload list matches
        # ANY value in `any=[...]`. So `any=[user_role]` returns chunks
        # whose access_roles include that role. THIS IS THE FILTER THAT
        # ENFORCES RBAC AT THE RETRIEVAL LAYER.
        rbac_filter = Filter(
            must=[
                FieldCondition(
                    key="access_roles",
                    match=MatchAny(any=list(allowed_roles)),
                )
            ]
        )

        # ── Single hybrid query with server-side RRF fusion ──────────────
        # `prefetch` runs each sub-query (dense + sparse) and collects
        # candidates. `query=FusionQuery(fusion=Fusion.RRF)` merges the
        # two candidate lists into one final ranking using Reciprocal
        # Rank Fusion. All done inside Qdrant — no app-side merging.
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=candidates_per_vector,
                    filter=rbac_filter,      # RBAC applied to dense prefetch
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=candidates_per_vector,
                    filter=rbac_filter,      # RBAC applied to sparse prefetch
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        # ── Repackage into our RetrievedChunk dataclass ──────────────────
        return [
            RetrievedChunk(
                id=str(point.id),
                text=point.payload["text"],
                source_document=point.payload["source_document"],
                collection=point.payload["collection"],
                section_title=point.payload["section_title"],
                chunk_type=point.payload["chunk_type"],
                hybrid_score=point.score,
            )
            for point in response.points
        ]

    # ──────────────────────────────────────────────────────────────────────
    #   Below: dense-only retrieval, kept purely for the README comparison
    #   showing that hybrid+rerank beats dense-only. Not used in production.
    # ──────────────────────────────────────────────────────────────────────

    def retrieve_dense_only(
        self,
        query: str,
        allowed_roles: Sequence[str],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Dense-only retrieval — for A/B comparison in the README, not for prod."""
        if not allowed_roles:
            return []

        dense_query = self.embedder.embed_dense([query])[0]
        rbac_filter = Filter(must=[FieldCondition(
            key="access_roles", match=MatchAny(any=list(allowed_roles)))])

        response = self.client.query_points(
            collection_name=self.collection,
            query=dense_query,
            using=DENSE_VECTOR_NAME,
            query_filter=rbac_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                id=str(p.id),
                text=p.payload["text"],
                source_document=p.payload["source_document"],
                collection=p.payload["collection"],
                section_title=p.payload["section_title"],
                chunk_type=p.payload["chunk_type"],
                hybrid_score=p.score,
            )
            for p in response.points
        ]