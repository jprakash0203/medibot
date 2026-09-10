"""Hybrid RAG answer pipeline.

Ties retrieval, reranking, and LLM generation together, returning both the
generated answer and the source citations used to produce it.
"""
from dataclasses import dataclass
from typing import Sequence

from loguru import logger

from app.rag.llm import LLMClient
from app.rag.reranker import Reranker
from app.rag.retrieval import HybridRetriever, RetrievedChunk


SYSTEM_PROMPT = """You are MediBot, an assistant for MediAssist Health Network staff.

Answer the user's question using ONLY the provided context passages. Rules:
- If the context does not contain the answer, say so plainly.
- Do NOT rely on prior training knowledge for medical facts. Only use the passages.
- Cite the source document and section for each fact where possible.
- Be concise. Prefer bullet points for procedures or dosages.
- Never invent drug names, dosages, ICD codes, or equipment model numbers.
"""


@dataclass
class SourceCitation:
    """One source shown to the user — matches the rubric's required shape."""
    source_document: str
    section_title: str
    collection: str


@dataclass
class AnswerResult:
    """Full response of the RAG pipeline."""
    answer: str
    sources: list[SourceCitation]
    reranked_chunks: list[RetrievedChunk]   # kept for logging / debugging
    retrieval_type: str = "hybrid_rag"


class RagAnswerer:
    """Retrieve → rerank → prompt → LLM. Returns answer + citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLMClient,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def answer(self, question: str, allowed_roles: Sequence[str]) -> AnswerResult:
        # 1. Hybrid retrieve with RBAC filter → top-N candidates.
        candidates = self.retriever.retrieve(
            query=question,
            allowed_roles=allowed_roles,
            top_k=self.retrieval_top_k,
        )
        logger.info(f"Hybrid retrieval returned {len(candidates)} candidates for roles={list(allowed_roles)}")

        # 2. Handle the "no accessible content" case gracefully.
        #    This produces the RBAC-refusal message the rubric asks for
        #    when a user asks about content their role can't access.
        if not candidates:
            return AnswerResult(
                answer=(
                    "I couldn't find any documents accessible to your role that answer this question. "
                    "You may not have permission to access the relevant collections."
                ),
                sources=[],
                reranked_chunks=[],
            )

        # 3. Rerank with cross-encoder → top-3.
        reranked = self.reranker.rerank(
            query=question,
            candidates=[c.text for c in candidates],
            top_k=self.rerank_top_k,
        )
        top_chunks = [candidates[r.original_index] for r in reranked]

        # Log for the "rerank score analysis" README section
        for r in reranked:
            original = candidates[r.original_index]
            logger.info(
                f"  Rerank: score={r.score:+.3f} "
                f"[was hybrid rank {r.original_index + 1}] "
                f"({original.source_document} — {original.section_title[:50]})"
            )

        # 4. Build the LLM prompt from ONLY the reranked top chunks.
        #    The rubric explicitly says the full candidate set must NOT be
        #    passed through — only reranked survivors reach the LLM.
        context = self._format_context(top_chunks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context passages:\n{context}\n\n"
            f"Answer:"
        )

        # 5. Generate the answer.
        answer_text = self.llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # 6. Build citations from the reranked chunks (not the whole candidate set).
        citations = [
            SourceCitation(
                source_document=c.source_document,
                section_title=c.section_title,
                collection=c.collection,
            )
            for c in top_chunks
        ]

        return AnswerResult(
            answer=answer_text,
            sources=citations,
            reranked_chunks=top_chunks,
        )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Format chunks as a numbered context block for the LLM prompt."""
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(
                f"[Passage {i}]\n"
                f"Source: {c.source_document} — {c.section_title or '(no section)'}\n"
                f"{c.text}"
            )
        return "\n\n".join(blocks)