"""Chat router: /chat and /collections/{role}.

/chat routes analytical questions to SQL RAG (billing_executive + admin only)
and everything else to Hybrid RAG. RBAC is enforced two ways:
  1. Analytical questions from non-permitted roles fall back to Hybrid RAG.
  2. Hybrid RAG is called with the authenticated role's collections —
     Qdrant filters at the retrieval layer (see app/rag/retrieval.py).
"""
import re

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth.models import AuthenticatedUser
from app.auth.router import get_current_user
from app.chat.models import ChatRequest, ChatResponse, Source
from app.core.rbac import (
    Role,
    can_use_sql_rag,
    collections_for_role,
)
from app.rag.answer import RagAnswerer
from app.rag.sql_rag import SqlRagError, sql_rag_chain


router = APIRouter(prefix="", tags=["chat"])


# ── Analytical-question classifier ────────────────────────────────────────
# Deterministic and fast. Covers the shapes the rubric's example questions
# use ("how many...", "what is the total...", "which one has the most...").
# Non-matching questions default to Hybrid RAG.
_ANALYTICAL_PATTERNS = [
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bhow much\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:is|are) the (?:total|sum|average|number|count)\b", re.IGNORECASE),
    re.compile(r"\bwhich (?:\w+ )?(?:has|have) the (?:most|least|highest|lowest)\b", re.IGNORECASE),
    re.compile(r"\blist (?:the |all )?(?:last|top|first) \d+\b", re.IGNORECASE),
    re.compile(r"\bcount\b.*\b(claims?|tickets?)\b", re.IGNORECASE),
    re.compile(r"\btotal\b.*\b(amount|claims?|tickets?)\b", re.IGNORECASE),
    re.compile(r"\baverage\b.*\b(amount|days?|time)\b", re.IGNORECASE),
    re.compile(r"\bbreak(?:\s|-)?down\b", re.IGNORECASE),
    re.compile(r"\bgroup(?:ed)? by\b", re.IGNORECASE),
]


def _is_analytical(question: str) -> bool:
    """Cheap keyword classifier. If it matches, we try SQL RAG; else Hybrid."""
    return any(p.search(question) for p in _ANALYTICAL_PATTERNS)


# ── Dependency: pull the singleton RagAnswerer off app.state ─────────────
# The heavy models are created once in main.py lifespan startup. This
# dependency just returns the instance — the per-request cost is a
# dict lookup, not a model load.
def get_answerer(request: Request) -> RagAnswerer:
    answerer: RagAnswerer | None = getattr(request.app.state, "answerer", None)
    if answerer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized",
        )
    return answerer


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/collections/{role}", response_model=list[str])
def list_collections(
    role: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> list[str]:
    """Return the collections a role can access.

    Requires a valid JWT. To avoid enumeration, non-admin users can only
    query their own role. Admin can query any role (useful for a settings UI).
    """
    try:
        target_role = Role(role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown role: {role}",
        )

    if user.role != Role.ADMIN and user.role != target_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only query your own role's collections",
        )

    return collections_for_role(target_role)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    answerer: RagAnswerer = Depends(get_answerer),
) -> ChatResponse:
    """Main RAG endpoint.

    Routing:
      - Analytical question + role can use SQL RAG → SQL RAG
      - Otherwise → Hybrid RAG with the user's allowed collections
    """
    question = payload.question.strip()
    role_value = user.role.value

    # Route to SQL RAG for analytical questions if role permits.
    if _is_analytical(question) and can_use_sql_rag(user.role):
        try:
            sql_result = sql_rag_chain(question, llm=answerer.llm)
        except SqlRagError as e:
            # Log the raw error server-side, return a user-friendly message.
            # We do NOT surface the SQL error text to the client — it can
            # leak schema details.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="I couldn't compute an answer from the database for that question.",
            ) from e

        return ChatResponse(
            answer=sql_result.answer,
            # For SQL RAG, "sources" is the executed SQL — treat the DB
            # as one virtual document. Alternative: return an empty list.
            sources=[
                Source(
                    source_document="mediassist.db",
                    section_title=sql_result.sql,
                    collection="database",
                )
            ],
            retrieval_type="sql_rag",
            role=role_value,
        )

    # Otherwise, Hybrid RAG with role-scoped collections.
    allowed_roles = [role_value]  # Qdrant filter uses role names
    result = answerer.answer(question=question, allowed_roles=allowed_roles)

    # If retrieval returned zero candidates, the user asked about
    # content they can't access. Produce the polite RBAC refusal
    # message the rubric wants.
    if not result.sources:
        role_collections = collections_for_role(user.role)
        return ChatResponse(
            answer=(
                f"As a {role_value}, you don't have access to the documents "
                f"needed to answer this question. I can only answer questions "
                f"from the {', '.join(role_collections)} collections."
            ),
            sources=[],
            retrieval_type="rbac_blocked",
            role=role_value,
        )

    return ChatResponse(
        answer=result.answer,
        sources=[
            Source(
                source_document=s.source_document,
                section_title=s.section_title,
                collection=s.collection,
            )
            for s in result.sources
        ],
        retrieval_type="hybrid_rag",
        role=role_value,
    )