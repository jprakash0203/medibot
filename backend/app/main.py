"""FastAPI application entry point.

The `lifespan` context manager runs at startup and shutdown — we use it
to warm up the heavy models (Docling not needed at query time, but
the FastEmbed dense/sparse encoders and the cross-encoder reranker are).
Loading them once at startup means the first user's /chat call is fast,
not a cold-start 30-second wait.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from app.auth.router import router as auth_router
from app.chat.router import router as chat_router
from app.config import get_settings
from app.rag.answer import RagAnswerer
from app.rag.embeddings import HybridEmbedder
from app.rag.llm import LLMClient
from app.rag.reranker import Reranker
from app.rag.retrieval import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown hooks.

    Startup: load models, build the RagAnswerer, stash on app.state.
    Shutdown: nothing special (Qdrant client uses HTTP, no connection to close).
    """
    settings = get_settings()
    logger.info("Starting MediBot backend...")

    # Fail fast if the vector collection doesn't exist yet.
    # Better a clear error at startup than an obscure one on first /chat.
    qdrant = QdrantClient(url=settings.qdrant_url)
    existing = {c.name for c in qdrant.get_collections().collections}
    if settings.qdrant_collection not in existing:
        raise RuntimeError(
            f"Qdrant collection '{settings.qdrant_collection}' not found. "
            f"Run `python scripts/ingest.py` first."
        )

    logger.info("Warming up embedder...")
    embedder = HybridEmbedder(
        dense_model=settings.dense_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )

    logger.info("Warming up reranker...")
    reranker = Reranker(settings.reranker_model)

    llm = LLMClient()

    retriever = HybridRetriever(
        client=qdrant,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
    )

    answerer = RagAnswerer(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )
    app.state.answerer = answerer
    logger.info("MediBot backend ready.")

    yield

    logger.info("Shutting down MediBot backend.")


app = FastAPI(
    title="MediBot",
    description="Advanced RAG assistant for MediAssist Health Network",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS for the Next.js frontend ────────────────────────────────────────
# In dev the frontend runs on localhost:3000; in prod you'd narrow this
# to the actual deployment origin. Never use "*" with allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness check. Deliberately does NOT hit Qdrant or the LLM —
    kubectl / load balancer probes should be cheap. Add a /readiness
    if you want a deeper check."""
    return {"status": "ok"}


# Register routers
app.include_router(auth_router)
app.include_router(chat_router)