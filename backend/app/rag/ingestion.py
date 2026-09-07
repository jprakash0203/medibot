"""Ingestion pipeline: PDFs on disk → parsed chunks → embedded → Qdrant.

Iterates every file in the data directory, classifies it by collection
(from its folder name), attaches RBAC metadata (from rbac.py), parses via
Docling, embeds via FastEmbed, and upserts to Qdrant.
"""
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.core.rbac import Collection, roles_for_collection
from app.rag.embeddings import HybridEmbedder
from app.rag.parser import DocumentParser
from app.rag.schemas import Chunk, ChunkMetadata
from app.rag.vector_store import VectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def build_metadata_for_file(file_path: Path, data_root: Path) -> ChunkMetadata:
    """Derive collection + access_roles from the file's folder location.

    Convention: files live under `{data_root}/{collection_name}/{filename}`,
    so the collection is the immediate parent folder.
    """
    # Get the folder name (e.g., 'clinical', 'billing')
    collection_name = file_path.relative_to(data_root).parts[0]

    # Validate against our known collections — fail loudly if unexpected
    try:
        collection = Collection(collection_name)
    except ValueError as e:
        raise ValueError(
            f"File {file_path} is under unknown collection '{collection_name}'. "
            f"Expected one of: {[c.value for c in Collection]}"
        ) from e

    # Look up which roles can access this collection
    access_roles = roles_for_collection(collection)

    return ChunkMetadata(
        source_document=file_path.name,
        collection=collection.value,
        access_roles=access_roles,
        section_title="",   # filled in per chunk by the parser
        chunk_type="text",  # filled in per chunk by the parser
    )


def ingest_all(data_dir: Path, vector_store: VectorStore,
               parser: DocumentParser, embedder: HybridEmbedder) -> None:
    """Full ingestion run: walk data_dir, parse, embed, upsert. Idempotent."""

    files = sorted(
        f for f in data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No PDF/MD files found under {data_dir}")

    logger.info(f"Found {len(files)} files to ingest.")

    total_chunks = 0
    for file_path in files:
        try:
            metadata_template = build_metadata_for_file(file_path, data_dir)
        except ValueError as e:
            logger.error(f"Skipping {file_path.name}: {e}")
            continue

        # Parse this file into chunks
        chunks: list[Chunk] = list(parser.parse(file_path, metadata_template))
        if not chunks:
            logger.warning(f"No chunks produced from {file_path.name}")
            continue

        # Embed all chunks (dense + sparse) in a batch — much faster than one-by-one
        texts = [c.text for c in chunks]
        logger.info(f"  Embedding {len(texts)} chunks (dense + sparse)...")
        dense_vecs = embedder.embed_dense(texts)
        sparse_vecs = embedder.embed_sparse(texts)

        # Upsert into Qdrant
        logger.info(f"  Upserting {len(chunks)} chunks into Qdrant...")
        vector_store.upsert_chunks(chunks, dense_vecs, sparse_vecs)
        total_chunks += len(chunks)

    logger.info(f"✓ Ingestion complete. Total chunks upserted: {total_chunks}")
    logger.info(f"✓ Collection now contains {vector_store.count()} points.")


def run() -> None:
    """Entry point used by scripts/ingest.py."""
    settings = get_settings()

    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.qdrant_url)

    embedder = HybridEmbedder(
        dense_model=settings.dense_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )

    vector_store = VectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        dense_dim=embedder.dense_dim,
    )
    vector_store.ensure_collection()

    parser = DocumentParser(max_tokens=500)  # under BGE-small's 512 limit

    ingest_all(
        data_dir=settings.data_dir,
        vector_store=vector_store,
        parser=parser,
        embedder=embedder,
    )