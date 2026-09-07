"""Dense (semantic) and sparse (BM25) embedding via FastEmbed.

FastEmbed runs models locally on CPU using ONNX Runtime — no GPU needed,
no API cost. First run downloads model weights (~200 MB dense, ~50 MB sparse)
and caches them under ~/.cache/fastembed/.
"""
from typing import Iterable

from fastembed import SparseTextEmbedding, TextEmbedding
from loguru import logger


class HybridEmbedder:
    """Produces dense vectors (semantic) and sparse vectors (BM25) for chunks.

    Design note: FastEmbed's `embed()` methods return generators — we materialize
    them to lists so callers can iterate multiple times. For large corpora
    (~10k+ chunks), stream them in batches instead.
    """

    def __init__(self, dense_model: str, sparse_model: str):
        logger.info(f"Loading dense embedding model: {dense_model}")
        self.dense = TextEmbedding(model_name=dense_model)
        logger.info(f"Loading sparse embedding model: {sparse_model}")
        self.sparse = SparseTextEmbedding(model_name=sparse_model)

        # Cache the dense vector dimension — Qdrant needs it at collection creation.
        # Trigger one embedding to discover the dim (avoids hardcoding).
        sample = list(self.dense.embed(["probe"]))
        self.dense_dim = len(sample[0])
        logger.info(f"Dense embedding dimension: {self.dense_dim}")

    def embed_dense(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one dense vector per input text."""
        return [vec.tolist() for vec in self.dense.embed(texts)]

    def embed_sparse(self, texts: Iterable[str]) -> list[dict]:
        """Return one sparse vector per input text.

        Each sparse vector is `{indices: [...], values: [...]}` — the format
        Qdrant expects for sparse vectors.
        """
        return [
            {"indices": vec.indices.tolist(), "values": vec.values.tolist()}
            for vec in self.sparse.embed(texts)
        ]
