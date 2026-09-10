"""Cross-encoder reranker.

Takes an initial candidate set (from hybrid retrieval) and re-scores each
candidate by reading query + candidate text jointly. Returns the top-K
best-scoring candidates.

Why this matters:
  - Bi-encoders (dense/sparse) score query and document independently.
  - Cross-encoders concatenate them and run a transformer over both,
    catching relevance signals that similarity search misses.
  - 100–1000x slower per pair, so only used on top-N candidates, not
    on the full corpus.
"""
from dataclasses import dataclass
from typing import Sequence

from loguru import logger
from sentence_transformers import CrossEncoder


@dataclass
class RerankedCandidate:
    """Result item after reranking. `original_index` lets callers map back
    to the payload/metadata they retrieved in the previous step."""
    original_index: int
    text: str
    score: float


class Reranker:
    """Wraps a HuggingFace cross-encoder for query-document relevance scoring.

    Model default: BAAI/bge-reranker-base — ~280 MB, strong quality-to-speed
    ratio for English. Downloaded on first use, cached under ~/.cache/.
    """

    def __init__(self, model_name: str):
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        # `max_length=512` — same limit as our BGE-small dense encoder.
        # Anything longer gets truncated; usually fine because we already
        # chunked to <512 tokens at ingestion.
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 3,
    ) -> list[RerankedCandidate]:
        """Score each (query, candidate) pair jointly; return top_k by score.

        Score interpretation: higher = more relevant. Absolute values are
        model-specific (bge-reranker returns unbounded logits, roughly -10 to +10).
        What matters is the *ordering*, not the absolute number.
        """
        if not candidates:
            return []

        # `predict` takes pairs, returns one score per pair.
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Zip, sort by score descending, take top_k
        ranked = sorted(
            (
                RerankedCandidate(original_index=i, text=candidates[i], score=float(scores[i]))
                for i in range(len(candidates))
            ),
            key=lambda x: x.score,
            reverse=True,
        )
        return ranked[:top_k]