"""Post-ingestion sanity checks. Run after scripts/ingest.py."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from app.config import get_settings
from app.core.rbac import Collection

REQUIRED_METADATA = {"source_document", "collection", "access_roles",
                     "section_title", "chunk_type", "text"}


def main() -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    coll = settings.qdrant_collection

    # 1. Collection exists and has vectors config
    info = client.get_collection(coll)
    print(f"✓ Collection '{coll}' has {info.points_count} points")
    assert "dense" in info.config.params.vectors, "Missing 'dense' named vector!"
    assert "sparse" in info.config.params.sparse_vectors, "Missing 'sparse' named vector!"
    print("✓ Both dense and sparse named vectors are configured")

    # 2. Sample 100 points and verify metadata schema
    sample, _ = client.scroll(collection_name=coll, limit=100, with_payload=True)
    assert sample, "No points found!"

    for point in sample:
        missing = REQUIRED_METADATA - set(point.payload.keys())
        assert not missing, f"Point {point.id} missing metadata: {missing}"
        assert isinstance(point.payload["access_roles"], list) and point.payload["access_roles"], \
            f"Point {point.id} has empty access_roles"
        assert point.payload["collection"] in [c.value for c in Collection], \
            f"Point {point.id} has invalid collection: {point.payload['collection']}"
    print(f"✓ All 100 sampled points have complete metadata schema")

    # 3. Collection distribution — every collection should have chunks
    all_points, _ = client.scroll(collection_name=coll, limit=10_000, with_payload=["collection"])
    dist = Counter(p.payload["collection"] for p in all_points)
    print(f"✓ Chunk distribution: {dict(dist)}")
    for c in Collection:
        assert dist.get(c.value, 0) > 0, f"No chunks for collection '{c.value}'!"
    print("✓ Every collection has at least one chunk")

    # 4. Spot check: an ingested chunk includes its section title
    text_sample, _ = client.scroll(collection_name=coll, limit=5, with_payload=True)
    print("\nSample chunks (first 200 chars of each):")
    for p in text_sample:
        print(f"  [{p.payload['collection']}] {p.payload['source_document']}")
        print(f"    section: {p.payload['section_title'] or '(none)'}")
        print(f"    text: {p.payload['text'][:200]}...\n")

    print("\n🎉 Ingestion verified.")


if __name__ == "__main__":
    main()