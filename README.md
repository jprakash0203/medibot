## Document Ingestion

Structural parsing via Docling + hierarchical chunking via HybridChunker.
Every chunk carries its parent section heading both in metadata AND
prepended into the chunk text (critical for retrieval quality).

Run once:
```bash
python scripts/ingest.py
python scripts/verify_ingestion.py
```

### Metadata schema (every chunk)

| Field | Example | Purpose |
|---|---|---|
| `source_document` | `drug_formulary.pdf` | Citation display |
| `collection` | `clinical` | Routing / filtering |
| `access_roles` | `["doctor", "admin"]` | RBAC filter (Qdrant metadata query) |
| `section_title` | `Amoxicillin > Adult Dosage` | Citation + context |
| `chunk_type` | `text` \| `table` \| `heading` \| `code` | Analytics |

### Storage

One Qdrant collection (`medibot_chunks`) with two named vectors per point:
- `dense` — 384-d BGE-small semantic vector
- `sparse` — BM25 keyword vector

This dual-vector setup enables Qdrant-native hybrid search in Phase 3 —
no application-level merging.