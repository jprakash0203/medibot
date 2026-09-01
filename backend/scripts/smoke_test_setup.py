import sqlite3
import sys
from pathlib import Path

# Make `app` importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from app.config import get_settings
from app.core.rbac import Role, collections_for_role, can_use_sql_rag


def main() -> None:
    settings = get_settings()
    print(f"✓ Config loaded. LLM provider: {settings.llm_provider}, model: {settings.llm_model}")

    # RBAC sanity
    print(f"✓ Nurse collections: {collections_for_role(Role.NURSE)}")
    print(f"✓ Admin collections: {collections_for_role(Role.ADMIN)}")
    print(f"✓ Nurse SQL RAG? {can_use_sql_rag(Role.NURSE)} (should be False)")
    print(f"✓ Admin SQL RAG? {can_use_sql_rag(Role.ADMIN)} (should be True)")

    # Data files present
    assert settings.data_dir.exists(), f"Missing data dir: {settings.data_dir}"
    pdfs = list(settings.data_dir.rglob("*.pdf"))
    print(f"✓ Found {len(pdfs)} PDFs")

    # SQLite DB reachable
    conn = sqlite3.connect(settings.sqlite_db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"✓ SQLite tables: {[t[0] for t in tables]}")
    conn.close()

    # Qdrant reachable
    client = QdrantClient(url=settings.qdrant_url)
    collections = client.get_collections()
    print(f"✓ Qdrant reachable. Existing collections: {[c.name for c in collections.collections]}")

    print("\n🎉 Phase 1 setup complete!")


if __name__ == "__main__":
    main()