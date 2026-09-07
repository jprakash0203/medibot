"""Standalone ingestion runner.

Run once (or whenever documents change):
    python scripts/ingest.py

Idempotent — re-running updates existing chunks rather than duplicating.
"""
import sys
from pathlib import Path

# Make `app` importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingestion import run

if __name__ == "__main__":
    run()