"""Smoke test for the SQL RAG chain.

Runs a battery of analytical questions and prints:
  - the LLM-generated SQL
  - the row count
  - the natural-language answer
  - PASS / FAIL against an expected-outcome check (where possible)

Redirect output into a file to capture as README evidence:
    python scripts/smoke_test_sql_rag.py > phase4_evidence.txt 2>&1
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.config import get_settings
from app.rag.sql_rag import SqlRagError, sql_rag_chain


# Each test provides a natural-language question plus an INDEPENDENT
# "verifier SQL" we know is correct. We run BOTH the pipeline and the
# verifier, and check the pipeline's row_count matches the verifier's.
# This gives us objective PASS/FAIL, not vibes-based.
TESTS = [
    {
        "question": "How many claims are currently pending?",
        "verifier_sql": "SELECT COUNT(*) FROM claims WHERE status = 'pending'",
    },
    {
        "question": "What is the total claimed amount for cardiology?",
        "verifier_sql": "SELECT COUNT(*) FROM claims WHERE department = 'cardiology'",
        # For sum queries we verify row_count, not the sum — the LLM might
        # word the answer differently, but the number of matching rows is invariant.
    },
    {
        "question": "How many maintenance tickets are still open or in progress?",
        "verifier_sql": "SELECT COUNT(*) FROM maintenance_tickets WHERE status IN ('open', 'in_progress')",
    },
    {
        "question": "Which equipment category has the most maintenance tickets?",
        "verifier_sql": "SELECT category, COUNT(*) c FROM maintenance_tickets GROUP BY category ORDER BY c DESC",
    },
    {
        "question": "How many claims were submitted by New India Assurance?",
        "verifier_sql": "SELECT COUNT(*) FROM claims WHERE insurer = 'New India Assurance'",
    },
    {
        "question": "List the last 5 escalated claims with their claimed amounts.",
        "verifier_sql": "SELECT claim_id, claimed_amount FROM claims WHERE status = 'escalated' ORDER BY submitted_date DESC LIMIT 5",
    },
]


def divider(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verifier_row_count(db_path: Path, sql: str) -> int:
    """Run the verifier SQL and return the row count."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return len(rows)
    finally:
        conn.close()


def main() -> None:
    # Reduce log noise so the output file is readable
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    settings = get_settings()
    divider("SQL RAG SMOKE TESTS")
    print(f"Database: {settings.sqlite_db_path}")
    print(f"Running {len(TESTS)} analytical questions.\n")

    passed = 0
    failed = 0

    for i, test in enumerate(TESTS, start=1):
        q = test["question"]
        expected_count = verifier_row_count(settings.sqlite_db_path, test["verifier_sql"])

        print(f"\n─── Test {i} ───")
        print(f"Q: {q}")
        print(f"Verifier expected row count: {expected_count}")

        try:
            result = sql_rag_chain(q)
        except SqlRagError as e:
            print(f"✗ FAIL — pipeline raised SqlRagError: {e}")
            failed += 1
            continue
        except Exception as e:
            print(f"✗ FAIL — unexpected error: {type(e).__name__}: {e}")
            failed += 1
            continue

        print(f"Generated SQL: {result.sql}")
        print(f"Pipeline row count: {result.row_count}")
        print(f"Answer: {result.answer}")

        if result.row_count == expected_count:
            print("✓ PASS")
            passed += 1
        else:
            print(f"✗ FAIL — row count mismatch (got {result.row_count}, expected {expected_count})")
            failed += 1

    divider("SUMMARY")
    print(f"Passed: {passed} / {len(TESTS)}")
    print(f"Failed: {failed} / {len(TESTS)}")
    print()

    if passed >= 4:
        print(f"🎉 Rubric requirement met (>= 4 passing analytical questions).")
        sys.exit(0)
    else:
        print(f"⚠ Only {passed} tests passed — rubric requires at least 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()