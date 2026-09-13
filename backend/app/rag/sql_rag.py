"""SQL RAG chain.

Natural-language question → LLM-generated SQL → executed against SQLite →
LLM turns result back into natural language. Three explicit steps, per rubric.

Design decisions:
  - Read-only SQLite connection. The LLM cannot damage data even if it
    generates DDL/DML — the connection refuses it.
  - Schema shown to the LLM includes VALUE EXAMPLES for TEXT columns
    (e.g. `status IN ('pending', 'approved', ...)`). Without this the
    LLM guesses casing and returns zero rows.
  - Raw SQL is cleaned via regex + validation before executing. LLMs
    love to wrap output in markdown fences or prefix it with commentary.
  - Empty results are handled explicitly — the LLM must not invent numbers.
"""
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.rag.llm import LLMClient


# ── Schema description shown to the LLM ────────────────────────────────────
# This is hand-written on purpose. Auto-generated schema (INFORMATION_SCHEMA
# style) doesn't include value examples, and without value examples the LLM
# guesses casing and enum values wrong. This is the single highest-leverage
# thing you can do for text-to-SQL quality.
SCHEMA_DESCRIPTION = """
Database: SQLite. Two tables.

Table: claims (billing claims)
  claim_id           TEXT PRIMARY KEY    e.g. 'CLM-2024-1000'
  patient_id         TEXT                e.g. 'PAT-51347'
  patient_name       TEXT                e.g. 'Kavya Pillai'
  department         TEXT                one of: 'cardiology', 'nephrology',
                                          'neurology', 'orthopedics', 'oncology',
                                          'general_medicine', 'pediatrics',
                                          'gastroenterology', 'pulmonology'
  claim_type         TEXT                one of: 'cashless', 'reimbursement'
  diagnosis_code     TEXT                ICD-10 codes, e.g. 'I21.4', 'N17.9'
  insurer            TEXT                e.g. 'New India Assurance', 'Bajaj Allianz',
                                          'United India', 'Star Health', 'HDFC Ergo'
  claimed_amount     REAL                in INR, e.g. 72700.0
  approved_amount    REAL                in INR, NULL if not approved yet
  status             TEXT                one of: 'pending', 'approved',
                                          'rejected', 'escalated'
  submitted_date     TEXT                ISO date, e.g. '2024-01-26'
  resolved_date      TEXT                ISO date, NULL if unresolved

Table: maintenance_tickets (equipment maintenance)
  ticket_id          TEXT PRIMARY KEY    e.g. 'TKT-2024-2000'
  equipment_name     TEXT                e.g. 'SterilPro 3000', 'DriveFlow IP-200'
  equipment_id       TEXT                e.g. 'EQ-HC-3588'
  category           TEXT                one of: 'sterilisation', 'infusion',
                                          'imaging', 'monitoring', 'diagnostic',
                                          'respiratory'
  campus             TEXT                e.g. 'MediAssist Hyderabad Central'
  issue_type         TEXT                one of: 'preventive_maintenance',
                                          'sensor_failure', 'battery_replacement',
                                          'calibration', 'firmware_update',
                                          'physical_damage'
  fault_code         TEXT                e.g. 'F-05', NULL if none
  raised_by          TEXT                staff name
  raised_date        TEXT                ISO date
  resolved_date      TEXT                ISO date, NULL if open
  status             TEXT                one of: 'open', 'in_progress', 'resolved'
  resolution_note    TEXT                free text, NULL if unresolved

Important rules for writing SQL:
  - All string comparisons are case-sensitive in SQLite. Use exact casing
    as shown above (e.g. 'pending' not 'Pending').
  - Dates are stored as ISO TEXT ('YYYY-MM-DD'). Compare with string
    operators or use SQLite date functions: date(submitted_date),
    strftime('%Y-%m', submitted_date), etc.
  - SQLite has NO 'INTERVAL' keyword. Use date('now', '-30 days') or
    date('{today}', '-30 days') instead. Today's date is provided below.
  - A "resolved" ticket has status = 'resolved'. An "open" ticket has
    status IN ('open', 'in_progress').
  - "Last month" means the previous calendar month (e.g. if today is
    2025-03-14, last month is 2025-02-01 to 2025-02-28).
"""


TEXT_TO_SQL_PROMPT = """You are a SQL expert. Convert the user's question into a
single SQLite SELECT statement.

{schema}

Today's date is: {today}

Rules:
  - Return ONLY the SQL statement. No markdown, no explanation, no code fences.
  - Use SELECT only. Never write INSERT, UPDATE, DELETE, DROP, or any DDL.
  - Use exact column names and exact string values from the schema.
  - If the question is ambiguous, make the most reasonable interpretation and proceed.
  - End the statement with a semicolon.

Question: {question}

SQL:"""


ANSWER_FROM_RESULT_PROMPT = """You are an analyst summarizing a query result for a
non-technical user.

Original question: {question}

SQL executed:
{sql}

Result rows ({row_count} rows):
{result_preview}

Write a concise, natural-language answer. Rules:
  - State the numeric answer clearly if the question was a count / sum / average.
  - If the result is empty, say so plainly — do not invent numbers.
  - Do not restate the SQL. Do not mention 'the query'.
  - Keep it under 3 sentences unless the user asked for a breakdown.

Answer:"""


# Sentinel status values returned to the caller when the pipeline aborts.
class SqlRagError(Exception):
    """Raised when the pipeline cannot produce a valid answer."""


@dataclass
class SqlRagResult:
    """Structured return type — used by the /chat endpoint in Phase 5."""
    answer: str
    sql: str
    row_count: int


# ── Step 2 helpers: SQL extraction and validation ─────────────────────────

MARKDOWN_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Matches "Here's the SQL:", "SQL Query:", "Answer:", etc — anything before SELECT
PREAMBLE_RE = re.compile(r"^.*?(?=\bSELECT\b)", re.IGNORECASE | re.DOTALL)
FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "replace", "attach", "detach", "pragma",
}


def _extract_sql(raw: str) -> str:
    """Pull a single, executable SELECT statement out of the LLM's output.

    Defends against: markdown fences, preamble text, trailing explanations,
    multiple statements. Raises SqlRagError if nothing valid is found.
    """
    text = raw.strip()

    # 1. If the LLM wrapped it in ```sql ... ```, take the fenced content.
    fence_match = MARKDOWN_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Strip anything before the first SELECT ("Here is the SQL: SELECT ...")
    text = PREAMBLE_RE.sub("", text).strip()

    # 3. Must start with SELECT after cleaning.
    if not text.lower().startswith("select"):
        raise SqlRagError(f"LLM did not produce a SELECT statement. Raw output: {raw!r}")

    # 4. Only keep the first statement — split on semicolon, take first non-empty.
    #    (SQLite refuses multi-statements via execute() anyway, but we're
    #     explicit about it here so the error message is helpful.)
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if not parts:
        raise SqlRagError(f"No statement after cleaning. Raw output: {raw!r}")
    sql = parts[0] + ";"

    # 5. Belt-and-braces: reject any forbidden keyword. This catches an LLM
    #    that produces `SELECT ... ; DROP TABLE claims;` — the first statement
    #    is a valid SELECT so extraction would keep it, but we still want to
    #    log and reject if we see something like `SELECT ... UNION DELETE ...`.
    lowered = sql.lower()
    for kw in FORBIDDEN_KEYWORDS:
        # Word-boundary match so 'created_date' doesn't trip on 'create'.
        if re.search(rf"\b{kw}\b", lowered):
            raise SqlRagError(f"SQL contains forbidden keyword '{kw}': {sql!r}")

    return sql


# ── Step 3 helpers: read-only execution ───────────────────────────────────

def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open SQLite in read-only mode via URI.

    A read-only connection is our second line of defense: even if the
    LLM produced destructive SQL and our extraction missed it, SQLite
    refuses to execute anything that would modify data or schema.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row  # dict-like access to columns
    return conn


def _execute(conn: sqlite3.Connection, sql: str, row_limit: int = 100
             ) -> list[dict]:
    """Execute the SQL and return rows as dicts. Capped at row_limit."""
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchmany(row_limit)
    return [dict(r) for r in rows]


def _format_result_preview(rows: list[dict], limit: int = 20) -> str:
    """Human-readable preview of rows for the answer-generation prompt.

    Kept small — for a 'count' query it's usually one row, for a listing
    query we cap at `limit` rows to keep the prompt manageable.
    """
    if not rows:
        return "(no rows)"
    preview_rows = rows[:limit]
    # Simple pipe-separated formatting — cheaper on tokens than JSON.
    headers = list(preview_rows[0].keys())
    lines = [" | ".join(headers)]
    for r in preview_rows:
        lines.append(" | ".join("" if r[h] is None else str(r[h]) for h in headers))
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


# ── The public function ───────────────────────────────────────────────────

def sql_rag_chain(question: str, llm: LLMClient | None = None) -> SqlRagResult:
    """Natural language → SQL → result → natural language.

    Three explicit steps per rubric. Read-only DB access. Returns the
    final answer plus the SQL that was executed (used for /chat's
    sources field in Phase 5).
    """
    settings = get_settings()
    llm = llm or LLMClient()

    # ── Step 1: NL → SQL ─────────────────────────────────────────────
    text_to_sql_prompt = TEXT_TO_SQL_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        today=date.today().isoformat(),
        question=question,
    )
    raw_sql = llm.complete(
        system_prompt="You are a precise SQL generator. Output only SQL.",
        user_prompt=text_to_sql_prompt,
        temperature=0.0,   # deterministic — text-to-SQL should not be creative
        max_tokens=300,
    )
    logger.info(f"LLM raw SQL output: {raw_sql!r}")

    # ── Step 2: Clean & validate ─────────────────────────────────────
    sql = _extract_sql(raw_sql)
    logger.info(f"Cleaned SQL: {sql}")

    # ── Step 3: Execute → NL answer ──────────────────────────────────
    conn = _open_readonly(settings.sqlite_db_path)
    try:
        rows = _execute(conn, sql)
    except sqlite3.Error as e:
        # Surface a helpful error to the caller. The /chat endpoint in
        # Phase 5 will turn this into a user-friendly message.
        raise SqlRagError(f"SQL execution failed: {e}. SQL was: {sql}") from e
    finally:
        conn.close()

    logger.info(f"SQL returned {len(rows)} rows")

    answer_prompt = ANSWER_FROM_RESULT_PROMPT.format(
        question=question,
        sql=sql,
        row_count=len(rows),
        result_preview=_format_result_preview(rows),
    )
    natural_answer = llm.complete(
        system_prompt="You are a clear, concise analyst.",
        user_prompt=answer_prompt,
        temperature=0.1,
        max_tokens=400,
    )

    return SqlRagResult(answer=natural_answer, sql=sql, row_count=len(rows))