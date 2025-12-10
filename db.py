# db.py
import os
import json
import uuid
from typing import List

import psycopg2

from ingest_paper import IngestedPaper
from section_summarizer import PaperSummary, RAGDocument


# --- Connection helper ---

def get_connection():
    """
    Create a new psycopg2 connection to the PostgreSQL database.
    Update the password to match your local postgres password.
    setx ST_AGENT_DB_PASSWORD "YOUR_POSTGRES_PASSWORD_HERE"
    """
    db_password = os.getenv("ST_AGENT_DB_PASSWORD")
    if not db_password:
        raise RuntimeError("ST_AGENT_DB_PASSWORD environment variable is not set.")
    
    conn = psycopg2.connect(
        dbname="st_agent",
        user="postgres",
        password=db_password,  # <-- CHANGE THIS
        host="localhost",
        port=5432,
    )
    return conn


# --- Schema setup ---

def init_db():
    """
    Create tables if they don't exist.
    Safe to call every time at startup.
    """
    schema_sql = """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id    UUID PRIMARY KEY,
        title       TEXT,
        source_path TEXT NOT NULL,
        num_pages   INTEGER NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS section_summaries (
        id           SERIAL PRIMARY KEY,
        paper_id     UUID REFERENCES papers(paper_id) ON DELETE CASCADE,
        section_name TEXT NOT NULL,
        summary_text TEXT NOT NULL,
        key_points   JSONB
    );

    -- ensure one summary per (paper, section)
    CREATE UNIQUE INDEX IF NOT EXISTS section_summaries_unique
    ON section_summaries (paper_id, section_name);

    CREATE TABLE IF NOT EXISTS rag_documents (
        doc_id       TEXT PRIMARY KEY,
        paper_id     UUID REFERENCES papers(paper_id) ON DELETE CASCADE,
        section_name TEXT NOT NULL,
        chunk_index  INTEGER NOT NULL,
        text         TEXT NOT NULL,
        metadata     JSONB
        -- embedding column will be added later
    );
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
    finally:
        conn.close()


# --- Paper ID helper (stable per PDF) ---

def get_or_create_paper_id(conn, source_path: str) -> uuid.UUID:
    """
    Look up a paper_id by source_path.
    If it exists, reuse it.
    If not, generate a new UUID (insert happens via insert_paper).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id FROM papers WHERE source_path = %s;",
            (source_path,),
        )
        row = cur.fetchone()

    if row:
        return uuid.UUID(row[0])

    return uuid.uuid4()


# --- Insert helpers ---

def insert_paper(conn, paper_id: uuid.UUID, ingested: IngestedPaper, title: str | None = None):
    """
    Insert (or upsert) a paper row.
    """
    sql = """
    INSERT INTO papers (paper_id, title, source_path, num_pages)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (paper_id) DO UPDATE
    SET title = EXCLUDED.title,
        source_path = EXCLUDED.source_path,
        num_pages = EXCLUDED.num_pages;
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                str(paper_id),
                title,
                ingested.source_path,
                ingested.num_pages,
            ),
        )


def insert_section_summaries(conn, paper_id: uuid.UUID, paper_summary: PaperSummary):
    """
    Insert all section summaries for a given paper.
    Uses upsert to avoid duplicates.
    """
    sql = """
    INSERT INTO section_summaries (paper_id, section_name, summary_text, key_points)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (paper_id, section_name) DO UPDATE
    SET summary_text = EXCLUDED.summary_text,
        key_points   = EXCLUDED.key_points;
    """

    with conn.cursor() as cur:
        for section in paper_summary.sections:
            key_points_json = json.dumps(section.key_points) if section.key_points else None
            cur.execute(
                sql,
                (
                    str(paper_id),
                    section.section_name,
                    section.summary_text,
                    key_points_json,
                ),
            )


def insert_rag_documents(conn, paper_id: uuid.UUID, rag_docs: List[RAGDocument]):
    """
    Insert all RAGDocument rows for a given paper.
    Upserts on doc_id.
    """
    sql = """
    INSERT INTO rag_documents (doc_id, paper_id, section_name, chunk_index, text, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (doc_id) DO UPDATE
    SET section_name = EXCLUDED.section_name,
        chunk_index  = EXCLUDED.chunk_index,
        text         = EXCLUDED.text,
        metadata     = EXCLUDED.metadata;
    """

    with conn.cursor() as cur:
        for rd in rag_docs:
            cur.execute(
                sql,
                (
                    rd.doc_id,
                    str(paper_id),
                    rd.metadata.get("section_name"),
                    int(rd.metadata.get("chunk_index", 0)),
                    rd.text,
                    json.dumps(rd.metadata),
                ),
            )
