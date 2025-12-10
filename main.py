# main.py

import os
import sys
import uuid

from agents import Agent, Runner

from ingest_paper import ingest_paper
from section_summarizer import summarize_paper_sections
from db import init_db, get_connection, get_or_create_paper_id, insert_paper, insert_section_summaries, insert_rag_documents


def main(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    # 0) Ensure DB schema exists
    init_db()

    # 1) Create the LLM agent
    agent = Agent(
        name="SpatialTranscriptomicsAssistant",
        instructions=(
            "You are a spatial transcriptomics research assistant. "
            "You summarize scientific paper sections clearly and concisely, "
            "with emphasis on methods, datasets, tissues, and benchmarks."
        ),
    )

    print(f"\n[1/4] Ingesting PDF: {pdf_path}")
    ingested = ingest_paper(pdf_path)
    print(f"  - Pages extracted: {ingested.num_pages}")
    print(f"  - Figures extracted: {len(ingested.figures)}")

    # 2) Get a stable paper_id
    conn = get_connection()
    try:
        paper_id = get_or_create_paper_id(conn, ingested.source_path)
    finally:
        conn.close()

    print("\n[2/4] Summarizing sections and preparing RAG documents...")
    paper_summary, rag_docs = summarize_paper_sections(
        agent,
        ingested,
        paper_id=str(paper_id),
    )

    print("\n[3/4] Previewing section summaries:")
    for section in paper_summary.sections:
        print("\n======================================")
        print(f"SECTION: {section.section_name}")
        print("======================================")
        preview = section.summary_text[:800]
        print(preview)
        if len(section.summary_text) > 800:
            print("... [truncated]")

    print(f"\n[4/4] Saving to database...")
    conn = get_connection()
    try:
        with conn:
            # Nicer title later; for now just None
            insert_paper(conn, paper_id, ingested, title=None)
            insert_section_summaries(conn, paper_id, paper_summary)
            insert_rag_documents(conn, paper_id, rag_docs)
    finally:
        conn.close()

    print(f"Done. Total RAG documents created: {len(rag_docs)} and saved to DB.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path_arg = r"E:\WangResearch\EllaModel_Wang.pdf"
        print(f"No command-line argument detected. Using default PDF:\n{pdf_path_arg}")
    else:
        pdf_path_arg = sys.argv[1]

    main(pdf_path_arg)
