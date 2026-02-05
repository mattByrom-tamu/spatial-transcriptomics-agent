# batch_ingest_papers_rag.py

import os
from typing import List

from ingest_paper import ingest_paper
from section_summarizer import build_rag_docs_only
from db import (
    init_db,
    get_connection,
    get_or_create_paper_id,
    insert_paper,
    insert_rag_documents,
)

# CHANGE this to your actual folder of PDFs
PDF_FOLDER = r"E:\WangResearch\Papers for GPT"


def find_pdfs(folder: str) -> List[str]:
    """
    Return full paths of all .pdf files in the given folder.
    """
    pdf_paths: List[str] = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(folder, name))
    return pdf_paths


def process_single_paper(pdf_path: str):
    """
    Ingest and index a single PDF in the database (no LLM).
    """
    print(f"\n=== Processing PDF: {pdf_path} ===")

    # 1) Ingest
    ingested = ingest_paper(pdf_path)
    print(f"  - Pages extracted: {ingested.num_pages}")
    print(f"  - Figures extracted: {len(ingested.figures)}")

    # 2) Get stable paper_id based on source_path
    conn = get_connection()
    try:
        paper_id = get_or_create_paper_id(conn, ingested.source_path)
    finally:
        conn.close()

    # 3) Build RAG chunks (no summaries, no Agent)
    print("  - Building RAG chunks...")
    rag_docs = build_rag_docs_only(
        ingested,
        paper_id=str(paper_id),
    )

    # 4) Save to DB
    conn = get_connection()
    try:
        with conn:
            insert_paper(conn, paper_id, ingested, title=None)
            insert_rag_documents(conn, paper_id, rag_docs)
    finally:
        conn.close()

    print(f"  - Stored {len(rag_docs)} RAG chunks for this paper.")


def main():
    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder does not exist: {PDF_FOLDER}")

    # 0) Ensure schema exists
    init_db()

    pdf_paths = find_pdfs(PDF_FOLDER)
    if not pdf_paths:
        print(f"No PDFs found in folder: {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_paths)} PDF(s) in {PDF_FOLDER}.")

    for pdf_path in pdf_paths:
        process_single_paper(pdf_path)

    print("\nAll PDFs processed.")


if __name__ == "__main__":
    main()
