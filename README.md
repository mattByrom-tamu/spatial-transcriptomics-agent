# Spatial Transcriptomics Paper Agent

This project ingests spatial transcriptomics papers (PDFs), extracts text and figures,
summarizes sections with an LLM, and stores RAG-ready chunks in PostgreSQL for
question answering. The end goal will be to build an  agent that can refer to multiple papers 
in order to output a literature review comparing multiple type of methodology. 

## Components

- `ingest_paper.py` – PDF parsing and image extraction using PyMuPDF.
- `section_summarizer.py` – chunking + LLM-based section summarization + RAG docs.
- `db.py` – PostgreSQL schema + insert helpers.
- `main.py` – end-to-end pipeline: ingest → summarize → store.
- `embed_rag_docs.py` – generate and store embeddings for RAG chunks.
- `retrieval.py` – query-time retrieval and QA over stored chunks.

## Requirements

- Python (Anaconda)
- PostgreSQL (local), database `st_agent`
- Environment variables:
  - `OPENAI_API_KEY` – OpenAI API key
  - `ST_AGENT_DB_PASSWORD` – password for the local postgres user

## Usage

```bash
# Ingest and index a paper
python main.py "paper directory"

# Embed stored chunks
python embed_rag_docs.py

# Ask a question using RAG
python retrieval.py
