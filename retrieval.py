# retrieval.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI

from db import get_connection
from agents import Agent, Runner


# ----- Data model for retrieved docs -----

@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    score: float  # similarity score


# ----- OpenAI client (for query embeddings) -----

client = OpenAI()  # uses OPENAI_API_KEY from environment


# ----- Embedding helpers -----

def embed_query(query: str) -> np.ndarray:
    """
    Embed a user question using the same model you used for rag_documents.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec


# ----- DB fetch helpers -----

def fetch_candidate_docs(
    paper_id: Optional[str] = None,
) -> List[Tuple[str, str, dict, np.ndarray]]:
    """
    Fetch all rag_documents that have an embedding, optionally filtered by paper_id.

    Returns a list of (doc_id, text, metadata, embedding_vector).
    """
    conn = get_connection()
    try:
        if paper_id:
            sql = """
            SELECT doc_id, text, metadata, embedding
            FROM rag_documents
            WHERE embedding IS NOT NULL
              AND paper_id = %s;
            """
            params = (paper_id,)
        else:
            sql = """
            SELECT doc_id, text, metadata, embedding
            FROM rag_documents
            WHERE embedding IS NOT NULL;
            """
            params = ()

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        docs: List[Tuple[str, str, dict, np.ndarray]] = []
        for doc_id, text, metadata, emb_list in rows:
            # psycopg2 returns embedding as a Python list (double precision[])
            emb_vec = np.array(emb_list, dtype=np.float32)
            docs.append((doc_id, text, metadata, emb_vec))

        return docs

    finally:
        conn.close()


# ----- Similarity + retrieval -----

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def retrieve_top_k(
    query: str,
    paper_id: Optional[str] = None,
    k: int = 5,
) -> List[RetrievedDoc]:
    """
    Embed the query, fetch candidate docs from DB, compute cosine similarity,
    and return top-k RetrievedDoc objects.
    """
    query_vec = embed_query(query)
    candidates = fetch_candidate_docs(paper_id=paper_id)

    if not candidates:
        return []

    scored: List[RetrievedDoc] = []
    for doc_id, text, metadata, emb_vec in candidates:
        score = cosine_similarity(query_vec, emb_vec)
        scored.append(
            RetrievedDoc(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                score=score,
            )
        )

    # Sort by score descending and return top-k
    scored.sort(key=lambda d: d.score, reverse=True)
    return scored[:k]


# ----- QA using Agent + retrieved docs -----

def answer_question_with_rag(
    agent: Agent,
    question: str,
    paper_id: Optional[str] = None,
    k: int = 5,
    max_context_chars: int = 8000,
) -> Tuple[str, List[RetrievedDoc]]:
    """
    High-level helper:

    - Retrieve top-k docs for the question (optionally limited to one paper)
    - Build a context block
    - Ask the Agent to answer using that context
    - Return (answer_text, retrieved_docs)
    """
    retrieved = retrieve_top_k(question, paper_id=paper_id, k=k)

    if not retrieved:
        return "I couldn't find any relevant chunks in the database.", []

    # Build context string from retrieved docs, truncated to max_context_chars
    context_pieces = []
    total_len = 0

    for doc in retrieved:
        chunk_header = f"[Doc: {doc.doc_id} | Score: {doc.score:.3f}]\n"
        chunk_text = doc.text.strip()
        piece = chunk_header + chunk_text + "\n\n"

        if total_len + len(piece) > max_context_chars:
            # stop if adding this would exceed our context budget
            break

        context_pieces.append(piece)
        total_len += len(piece)

    context_block = "".join(context_pieces)

    prompt = f"""
You are a spatial transcriptomics research assistant.

You will be given a user question and a set of retrieved text chunks from
spatial transcriptomics papers. Use ONLY the information in these chunks as
evidence. If the answer is uncertain or not fully supported, say so.

Question:
{question}

Retrieved context:
\"\"\" 
{context_block}
\"\"\"

Now answer the question as clearly and concisely as possible. Where relevant,
mention specific methods, datasets, or comparisons and base them on the context.
"""

    result = Runner.run_sync(agent, prompt)
    answer = result.final_output.strip()
    return answer, retrieved


# ----- Simple CLI entrypoint for testing -----

def main():
    """
    Quick test runner:
    - Creates an Agent
    - Asks a hard-coded test question
    - Prints answer + which docs were used
    """
    # You can change this question or make it interactive
    test_question = (
        "What are the main advantages of ELLA compared to previous methods "
        "for modeling subcellular spatial variation of gene expression?"
    )

    # If you want to restrict to one paper, set paper_id here:
    # paper_id = "b9000f83-2642-4f82-9e90-9022ad1ed0ce"
    paper_id = None  # search across all indexed papers

    agent = Agent(
        name="SpatialTranscriptomicsQAAssistant",
        instructions=(
            "You are a spatial transcriptomics research assistant. "
            "You answer questions using provided context from scientific papers "
            "and you avoid hallucinating details that are not supported."
        ),
    )

    print(f"Question: {test_question}\n")
    answer, retrieved = answer_question_with_rag(
        agent,
        test_question,
        paper_id=paper_id,
        k=5,
    )

    print("=== ANSWER ===")
    print(answer)
    print("\n=== RETRIEVED CHUNKS USED ===")
    for doc in retrieved:
        print(f"- {doc.doc_id}  (score={doc.score:.3f}, section={doc.metadata.get('section_name')})")


if __name__ == "__main__":
    main()
