import time
import numpy as np
from typing import List, Tuple

from openai import OpenAI
import psycopg2

from db import get_connection

# Use existing OpenAI API key from environment
client = OpenAI()


def fetch_unembedded_chunks(conn, batch_size: int = 50) -> List[Tuple[str, str]]:
    """
    Fetch a batch of rag_documents rows where embedding IS NULL.
    Returns a list of (doc_id, text).
    """
    sql = """
    SELECT doc_id, text
    FROM rag_documents
    WHERE embedding IS NULL
    ORDER BY doc_id
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (batch_size,))
        rows = cur.fetchall()
    return rows


def update_embeddings(conn, embeddings: List[np.ndarray], doc_ids: List[str]):
    """
    Store embeddings back into rag_documents.embedding as double precision[].
    """
    sql = """
    UPDATE rag_documents
    SET embedding = %s
    WHERE doc_id = %s;
    """
    with conn.cursor() as cur:
        for emb, doc_id in zip(embeddings, doc_ids):
            # Convert numpy array to Python list for psycopg2
            emb_list = emb.tolist()
            cur.execute(sql, (emb_list, doc_id))


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Call OpenAI embeddings API on a batch of texts.
    Returns a list of numpy arrays.
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    vectors = []
    for item in response.data:
        vectors.append(np.array(item.embedding, dtype=np.float32))
    return vectors


def main():
    conn = get_connection()
    try:
        total_embedded = 0

        while True:
            rows = fetch_unembedded_chunks(conn, batch_size=50)
            if not rows:
                print("No more unembedded rows. Done.")
                break

            doc_ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]

            print(f"Embedding batch of {len(texts)} documents...")
            embeddings = embed_texts(texts)

            with conn:
                update_embeddings(conn, embeddings, doc_ids)

            total_embedded += len(texts)
            print(f"Total embedded so far: {total_embedded}")

            # Small pause to be polite to API / avoid rate limits
            time.sleep(0.5)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
