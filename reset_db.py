# reset_db.py

from db import get_connection, init_db


def reset_db():
    """
    Remove all data from our app tables in st_agent.
    Keeps the tables and schema (including embedding column).
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # Truncate in dependency-safe order with CASCADE
                cur.execute(
                    """
                    TRUNCATE TABLE rag_documents, section_summaries, papers
                    RESTART IDENTITY CASCADE;
                    """
                )
        print("All data cleared from papers, section_summaries, rag_documents.")
    finally:
        conn.close()

    # Ensure schema still exists/ok (safe to call)
    init_db()
    print("Schema checked/initialized.")


if __name__ == "__main__":
    reset_db()
