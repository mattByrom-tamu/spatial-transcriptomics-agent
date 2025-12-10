import psycopg2

try:
    conn = psycopg2.connect(
        dbname="st_agent",
        user="postgres",
        password="riven",
        host="localhost",
        port=5432,
    )
    print("✔ Connected to st_agent successfully!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:")
    print(e)