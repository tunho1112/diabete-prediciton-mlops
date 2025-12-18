import os

from dotenv import load_dotenv
from postgresql_client import PostgresSQLClient

POSTGRES_DB = "k6"
POSTGRES_USER = "k6"
POSTGRES_PASSWORD = "k6"

def main():
    pc = PostgresSQLClient(
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

    # Create devices table
    create_table_query = """
        CREATE TABLE IF NOT EXISTS diabetes (
        created TIMESTAMPTZ DEFAULT NOW(),
        diabete_id INTEGER,
        pregnancies INTEGER,
        glucose INTEGER,
        bloodpressure INTEGER,
        skinthickness INTEGER,
        insulin INTEGER,
        bmi DOUBLE PRECISION,
        diabetespedigreefunction DOUBLE PRECISION,
        age INTEGER
    );  
    """

    try:
        pc.execute_query(create_table_query)
    except Exception as e:
        print(f"Failed to create table with error: {e}")


if __name__ == "__main__":
    main()
